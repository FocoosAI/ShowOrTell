import logging
import os
import pickle
from collections import namedtuple

import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

Label = namedtuple("Label", ["name", "id", "color"])


class DatasetVOC(Dataset):
    def __init__(
        self,
        datapath,
        transform,
        split,
        nprompts,
        use_original_imgsize,
        benchmark="voc",
    ):
        self.split = "val" if split in ["val", "test"] else "trn"
        self.nclass = 21
        self.benchmark = benchmark
        self.nprompts = nprompts
        self.base_path = os.path.join(datapath, "VOCdevkit", "VOC2012")
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize
        self.ignore_label = 255
        self.ignore_background = False
        self.pixel_percentage = 10

        if self.ignore_background:
            self.class_ids = np.arange(1, self.nclass + 1)
        else:
            self.class_ids = np.arange(0, self.nclass)

        self.__get_img_metadata()
        self.__get_class()

    def __get_img_metadata(self):
        with open("./data/splits/pascalvoc_train.pkl", "rb") as f:
            img_list_classwise_trn, img_list_trn = pickle.load(f)

        with open("./data/splits/pascalvoc_val.pkl", "rb") as f:
            img_list_classwise_val, img_list_val = pickle.load(f)

        self.img_list_classwise_trn = img_list_classwise_trn
        self.img_list_trn = img_list_trn
        self.img_list_classwise_val = img_list_classwise_val
        self.img_list_val = img_list_val

    def __get_class(self):
        self.labels = [
            Label("background", 0, (0, 0, 0)),
            Label("aeroplane", 1, (128, 0, 0)),
            Label("bicycle", 2, (0, 128, 0)),
            Label("bird", 3, (128, 128, 0)),
            Label("boat", 4, (0, 0, 128)),
            Label("bottle", 5, (128, 0, 128)),
            Label("bus", 6, (0, 128, 128)),
            Label("car", 7, (128, 128, 128)),
            Label("cat", 8, (64, 0, 0)),
            Label("chair", 9, (192, 0, 0)),
            Label("cow", 10, (64, 128, 0)),
            Label("diningtable", 11, (192, 128, 0)),
            Label("dog", 12, (64, 0, 128)),
            Label("horse", 13, (192, 0, 128)),
            Label("motorbike", 14, (64, 128, 128)),
            Label("person", 15, (192, 128, 128)),
            Label("pottedplant", 16, (0, 64, 0)),
            Label("sheep", 17, (128, 64, 0)),
            Label("sofa", 18, (0, 192, 0)),
            Label("train", 19, (128, 192, 0)),
            Label("tvmonitor", 20, (0, 64, 128)),
        ]
        self.class_names = [label.name for label in self.labels]
    def id_2_label(self, id):
        id2label = {label.id: label for label in self.labels}
        return id2label[id]

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, "SegmentationClass", name)
        mask = torch.tensor(np.array(Image.open(mask_path)))
        return mask

    def load_support_set(self, pixels_percentage=2):
        assert len(self.img_list_trn) >= (self.nprompts * self.nclass), (
            "Not enough training images to generate the support set. Lower the number of nprompts or increase the number of training images."
        )

        support_set = []
        warning_classes = []
        for class_id in tqdm(self.class_ids):
            if len(self.img_list_classwise_trn[class_id]) < self.nprompts:
                warning_classes.append(int(class_id))
            support_names = []
            support_imgs = []
            support_masks = []
            while len(support_names) < self.nprompts:
                support_name = np.random.choice(
                    self.img_list_classwise_trn[class_id], 1, replace=False
                )[0]

                if (
                    len(self.img_list_classwise_trn[class_id]) >= self.nprompts
                    and support_name in support_names
                ):
                    continue

                support_img = Image.open(
                    os.path.join(
                        self.base_path, "JPEGImages", support_name.replace("png", "jpg")
                    )
                ).convert("RGB")
                support_img = self.transform(support_img)

                support_mask = self.read_mask(support_name)
                # This three steps are necessary to convert the mask to binary mask since 0 is the mask for a class.
                # support_mask[support_mask != class_id] = -1
                # support_mask[support_mask == class_id] = 1
                # support_mask[support_mask == -1] = 0
                support_mask = support_mask == class_id

                if pixels_percentage != 0:
                    pixels_proportion = (
                        support_mask.shape[0]
                        * support_mask.shape[1]
                        * (pixels_percentage / 100)
                    )
                    if support_mask.sum() <= pixels_proportion:
                        continue

                support_mask = F.interpolate(
                    support_mask.unsqueeze(0).unsqueeze(0).float(),
                    support_img.size()[-2:],
                    mode="nearest",
                ).squeeze()

                support_imgs.append(support_img)
                support_masks.append(support_mask)
                support_names.append(support_name)

            support_set.append(
                {
                    "support_imgs": torch.stack(support_imgs),
                    "support_masks": torch.stack(support_masks),
                    "support_names": support_names,
                }
            )

        if len(warning_classes) > 0:
            logging.warning(
                f"Class {warning_classes} have less than {self.nprompts} distinct training images."
            )
        return support_set

    def __len__(self):
        return len(self.img_metadata) if self.split == "trn" else len(self.img_list_val)

    def __getitem__(self, idx):
        query_name = self.img_list_val[idx]
        query_img = Image.open(
            os.path.join(self.base_path, "JPEGImages", query_name + ".jpg")
        ).convert("RGB")
        query_mask = self.read_mask(query_name + ".png")

        org_qry_imsize = query_img.size

        query_img = self.transform(query_img)

        query_mask = query_mask.float()
        if not self.use_original_imgsize:
            query_mask = F.interpolate(
                query_mask.unsqueeze(0).unsqueeze(0).float(),
                query_img.size()[-2:],
                mode="nearest",
            ).squeeze()

        batch = {
            "query_img": query_img,
            "query_mask": query_mask,
            "query_name": query_name,
            "org_query_imsize": org_qry_imsize,
        }

        return batch

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


class DatasetADE(Dataset):
    def __init__(
        self,
        datapath,
        transform,
        split,
        nprompts,
        use_original_imgsize,
        benchmark="ade",
    ):
        self.split = "val" if split in ["val", "test"] else "trn"
        self.nclass = 150
        self.benchmark = benchmark
        self.nprompts = nprompts
        self.base_path = os.path.join(datapath, "ADEChallengeData2016")
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize
        self.ignore_background = True
        self.pixel_percentage = 4

        if self.ignore_background:
            self.class_ids = np.arange(1, self.nclass + 1)
        else:
            self.class_ids = np.arange(0, self.nclass)

        self.__get_img_metadata()
        self.__get_class()

    def __get_img_metadata(self):
        with open("./data/splits/ade_train.pkl", "rb") as f:
            img_list_classwise_trn, img_list_trn = pickle.load(f)

        with open("./data/splits/ade_val.pkl", "rb") as f:
            img_list_classwise_val, img_list_val = pickle.load(f)

        self.img_list_classwise_trn = img_list_classwise_trn
        self.img_list_trn = img_list_trn
        self.img_list_classwise_val = img_list_classwise_val
        self.img_list_val = img_list_val

    def __get_class(self):
        self.labels = [
            Label("background", 0, (0, 0, 0)),
            Label("wall", 1, (120, 120, 120)),
            Label("building;edifice", 2, (180, 120, 120)),
            Label("sky", 3, (135, 206, 235)),
            Label("floor", 4, (209, 204, 155)),
            Label("tree", 5, (34, 139, 34)),
            Label("ceiling", 6, (255, 255, 255)),
            Label("road", 7, (128, 128, 128)),
            Label("bed", 8, (139, 69, 19)),
            Label("windowpane", 9, (173, 216, 230)),
            Label("grass", 10, (0, 128, 0)),
            Label("cabinet", 11, (139, 69, 19)),
            Label("sidewalk", 12, (169, 169, 169)),
            Label("person", 13, (255, 228, 225)),
            Label("bicycle", 14, (255, 0, 0)),
            Label("car", 15, (255, 0, 0)),
            Label("motorcycle", 16, (255, 215, 0)),
            Label("airplane", 17, (169, 169, 169)),
            Label("bus", 18, (255, 215, 0)),
            Label("train", 19, (205, 133, 63)),
            Label("truck", 20, (128, 128, 0)),
            Label("boat", 21, (0, 0, 128)),
            Label("traffic light", 22, (255, 0, 0)),
            Label("fire hydrant", 23, (255, 0, 0)),
            Label("street sign", 24, (255, 255, 255)),
            Label("pole", 25, (105, 105, 105)),
            Label("landmark", 26, (255, 215, 0)),
            Label("bench", 27, (160, 82, 45)),
            Label("bird", 28, (255, 255, 255)),
            Label("cat", 29, (0, 0, 0)),
            Label("dog", 30, (255, 255, 255)),
            Label("horse", 31, (139, 69, 19)),
            Label("sheep", 32, (255, 255, 255)),
            Label("cow", 33, (255, 255, 255)),
            Label("elephant", 34, (169, 169, 169)),
            Label("bear", 35, (139, 69, 19)),
            Label("zebra", 36, (255, 255, 255)),
            Label("giraffe", 37, (255, 255, 255)),
            Label("hat", 38, (255, 228, 225)),
            Label("laptop", 39, (0, 0, 0)),
            Label("mouse", 40, (0, 0, 0)),
            Label("remote", 41, (169, 169, 169)),
            Label("keyboard", 42, (0, 0, 0)),
            Label("cell phone", 43, (0, 0, 0)),
            Label("microwave", 44, (169, 169, 169)),
            Label("oven", 45, (169, 169, 169)),
            Label("toaster", 46, (169, 169, 169)),
            Label("sink", 47, (169, 169, 169)),
            Label("refrigerator", 48, (169, 169, 169)),
            Label("book", 49, (255, 255, 255)),
            Label("clock", 50, (169, 169, 169)),
            Label("vase", 51, (255, 215, 0)),
            Label("scissors", 52, (169, 169, 169)),
            Label("teddy bear", 53, (255, 228, 225)),
            Label("hair drier", 54, (169, 169, 169)),
            Label("toothbrush", 55, (255, 255, 255)),
            Label("scissors", 56, (169, 169, 169)),
            Label("computer", 57, (0, 0, 0)),
            Label("plate", 58, (255, 255, 255)),
            Label("bottle", 59, (0, 0, 0)),
            Label("can", 60, (255, 255, 255)),
            Label("cup", 61, (255, 255, 255)),
            Label("glasses", 62, (0, 0, 0)),
            Label("pan", 63, (169, 169, 169)),
            Label("pot", 64, (169, 169, 169)),
            Label("lamp", 65, (255, 255, 255)),
            Label("fork", 66, (169, 169, 169)),
            Label("spoon", 67, (169, 169, 169)),
            Label("knife", 68, (169, 169, 169)),
            Label("plate", 69, (255, 255, 255)),
            Label("table", 70, (160, 82, 45)),
            Label("chair", 71, (160, 82, 45)),
            Label("towel", 72, (255, 255, 255)),
            Label("cup", 73, (255, 255, 255)),
            Label("teapot", 74, (255, 255, 255)),
            Label("furniture", 75, (160, 82, 45)),
            Label("plate", 76, (255, 255, 255)),
            Label("tray", 77, (255, 255, 255)),
            Label("furniture", 78, (160, 82, 45)),
            Label("bottle", 79, (255, 255, 255)),
            Label("cup", 80, (255, 255, 255)),
            Label("plate", 81, (255, 255, 255)),
            Label("book", 82, (255, 255, 255)),
            Label("bottle", 83, (255, 255, 255)),
            Label("can", 84, (255, 255, 255)),
            Label("clock", 85, (169, 169, 169)),
            Label("mirror", 86, (169, 169, 169)),
            Label("plate", 87, (255, 255, 255)),
            Label("lamp", 88, (255, 255, 255)),
            Label("spoon", 89, (169, 169, 169)),
            Label("fork", 90, (169, 169, 169)),
            Label("knife", 91, (169, 169, 169)),
            Label("clock", 92, (169, 169, 169)),
            Label("mirror", 93, (169, 169, 169)),
            Label("keyboard", 94, (0, 0, 0)),
            Label("cell phone", 95, (0, 0, 0)),
            Label("microwave", 96, (169, 169, 169)),
            Label("oven", 97, (169, 169, 169)),
            Label("toaster", 98, (169, 169, 169)),
            Label("sink", 99, (169, 169, 169)),
            Label("refrigerator", 100, (169, 169, 169)),
            Label("book", 101, (255, 255, 255)),
            Label("clock", 102, (169, 169, 169)),
            Label("vase", 103, (255, 215, 0)),
            Label("scissors", 104, (169, 169, 169)),
            Label("teddy bear", 105, (255, 228, 225)),
            Label("hair drier", 106, (169, 169, 169)),
            Label("toothbrush", 107, (255, 255, 255)),
            Label("mirror", 108, (169, 169, 169)),
            Label("plate", 109, (255, 255, 255)),
            Label("cup", 110, (255, 255, 255)),
            Label("bottle", 111, (255, 255, 255)),
            Label("plate", 112, (255, 255, 255)),
            Label("tray", 113, (255, 255, 255)),
            Label("furniture", 114, (160, 82, 45)),
            Label("lamp", 115, (255, 255, 255)),
            Label("fork", 116, (169, 169, 169)),
            Label("spoon", 117, (169, 169, 169)),
            Label("knife", 118, (169, 169, 169)),
            Label("book", 119, (255, 255, 255)),
            Label("bottle", 120, (255, 255, 255)),
            Label("can", 121, (255, 255, 255)),
            Label("plate", 122, (255, 255, 255)),
            Label("clock", 123, (169, 169, 169)),
            Label("mirror", 124, (169, 169, 169)),
            Label("plate", 125, (255, 255, 255)),
            Label("lamp", 126, (255, 255, 255)),
            Label("spoon", 127, (169, 169, 169)),
            Label("fork", 128, (169, 169, 169)),
            Label("knife", 129, (169, 169, 169)),
            Label("clock", 130, (169, 169, 169)),
            Label("mirror", 131, (169, 169, 169)),
            Label("keyboard", 132, (0, 0, 0)),
            Label("cell phone", 133, (0, 0, 0)),
            Label("microwave", 134, (169, 169, 169)),
            Label("oven", 135, (169, 169, 169)),
            Label("toaster", 136, (169, 169, 169)),
            Label("sink", 137, (169, 169, 169)),
            Label("refrigerator", 138, (169, 169, 169)),
            Label("book", 139, (255, 255, 255)),
            Label("clock", 140, (169, 169, 169)),
            Label("vase", 141, (255, 215, 0)),
            Label("scissors", 142, (169, 169, 169)),
            Label("teddy bear", 143, (255, 228, 225)),
            Label("hair drier", 144, (169, 169, 169)),
            Label("toothbrush", 145, (255, 255, 255)),
            Label("scissors", 146, (169, 169, 169)),
            Label("computer", 147, (0, 0, 0)),
            Label("plate", 148, (255, 255, 255)),
            Label("bottle", 149, (0, 0, 0)),
            Label("can", 150, (255, 255, 255)),
        ]
        self.class_names = [label.name for label in self.labels]

    def id_2_label(self, id):
        id2label = {label.id: label for label in self.labels}
        return id2label[id]

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, "annotations", name)
        mask = torch.tensor(np.array(Image.open(mask_path)))
        return mask

    def load_support_set(self):
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
                    os.path.join(self.base_path, "images", support_name + ".jpg")
                ).convert("RGB")
                support_img = self.transform(support_img)

                support_mask = self.read_mask(support_name + ".png")
                # This three steps are necessary to convert the mask to binary mask since 0 is the mask for a class.
                # support_mask[support_mask != class_id] = -1
                # support_mask[support_mask == class_id] = 1
                # support_mask[support_mask == -1] = 0
                # real_class_id = class_id + 1
                support_mask = support_mask == class_id

                if self.pixel_percentage != 0:
                    pixels_proportion = (
                        support_mask.shape[0]
                        * support_mask.shape[1]
                        * (self.pixel_percentage / 100)
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
            os.path.join(self.base_path, "images", query_name + ".jpg")
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

import colorsys
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


class DatasetGENERAL(Dataset):
    def __init__(
        self,
        datapath,
        transform,
        split,
        nprompts,
        use_original_imgsize,
        benchmark,
    ):
        dict_benchmark = {
            "lovedarural": "LoveDA-Rural",
            "lovedaurban": "LoveDA-Urban",
            "mhpv1": "MHPv1",
            "pidray": "PIDray",
            "uecfood": "UECFOOD",
            "zerowaste": "ZeroWaste",
            "toolkits": "Toolkits",
            "trash": "Trash",
            "pizza": "Pizza",
            "houseparts": "House-Parts",
            "uavid": "uavid",
        }
        self.split = "val" if split in ["val", "test"] else "trn"
        self.benchmark = benchmark
        self.nprompts = nprompts
        self.base_path = os.path.join(datapath, dict_benchmark[benchmark])
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize
        self.ignore_label = 255
        self.ignore_background = True
        self.pixel_percentage = 0

        self.__get_img_metadata()
        self.__get_class()

        if self.ignore_background:
            self.class_ids = np.arange(1, self.nclass + 1, dtype=int)
        else:
            self.class_ids = np.arange(0, self.nclass, dtype=int)

    def __get_img_metadata(self):
        with open(f"./data/splits/{self.benchmark}_train.pkl", "rb") as f:
            img_list_classwise_trn, img_list_trn = pickle.load(f)

        with open(f"./data/splits/{self.benchmark}_val.pkl", "rb") as f:
            img_list_classwise_val, img_list_val = pickle.load(f)

        self.img_list_classwise_trn = img_list_classwise_trn
        self.img_list_trn = img_list_trn
        self.img_list_classwise_val = img_list_classwise_val
        self.img_list_val = img_list_val

    def __get_class(self):
        def generate_palette(num_classes):
            """
            Generate a palette with `num_classes` distinct colors.

            Args:
                num_classes (int): Number of classes (colors) required.

            Returns:
                list: A list of lists, where each inner list represents an [R, G, B] color.
            """
            palette = []
            for i in range(num_classes):
                # Evenly distribute the hue value
                hue = i / num_classes
                # Convert HSV (with full saturation and value) to RGB
                rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                # Scale RGB values to 0-255 and convert to integers
                rgb = tuple(int(x * 255) for x in rgb)
                palette.append(rgb)
            return palette

        class_file_path = os.path.join(self.base_path, "classes.txt")

        classes = []
        with open(class_file_path, "r") as f:
            for line in f:
                if line.strip() and line.strip() != "background":
                    classes.append(line.strip())
        self.nclass = len(classes)

        self.labels = []
        self.labels.append(Label("background", 0, (0, 0, 0)))
        palette = generate_palette(len(classes))
        for i, c in enumerate(classes):
            self.labels.append(Label(c, i + 1, palette[i]))

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
                warning_classes.append(class_id)
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

                try:
                    support_img = Image.open(
                        os.path.join(self.base_path, support_name + ".jpg")
                    ).convert("RGB")
                except FileNotFoundError:
                    support_img = Image.open(
                        os.path.join(self.base_path, support_name + ".png")
                    ).convert("RGB")
                support_img = self.transform(support_img)

                support_mask = self.read_mask(support_name + ".png")
                # This three steps are necessary to convert the mask to binary mask since 0 is the mask for a class.
                # support_mask[support_mask != class_id] = -1
                # support_mask[support_mask == class_id] = 1
                # support_mask[support_mask == -1] = 0
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
        try:
            query_img = Image.open(
                os.path.join(self.base_path, query_name + ".jpg")
            ).convert("RGB")
        except FileNotFoundError:
            query_img = Image.open(
                os.path.join(self.base_path, query_name + ".png")
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

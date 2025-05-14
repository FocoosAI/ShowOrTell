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
            Label("floor;flooring", 4, (209, 204, 155)),
            Label("tree", 5, (34, 139, 34)),
            Label("ceiling", 6, (255, 255, 255)),
            Label("road;route", 7, (128, 128, 128)),
            Label("bed", 8, (139, 69, 19)),
            Label("windowpane;window", 9, (173, 216, 230)),
            Label("grass", 10, (0, 128, 0)),
            Label("cabinet", 11, (139, 69, 19)),
            Label("sidewalk;pavement", 12, (169, 169, 169)),
            Label(
                "person;individual;someone;somebody;mortal;soul", 13, (255, 228, 225)
            ),
            Label("earth;ground", 14, (255, 0, 0)),
            Label("door;double;door", 15, (255, 0, 0)),
            Label("table", 16, (255, 215, 0)),
            Label("mountain;mount", 17, (169, 169, 169)),
            Label("plant;flora;plant;life", 18, (255, 215, 0)),
            Label("curtain;drape;drapery;mantle;pall", 19, (205, 133, 63)),
            Label("chair", 20, (128, 128, 0)),
            Label("car;auto;automobile;machine;motorcar", 21, (0, 0, 128)),
            Label("water", 22, (255, 0, 0)),
            Label("painting;picture", 23, (255, 0, 0)),
            Label("sofa;couch;lounge", 24, (255, 255, 255)),
            Label("shelf", 25, (105, 105, 105)),
            Label("house", 26, (255, 215, 0)),
            Label("sea", 27, (160, 82, 45)),
            Label("mirror", 28, (255, 255, 255)),
            Label("rug;carpet;carpeting", 29, (0, 0, 0)),
            Label("field", 30, (255, 255, 255)),
            Label("armchair", 31, (139, 69, 19)),
            Label("seat", 32, (255, 255, 255)),
            Label("fence;fencing", 33, (255, 255, 255)),
            Label("desk", 34, (169, 169, 169)),
            Label("rock;stone", 35, (139, 69, 19)),
            Label("wardrobe;closet;press", 36, (255, 255, 255)),
            Label("lamp", 37, (255, 255, 255)),
            Label("bathtub;bathing;tub;bath;tub", 38, (255, 228, 225)),
            Label("railing;rail", 39, (0, 0, 0)),
            Label("cushion", 40, (0, 0, 0)),
            Label("base;pedestal;stand", 41, (169, 169, 169)),
            Label("box", 42, (0, 0, 0)),
            Label("column;pillar", 43, (0, 0, 0)),
            Label("signboard;sign", 44, (169, 169, 169)),
            Label("chest;of;drawers;chest;bureau;dresser", 45, (169, 169, 169)),
            Label("counter", 46, (169, 169, 169)),
            Label("sand", 47, (169, 169, 169)),
            Label("sink", 48, (169, 169, 169)),
            Label("skyscraper", 49, (255, 255, 255)),
            Label("fireplace;hearth;open;fireplace", 50, (169, 169, 169)),
            Label("refrigerator;icebox", 51, (255, 215, 0)),
            Label("grandstand;covered;stand", 52, (169, 169, 169)),
            Label("path", 53, (255, 228, 225)),
            Label("stairs;steps", 54, (169, 169, 169)),
            Label("runway", 55, (255, 255, 255)),
            Label("case;display;case;showcase;vitrine", 56, (169, 169, 169)),
            Label("pool;table;billiard;table;snooker;table", 57, (0, 0, 0)),
            Label("pillow", 58, (255, 255, 255)),
            Label("screen;door;screen", 59, (0, 0, 0)),
            Label("stairway;staircase", 60, (255, 255, 255)),
            Label("river", 61, (255, 255, 255)),
            Label("bridge;span", 62, (0, 0, 0)),
            Label("bookcase", 63, (169, 169, 169)),
            Label("blind;screen", 64, (169, 169, 169)),
            Label("coffee;table;cocktail;table", 65, (255, 255, 255)),
            Label(
                "toilet;can;commode;crapper;pot;potty;stool;throne", 66, (169, 169, 169)
            ),
            Label("flower", 67, (169, 169, 169)),
            Label("book", 68, (169, 169, 169)),
            Label("hill", 69, (255, 255, 255)),
            Label("bench", 70, (160, 82, 45)),
            Label("countertop", 71, (160, 82, 45)),
            Label(
                "stove;kitchen;stove;range;kitchen;range;cooking;stove",
                72,
                (255, 255, 255),
            ),
            Label("palm;palm;tree", 73, (255, 255, 255)),
            Label("kitchen;island", 74, (255, 255, 255)),
            Label(
                "computer;computing;machine;computing;device;data;processor;electronic;computer;information;processing;system",
                75,
                (160, 82, 45),
            ),
            Label("swivel;chair", 76, (255, 255, 255)),
            Label("boat", 77, (255, 255, 255)),
            Label("bar", 78, (160, 82, 45)),
            Label("arcade;machine", 79, (255, 255, 255)),
            Label("hovel;hut;hutch;shack;shanty", 80, (255, 255, 255)),
            Label(
                "bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle",
                81,
                (255, 255, 255),
            ),
            Label("towel", 82, (255, 255, 255)),
            Label("light;light;source", 83, (255, 255, 255)),
            Label("truck;motortruck", 84, (255, 255, 255)),
            Label("tower", 85, (169, 169, 169)),
            Label("chandelier;pendant;pendent", 86, (169, 169, 169)),
            Label("awning;sunshade;sunblind", 87, (255, 255, 255)),
            Label("streetlight;street;lamp", 88, (255, 255, 255)),
            Label("booth;cubicle;stall;kiosk", 89, (169, 169, 169)),
            Label(
                "television;television;receiver;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box",
                90,
                (169, 169, 169),
            ),
            Label("airplane;aeroplane;plane", 91, (169, 169, 169)),
            Label("dirt;track", 92, (169, 169, 169)),
            Label("apparel;wearing;apparel;dress;clothes", 93, (169, 169, 169)),
            Label("pole", 94, (0, 0, 0)),
            Label("land;ground;soil", 95, (0, 0, 0)),
            Label(
                "bannister;banister;balustrade;balusters;handrail", 96, (169, 169, 169)
            ),
            Label("escalator;moving;staircase;moving;stairway", 97, (169, 169, 169)),
            Label("ottoman;pouf;pouffe;puff;hassock", 98, (169, 169, 169)),
            Label("bottle", 99, (169, 169, 169)),
            Label("buffet;counter;sideboard", 100, (169, 169, 169)),
            Label("poster;posting;placard;notice;bill;card", 101, (255, 255, 255)),
            Label("stage", 102, (169, 169, 169)),
            Label("van", 103, (255, 215, 0)),
            Label("ship", 104, (169, 169, 169)),
            Label("fountain", 105, (255, 228, 225)),
            Label(
                "conveyer;belt;conveyor;belt;conveyer;conveyor;transporter",
                106,
                (169, 169, 169),
            ),
            Label("canopy", 107, (255, 255, 255)),
            Label("washer;automatic;washer;washing;machine", 108, (169, 169, 169)),
            Label("plaything;toy", 109, (255, 255, 255)),
            Label("swimming;pool;swimming;bath;natatorium", 110, (255, 255, 255)),
            Label("stool", 111, (255, 255, 255)),
            Label("barrel;cask", 112, (255, 255, 255)),
            Label("basket;handbasket", 113, (255, 255, 255)),
            Label("waterfall;falls", 114, (160, 82, 45)),
            Label("tent;collapsible;shelter", 115, (255, 255, 255)),
            Label("bag", 116, (169, 169, 169)),
            Label("minibike;motorbike", 117, (169, 169, 169)),
            Label("cradle", 118, (169, 169, 169)),
            Label("oven", 119, (255, 255, 255)),
            Label("ball", 120, (255, 255, 255)),
            Label("food;solid;food", 121, (255, 255, 255)),
            Label("step;stair", 122, (255, 255, 255)),
            Label("tank;storage;tank", 123, (169, 169, 169)),
            Label("trade;name;brand;name;brand;marque", 124, (169, 169, 169)),
            Label("microwave;microwave;oven", 125, (255, 255, 255)),
            Label("pot;flowerpot", 126, (255, 255, 255)),
            Label(
                "animal;animate;being;beast;brute;creature;fauna", 127, (169, 169, 169)
            ),
            Label("bicycle;bike;wheel;cycle", 128, (169, 169, 169)),
            Label("lake", 129, (169, 169, 169)),
            Label("dishwasher;dish;washer;dishwashing;machine", 130, (169, 169, 169)),
            Label("screen;silver;screen;projection;screen", 131, (169, 169, 169)),
            Label("blanket;cover", 132, (0, 0, 0)),
            Label("sculpture", 133, (0, 0, 0)),
            Label("hood;exhaust;hood", 134, (169, 169, 169)),
            Label("sconce", 135, (169, 169, 169)),
            Label("vase", 136, (169, 169, 169)),
            Label("traffic;light;traffic;signal;stoplight", 137, (169, 169, 169)),
            Label("tray", 138, (169, 169, 169)),
            Label(
                "ashcan;trash;can;garbage;can;wastebin;ash;bin;ash-bin;ashbin;dustbin;trash;barrel;trash;bin",
                139,
                (255, 255, 255),
            ),
            Label("fan", 140, (169, 169, 169)),
            Label("pier;wharf;wharfage;dock", 141, (255, 215, 0)),
            Label("crt;screen", 142, (169, 169, 169)),
            Label("plate", 143, (255, 228, 225)),
            Label("monitor;monitoring;device", 144, (169, 169, 169)),
            Label("bulletin;board;notice;board", 145, (255, 255, 255)),
            Label("shower", 146, (169, 169, 169)),
            Label("radiator", 147, (0, 0, 0)),
            Label("glass;drinking;glass", 148, (255, 255, 255)),
            Label("clock", 149, (0, 0, 0)),
            Label("flag", 150, (255, 255, 255)),
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

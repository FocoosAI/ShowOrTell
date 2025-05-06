import argparse
import os
import os.path as osp
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm


class UAVidColorTransformer:
    def __init__(self):
        # color table.
        self.clr_tab = self.createColorTable()
        # id table.
        id_tab = {}
        for k, v in self.clr_tab.items():
            id_tab[k] = self.clr2id(v)
        self.id_tab = id_tab

    def createColorTable(self):
        clr_tab = {}
        clr_tab["Clutter"] = [0, 0, 0]
        clr_tab["Building"] = [128, 0, 0]
        clr_tab["Road"] = [128, 64, 128]
        clr_tab["Static_Car"] = [192, 0, 192]
        clr_tab["Tree"] = [0, 128, 0]
        clr_tab["Vegetation"] = [128, 128, 0]
        clr_tab["Human"] = [64, 64, 0]
        clr_tab["Moving_Car"] = [64, 0, 128]
        return clr_tab

    def colorTable(self):
        return self.clr_tab

    def clr2id(self, clr):
        return clr[0] + clr[1] * 255 + clr[2] * 255 * 255

    # transform to uint8 integer label
    def transform(self, label, dtype=np.int32):
        height, width = label.shape[:2]
        # default value is index of clutter.
        newLabel = np.zeros((height, width), dtype=dtype)
        id_label = label.astype(np.int64)
        id_label = (
            id_label[:, :, 0] + id_label[:, :, 1] * 255 + id_label[:, :, 2] * 255 * 255
        )
        for tid, key in enumerate(self.clr_tab.keys()):
            val = self.id_tab[key]
            mask = id_label == val
            newLabel[mask] = tid
        return newLabel

    # transform back to 3 channels uint8 label
    def inverse_transform(self, label):
        label_img = np.zeros(shape=(label.shape[0], label.shape[1], 3), dtype=np.uint8)
        values = list(self.clr_tab.values())
        for tid, val in enumerate(values):
            mask = label == tid
            label_img[mask] = val
        return label_img


def prepareTrainIDForDir(gtDirPath, saveDirPath):
    clrEnc = UAVidColorTransformer()

    gt_paths = [p for p in os.listdir(gtDirPath) if p.startswith("seq")]
    img_index = 0
    for pd in gt_paths:
        lbl_dir = osp.join(gtDirPath, pd, "Labels")
        lbl_paths = os.listdir(lbl_dir)
        # if not osp.isdir(osp.join(saveDirPath, pd, "TrainId")):
        #     os.makedirs(osp.join(saveDirPath, pd, "TrainId"))
        #     assert osp.isdir(osp.join(saveDirPath, pd, "TrainId")), (
        #         "Fail to create directory:%s" % (osp.join(saveDirPath, pd, "TrainId"))
        #     )
        for lbl_p in tqdm(lbl_paths, desc=f"Processing {pd} split"):
            if "shifted" in lbl_p or "flipped" in lbl_p:
                continue
            lbl_path = osp.abspath(osp.join(lbl_dir, lbl_p))
            # trainId_path = osp.join(saveDirPath, pd, "TrainId", lbl_p)
            trainId_path = osp.join(saveDirPath, lbl_p)
            gt = np.array(Image.open(lbl_path))
            trainId = clrEnc.transform(gt, dtype=np.uint8)
            Image.fromarray(trainId).save(trainId_path)
            img_index += 1


def prepare_annotations(base_path, annot_path):
    annot_train_path = os.path.join(annot_path, "train")
    annot_val_path = os.path.join(annot_path, "valid")
    os.makedirs(annot_train_path, exist_ok=True)
    os.makedirs(annot_val_path, exist_ok=True)

    prepareTrainIDForDir(os.path.join(base_path, "uavid_train"), annot_train_path)
    prepareTrainIDForDir(os.path.join(base_path, "uavid_val"), annot_val_path)
    # os.rename(os.path.join(annot_path, "val"), os.path.join(annot_path, "valid"))


def move_images(base_path, train_path, val_path):
    # Function to copy images from sequence folders
    def copy_sequence_images(src_base_path, dst_path):
        # img_index = 0
        seq_dirs = [d for d in os.listdir(src_base_path) if d.startswith("seq")]

        for seq_dir in seq_dirs:
            images_dir = os.path.join(src_base_path, seq_dir, "Images")
            if not os.path.exists(images_dir):
                print(f"Warning: {images_dir} does not exist")
                continue

            image_files = os.listdir(images_dir)
            for img_file in tqdm(image_files):
                src_file = os.path.join(images_dir, img_file)
                dst_file = os.path.join(dst_path, img_file)

                if "shifted" in src_file or "flipped" in src_file:
                    continue

                shutil.move(src_file, dst_file)

    print("Copying training images...")
    copy_sequence_images(os.path.join(base_path, "uavid_train"), train_path)
    print("Copying validation images...")
    copy_sequence_images(os.path.join(base_path, "uavid_val"), val_path)

    print("Image copying complete!")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert UAVid dataset to mask images"
    )
    parser.add_argument(
        "--base-path",
        type=str,
        required=True,
        help="Base path containing the UAVid dataset",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_path = args.base_path
    imgs_train_path = os.path.join(base_path, "train")
    imgs_val_path = os.path.join(base_path, "valid")
    os.makedirs(imgs_train_path, exist_ok=True)
    os.makedirs(imgs_val_path, exist_ok=True)

    annot_path = os.path.join(base_path, "annotations")
    os.makedirs(annot_path, exist_ok=True)

    prepare_annotations(base_path, annot_path)
    move_images(base_path, imgs_train_path, imgs_val_path)

    # Write UAVid class names to classes.txt file
    class_names = [
        "background",
        "Building",
        "Road",
        "Static_Car",
        "Tree",
        "Vegetation",
        "Human",
        "Moving_Car",
    ]
    with open(os.path.join(base_path, "classes.txt"), "w") as f:
        for name in class_names:
            f.write(name + "\n")

    # Count files in each directory
    train_count = len(os.listdir(imgs_train_path))
    val_count = len(os.listdir(imgs_val_path))
    print(f"Train images: {train_count}")
    print(f"Validation images: {val_count}")


if __name__ == "__main__":
    main()

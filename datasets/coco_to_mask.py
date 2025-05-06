import argparse
import contextlib
import os

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert COCO annotations to mask images"
    )
    parser.add_argument(
        "--base-path", type=str, required=True, help="Base path containing the dataset"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "valid"],
        help="Names of the splits to process",
    )
    parser.add_argument(
        "--annotation-paths",
        type=str,
        nargs="+",
        default=None,
        help="Paths to the annotation files for each split relative to base-path. If only one path provided, it will be used for all splits",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_path = args.base_path
    annot_path = os.path.join(base_path, "annotations")

    if args.annotation_paths is None:
        raise ValueError("Annotation paths must be provided")

    if len(args.annotation_paths) == 1:
        # Use the same annotation file for all splits
        annotation_paths = [args.annotation_paths[0]] * len(args.splits)
    else:
        if len(args.annotation_paths) != len(args.splits):
            raise ValueError("Number of annotation paths must match number of splits")
        annotation_paths = args.annotation_paths

    for i, split in enumerate(args.splits):
        annotation_file = os.path.join(base_path, annotation_paths[i])

        # Suppress print statements from COCO initialization
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            coco = COCO(annotation_file)
        cat_ids = coco.getCatIds()
        img_dir = os.path.join(base_path, split)

        os.makedirs(os.path.join(annot_path, split), exist_ok=True)

        for id, meta in tqdm(coco.imgs.items(), desc=f"Processing {split} split"):
            anns_ids = coco.getAnnIds(imgIds=meta["id"], catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(anns_ids)

            # if there are no annotations, remove the image
            if len(anns) == 0:
                image_path = os.path.join(img_dir, meta["file_name"])
                if os.path.exists(image_path):
                    os.remove(image_path)
                    continue

            anns_img = np.zeros((meta["height"], meta["width"]))
            for ann in anns:
                mask = coco.annToMask(ann)
                anns_img[mask == 1] = ann["category_id"]

            if anns_img.sum() > 0:
                output_file = os.path.join(
                    annot_path,
                    split,
                    f"{meta['file_name'].rsplit('.', maxsplit=1)[0]}.png",
                )
                Image.fromarray(anns_img.astype(np.uint8)).save(output_file)

        # Move the annotation file to the annotations directory
        os.rename(
            annotation_file, os.path.join(annot_path, f"annotations_{split}.json")
        )

    category_names = [cat["name"] for cat in coco.dataset["categories"]]
    with open(os.path.join(base_path, "classes.txt"), "w") as f:
        f.write("background\n")
        for name in category_names[1:]:
            f.write(name + "\n")


if __name__ == "__main__":
    main()

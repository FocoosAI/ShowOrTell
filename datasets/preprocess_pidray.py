import argparse
import os

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm


def process_annotations(base_path):
    """Generate segmentation masks from COCO annotations."""
    annot_path = os.path.join(base_path, "annotations")
    coco = None

    for split in ["train", "test"]:
        if split == "train":
            coco = COCO(os.path.join(annot_path, f"{split}.json"))
        else:
            coco = COCO(os.path.join(annot_path, f"{split}_hard.json"))
        cat_ids = coco.getCatIds()
        img_dir = os.path.join(base_path, split)

        os.makedirs(os.path.join(annot_path, split), exist_ok=True)

        for id, meta in tqdm(coco.imgs.items(), desc=f"Processing {split} split"):
            anns_ids = coco.getAnnIds(imgIds=meta["id"], catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(anns_ids)

            if len(anns) == 0:
                image_path = os.path.join(img_dir, meta["file_name"])
                if os.path.exists(image_path):
                    os.remove(image_path)
                    continue

            anns_img = np.zeros((meta["height"], meta["width"]))
            for ann in anns:
                anns_img = np.maximum(
                    anns_img, coco.annToMask(ann) * ann["category_id"]
                )

            if anns_img.sum() > 0:
                output_file = os.path.join(
                    annot_path,
                    split,
                    f"{meta['file_name'].rsplit('.', maxsplit=1)[0]}.png",
                )
                Image.fromarray(anns_img.astype(np.uint8)).save(output_file)

    return coco, annot_path


def create_class_definitions(base_path, coco):
    """Create class definitions file based on categories."""
    category_names = [cat["name"] for cat in coco.dataset["categories"]]
    with open(os.path.join(base_path, "classes.txt"), "w") as f:
        f.write("background\n")
        for name in category_names:
            f.write(name + "\n")


def rename_test_dirs(base_path, annot_path):
    """Rename 'test' directories to 'valid'."""
    os.rename(os.path.join(annot_path, "test"), os.path.join(annot_path, "valid"))
    os.rename(os.path.join(base_path, "test"), os.path.join(base_path, "valid"))


def clean_valid_directory(base_path):
    """Remove 'easy' and 'hidden' images from the valid directory."""
    for image in tqdm(os.listdir(os.path.join(base_path, "valid"))):
        if "easy" in image or "hidden" in image:
            os.remove(os.path.join(base_path, "valid", image))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess PIDray dataset")
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Path to the base PIDray dataset directory",
    )
    return parser.parse_args()


def main():
    """Main function to preprocess the PIDray dataset."""
    args = parse_args()
    base_path = args.base_path

    print(f"Processing dataset at: {base_path}")

    # Step 1: Process annotations and generate segmentation masks
    coco, annot_path = process_annotations(base_path)

    # Step 2: Create class definitions
    create_class_definitions(base_path, coco)

    # Step 3: Rename test directories to valid
    rename_test_dirs(base_path, annot_path)

    # Step 4: Clean valid directory
    clean_valid_directory(base_path)

    print("PIDray dataset preprocessing completed!")


if __name__ == "__main__":
    main()

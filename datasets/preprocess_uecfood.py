import argparse
import os
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm


def setup_directories(base_path):
    """Create necessary directories for annotations."""
    annot_path = os.path.join(base_path, "annotations")
    os.makedirs(os.path.join(annot_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(annot_path, "test"), exist_ok=True)
    return annot_path


def reorganize_files(base_path, annot_path):
    """Move files from img/mask to their respective directories."""
    for split in ["train", "test"]:
        original_image_dir = os.path.join(base_path, split, "img")
        original_mask_dir = os.path.join(base_path, split, "mask")

        target_image_dir = os.path.join(base_path, split)
        annotation_dir = os.path.join(base_path, "annotations", split)

        if os.path.exists(original_mask_dir):
            for file in os.listdir(original_mask_dir):
                shutil.move(os.path.join(original_mask_dir, file), annotation_dir)
            os.rmdir(original_mask_dir)

        if os.path.exists(original_image_dir):
            for file in os.listdir(original_image_dir):
                shutil.move(os.path.join(original_image_dir, file), target_image_dir)
            os.rmdir(original_image_dir)


def process_labels(base_path):
    """Process label files to extract index channel."""
    for split in ["train", "test"]:
        labels_path = os.path.join(base_path, "annotations", split)
        for label_file in tqdm(os.listdir(labels_path), desc=f"Processing {split} split"):
            label = Image.open(os.path.join(labels_path, label_file))
            label = np.array(label)

            index_label = label[:, :, 0]

            Image.fromarray(index_label.astype(np.uint8)).save(
                os.path.join(labels_path, label_file)
            )


def rename_test_dirs(base_path):
    """Rename 'test' directories to 'valid'."""
    os.rename(os.path.join(base_path, "test"), os.path.join(base_path, "valid"))

    os.rename(
        os.path.join(base_path, "annotations", "test"),
        os.path.join(base_path, "annotations", "valid"),
    )


def convert_images(base_path):
    """Convert jpg images to png format."""
    for split in ["train", "valid"]:
        for img in tqdm(os.listdir(os.path.join(base_path, split)), desc=f"Converting images to png for {split} split"):
            img_path = os.path.join(base_path, split, img)
            if img_path.endswith(".jpg"):
                img = Image.open(img_path)
                img.save(img_path.replace(".jpg", ".png"), "PNG", icc_profile=None)
                os.remove(img_path)


def create_class_definitions(base_path):
    """Create class definitions file based on category.txt."""
    with (
        open(os.path.join(base_path, "category.txt"), "r", encoding="utf-8") as infile,
        open(os.path.join(base_path, "classes.txt"), "w", encoding="utf-8") as outfile,
    ):
        next(infile)
        outfile.write("background\n")
        for line in infile:
            name = line.split("\t", 1)[1].strip()
            outfile.write(name + "\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess UECFood dataset")
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Path to the base UECFood dataset directory",
    )
    return parser.parse_args()


def main():
    """Main function to preprocess the UECFood dataset."""
    args = parse_args()
    base_path = args.base_path

    print(f"Processing dataset at: {base_path}")

    # Step 1: Setup directories
    annot_path = setup_directories(base_path)

    # Step 2: Reorganize files
    reorganize_files(base_path, annot_path)

    # Step 3: Process labels
    process_labels(base_path)

    # Step 4: Rename test directories to valid
    rename_test_dirs(base_path)

    # Step 5: Convert jpg images to png
    convert_images(base_path)

    # Step 6: Create class definitions
    create_class_definitions(base_path)

    print("UECFood dataset preprocessing completed!")


if __name__ == "__main__":
    main()

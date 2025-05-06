import argparse
import os
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm


def reorganize_dataset(base_path):
    """Reorganize the dataset by separating rural and urban data."""
    rural_target = os.path.join(os.path.dirname(base_path), "LoveDA-Rural")
    urban_target = os.path.join(os.path.dirname(base_path), "LoveDA-Urban")
    os.makedirs(os.path.join(rural_target, "Train"), exist_ok=True)
    os.makedirs(os.path.join(rural_target, "Val"), exist_ok=True)

    for subdir in ["Train", "Val"]:
        rural_source = os.path.join(base_path, subdir, "Rural")
        rural_dest = os.path.join(rural_target, subdir)
        if os.path.exists(rural_source):
            shutil.move(rural_source, rural_dest)

    os.rename(base_path, urban_target)

    return rural_target, urban_target


def restructure_directories(rural_target, urban_target):
    """Restructure directories to follow a consistent pattern."""
    for base in [rural_target, urban_target]:
        for subdir in ["Train", "Val"]:
            original_image_dir = os.path.join(
                base, subdir, base.split("-")[1], "images_png"
            )
            original_mask_dir = os.path.join(
                base, subdir, base.split("-")[1], "masks_png"
            )
            target_image_dir = os.path.join(base, subdir)

            annotation_dir = os.path.join(base, "annotations", subdir.lower())

            os.makedirs(annotation_dir, exist_ok=True)

            if os.path.exists(original_mask_dir):
                for file in os.listdir(original_mask_dir):
                    shutil.move(os.path.join(original_mask_dir, file), annotation_dir)
                os.rmdir(original_mask_dir)

            if os.path.exists(original_image_dir):
                for file in os.listdir(original_image_dir):
                    shutil.move(
                        os.path.join(original_image_dir, file), target_image_dir
                    )
                os.rmdir(original_image_dir)

            if os.path.exists(os.path.join(base, subdir, base.split("-")[1])):
                os.rmdir(os.path.join(base, subdir, base.split("-")[1]))

            os.rename(os.path.join(base, subdir), os.path.join(base, subdir.lower()))

    print("Dataset restructuring completed successfully!")


def process_labels(rural_target, urban_target):
    """Process label files and create class definitions."""
    for base in [rural_target, urban_target]:
        for split in ["train", "val"]:
            labels_path = os.path.join(base, "annotations", split)
            for label_file in tqdm(os.listdir(labels_path), desc=f"Processing {split} split"):
                label = Image.open(os.path.join(labels_path, label_file))
                label = np.array(label)

                label -= 1
                label = np.where(label == 255, 0, label)

                Image.fromarray(label.astype(np.uint8)).save(
                    os.path.join(labels_path, label_file)
                )

        category_names = [
            "background",
            "building",
            "road",
            "water",
            "barren",
            "forest",
            "agriculture",
        ]
        with open(os.path.join(base, "classes.txt"), "w") as f:
            for name in category_names:
                f.write(name + "\n")

    # Rename val directories to valid
    os.rename(os.path.join(rural_target, "val"), os.path.join(rural_target, "valid"))
    os.rename(os.path.join(urban_target, "val"), os.path.join(urban_target, "valid"))

    os.rename(
        os.path.join(rural_target, "annotations", "val"),
        os.path.join(rural_target, "annotations", "valid"),
    )
    os.rename(
        os.path.join(urban_target, "annotations", "val"),
        os.path.join(urban_target, "annotations", "valid"),
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess LoveDA dataset")
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Path to the base LoveDA dataset directory",
    )
    return parser.parse_args()


def main():
    """Main function to preprocess the LoveDA dataset."""
    args = parse_args()
    base_path = args.base_path

    print(f"Processing dataset at: {base_path}")

    # Step 1: Reorganize dataset
    rural_target, urban_target = reorganize_dataset(base_path)

    # Step 2: Restructure directories
    restructure_directories(rural_target, urban_target)

    # Step 3: Process labels and create class definitions
    process_labels(rural_target, urban_target)

    print("LoveDA dataset preprocessing completed!")


if __name__ == "__main__":
    main()

import argparse
import glob
import os
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm


def read_image_lists(base_path):
    """Read train and validation image lists."""
    train_file = os.path.join(base_path, "train_list.txt")
    with open(train_file, "r") as f:
        lines = f.readlines()
    train_images = [line.strip() for line in lines]

    val_file = os.path.join(base_path, "test_list.txt")
    with open(val_file, "r") as f:
        lines = f.readlines()
    val_images = [line.strip() for line in lines]

    return train_images, val_images


def setup_directories(base_path):
    """Create necessary directories for images and annotations."""
    annot_path = os.path.join(base_path, "annotations")
    os.makedirs(os.path.join(base_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "valid"), exist_ok=True)

    os.makedirs(os.path.join(annot_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(annot_path, "valid"), exist_ok=True)

    return annot_path


def organize_images(base_path, train_images, val_images):
    """Move images to their respective train/valid directories."""
    for image in tqdm(os.listdir(os.path.join(base_path, "images")), desc="Moving images"):
        if image in train_images:
            shutil.move(
                os.path.join(base_path, "images", image),
                os.path.join(base_path, "train", image),
            )
        elif image in val_images:
            shutil.move(
                os.path.join(base_path, "images", image),
                os.path.join(base_path, "valid", image),
            )

    os.rmdir(os.path.join(base_path, "images"))


def process_masks(base_path, annot_path, train_images, val_images):
    """Process mask files and create consolidated masks."""
    for img in tqdm(train_images + val_images, desc="Processing masks"):
        img_id = img.split(".")[0]
        mask_paths = glob.glob(os.path.join(annot_path, f"{img_id}*.png"))
        mask = np.stack([np.array(Image.open(mask_path)) for mask_path in mask_paths])
        mask = Image.fromarray(mask.max(axis=0).astype(np.uint8))

        if img in train_images:
            mask.save(os.path.join(annot_path, "train", img_id + ".png"))
        else:
            mask.save(os.path.join(annot_path, "valid", img_id + ".png"))


def create_class_definitions(base_path):
    """Create class definitions file."""
    category_names = [
        "background",
        "hat",
        "hair",
        "sunglasses",
        "upper clothes",
        "skirt",
        "pants",
        "dress",
        "belt",
        "left shoe",
        "right shoe",
        "face",
        "left leg",
        "right leg",
        "left arm",
        "right arm",
        "bag",
        "scarf",
    ]

    with open(os.path.join(base_path, "classes.txt"), "w") as f:
        for name in category_names:
            f.write(name + "\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess MHP dataset")
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Path to the base MHP dataset directory",
    )
    return parser.parse_args()


def main():
    """Main function to preprocess the MHP dataset."""
    args = parse_args()
    base_path = args.base_path

    print(f"Processing dataset at: {base_path}")

    # Step 1: Read image lists
    train_images, val_images = read_image_lists(base_path)

    # Step 2: Setup directories
    annot_path = setup_directories(base_path)

    # Step 3: Organize images
    organize_images(base_path, train_images, val_images)

    # Step 4: Process masks
    process_masks(base_path, annot_path, train_images, val_images)

    # Step 5: Create class definitions
    create_class_definitions(base_path)

    print("MHP dataset preprocessing completed!")


if __name__ == "__main__":
    main()

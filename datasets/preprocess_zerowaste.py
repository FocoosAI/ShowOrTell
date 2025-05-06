import argparse
import json
import os
import shutil

from tqdm import tqdm


def reorganize_files(base_path):
    """Move files from data/sem_seg to their respective directories."""
    annot_path = os.path.join(base_path, "annotations")

    os.makedirs(os.path.join(annot_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(annot_path, "val"), exist_ok=True)

    for split in ["train", "val"]:
        for image in tqdm(os.listdir(os.path.join(base_path, split, "data")), desc=f"Processing {split} split"):
            shutil.move(
                os.path.join(base_path, split, "data", image),
                os.path.join(base_path, split, image.replace("PNG", "png")),
            )

            shutil.move(
                os.path.join(base_path, split, "sem_seg", image),
                os.path.join(annot_path, split, image.replace("PNG", "png")),
            )

        os.rmdir(os.path.join(base_path, split, "data"))
        os.rmdir(os.path.join(base_path, split, "sem_seg"))

    return annot_path


def rename_validation_dirs(base_path, annot_path):
    """Rename 'val' directories to 'valid'."""
    os.rename(
        os.path.join(annot_path, "val"),
        os.path.join(annot_path, "valid"),
    )
    os.rename(
        os.path.join(base_path, "val"),
        os.path.join(base_path, "valid"),
    )


def create_class_definitions(base_path):
    """Create class definitions file based on categories."""
    with open(os.path.join(base_path, "train", "labels.json")) as f:
        data = json.load(f)

    category_names = [cat["name"] for cat in data["categories"]]
    with open(os.path.join(base_path, "classes.txt"), "w") as f:
        f.write("background\n")
        for name in category_names:
            f.write(name + "\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess ZeroWaste dataset")
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Path to the base ZeroWaste dataset directory",
    )
    return parser.parse_args()


def main():
    """Main function to preprocess the ZeroWaste dataset."""
    args = parse_args()
    base_path = args.base_path

    print(f"Processing dataset at: {base_path}")

    # Step 1: Reorganize files
    annot_path = reorganize_files(base_path)

    # Step 2: Rename validation directories
    rename_validation_dirs(base_path, annot_path)

    # Step 3: Create class definitions
    create_class_definitions(base_path)

    print("ZeroWaste dataset preprocessing completed!")


if __name__ == "__main__":
    main()

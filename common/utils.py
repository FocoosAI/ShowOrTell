r"""Helper functions"""

import csv
import datetime
import json
import logging
import os
import random
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from prettytable import PrettyTable


def fix_randseed(seed):
    r"""Set random seeds for reproducibility"""
    if seed is None:
        seed = int(random.random() * 1e5)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mean(x):
    return sum(x) / len(x) if len(x) > 0 else 0.0


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch


def to_cpu(tensor):
    return tensor.detach().clone().cpu()


def save_results(results, log_dir, args):
    # Only the main process save the results
    logger = logging.getLogger("process_0")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{args.benchmark}_{args.model_name}_seed{args.seed}_{args.nprompts}prompts_{timestamp}.json"
    result_path = os.path.join(log_dir, filename)

    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {result_path}")

    header = ["model", "benchmark", "prompts", "seed", "mIoU", "", "log_dir"]
    result_csv_path = os.path.join(args.log_root, f"results_{args.nprompts}prompts.csv")
    file_exists = os.path.exists(result_csv_path)
    with open(result_csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(header)

        writer.writerow(
            [
                args.model_name,
                args.benchmark,
                args.nprompts,
                args.seed,
                round(results["miou"], 3),
                "",
                log_dir,
            ]
        )


def visualize_semseg_pred(
    query_dict,
    support_set_names,
    segmentation_map,
    labels,
    label_text=False,
    save_path=None,
):
    def __add_colors_and_name(ax, mask, alpha=0.8):
        # Create an overlay image from the segmentation map
        overlay = np.zeros_like(original_image, dtype=np.uint8)
        for label in labels:
            # Assume label.color is a tuple in (R, G, B) format.
            overlay[mask == label.id] = label.color

        ax.imshow(overlay, alpha=alpha)

        if label_text:
            for label in labels:
                if label.name == "background":
                    continue
                mask_positions = np.argwhere(mask == label.id)
                if mask_positions.any():
                    y, x = mask_positions.mean(axis=0)
                    normalized_color = tuple(c / 255 for c in label.color)
                    ax.text(
                        x,
                        y,
                        label.name,
                        color=normalized_color,
                        fontsize=10,
                        ha="center",
                        va="center",
                        bbox=dict(facecolor="black", alpha=alpha, edgecolor="none"),
                    )

    original_size = (
        query_dict["org_query_imsize"][1].item(),
        query_dict["org_query_imsize"][0].item(),
    )
    query_img_resized = F.interpolate(
        query_dict["query_img"],
        size=original_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    query_mask_resized = (
        F.interpolate(
            query_dict["query_mask"].unsqueeze(0).float(),
            size=original_size,
            mode="nearest",
        )
        .squeeze()
        .cpu()
        .numpy()
    )
    segmentation_map = (
        F.interpolate(
            segmentation_map.unsqueeze(0).unsqueeze(0).float(),
            size=original_size,
            mode="nearest",
        )
        .squeeze()
        .cpu()
        .numpy()
    )

    original_image = query_img_resized.permute(1, 2, 0).cpu().numpy()

    if save_path is None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 7))
        ax[0].imshow(original_image)
        __add_colors_and_name(ax[0], query_mask_resized)
        ax[0].set_title("Ground Truth")
        ax[0].axis("off")

        ax[1].imshow(original_image)
        __add_colors_and_name(ax[1], segmentation_map)
        ax[1].set_title("Prediction")
        ax[1].axis("off")

        plt.show()
    else:
        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, "support_images_names.txt"), "w") as file:
            for name in support_set_names:
                file.write(f"{name}\n")

        query_name = query_dict["query_name"][0].split("/")[-1].split(".")[0]

        plt.imshow(original_image)
        __add_colors_and_name(plt.gca(), segmentation_map)
        plt.axis("off")
        plt.savefig(
            os.path.join(save_path, f"{query_name}_pred.png"),
            bbox_inches="tight",
            pad_inches=0,
            dpi=200,
        )
        plt.close()

        plt.imshow(original_image)
        __add_colors_and_name(plt.gca(), query_mask_resized)
        plt.axis("off")
        plt.savefig(
            os.path.join(save_path, f"{query_name}_gt.png"),
            bbox_inches="tight",
            pad_inches=0,
            dpi=200,
        )
        plt.close()


def format_results_table(results, dataset_labels):
    class_table = PrettyTable()
    class_table.field_names = ["Index", "Class Name", "IoU"]

    for class_id, iou in results["per_class_iou"].items():
        class_name = None
        for label in dataset_labels:
            if str(label.id) == str(class_id):
                class_name = label.name
                break

        if class_name is None:
            class_name = f"Unknown ({class_id})"

        if isinstance(iou, float):
            class_table.add_row([class_id, class_name, f"{iou:.2f}"])
        else:
            class_table.add_row([class_id, class_name, str(iou)])

    return class_table, results["miou"]


def download_file(url, output_path, use_gdown=False):
    """Download a file if it doesn't exist.

    Args:
        url: URL or file ID
        output_path: Destination path
        use_gdown: Whether to use gdown instead of wget

    Returns:
        bool: Success status
    """
    output_path = Path(output_path)

    if output_path.exists():
        logging.info(f"Found existing file: {output_path}")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        cmd = (
            ["python3", "-m", "gdown", url, "-O", str(output_path)]
            if use_gdown
            else ["wget", "-q", "--show-progress", url, "-O", str(output_path)]
        )

        subprocess.run(cmd, check=True)

        if output_path.exists() and output_path.stat().st_size > 0:
            size_mb = output_path.stat().st_size / (1024 * 1024)
            logging.info(f"Downloaded {output_path.name} ({size_mb:.1f} MB)")
            return True
        else:
            logging.error("Download failed: empty or missing file")
            return False
    except Exception as e:
        logging.error(f"Download error: {str(e)}")
        return False


def download_all_models(checkpointspath):
    """Download all models from the list.

    Args:
        checkpointspath: Path to save the downloaded files
    """
    models = [
        {
            "name": "DINOv2",
            "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
            "path": os.path.join(checkpointspath, "dinov2_vitl14_pretrain.pth"),
            "use_gdown": False,
        },
        {
            "name": "SAM",
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "path": os.path.join(checkpointspath, "sam_vit_h_4b8939.pth"),
            "use_gdown": False,
        },
        {
            "name": "SINE",
            "url": "1GYQbbUZClbmhVESDLpRwqe-TyijW2kKb",
            "path": os.path.join(checkpointspath, "sine_checkpoint.bin"),
            "use_gdown": True,
        },
    ]

    for model in models:
        if not download_file(model["url"], model["path"], model["use_gdown"]):
            logging.error(f"Failed to download {model['name']} model")

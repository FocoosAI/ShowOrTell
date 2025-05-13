import argparse
import os
import random

import torch
from common import utils
from common.evaluation import SegmentationMetrics
from common.logger import DistributedLogger
from common.utils import format_results_table, save_results, visualize_semseg_pred
from data.dataset import PromptingDataset
from models.base_model import BaseModel
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


def test(model, dataloader_test, args):
    tqdm_disabled = (
        torch.distributed.is_initialized() and torch.distributed.get_rank() != 0
    )

    logger.info("=================== Starting Evaluation ====================")
    with tqdm(dataloader_test, leave=True, disable=tqdm_disabled) as pbar:
        for idx, query_dict in enumerate(pbar):
            pred = model.evaluate(query_dict)

            gt = query_dict["query_mask"].squeeze().clone().int()

            val_metrics.update(
                gt.unsqueeze(0).cpu().numpy(),
                pred.unsqueeze(0).cpu().numpy(),
            )

            if torch.distributed.is_initialized() and idx % 100 == 0:
                torch.distributed.barrier()
                val_metrics.synch(device)

                score = val_metrics.get_results()
                miou = score["Mean IoU"] * 100
                metrics_str = f"{miou:.3f} (idx: {idx})"
            else:
                score = val_metrics.get_results()
                miou = score["Mean IoU"] * 100
                metrics_str = f"{miou:.3f}"

            if idx % 100 == 0:
                logger.info(f"Batch {idx}/{len(dataloader_test)} | mIoU: {miou:.3f}")
            pbar.set_postfix({"mIoU": metrics_str})

            if args.save_visualization:
                visualize_semseg_pred(
                    query_dict,
                    model.support_set[0]["support_names"],
                    pred,
                    labels=dataset.labels,
                    label_text=False,
                    save_path=f"predictions/{args.model_name}/{args.nprompts}prompt/{args.benchmark}/",
                )
                if idx == 50:
                    break

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        val_metrics.synch(device)

    score = val_metrics.get_results()
    logger.info(f"Total samples: {score['Total samples']}")
    results = {
        "miou": score["Mean IoU"] * 100,
        "per_class_iou": {
            k: v * 100 if isinstance(v, float) else v
            for k, v in score["Class IoU"].items()
        },
    }

    table, miou = format_results_table(results, dataset.labels)
    logger.info(f"Per-class results:\n{table}")
    logger.info(f"Mean IoU: {miou:.2f}%")

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        if not args.save_visualization:
            save_results(results, logger.log_dir, args)


if __name__ == "__main__":
    # Arguments parsing
    parser = argparse.ArgumentParser(
        description="Show or Tell? A Benchmark To Evaluate Visual and Textual Prompts in Semantic Segmentation"
    )

    # Dataset parameters
    parser.add_argument("--datapath", type=str, default="./datasets")
    parser.add_argument("--checkpointspath", type=str, default="./models/checkpoints")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="cityscapes",
        choices=[
            "pascal",
            "cityscapes",
            "ade20k",
            "lovedarural",
            "lovedaurban",
            "mhpv1",
            "pidray",
            "houseparts",
            "pizza",
            "toolkits",
            "trash",
            "uecfood",
            "zerowaste",
            "uavid",
        ],
    )
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--nworker", type=int, default=4)
    parser.add_argument(
        "--nprompts",
        type=int,
        default=1,
        choices=[1, 5],
        help="Number of prompts to use (1 or 5)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--log-root", type=str, default="output/")
    parser.add_argument("--save-visualization", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    utils.fix_randseed(args.seed)

    name = args.benchmark
    log_dir = os.path.join(args.log_root, f"{name}_{args.model_name}")

    if os.environ.get("RANK") is not None:
        distributed.init_process_group(backend="nccl")
        device_id, device = (
            int(os.environ["LOCAL_RANK"]),
            torch.device(int(os.environ["LOCAL_RANK"])),
        )
        rank, world_size = distributed.get_rank(), distributed.get_world_size()
        torch.cuda.set_device(device_id)

        logger = DistributedLogger(log_dir=log_dir, rank=rank)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args.device = device

        logger = DistributedLogger(log_dir=log_dir)

    logger.log_args(args)
    logger.info(
        f"Rank of current process: {rank if torch.distributed.is_initialized() else 0}"
    )
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    logger.info(f"Available models: {list(BaseModel.NAME_TO_MODEL.keys())}\n")
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        utils.download_all_models(args.checkpointspath)

    PromptingDataset.initialize(args.img_size, args.datapath)
    dataset = PromptingDataset.build_dataset(args.benchmark, args.nprompts)
    support_set = PromptingDataset.load_support_set(dataset, logger)

    logger.info(f"Initializing {args.model_name} model...")
    model_config = {
        "model_name": args.model_name,
        "config_path": os.path.join("models", args.model_name, "config.json"),
        "checkpointspath": args.checkpointspath,
        "device": device,
        "support_set": support_set,
        "class_ids": dataset.class_ids,
        "class_names": dataset.class_names,
        "ignore_background": dataset.ignore_background,
        "logger": logger,
    }
    model = BaseModel.from_name(**model_config)

    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        if torch.distributed.is_initialized()
        else None
    )

    dataloader_test = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.nworker,
        sampler=sampler,
    )

    nclass = dataset.nclass + 1 if dataset.ignore_background else dataset.nclass
    val_metrics = SegmentationMetrics(
        n_classes=nclass, ignore_background=dataset.ignore_background
    )

    with torch.no_grad():
        test(model, dataloader_test, args=args)
    logger.info("==================== Finished Evaluation ====================")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

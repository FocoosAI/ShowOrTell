import json
import warnings
from dataclasses import dataclass
from typing import Dict

import torch
from tqdm import tqdm

warnings.filterwarnings(
    "ignore", message="Importing from timm.models.registry is deprecated"
)
warnings.filterwarnings(
    "ignore", message="Importing from timm.models.layers is deprecated"
)
warnings.filterwarnings(
    "ignore", message="Overwriting tiny_vit.*in registry", category=UserWarning
)


@dataclass
class ModelConfig:
    model_name: str = ""
    device: str = ""

    @classmethod
    def from_json(cls, config_path: str) -> "ModelConfig":
        """Load configuration from a JSON file.

        Args:
            config_path: Path to the config JSON file.

        Returns:
            ModelConfig: Configuration object with loaded parameters.
        """

        with open(config_path, "r") as f:
            config = json.load(f)
        return cls(**config)


class BaseModel:
    NAME_TO_MODEL: Dict = {}

    def __init__(
        self,
        config_path,
        checkpointspath,
        device,
        support_set,
        class_ids,
        class_names,
        ignore_background,
        logger,
    ):
        self.config_path = config_path
        self.checkpointspath = checkpointspath
        self.device = device
        self.support_set = support_set
        self.class_ids = class_ids
        self.class_names = class_names
        self.ignore_background = ignore_background
        self.logger = logger

    @classmethod
    def register(cls, name, class_type):
        cls.NAME_TO_MODEL[name] = class_type

    @classmethod
    def from_name(cls, **kwargs):
        model_name = kwargs.pop("model_name")

        try:
            model_class = cls.NAME_TO_MODEL[model_name]()
        except KeyError:
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {list(cls.NAME_TO_MODEL.keys())}"
            )

        return model_class(**kwargs)

    def evaluate(self, query_dict):
        tqdm_disabled = (
            torch.distributed.is_initialized() and torch.distributed.get_rank() != 0
        )
        segm_mask = []
        for support_element, class_id in tqdm(
            zip(self.support_set, self.class_ids),
            total=len(self.support_set),
            leave=False,
            disable=tqdm_disabled,
        ):
            support_imgs = support_element["support_imgs"]
            support_masks = support_element["support_masks"]

            data = {
                "support_imgs": support_imgs.unsqueeze(0).to(self.device),
                "support_masks": support_masks.unsqueeze(0).to(self.device),
                "query_img": query_dict["query_img"].to(self.device),
                "query_name": query_dict["query_name"],
                "query_mask": query_dict["query_mask"].to(self.device),
                "class_id": class_id,
            }
            pred_mask, mask_probs = self.forward(data)

            query_mask = query_dict["query_mask"].clone()
            query_mask = (query_mask == class_id).int().to(pred_mask.device)

            segm_mask.append(mask_probs.squeeze())

        if self.ignore_background:
            segm_mask.insert(
                0, torch.zeros_like(pred_mask.squeeze(), device=pred_mask.device)
            )

            for i in range(len(segm_mask)):
                mask_copy = segm_mask[i].clone()
                segm_mask[i] = torch.where(mask_copy > 0.2, mask_copy, 0)

        pred_mask = torch.stack(segm_mask).argmax(dim=0)

        return pred_mask

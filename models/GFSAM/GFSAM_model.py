import os
import sys
from dataclasses import dataclass
from typing import Tuple

sys.path.append(os.path.abspath(os.path.join("models", "GFSAM")))

import torch

from models.base_model import BaseModel, ModelConfig
from models.GFSAM.matcher.GFSAM import build_model


@dataclass
class GFSAMConfig(ModelConfig):
    dinov2_weights: str = ""
    sam_weights: str = ""
    dinov2_size: str = "vit_large"
    sam_size: str = "vit_h"
    img_size: int = 518


class GFSAMModel(BaseModel):
    def __init__(
        self,
        config_path: str,
        checkpointspath,
        device,
        support_set,
        class_ids,
        class_names,
        ignore_background,
        logger,
    ):
        super().__init__(
            config_path,
            checkpointspath,
            device,
            support_set,
            class_ids,
            class_names,
            ignore_background,
            logger,
        )
        config = GFSAMConfig.from_json(config_path)
        config.device = device
        config.dinov2_weights = os.path.join(
            checkpointspath, "dinov2_vitl14_pretrain.pth"
        )
        config.sam_weights = os.path.join(checkpointspath, "sam_vit_h_4b8939.pth")

        self.model = build_model(config)

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the GFSAM model.

        Args:
            data (Dict): Dictionary containing:
                - query_img: The target image to segment
                - query_mask: The mask of the target image
                - query_name: Name/identifier of the query image

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - pred_mask: The predicted segmentation mask
                - prob_masks: Probability maps for the segmentation
        """
        with torch.no_grad():
            query_img, query_name, support_imgs, support_masks = (
                data["query_img"],
                data["query_name"],
                data["support_imgs"],
                data["support_masks"],
            )
            self.model.set_reference(support_imgs, support_masks)
            self.model.set_target(query_img, query_name)
            pred_mask, prob_masks, (points, sel_points) = self.model.predict()

        return pred_mask, prob_masks

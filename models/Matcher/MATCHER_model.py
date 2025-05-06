import os
import sys
from dataclasses import dataclass, field
from typing import Tuple

sys.path.append(os.path.abspath(os.path.join("models", "Matcher")))

import torch

from models.Matcher.matcher.Matcher import build_matcher_oss
from models.base_model import BaseModel, ModelConfig


@dataclass
class MatcherConfig(ModelConfig):
    dinov2_weights: str = ""
    sam_weights: str = ""
    dinov2_size: str = "vit_large"
    sam_size: str = "vit_h"
    points_per_side: int = 64
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    sel_stability_score_thresh: float = 0.9
    iou_filter: float = 0.85
    box_nms_thresh: float = 0.65
    output_layer: int = 3
    dense_multimask_output: int = 0
    use_dense_mask: int = 1
    multimask_output: int = 1
    num_centers: int = 8
    use_box: bool = False
    use_points_or_centers: bool = True
    sample_range: list = field(default_factory=lambda: [1, 6])
    max_sample_iterations: int = 64
    alpha: float = 1.0
    beta: float = 0.0
    exp: float = 0.0
    emd_filter: float = 0.0
    purity_filter: float = 0.02
    coverage_filter: float = 0.0
    use_score_filter: bool = True
    deep_score_filter: float = 0.33
    deep_score_norm_filter: float = 0.1
    topk_scores_threshold: float = 0.0
    num_merging_mask: int = 9


class MatcherModel(BaseModel):
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
        config = MatcherConfig.from_json(config_path)
        config.device = device
        config.dinov2_weights = os.path.join(
            checkpointspath, "dinov2_vitl14_pretrain.pth"
        )
        config.sam_weights = os.path.join(checkpointspath, "sam_vit_h_4b8939.pth")

        self.model = build_matcher_oss(config)

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the Matcher model.

        Args:
            data (Dict): Dictionary containing:
                - query_img: The target image to segment
                - query_name: Name/identifier of the query image
                - query_mask: The mask of the target image
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - pred_mask: The predicted segmentation mask
                - prob_masks: Probability maps for the segmentation
        """
        with torch.no_grad():
            query_img, _, support_imgs, support_masks = (
                data["query_img"],
                data["query_name"],
                data["support_imgs"],
                data["support_masks"],
            )
            self.model.set_reference(support_imgs, support_masks)
            self.model.set_target(query_img)
            pred_mask, prob_masks = self.model.predict()

        return pred_mask[0], prob_masks

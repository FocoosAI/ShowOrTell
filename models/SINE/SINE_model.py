import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Tuple

sys.path.append(os.path.abspath(os.path.join("models", "SINE")))

import torch

from models.base_model import BaseModel, ModelConfig
from models.SINE.inference_fss.model.model import build_model
from models.SINE.tools.eval_fss import wrap_data


@dataclass
class SINEConfig(ModelConfig):
    dinov2_weights: str = ""
    sine_weights: str = ""
    img_size: int = 518
    pad_size: int = 896
    pt_model: str = "dinov2"
    image_enc_use_fc: bool = False
    score_threshold: float = 0.7
    dinov2_size: str = "vit_large"
    transformer_depth: int = 6
    feat_chans: int = 256
    transformer_nheads: int = 8
    transformer_mlp_dim: int = 2048
    transformer_mask_dim: int = 256
    transformer_fusion_layer_depth: int = 1
    transformer_num_queries: int = 200
    transformer_pre_norm: bool = True


class SINEModel(BaseModel):
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
        config = SINEConfig.from_json(config_path)
        config.device = device
        config.dinov2_weights = os.path.join(
            checkpointspath, "dinov2_vitl14_pretrain.pth"
        )
        config.sine_weights = os.path.join(checkpointspath, "sine_checkpoint.bin")

        self.model = build_model(config)
        state_dict = torch.load(config.sine_weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the SINE model.

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
            args = SimpleNamespace()
            args.pad_size = 896
            wrap_batch = wrap_data(data, args)

            pred = self.model(wrap_batch)
            res = pred["sem_seg"] > self.model.score_threshold
            pred_mask = res.float()

        return pred_mask, pred["sem_seg"]

import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join("models", "PersonalizeSAM")))

from models.base_model import BaseModel, ModelConfig
from models.PersonalizeSAM.per_segment_anything import SamPredictor, sam_model_registry


@dataclass
class PersonalizeSAMConfig(ModelConfig):
    sam_weights: str = ""
    sam_size: str = "vit_h"


class PersonalizeSAMModel(BaseModel):
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
        config = PersonalizeSAMConfig.from_json(config_path)
        config.device = device
        config.sam_weights = os.path.join(checkpointspath, "sam_vit_h_4b8939.pth")

        sam = sam_model_registry[config.sam_size](checkpoint=config.sam_weights).to(
            config.device
        )
        self.model = SamPredictor(sam)

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the PersonalizeSAM model.

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
        ref_image, ref_mask, test_image = (
            data["support_imgs"],
            data["support_masks"],
            data["query_img"],
        )

        def point_selection(mask_sim, topk=1):
            # Top-1 point selection
            w, h = mask_sim.shape
            topk_xy = mask_sim.flatten(0).topk(topk)[1]
            topk_x = (topk_xy // h).unsqueeze(0)
            topk_y = topk_xy - topk_x * h
            topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
            topk_label = np.array([1] * topk)
            topk_xy = topk_xy.cpu().numpy()

            # Top-last point selection
            last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
            last_x = (last_xy // h).unsqueeze(0)
            last_y = last_xy - last_x * h
            last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
            last_label = np.array([0] * topk)
            last_xy = last_xy.cpu().numpy()

            return topk_xy, topk_label, last_xy, last_label

        # print("======> Obtain Location Prior")
        ref_image = ref_image.squeeze().permute(1, 2, 0).cpu().numpy()
        ref_mask = ref_mask.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # Image features encoding
        ref_mask = self.model.set_image(ref_image, ref_mask)
        ref_feat = self.model.features.squeeze().permute(1, 2, 0)

        ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0:2], mode="bilinear")
        ref_mask = ref_mask.squeeze()[0]

        # Target feature extraction
        target_feat = ref_feat[ref_mask > 0]
        target_embedding = target_feat.mean(0).unsqueeze(0)
        target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
        target_embedding = target_embedding.unsqueeze(0)

        # print('======> Start Testing')
        # Image feature encoding
        test_image = test_image.squeeze().permute(1, 2, 0).cpu().numpy()
        self.model.set_image(test_image)
        test_feat = self.model.features.squeeze()

        # Cosine similarity
        C, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(C, h * w)
        sim = target_feat @ test_feat

        similarity = sim.reshape(1, 1, h, w)  # 1,1,64,64
        sim = F.interpolate(similarity, scale_factor=4, mode="bilinear")
        sim = self.model.model.postprocess_masks(
            sim,
            input_size=self.model.input_size,
            original_size=self.model.original_size,
        ).squeeze()

        # Positive-negative location prior
        topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=1)
        topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
        topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

        # Obtain the target guidance for cross-attention layers
        sim = (sim - sim.mean()) / torch.std(sim)
        sim = F.interpolate(
            sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear"
        )
        attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)

        # First-step prediction
        masks, scores, logits, _ = self.model.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            multimask_output=False,
            attn_sim=attn_sim,  # Target-guided Attention
            target_embedding=target_embedding,  # Target-semantic Prompting
        )
        best_idx = 0

        # Cascaded Post-refinement-1
        masks, scores, logits, _ = self.model.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            mask_input=logits[best_idx : best_idx + 1, :, :],
            multimask_output=True,
        )
        best_idx = np.argmax(scores)

        # Cascaded Post-refinement-2
        y, x = np.nonzero(masks[best_idx])
        if x.shape[0] != 0 or y.shape[0] != 0:
            x_min = x.min()
            x_max = x.max()
            y_min = y.min()
            y_max = y.max()
            input_box = np.array([x_min, y_min, x_max, y_max])
            masks, scores, logits, _ = self.model.predict(
                point_coords=topk_xy,
                point_labels=topk_label,
                box=input_box[None, :],
                mask_input=logits[best_idx : best_idx + 1, :, :],
                multimask_output=True,
            )
            best_idx = np.argmax(scores)

        best_mask = torch.tensor(masks[best_idx], device=similarity.device)  # H,W

        confidence_rsz = F.interpolate(
            similarity,
            best_mask.shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        curr_prob_mask = (
            (confidence_rsz * best_mask).sum() / best_mask.sum() * best_mask
        )

        return best_mask, curr_prob_mask

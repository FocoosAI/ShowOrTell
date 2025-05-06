from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY

from dinov2.models import vision_transformer as vits
import dinov2.utils.utils as dinov2_utils
from fsdet.model.image_encoder import DINOv2EncoderViT
from fsdet.model.transofrmer_decoder import build_mformer


@META_ARCH_REGISTRY.register()
class SINE(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        image_encoder: nn.Module,
        transformer_decoder: nn.Module = None,
        # inference
        preprocess: bool = True,
        sem_seg_postprocess_before_inference: bool = True,
        num_classes: int = 1,
        test_topk_per_image: int = 100,
        score_threshold: float = 0.,
        use_id_query: bool = True,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ):
        super(SINE, self).__init__()

        self.image_encoder = image_encoder
        self.transformer_decoder = transformer_decoder
        self.preprocess = preprocess
        self.num_classes = num_classes
        self.test_topk_per_image = test_topk_per_image
        self.score_threshold = score_threshold
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        if preprocess:
            self.ref_feat_dict = {}
            self.ref_mask_dict = {}

        # additional args
        self.use_id_query = use_id_query
        assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):

        # DINOv2, Image Encoder
        dinov2_kwargs = dict(
            img_size=518,
            patch_size=14,
            init_values=1e-5,
            ffn_layer='mlp',
            block_chunks=0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
        )
        dinov2 = vits.__dict__["vit_large"](**dinov2_kwargs)

        dinov2_utils.load_pretrained_weights(dinov2, cfg.MODEL.DINO.WEIGHTS, "teacher")
        dinov2.eval()
        image_encoder = DINOv2EncoderViT(dinov2, out_chans=cfg.MODEL.DINO.OUT_CHANS)

        # transformer
        transformer = build_mformer(cfg)

        return {
            "image_encoder": image_encoder,
            "transformer_decoder": transformer,
            "preprocess": cfg.MODEL.SINE.preprocess,
            "sem_seg_postprocess_before_inference": cfg.MODEL.SINE.sem_seg_postprocess_before_inference,
            "score_threshold": cfg.MODEL.SINE.score_threshold,
            "use_id_query": cfg.MODEL.SINE.use_id_query
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def get_sam_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings = self.segmenter.image_encoder(pixel_values)
        return image_embeddings

    def get_enc_embs(self, pixel_values: torch.FloatTensor):

        with torch.no_grad():
            image_embeddings = self.image_encoder.get_enc_embs(pixel_values)
        image_embeddings = self.image_encoder.neck(image_embeddings)

        return image_embeddings

    def do_preprocess(self, features, images, batched_inputs):
        assert "instances" in batched_inputs[0], "reference masks must be provided!"
        instances = [x["instances"].to(self.device) for x in batched_inputs]
        targets = self.prepare_targets(instances, images)
        masks = [target['masks'] for target in targets]
        labels = [target['labels'] for target in targets]

        bs, c, h, w = features.shape

        for feature, mask, label in zip(features, masks, labels):
            nm = len(mask)

            mask_for_pooling = F.interpolate(mask[None, ...].to(feature.dtype), size=(h*4, w*4),
                                             mode='bilinear', align_corners=False).to(feature.dtype)[0]
            feat_for_pooling = F.interpolate(feature[None, ...], size=mask_for_pooling.shape[-2:],
                                                 mode='bilinear', align_corners=False).repeat(nm, 1, 1, 1)

            for m, f, l in zip(mask_for_pooling, feat_for_pooling, label.cpu().tolist()):
                if l not in self.ref_feat_dict:
                    self.ref_feat_dict[l] = []
                    self.ref_mask_dict[l] = []
                self.ref_feat_dict[l].append(f.cpu())
                self.ref_mask_dict[l].append(m.cpu())

    def integrate_queries(self):

        id_labels = []
        id_queries = []

        seg_labels = []
        seg_queries = []

        for label in self.ref_feat_dict.keys():
            feats = self.ref_feat_dict[label]
            masks = self.ref_mask_dict[label]

            # cal id query
            for feat, mask in zip(feats, masks):
                id_query = self.transformer_decoder.mask_pooling(feat[None, ...], mask[None, None, ...])[0]
                id_queries.append(id_query)
                id_labels.append(label)

            # cal seg query
            masks = [(mask > 0).to(mask.dtype) for mask in masks]
            denorm = sum([mask.sum() for mask in masks]) + 1e-8
            masks = [mask / denorm for mask in masks]

            seq_query = 0
            for feat, mask in zip(feats, masks):
                q = (feat * mask[None, ...]).sum(dim=(-1, -2))
                seq_query += q

            seg_queries.append(seq_query[None, ...])
            seg_labels.append(label)

        self.register_buffer("id_labels", torch.tensor(id_labels, dtype=torch.int64, device=self.device))
        self.register_buffer("id_queries", torch.cat(id_queries, dim=0).to(self.device))
        self.register_buffer("seg_labels", torch.tensor(seg_labels, dtype=torch.int64, device=self.device))
        self.register_buffer("seg_queries", torch.cat(seg_queries, dim=0).to(self.device))

        return self.id_labels, self.id_queries, self.seg_labels, self.seg_queries

    def forward(self, batched_inputs):

        assert not self.training

        # prepare dinov2 features
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = ImageList.from_tensors(images, size_divisibility=self.image_encoder.patch_size)

        features = self.get_enc_embs(images.tensor)

        if self.preprocess:
            self.do_preprocess(features, images, batched_inputs)
            return None

        if not self.use_id_query:
            id_queries = torch.randn((0,self.id_queries.shape[-1])).to(self.id_queries)
            id_labels = torch.randn((0, self.id_labels.shape[-1])).to(self.id_labels)
        else:
            id_queries = self.id_queries
            id_labels = self.id_labels

        outputs = self.transformer_decoder(
            id_queries=id_queries[None, ...],
            seg_queries=self.seg_queries[None, ...],
            tar_features=features,
            id_labels=id_labels[None, ...],
            seg_labels=self.seg_labels[None, ...]
        )


        processed_results = []

        for output, input_per_image, image_size in zip(
                outputs, batched_inputs, images.image_sizes
        ):
            ref_sem_labels = output['ref_sem_labels']
            pred_ins_logits = output['pred_ins_logits']
            pred_ins_masks = output['pred_ins_masks']
            processed_results.append({})


            # upsample masks
            pred_ins_masks = F.interpolate(
                pred_ins_masks,
                size=image_size,
                mode="bilinear",
                align_corners=False,
            )

            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])

            new_size = (input_per_image['instances'].image_size[0], input_per_image['instances'].image_size[1])

            if self.sem_seg_postprocess_before_inference:

                pred_ins_masks = retry_if_cuda_oom(sem_seg_postprocess)(
                    pred_ins_masks[0], new_size, height, width
                )
                pred_ins_logits = pred_ins_logits.to(pred_ins_masks)[0]
                ref_sem_labels = ref_sem_labels.to(pred_ins_masks)[0]


            r = retry_if_cuda_oom(self.instance_inference)(pred_ins_masks, pred_ins_logits, ref_sem_labels)
            if not self.sem_seg_postprocess_before_inference:
                r = retry_if_cuda_oom(sem_seg_postprocess)(
                    r, new_size, height, width
                )
            processed_results[-1]["instances"] = r

        return processed_results

    def instance_inference(self, pred_masks, pred_logits, labels):

        # inference instances
        image_size = pred_masks.shape[-2:]
        # [Q, K]
        scores = F.softmax(pred_logits, dim=-1)[:, :-1]
        scores = scores.flatten(0, 1)
        num_classes = len(labels)
        labels = labels.unsqueeze(0).repeat(self.transformer_decoder.num_queries, 1).flatten(0, 1)

        # First, select top-k based on score
        scores_per_image, topk_indices = scores.topk(min(self.test_topk_per_image, len(scores)), sorted=False)  # select top-100
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // num_classes
        pred_masks = pred_masks[topk_indices]
        pred_masks_ = (pred_masks > 0).float()

        # Second, filter instances below the threshold

        # 1. calculate average mask prob
        mask_scores_per_image = (pred_masks.sigmoid().flatten(1) * pred_masks_.flatten(1)).sum(1) / (
                pred_masks_.flatten(1).sum(1) + 1e-6)
        # 2. calculate scores
        scores = scores_per_image * mask_scores_per_image
        # 3. filter
        vaild = scores > self.score_threshold
        scores_per_image = scores[vaild]
        pred_masks_ = pred_masks_[vaild]
        labels_per_image = labels_per_image[vaild]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = pred_masks_

        result.scores = scores_per_image
        result.pred_classes = labels_per_image.to(torch.int64)

        # get bbox from mask
        pred_boxes = torch.zeros(pred_masks_.size(0), 4)
        for i in range(pred_masks_.size(0)):
           mask = pred_masks_[i].squeeze()
           ys, xs = torch.where(mask)
           try:
                pred_boxes[i] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()]).float()
           except:
               pred_boxes[i] = torch.tensor([0,0,1,1]).to(mask).float()

        result.pred_boxes = Boxes(pred_boxes)

        return result

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets


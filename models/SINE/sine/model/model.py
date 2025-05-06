from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from dinov2.models import vision_transformer as vits
import dinov2.utils.utils as dinov2_utils

from sine.model.image_encoder import DINOv2EncoderViT
from sine.model.transformer_decoder.mformer import build_mformer, MLP
from sine.model.matcher import HungarianMatcher
from sine.model.criterion import SetCriterion

class SINE(nn.Module):

    def __init__(
        self,
        image_encoder: nn.Module,
        transformer_decoder: nn.Module = None,
        criterion: nn.Module = None,
        # inference
        semantic_on: bool = True,
        instance_on: bool = False,
        identity_on: bool = False,
        sem_seg_postprocess_before_inference: bool = False,
        num_classes: int = 1,
        test_topk_per_image: int = 100,
        score_threshold: float = 0.8,

        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ):
        super(SINE, self).__init__()

        self.image_encoder = image_encoder
        self.transformer_decoder = transformer_decoder
        self.criterion = criterion
        self.num_classes = num_classes
        self.test_topk_per_image = test_topk_per_image
        self.score_threshold = score_threshold
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.identity_on = identity_on

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        for n, param in self.image_encoder.encoder.named_parameters():
            param.requires_grad = False

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

    def forward(self, batched_inputs):
        batched_ref_inputs = [item['ref_dict'] for item in batched_inputs]
        batched_tar_inputs = [item['tar_dict'] for item in batched_inputs]

        # prepare dinov2 features
        ref_enc_images = [x["image"].to(self.device) for x in batched_ref_inputs]
        ref_enc_images = ImageList.from_tensors(ref_enc_images, size_divisibility=self.image_encoder.patch_size)

        tar_enc_images = [x["image"].to(self.device) for x in batched_tar_inputs]
        tar_enc_images = ImageList.from_tensors(tar_enc_images, size_divisibility=self.image_encoder.patch_size)

        ref_features = self.get_enc_embs(ref_enc_images.tensor)
        tar_features = self.get_enc_embs(tar_enc_images.tensor)


        assert "instances" in batched_ref_inputs[0], "reference masks must be provided!"
        ref_instances = [x["instances"].to(self.device) for x in batched_ref_inputs]
        ref_targets = self.prepare_targets(ref_instances, ref_enc_images)
        ref_masks = [ref_target['masks'] for ref_target in ref_targets]
        ref_labels = [ref_target['labels'] for ref_target in ref_targets]

        outputs = self.transformer_decoder(
            ref_features=ref_features,
            tar_features=tar_features,
            ref_masks=ref_masks,
            ref_labels=ref_labels
        )

        if self.training:
            # mask classification target
            if "instances" in batched_tar_inputs[0]:
                tar_instances = [x["instances"].to(self.device) for x in batched_tar_inputs]
                tar_targets = self.prepare_targets(tar_instances, tar_enc_images)
            else:
                tar_targets = None

            for out, tar_target in zip (outputs, tar_targets):

                ref_labels_unique = out['ref_sem_labels']
                tar_labels_unique = torch.unique(tar_target['labels'])
                assert all(torch.isin(tar_labels_unique, ref_labels_unique))

                out['ref_sem_labels_t'] = torch.tensor(list(range(out['ref_sem_labels'].shape[1]))).to(out['ref_sem_labels'])
                ref_sem_labels_list = out['ref_sem_labels'][0].cpu().tolist()
                tgt_ids = tar_target["labels"]
                tar_target['labels_t'] = torch.tensor([ref_sem_labels_list.index(tid) for tid in tgt_ids]).to(tgt_ids)

            # bipartite matching-based loss
            losses = self.criterion(outputs, ref_targets, tar_targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            processed_results = []

            for output, input_per_image, image_size in zip(
                    outputs, batched_tar_inputs, tar_enc_images.image_sizes
            ):
                ref_id_labels = output['ref_id_labels']
                ref_sem_labels = output['ref_sem_labels']
                pred_id_logits = output['pred_id_logits']
                pred_ins_logits = output['pred_ins_logits']
                pred_id_masks = output['pred_id_masks']
                pred_ins_masks = output['pred_ins_masks']
                processed_results.append({})


                # upsample masks
                pred_id_masks = F.interpolate(
                    pred_id_masks,
                    size=image_size,
                    mode="bilinear",
                    align_corners=False,
                )
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
                    pred_id_masks = retry_if_cuda_oom(sem_seg_postprocess)(
                        pred_id_masks[0], new_size, height, width
                    )
                    pred_ins_masks = retry_if_cuda_oom(sem_seg_postprocess)(
                        pred_ins_masks[0], new_size, height, width
                    )
                    pred_id_logits = pred_id_logits.to(pred_id_masks)[0]
                    pred_ins_logits = pred_ins_logits.to(pred_ins_masks)[0]
                    ref_id_labels = ref_id_labels.to(pred_id_masks)[0]
                    ref_sem_labels = ref_sem_labels.to(pred_ins_masks)[0]

                if self.identity_on:
                    r = retry_if_cuda_oom(self.id_inference)(pred_id_masks, pred_id_logits, ref_id_labels)
                    processed_results[-1]["id_seg"] = r
                elif self.instance_on:
                    r = retry_if_cuda_oom(self.instance_inference)(pred_ins_masks, pred_ins_logits, ref_sem_labels)
                    processed_results[-1]["ins_seg"] = r
                elif self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(pred_ins_masks, pred_ins_logits, ref_sem_labels)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(
                            r, new_size, height, width
                        )
                    processed_results[-1]["sem_seg"] = r

            return processed_results[0] if len(processed_results) == 1 else processed_results

    def id_inference(self, pred_masks, pred_logits, labels):

        # inference instances
        image_size = pred_masks.shape[-2:]
        # [Q, K]
        scores = F.softmax(pred_logits, dim=-1)[:, :-1]
        scores = scores.flatten(0, 1)

        # First, select top-k based on score
        scores_per_image, topk_indices = scores.topk(min(self.test_topk_per_image, len(scores)), sorted=False)  # select top-100
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // self.num_classes
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
        # result.pred_boxes = Boxes(torch.zeros(pred_masks_.size(0), 4))
        result.scores = scores_per_image
        result.pred_classes = labels_per_image

        return result

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
        # result.pred_boxes = Boxes(torch.zeros(pred_masks_.size(0), 4))
        result.scores = scores_per_image
        result.pred_classes = labels_per_image

        return result

    def semantic_inference(self, pred_masks, pred_logits, labels):

        pred_masks = pred_masks[0]
        pred_logits = pred_logits[0]

        pred_logits = F.softmax(pred_logits, dim=-1)[..., :-1]
        pred_masks = pred_masks.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", pred_logits, pred_masks)

        return semseg

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
                    "ids": targets_per_image.ins_ids,
                }
            )
        return new_targets


def build_model(args):

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

    dinov2 = vits.__dict__[args.dinov2_size](**dinov2_kwargs)

    dinov2_utils.load_pretrained_weights(dinov2, args.dinov2_weights, "teacher")
    dinov2.eval()
    image_encoder = DINOv2EncoderViT(dinov2, out_chans=args.feat_chans, use_fc=args.image_enc_use_fc)

    transformer = build_mformer(args)

    # building criterion
    matcher = HungarianMatcher(
        cost_class=args.class_weight,
        cost_mask=args.mask_weight,
        cost_dice=args.dice_weight,
        num_points=args.train_num_points,
    )

    weight_dict = {"loss_ce_ins": args.class_weight, "loss_ce_id": args.class_weight, "loss_mask": args.mask_weight,
                   "loss_dice": args.dice_weight}

    if args.deep_supervision:
        dec_layers = args.transformer_depth
        aux_weight_dict = {}
        for i in range(dec_layers):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "masks"]

    criterion = SetCriterion(
        num_classes=1,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.no_object_weight,
        losses=losses,
        num_points=args.train_num_points,
        oversample_ratio=args.oversample_ratio,
        importance_sample_ratio=args.importance_sample_ratio,
    )

    model = SINE(
        image_encoder=image_encoder,
        transformer_decoder=transformer,
        criterion=criterion,
        score_threshold=args.score_threshold
    )

    return model

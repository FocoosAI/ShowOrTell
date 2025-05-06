# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_sine_config(cfg):

    # matcher fsod inference config
    cfg.MODEL.META_ARCHITECTURE = "SINE"
    cfg.MODEL.DINO = CN(new_allowed=True)
    cfg.MODEL.DINO.WEIGHTS = ""
    cfg.MODEL.DINO.OUT_CHANS = 256
    cfg.MODEL.SINE = CN(new_allowed=True)
    cfg.MODEL.SINE.sem_seg_postprocess_before_inference = True
    cfg.MODEL.SINE.test_topk_per_image = 100
    cfg.MODEL.SINE.score_threshold = 0.7
    cfg.MODEL.Transformer = CN(new_allowed=True)
    
    # input
    cfg.INPUT.MIN_SIZE_TEST = 896
    cfg.INPUT.MAX_SIZE_TEST = 896
    cfg.INPUT.IMAGE_SIZE = 896


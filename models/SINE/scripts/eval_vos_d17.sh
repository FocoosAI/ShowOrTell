#!/bin/bash

TYPE=linear
RATIO=20
MODEL=sine
WEIGHT=outputs/pt_sine/pytorch_model.bin
dataset=D17
N_FRAME=6

LOG=${MODEL}_1036_30e_1110_${N_FRAME}_fixfirst
echo ${LOG}

CUDA_VISIBLE_DEVICES=0 python tools/eval_vos.py \
  --dataset ${dataset} \
  --sine-weights ${WEIGHT} \
  --output outputs/vos/${LOG} \
  --img-size 1036 \
  --pad-size 1036 \
  --num_frame ${N_FRAME} \
  --memory_decay_type ${TYPE} \
  --memory_decay_ratio ${RATIO} \
  --fix_first_frame
python inference_vos/davis2017-evaluation/evaluation_method.py \
  --davis_path datasets/vos/DAVIS17/trainval \
  --set val \
  --task semi-supervised \
  --results_path outputs/vos/${LOG}


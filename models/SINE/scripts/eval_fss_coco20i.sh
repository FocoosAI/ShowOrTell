#!/bin/bash

shot=1
data=coco
fold=0
weight=outputs/pt_sine/pytorch_model.bin
output=outputs/fss/${data}/${shot}shot/fold${fold}
echo ${output}

CUDA_VISIBLE_DEVICES=0 python tools/eval_fss.py  \
  --benchmark ${data} \
  --fold ${fold} \
  --nshot ${shot} \
  --sine-weights ${weight} \
  --log-root ${output}

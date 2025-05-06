#!/bin/bash

weights=outputs/pt_sine/pytorch_model.bin
shots=1
seed=0

config=configs/COCO/${shots}shots/seed${seed}.yaml
output_dir=outputs/fsod/coco/${shots}shot/seed${seed}

echo ${output_dir}

python tools/preprocess.py \
  --config-file ${config} \
  --opts OUTPUT_DIR ${output_dir} \
  MODEL.SINE.preprocess True \
  MODEL.WEIGHTS ${weights}

python tools/test.py \
  --num-gpus=8 \
  --config-file ${config} \
  --opts MODEL.SINE.preprocess False \
  OUTPUT_DIR ${output_dir} \
  MODEL.WEIGHTS ${weights}

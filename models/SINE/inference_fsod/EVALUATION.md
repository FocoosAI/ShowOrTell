## Evaluation - Few-shot Instance Segmentation


### Prepare data

Download following datasets:


> #### COCO 2014
> - Download [cocosplit](https://drive.google.com/file/d/12jGNdhdL8jz5YO8Gz5P-liNtY7eAz6Av/view).
> #### VOC
> - Download [voc-cocostyle.zip](https://drive.google.com/file/d/16P6ewBkjg5WguQEzH3-GfnhHVht1cN-9/view?usp=sharing).

Create a directory 'datasets' for the above datasets in 'inference_fsod/' and appropriately place each dataset to have following directory structure:

    datasets/
    ├── coco/           
    │   ├── annotations/
    │   │   └── instances_{train,val}2014.json
    │   └── {train,val}2014/
    ├── cocosplit/
    │   ├── datasplit/
    │   ├── seed0/
    │   ├── seed1/
    │   └── ...
    ├── VOCOutput/
    │   ├── annotations
    │   │   ├── train.json
    │   │   ├── val.json
    │   │   └── val_converted.json
    │   ├── train/
    │   ├── val/



### Testing


```
cd inference_fsod

python tools/preprocess.py \
  --config-file configs/COCO/1shots/seed0.yaml \
  --opts OUTPUT_DIR /path/to/fsod/coco/1shot/seed0 \
  MODEL.SINE.preprocess True \
  MODEL.WEIGHTS /path/to/pt_sine/pytorch_model.bin

python tools/test.py \
  --num-gpus=8 \
  --config-file configs/COCO/1shots/seed0.yaml \
  --opts MODEL.SINE.preprocess False \
  OUTPUT_DIR /path/to/fsod/coco/1shot/seed0 \
  MODEL.WEIGHTS /path/to/pt_sine/pytorch_model.bin
```
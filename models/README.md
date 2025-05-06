# 🚀 Implementing Your Model in ShowOrTell

> Guide to integrate your custom models into our benchmark framework

## 🧩 Choose Your Implementation Method

- [PyTorch Implementation](#-pytorch-implementation)
- [MMSegmentation Integration](#-mmsegmentation-integration)

## 🔥 PyTorch Implementation

Follow these simple steps to integrate your PyTorch model into our benchmark.

### 📁 Step 1: Add Your Model Files

```bash
cd models
git clone <git_of_your_model>
cd <your_model>
```

### 🏗️ Step 2: Create The Model Class

To ensure compatibility with our benchmark, your model must implement our interface. Create a new Python file `<YOUR_MODEL_NAME>_model.py` in your model folder with the following structure:

```python
from dataclasses import dataclass
from typing import Tuple

import torch
from models.base_model import BaseModel, ModelConfig

@dataclass
class <YOUR_MODEL_NAME>Config(ModelConfig):
    # Add here all parameters required by your model
    
    # Example:
    # dinov2_size: str = "vit_large"

class <YOUR_MODEL_NAME>Model(BaseModel):
  def __init__(
      self,
      config_path: str,
      checkpointspath,
      device,
      support_set,
      class_ids,
      ignore_background,
      logger,
  ):
      super().__init__(
          config_path,
          checkpointspath,
          device,
          support_set,
          class_ids,
          ignore_background,
          logger,
      )
      config = <YOUR_MODEL>Config.from_json(config_path)
      # Here build your model
      # self.model = ...
      
  def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
      """Forward pass for the model.

      Args:
          data (Dict): Dictionary containing:
              - query_img
              - query_name
              - query_mask

      Returns:
          Tuple[torch.Tensor, torch.Tensor]:
              - pred_mask: The predicted segmentation mask
              - prob_masks: Probability maps for the segmentation
      """

      # Implement your model's forward pass here
      # The output must be the predicted mask and the associated probability
```

### ⚙️ Step 3: Add Configuration File

Create a `config.json` file in the root folder of your model. This file should specify all parameters required by your model, as defined in your `<YOUR_MODEL_NAME>Config` dataclass.

### 📝 Step 4: Register Your Model

Create an `__init__.py` in the root of your model with the following:

```python
import os
from models.base_model import BaseModel

def import_model():
    from .<YOUR_MODEL_NAME>_model.py import <YOUR_MODEL_NAME>Model
    return <YOUR_MODEL_NAME>Model

path_components = os.path.abspath(__file__).split(os.sep)
models_index = path_components.index("models")
if models_index + 1 < len(path_components):
    model_name = path_components[models_index + 1]

BaseModel.register(model_name, import_model)
```

After that, add your model folder to the `__init__.py` file in the `models/` folder to register it.

### 🔄 Step 5: Add Required Checkpoints (Optional)

All checkpoints required by existing models are stored in the `models/checkpoints` folder. Add any checkpoints your model needs to this location.

## 📚 MMSegmentation Integration

Follow these detailed guides for each version of the MMSegmentation library.

For all MMSegmentation integrations, place datasets inside the `<YOUR_MODEL>/data` folder or create a symbolic link.

<details>
  <summary><b>MaskCLIP-based Models</b></summary>

  ### Step 1: Create the Dataset Class
  In the `mmseg/datasets` folder, create a file `prompting.py` with the following:

  ```python
  import colorsys
  import os

  from .builder import DATASETS
  from .custom import CustomDataset

  @DATASETS.register_module()
  class PromptingDataset(CustomDataset):
      def __init__(self, dataset_name, **kwargs):
          kwargs["data_root"] = os.path.join(kwargs.get("data_root"), dataset_name)
          img_dir = "valid"
          ann_dir = "annotations/valid"

          classes = self.__get_classes(kwargs["data_root"])
          palette = self.__generate_palette(len(classes))

          super(PromptingDataset, self).__init__(
              img_dir=img_dir,
              ann_dir=ann_dir,
              classes=classes,
              palette=palette,
              reduce_zero_label=False,
              **kwargs,
          )

      def __generate_palette(self, num_classes):
          palette = [[0, 0, 0]]
          for i in range(num_classes):
              hue = i / num_classes
              rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
              rgb_list = [int(x * 255) for x in rgb]
              palette.append(rgb_list)
          return palette

      def __get_classes(self, data_root):
          classes = ()
          with open(os.path.join(data_root, "classes.txt")) as f:
              for line in f:
                  classes = classes + (line.strip(),)
          print(classes)
          return classes
  ```
  Remember to import the newly created class in the `__init__.py` file.

  ### Step 2: Add Configuration Files

  Create a new configuration file for each dataset in the `configs/_base_/datasets` folder.

  > ⚠️ **IMPORTANT**
  >
  > - Customize the `dataset_name` and `img_suffix` accordingly.
  > - Some models may require different parameters in the test_pipeline or additional model settings. Check existing configuration files for examples

  Example configuration file:

  ```python
  dataset_type = "PromptingDataset"
  data_root = "./data"
  img_norm_cfg = dict(
      mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
  )
  crop_size = (512, 512)
  test_pipeline = [
      dict(type="LoadImageFromFile"),
      dict(
          type="MultiScaleFlipAug",
          img_scale=(2048, 512),
          flip=False,
          transforms=[
              dict(type="Resize", keep_ratio=True),
              dict(type="RandomFlip"),
              dict(type="Normalize", **img_norm_cfg),
              dict(type="ImageToTensor", keys=["img"]),
              dict(type="Collect", keys=["img"]),
          ],
      ),
  ]

  data = dict(
      samples_per_gpu=4,
      workers_per_gpu=4,
      test=dict(
          type=dataset_type,
          dataset_name="<DATASET_NAME>",  # Change this accordingly!
          data_root=data_root,
          pipeline=test_pipeline,
          img_suffix=".jpg/.png/.jpeg",  # Change this accordingly!
          seg_map_suffix=".png",
      )
  )
  ```

  In the `configs/<MODEL_NAME>` folder, create a new configuration file for each dataset:

  ```python
  _base_ = [
    '../_base_/models/<MODEL_CONFIG>.py', '../_base_/datasets/<DATASET_CONFIG>.py', 
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
  ]
  model = dict(
      decode_head=dict(
          num_classes=<DATASET_NUM_CLASSES>,
          text_categories=<DATASET_NUM_CLASSES>, 
          text_channels=512,
          text_embeddings_path='./pretrain/<PATH_TO_EXTRACTED_TEXT_EMBEDDINGS>.pth',
          visual_projs_path='./pretrain/ViT16_clip_weights.pth',
      ),
  )
  ```

  ### Step 3: Run Evaluation

  Use this command to run the evaluation:

  ```bash
  python tools/test.py configs/<MODEL_NAME>/<DATASET_CONFIG_STEP_2>.py <PRETRAIN_PATH> --eval mIoU
  ```
  > 📝 **NOTE**: Command may vary depending on your model. Check the model documentation.
</details>

<details>
  <summary><b>MMCV < 2.0</b></summary>

  ### Step 1: Create the Dataset Class
  In the `segmentation/datasets` folder, create a file `prompting.py` with the following:

  ```python
  import colorsys
  import os

  from mmseg.datasets import DATASETS, CustomDataset

  @DATASETS.register_module()
  class PromptingDataset(CustomDataset):
      def __init__(self, dataset_name, **kwargs):
          kwargs["data_root"] = os.path.join(kwargs.get("data_root"), dataset_name)
          img_dir = "valid"
          ann_dir = "annotations/valid"

          classes = self.__get_classes(kwargs["data_root"])
          palette = self.__generate_palette(len(classes))

          super(PromptingDataset, self).__init__(
              img_dir=img_dir,
              ann_dir=ann_dir,
              classes=classes,
              palette=palette,
              reduce_zero_label=False,
              **kwargs,
          )

      def __generate_palette(self, num_classes):
          palette = [[0, 0, 0]]
          for i in range(num_classes):
              hue = i / num_classes
              rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
              rgb_list = [int(x * 255) for x in rgb]
              palette.append(rgb_list)
          return palette

      def __get_classes(self, data_root):
          classes = ()
          with open(os.path.join(data_root, "classes.txt")) as f:
              for line in f:
                  classes = classes + (line.strip(),)
          print(classes)
          return classes
  ```
  Remember to import the newly created class in the `__init__.py` file.

  ### Step 2: Add Configuration Files

  In `segmentation/configs/_base_/custom_import.py` add the import of your dataset class (e.g., `segmentation.datasets.prompting`).
  
  > ⚠️ **IMPORTANT**
  >
  > - Customize the `dataset_name` and `img_suffix` accordingly.
  > - Some models may require different parameters in the test_pipeline or additional model settings. Check existing configuration files for examples.

  Create configuration files for each dataset in the `segmentation/configs/_base_/datasets` folder:

  ```python
  _base_ = ["../custom_import.py"]
  dataset_type = "PromptingDataset"
  data_root = "./data"

  test_pipeline = [
      dict(type="LoadImageFromFile"),
      dict(type="ToRGB"),
      dict(
          type="MultiScaleFlipAug",
          img_scale=(2048, 448),
          flip=False,
          transforms=[
              dict(type="Resize", keep_ratio=True),
              dict(type="RandomFlip"),
              dict(type="ImageToTensorV2", keys=["img"]),
              dict(
                  type="Collect",
                  keys=["img"],
                  meta_keys=["ori_shape", "img_shape", "pad_shape", "flip", "img_info"],
              ),
          ],
      ),
  ]

  data = dict(
      test=dict(
          type=dataset_type,
          dataset_name="<DATASET_NAME>",  # Change this accordingly!
          data_root=data_root,
          pipeline=test_pipeline,
          img_suffix=".jpg/.png/.jpeg",  # Change this accordingly!
          seg_map_suffix=".png",
      )
  )

  test_cfg = dict(mode="slide", stride=(224, 224), crop_size=(448, 448))
  ```

  ### Step 3: Modify the Model Config

  Update the model configs to enable evaluation on benchmark datasets:
  1. Add datasets to the `evaluate` section by adding the name with config path
  2. Add dataset names to the `evaluate.task` section

  ### Step 4: Run Evaluation

  Execute this command to run the evaluation:

  ```bash
  python main_eval.py <MODEL_CONFIG>.yaml
  ```
  > 📝 **NOTE**: Command may vary depending on your model. Check the model documentation.
</details>

<details>
  <summary><b>MMCV >= 2.0</b></summary>

  ### Step 1: Create the Dataset Class
  Add the following to `custom_datasets.py`:

  ```python
  @DATASETS.register_module()
  class PromptingDataset(BaseSegDataset):
    def __init__(self, dataset_name, **kwargs):
        kwargs["data_root"] = osp.join(kwargs.get("data_root"), dataset_name)
        img_dir = "valid"
        ann_dir = "annotations/valid"

        classes = self.__get_classes(kwargs["data_root"])
        palette = self.__generate_palette(len(classes) - 1)

        super(PromptingDataset, self).__init__(
            data_prefix=dict(img_path=img_dir, seg_map_path=ann_dir),
            metainfo=dict(classes=classes, palette=palette),
            **kwargs,
        )

    def __generate_palette(self, num_classes):
        palette = [[0, 0, 0]]
        for i in range(num_classes):
            hue = i / num_classes
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            rgb_list = [int(x * 255) for x in rgb]
            palette.append(rgb_list)
        return palette

    def __get_classes(self, data_root):
        classes = ()
        with open(osp.join(data_root, "classes.txt")) as f:
            for line in f:
                classes = classes + (line.strip(),)
        print(classes)
        return classes
  ```

  ### Step 2: Add Configuration Files

  > ⚠️ **IMPORTANT**
  >
  > - Customize the `dataset_name` and `img_suffix` accordingly.
  > - Some models may require different parameters in the test_pipeline or additional model settings. Check existing configuration files for examples.
  
  Create configuration files for each dataset in the `configs/` folder:

  ```python
  _base_ = './base_config.py'

  # model settings
  model = dict(
      name_path='./configs/cls_seginw_mhpv1.txt',
      prob_thd= 0.2
  )

  # dataset settings
  dataset_type = 'PromptingDataset'
  data_root = './data'

  test_pipeline = [
      dict(type='LoadImageFromFile'),
      dict(type='Resize', scale=(2048, 336), keep_ratio=True),
      dict(type='LoadAnnotations'),
      dict(type='PackSegInputs')
  ]

  test_dataloader = dict(
      batch_size=1,
      num_workers=4,
      persistent_workers=True,
      sampler=dict(type='DefaultSampler', shuffle=False),
      dataset=dict(
          type=dataset_type,
          dataset_name="<DATASET_NAME>",  # Change this accordingly!
          data_root=data_root,
          img_suffix=".jpg/.png/.jpeg",  # Change this accordingly!
          seg_map_suffix=".png",
          pipeline=test_pipeline))
  ```

  ### Step 3: Run Evaluation

  Execute this command to run evaluation:

  ```bash
  python eval.py --config ./configs/<CONFIG_NAME.py # --other_model_parameters
  ```
  > 📝 **NOTE**: Command may vary depending on your model. Check the model documentation.
</details>
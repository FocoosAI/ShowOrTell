import os
import pickle
import copy
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import detection_utils as utils
from detectron2.structures import BitMasks, Boxes, Instances
from detectron2.data import transforms as T

class COCOPanoDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        image_size,
        root='datasets',
        dataset_name='coco_2017_train_panoptic',
        is_train=True,
        crop_ratio=1.0,
        tfm_gens_crop_pair=None,
        tfm_gens_sel_pair=None,
        dino_transform=None,
        img_format='RGB',
        serialize=False,
    ):

        self.is_train = is_train
        assert is_train, "COCOPanoDataset only used in training"
        self._serialize = serialize

        self.root_ = root
        self.dataset_name = dataset_name
        coco_pano_meta = MetadataCatalog.get(dataset_name)
        coco_pano_data = {item['file_name']: item for item in DatasetCatalog.get(dataset_name)}

        self.catid2img = self.load_catid2img(coco_pano_data)
        self.class_ids = list(self.catid2img.keys())

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            coco_pano_data = {k: _serialize(v) for k, v in coco_pano_data.items()}

        self.coco_pano_data = coco_pano_data

        self.ignore_label = coco_pano_meta.get('ignore_label')

        self.crop_ratio = crop_ratio
        self.tfm_gens_crop_pair = tfm_gens_crop_pair
        self.tfm_gens_sel_pair = tfm_gens_sel_pair
        self.dino_transform = dino_transform

        self.img_format = img_format

        self.image_size = image_size

    def load_catid2img(self, coco_pano_data):
        if not os.path.exists(os.path.join(self.root_, 'sine_pkls', f'{self.dataset_name}_catid2img.pkl')):
            if not os.path.exists(os.path.join(self.root_, 'sine_pkls')) :os.makedirs(os.path.join(self.root_, 'sine_pkls'))
            catid2img = {}
            for item in coco_pano_data.values():
                for segment in item['segments_info']:
                    if segment['category_id'] not in catid2img:
                        catid2img[segment['category_id']] = []
                    if item['file_name'] not in catid2img[segment['category_id']]:
                        catid2img[segment['category_id']].append(item['file_name'])
            with open(os.path.join(self.root_, 'sine_pkls', f'{self.dataset_name}_catid2img.pkl'), 'wb') as file:
                pickle.dump(catid2img, file)
        else:
            with open(os.path.join(self.root_, 'sine_pkls', f'{self.dataset_name}_catid2img.pkl'), 'rb') as file:
                catid2img = pickle.load(file)
        return catid2img


    def __len__(self):
        return len(self.coco_pano_data)

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Uniform float random number between low and high.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

    def get_ref_tar_dict(self, class_sample, crop_pair_flag=True):

        def load_dict(file):
            dict = self.coco_pano_data[file]
            if self._serialize:
                dict = memoryview(dict)
                dict = pickle.loads(dict)
            return dict

        if len(self.catid2img[class_sample]) < 2: crop_pair_flag = True
        ref_file = np.random.choice(self.catid2img[class_sample], 1, replace=False)[0]
        ref_dict = load_dict(ref_file)
        if crop_pair_flag:
            tar_dict = copy.deepcopy(ref_dict)
        else:
            while True:
                tar_file = np.random.choice(self.catid2img[class_sample], 1, replace=False)[0]
                if ref_file != tar_file: break
            tar_dict = load_dict(tar_file)

        return ref_dict, tar_dict

    def pad_img(self, x, pad_size):

        assert isinstance(x, torch.Tensor)
        # Pad
        h, w = x.shape[-2:]
        padh = pad_size - h
        padw = pad_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def process_img_dict(self, dataset_dict, crop_pair_flag, keep_cat_ids=None):

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        while True:
            try:
                image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
            except OSError as e:
                print(f"Catched exception: {str(e)}. Re-trying...")
                import time
                time.sleep(3)
            else:
                break

        utils.check_image_size(dataset_dict, image)

        if crop_pair_flag:
            tfm_gens = self.tfm_gens_crop_pair
        else:
            tfm_gens = self.tfm_gens_sel_pair

        image, transforms = T.apply_transform_gens(tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        if self.dino_transform is None:
            dino_image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        else:
            dino_image = self.dino_transform(image)
        dataset_dict["image"] = self.pad_img(dino_image, self.image_size)

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]
            # sort
            cat_ids = [info['category_id'] for info in segments_info]
            segments_info = [segments_info[idx] for idx in sorted(range(len(cat_ids)), key=lambda  k: cat_ids[k])]

            # apply the same transformation to panoptic segmentation
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

            from panopticapi.utils import rgb2id

            pan_seg_gt = rgb2id(pan_seg_gt)

            instances = Instances(image_shape)
            classes = []
            masks = []
            ins_ids = []
            new_segments_info = []
            for segment_info in segments_info:
                class_id = segment_info["category_id"]
                if class_id in keep_cat_ids and not segment_info["iscrowd"]:
                    mask = pan_seg_gt == segment_info["id"]
                    if mask.sum() > 0:
                        classes.append(class_id)
                        masks.append(mask)
                        new_segments_info.append(segment_info)
                        ins_ids.append(segment_info["id"])

            dataset_dict['segments_info'] = new_segments_info
            classes = np.array(classes)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            ins_ids = np.array(ins_ids)
            instances.ins_ids = torch.tensor(ins_ids, dtype=torch.int64)
            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                # instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
                # instances.gt_masks = self.pad_img(torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1])), self.image_size)
                instances.gt_masks = torch.zeros((0, self.image_size, self.image_size))
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            else:
                masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                masks = self.pad_img(masks, self.image_size)
                masks = BitMasks(masks)
                instances.gt_masks = masks.tensor
                instances.gt_boxes = masks.get_bounding_boxes()

            dataset_dict["instances"] = instances

        return dataset_dict

    def __getitem__(self, idx):

        ref_mask_num = 0
        tar_mask_num = 0

        while ref_mask_num == 0 or tar_mask_num == 0:

            # sample category
            class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]

            #sample reference and target dataset dict
            crop_pair_flag = self._rand_range() < self.crop_ratio
            ref_dict, tar_dict = self.get_ref_tar_dict(class_sample, crop_pair_flag)
            ref_cat_ids = list(set([seg['category_id'] for seg in ref_dict['segments_info']]))
            ref_dict = self.process_img_dict(ref_dict, crop_pair_flag, keep_cat_ids=ref_cat_ids)
            ref_cat_ids = list(set([seg['category_id'] for seg in ref_dict['segments_info']]))
            tar_dict = self.process_img_dict(tar_dict, crop_pair_flag, keep_cat_ids=ref_cat_ids)

            ref_mask_num = len(ref_dict['instances'])
            tar_mask_num = len(tar_dict['instances'])

        return {"ref_dict": ref_dict, "tar_dict": tar_dict}

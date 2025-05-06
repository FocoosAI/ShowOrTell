import os
import pickle
import copy
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import detection_utils as utils
from detectron2.structures import BitMasks, Boxes, Instances, BoxMode
from detectron2.data import transforms as T
from detectron2.utils.file_io import PathManager

import sys
sys.path.append('./')

from pycocotools import mask as coco_mask

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

class O365Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        image_size,
        root='datasets',
        dataset_name='coco_2017_train_ins',
        is_train=True,
        crop_ratio=1.0,
        tfm_gens_crop_pair=None,
        tfm_gens_sel_pair=None,
        dino_transform=None,
        img_format='RGB',
        serialize=True,
    ):

        self.is_train = is_train
        assert is_train, "COCOInsDataset only used in training"
        self._serialize = serialize

        self.root_ = root
        self.dataset_name = dataset_name
        coco_ins_meta = MetadataCatalog.get(dataset_name)
        coco_ins_data = {item['file_name']: item for item in DatasetCatalog.get(dataset_name)}

        self.catid2img = self.load_catid2img(coco_ins_data)
        self.class_ids = list(self.catid2img.keys())

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            coco_ins_data = {k: _serialize(v) for k, v in coco_ins_data.items()}

        self.coco_ins_data = coco_ins_data

        self.ignore_label = coco_ins_meta.get('ignore_label')

        self.crop_ratio = crop_ratio
        self.tfm_gens_crop_pair = tfm_gens_crop_pair
        self.tfm_gens_sel_pair = tfm_gens_sel_pair
        self.dino_transform = dino_transform

        self.img_format = img_format

        self.image_size = image_size


    def load_catid2img(self, coco_ins_data):
        if not os.path.exists(os.path.join(self.root_, 'sine_pkls', f'{self.dataset_name}_catid2img.pkl')):
            if not os.path.exists(os.path.join(self.root_, 'sine_pkls')) :os.makedirs(os.path.join(self.root_, 'sine_pkls'))
            catid2img = {}
            for item in tqdm(coco_ins_data.values()):
                for anno in item['annotations']:
                    if anno['category_id'] not in catid2img:
                        catid2img[anno['category_id']] = []
                    if item['file_name'] not in catid2img[anno['category_id']]:
                        catid2img[anno['category_id']].append(item['file_name'])
            with open(os.path.join(self.root_, 'sine_pkls', f'{self.dataset_name}_catid2img.pkl'), 'wb') as file:
                pickle.dump(catid2img, file)
        else:
            with open(os.path.join(self.root_, 'sine_pkls', f'{self.dataset_name}_catid2img.pkl'), 'rb') as file:
                catid2img = pickle.load(file)
        return catid2img


    def __len__(self):
        return len(self.coco_ins_data)

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
            dict = self.coco_ins_data[file]
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


        if 'segmentation_file' in dataset_dict:
            with PathManager.open(dataset_dict['segmentation_file'], "rb") as f:
                segm_info = json.load(f)

            assert segm_info["image_id"] == dataset_dict["image_id"]
            for anno in dataset_dict['annotations']:
                anno_id = anno["id"]
                segm = segm_info["segmentations"][str(anno_id)]
                anno["segmentation"] = coco_mask.frPyObjects(segm, *segm["size"])


        if "annotations" in dataset_dict:
            annotations = dataset_dict.pop("annotations")
            # sort
            cat_ids = [info['category_id'] for info in annotations]
            annotations = [annotations[idx] for idx in sorted(range(len(cat_ids)), key=lambda k: cat_ids[k])]

            for anno in annotations:
                # Let's always keep mask
                # if not self.mask_on:
                #     anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            # annos = [
            #     utils.transform_instance_annotations(obj, transforms, image_shape)
            #     for obj in annotations if obj.get("iscrowd", 0) == 0 and obj['category_id'] in keep_cat_ids
            # ]
            annos = []
            for obj in annotations:
                if obj.get("iscrowd", 0) == 0 and obj['category_id'] in keep_cat_ids:
                    obj['bbox_mode'] = BoxMode.XYWH_ABS
                    if isinstance(obj['segmentation'], dict) and isinstance(obj['segmentation']['counts'], list):
                        # uncompressed_rle ---> rle
                        obj['segmentation'] = coco_mask.frPyObjects(obj['segmentation'], *obj['segmentation']['size'])

                    # rle ---> polygon
                    # obj['segmentation'] = convert_coco_mask_to_poly(coco_mask.decode(obj['segmentation']))
                    # if obj['segmentation'] == []:
                    #     continue

                    annos.append(utils.transform_instance_annotations(obj, transforms, image_shape))

            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            if annos != [] and isinstance(annos[0]['segmentation'], np.ndarray): # annos may be empty
                instances = utils.annotations_to_instances(annos, image_shape, mask_format='bitmask')
            elif annos == []:
                instances = utils.annotations_to_instances(annos, image_shape, mask_format='bitmask')
            else:
                instances = utils.annotations_to_instances(annos, image_shape, mask_format='polygon')
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            try:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            except:
                dataset_dict["instances"] = instances
                return dataset_dict

            ins_ids = [anno['id'] for anno in annos]
            ins_ids = np.array(ins_ids)
            instances.ins_ids = torch.tensor(ins_ids, dtype=torch.int64)
            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            # Generate masks from polygon
            h, w = instances.image_size
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                if isinstance(gt_masks, BitMasks):
                    gt_masks = gt_masks.tensor
                else:
                    gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = self.pad_img(gt_masks, self.image_size)
            if annos == []:
                instances.set('gt_masks', torch.zeros((0, h, w), dtype=torch.uint8))
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
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
            ref_cat_ids = list(set([anno['category_id'] for anno in ref_dict['annotations']]))
            ref_dict = self.process_img_dict(ref_dict, crop_pair_flag, keep_cat_ids=ref_cat_ids)
            ref_cat_ids = list(set(ref_dict['instances'].gt_classes.tolist()))
            tar_dict = self.process_img_dict(tar_dict, crop_pair_flag, keep_cat_ids=ref_cat_ids)

            ref_mask_num = len(ref_dict['instances'])
            tar_mask_num = len(tar_dict['instances'])

        return {"ref_dict": ref_dict, "tar_dict": tar_dict}



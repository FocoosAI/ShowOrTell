import torch
from torchvision import transforms
import numpy as np

from detectron2.data import transforms as T
from dinov2.data.transforms import MaybeToTensor, make_normalize_transform

from sine.data.coco import COCOPanoDataset
from sine.data.coco_ins import COCOInsDataset
from sine.data.paco import PACODataset
from sine.data.o365 import O365Dataset

class HybridDataset(torch.utils.data.Dataset):

    dino_transform = transforms.Compose([
        MaybeToTensor(),
        make_normalize_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img_format = 'RGB'
    pano_data_name_mapping_train = {
        'coco': 'coco_2017_train_panoptic',
        'ade20k': 'ade20k_panoptic_train',
    }

    pano_data_name_mapping_val = {
        'coco': 'coco_2017_val_panoptic',
        'ade20k': 'ade20k_panoptic_val',
    }

    ins_data_name_mapping_train = {
        'coco': 'coco_2017_train_ins',
        'lvis': 'lvis_v1_train_ins',
        'paco': 'paco_lvis_v1_train',
        'o365': 'object365_train',
    }

    ins_data_name_mapping_val= {
        'coco': 'coco_2017_val_ins',
        'lvis': 'lvis_v1_val_ins',
        'paco': 'paco_lvis_v1_val',
        'o365': 'object365_val'
    }

    def __init__(
        self,
        image_size,
        root='datasets',
        crop_ratio=0.5,
        samples_per_epoch=500 * 8 * 2 * 10,
        dataset="pano_seg||ins_seg",
        sample_rate=[1, 3],
        pano_seg_data="coco||ade20k",
        pano_sample_rate=[1, 1],
        ins_seg_data="o365||lvis||paco",
        ins_sample_rate=[1, 1, 1],
        tfm_gens_crop_pair=None,
        tfm_gens_sel_pair=None,
        is_train=True,
    ):
        DatasetDict = {
            'pano_seg': {
                'coco': COCOPanoDataset,
                'ade20k': COCOPanoDataset
            },
            'ins_seg': {
                'coco': COCOInsDataset,
                'o365': O365Dataset,
                'lvis': COCOInsDataset,
                'paco': PACODataset
            }
        }

        self.is_train = is_train
        self.pano_data_name_mapping = self.pano_data_name_mapping_train if is_train else self.pano_data_name_mapping_val
        self.ins_data_name_mapping = self.ins_data_name_mapping_train if is_train else self.ins_data_name_mapping_val

        self.root_ = root
        self.image_size = image_size
        self.samples_per_epoch = samples_per_epoch

        self.datasets = dataset.split("||") # just support pano_seg
        self.seg_dataset_dict = {
            'pano_seg': pano_seg_data.split("||") if 'pano_seg' in self.datasets else None,
            'ins_seg': ins_seg_data.split("||") if 'ins_seg' in self.datasets else None,
        }

        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        pano_sample_rate = np.array(pano_sample_rate)
        ins_sample_rate = np.array(ins_sample_rate)

        self.dataset_sample_rate_dict = {
            'pano_seg': pano_sample_rate / pano_sample_rate.sum() if 'pano_seg' in self.datasets else None,
            'ins_seg': ins_sample_rate / ins_sample_rate.sum() if 'ins_seg' in self.datasets else None,
        }

        self.all_datasets = {ds: [] for ds in self.datasets}
        for dataset in self.datasets:
            Dataset = DatasetDict[dataset]

            for seg_dataset in self.seg_dataset_dict[dataset]:
                dataset_name = self.pano_data_name_mapping[seg_dataset] \
                    if 'pano' in dataset else self.ins_data_name_mapping[seg_dataset]
                from sine.utils.utils import Print; Print(f"loading {seg_dataset} !")
                self.all_datasets[dataset].append(
                    Dataset[seg_dataset](
                        image_size=image_size,
                        root=root,
                        dataset_name=dataset_name,
                        crop_ratio=crop_ratio,
                        tfm_gens_crop_pair=tfm_gens_crop_pair,
                        tfm_gens_sel_pair=tfm_gens_sel_pair,
                        dino_transform=self.dino_transform,
                    )
                )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        seg_flag = np.random.choice(self.datasets, p=self.sample_rate)
        datasets = self.all_datasets[seg_flag]
        ind = np.random.choice(list(range(len(datasets))), p=self.dataset_sample_rate_dict[seg_flag])
        data = datasets[ind][idx]
        return data

def build_augmentation(args, is_train):
    augmentation = []

    if is_train:
        # LSJ aug
        if args.random_flip != "none":
            augmentation.append(
                T.RandomFlip(
                    horizontal=args.random_flip == "horizontal",
                    vertical=args.random_flip == "vertical",
                )
            )

        augmentation.extend([
            T.ResizeScale(
                min_scale=args.min_scale, max_scale=args.max_scale, target_height=args.image_size,
                target_width=args.image_size
            ),
            T.FixedSizeCrop(crop_size=(args.image_size, args.image_size), pad=False),
        ])

    else:
        augmentation.append(
            T.ResizeShortestEdge(
                short_edge_length=args.image_size,
                max_size=args.image_size
            )
        )

    return augmentation

def build_dataset(args, is_train):

    augmentation = build_augmentation(args, is_train)

    if is_train:
        dataset = HybridDataset(
            image_size=args.image_size,
            root=args.data_root,
            crop_ratio=args.crop_ratio,
            samples_per_epoch=args.steps_per_epoch * args.world_size * args.batch_size * args.update_freq,
            dataset=args.dataset,
            sample_rate=[float(x) for x in args.sample_rate.split(",")],
            pano_seg_data=args.pano_seg_data,
            pano_sample_rate=[float(x) for x in args.pano_sample_rate.split(",")],
            ins_seg_data=args.ins_seg_data,
            ins_sample_rate=[float(x) for x in args.ins_sample_rate.split(",")],
            tfm_gens_crop_pair=augmentation,
            tfm_gens_sel_pair=augmentation,
            is_train=is_train
        )
    else:
        dataset = HybridDataset(
            image_size=args.image_size,
            root=args.data_root,
            crop_ratio=args.crop_ratio,
            samples_per_epoch=args.steps_per_epoch * args.world_size * args.batch_size * args.update_freq,
            dataset="pano_seg",
            sample_rate=[1,],
            pano_seg_data="coco",
            pano_sample_rate=[1,],
            ins_seg_data="coco",
            ins_sample_rate=[1,],
            tfm_gens_crop_pair=augmentation,
            tfm_gens_sel_pair=augmentation,
            is_train=is_train
        )

    return dataset


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser('coco dataset', add_help=False)
    parser.add_argument('--random_flip', default="horizontal", type=str)
    parser.add_argument('--min_scale', default=0.1, type=float)
    parser.add_argument('--max_scale', default=2.0, type=float)
    parser.add_argument('--image_size', default=896, type=int)
    args = parser.parse_args()

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    is_train = True
    augmentation = build_augmentation(args, is_train)

    dataset = HybridDataset(
        is_train=is_train,
        image_size=args.image_size,
        root='datasets',
        crop_ratio=0.5,
        samples_per_epoch=5000 * 8 * 2 * 10,
        dataset="pano_seg||ins_seg",
        sample_rate=[1, 3],
        pano_seg_data="coco||ade20k",
        pano_sample_rate=[1, 1],
        ins_seg_data="o365||lvis||paco",
        ins_sample_rate=[1, 1, 1],
        tfm_gens_crop_pair=augmentation,
        tfm_gens_sel_pair=augmentation,
    )


    for id in range(len(dataset)):

        data = dataset[id]
        ref_ = data['ref_dict']
        tar_ = data['tar_dict']

        if len(ref_['instances'])>0:
            print(f"iter: {id} ", ref_['file_name'], f"ins num: {len(ref_['instances'])}")
        else:
            print()

        if len(tar_['instances'])>0:
            print(f"iter: {id} ", tar_['file_name'], f"ins num: {len(tar_['instances'])}")
        else:
            print()
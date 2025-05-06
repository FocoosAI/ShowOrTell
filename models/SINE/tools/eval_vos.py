from os import path
import os
import argparse
import shutil
import queue
import math
import numpy as np
from PIL import Image
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

from detectron2.structures import BitMasks, Instances
from torchvision import transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

import random
import sys
sys.path.append('./')

from dinov2.data.transforms import MaybeToTensor, make_normalize_transform

from inference_vos.inference.data.test_datasets_nonorm import DAVISTestDataset, YouTubeVOSTestDataset
from inference_vos.inference.data.mask_mapper import MaskMapper
from inference_vos.inference import ddp_utils
from inference_vos.inference.memory import Memory, Frame
from inference_vos.model.model import build_model

def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def pad_img(x, pad_size):

    assert isinstance(x, torch.Tensor)
    # Pad
    h, w = x.shape[-2:]
    padh = pad_size - h
    padw = pad_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def resize(x, size, mode='bilinear'):
    height, width = x.shape[-2:]
    aspect_ratio = width / height
    if aspect_ratio > 1:
        new_width = size
        new_height = int(size / aspect_ratio)
    else:
        new_width = int(size * aspect_ratio)
        new_height = size
    return F.interpolate(x, size=(new_height, new_width), mode=mode)
        

def wrap_instances(mask, id, resize_size):
    # ref
    ref_mask_num = mask.shape[0]
    ref_masks = resize(mask.unsqueeze(0), args.img_size, mode='nearest')[0]
    ref_masks = pad_img(ref_masks, args.pad_size)
    ref_instances = Instances(resize_size)
    ref_instances.gt_classes = torch.tensor([id], dtype=torch.int64)
    ref_instances.gt_masks = ref_masks
    ref_instances.gt_boxes = BitMasks(ref_masks).get_bounding_boxes()
    ref_instances.ins_ids = torch.tensor([id], dtype=torch.int64)
    
    return ref_instances
    

def wrap_data_ref(batch, args):
    # transforms for image encoder
    encoder_transform = transforms.Compose([
        MaybeToTensor(),
        make_normalize_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    ref_dict = {}

    # ref
    ref_img = resize(batch['support_imgs'][0], args.img_size)[0]   # 值在0~1之间
    ref_image_shape = batch['support_imgs'].shape[-2:]
    ref_dict["image"] = pad_img(encoder_transform(ref_img), args.pad_size)  # norm后pad

    # label
    ref_dict['height'], ref_dict['width'] = ref_image_shape
    ref_mask_num = batch['support_masks'].shape[1]
    ref_instances = Instances(ref_img.shape[-2:])
    ref_instances.gt_classes = batch['class_id']
    ref_masks = resize(batch['support_masks'], args.img_size, mode='nearest')[0]
    ref_masks = pad_img(ref_masks, args.pad_size)
    ref_masks = BitMasks(ref_masks)
    ref_instances.gt_masks = ref_masks.tensor
    ref_instances.gt_boxes = ref_masks.get_bounding_boxes()
    ref_instances.ins_ids = batch['class_id'] # TODO
    ref_dict["instances"] = ref_instances


    return ref_dict

def wrap_data_tar(batch, args):
    # transforms for image encoder
    encoder_transform = transforms.Compose([
        MaybeToTensor(),
        make_normalize_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    tar_dict = {}

    # tar
    tar_img = resize(batch['query_img'], args.img_size)[0]
    tar_image_shape = batch['query_img'].shape[-2:]  # h, w
    tar_dict["image"] = pad_img(encoder_transform(tar_img), args.pad_size)

    # # label
    tar_dict['height'], tar_dict['width'] = tar_image_shape
    tar_dict['resize_height'], tar_dict['resize_width'] = tar_img.shape[-2:]

    return tar_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data options
    parser.add_argument('--dataset', help='D16/D17/Y18', default='D17')
    parser.add_argument('--d16_path', default='datasets/vos/DAVIS16')
    parser.add_argument('--d17_path', default='datasets/vos/DAVIS17')
    parser.add_argument('--y18_path', default='datasets/vos/YouTubeVOS18')

    parser.add_argument('--split', help='val/test', default='val')
    parser.add_argument('--output', default="outputs/vos/debug")
    parser.add_argument('--save_all', action='store_true', 
                help='Save all frames. Useful only in YouTubeVOS/long-time video', )

    parser.add_argument('--fast_eval', action='store_true')
    parser.add_argument('--all_frame', action='store_true')
    parser.add_argument('--num_frame', default=6, type=int)
    parser.add_argument('--fix_first_frame', action='store_true')
    parser.add_argument('--fix_last_frame', action='store_true')
    parser.add_argument('--hard_tgt', action='store_true')
    parser.add_argument('--reweight', action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')

    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--size', default=518, type=int,
                help='Resize the shorter side to this size. -1 to use original resolution. ')
    parser.add_argument('--img-size', type=int, default=1036)
    parser.add_argument('--pad-size', type=int, default=1036)

    parser.add_argument('--fold_num', type=int, default=1)
    parser.add_argument('--fold_index', type=int, default=0)
    parser.add_argument('--fold_range', type=str, default=None)

    # Matcher parameters
    parser.add_argument('--feat_chans', default=256, type=int)
    parser.add_argument('--image_enc_use_fc', action="store_true")

    parser.add_argument('--dinov2-size', type=str, default="vit_large")
    parser.add_argument('--dinov2-weights', type=str, default="models/dinov2_vitl14_pretrain.pth")
    parser.add_argument('--sine-weights', type=str, default="outputs/pt_sine/pytorch_model.bin")

    parser.add_argument('--transformer_depth', default=6, type=int)
    parser.add_argument('--transformer_nheads', default=8, type=int)
    parser.add_argument('--transformer_mlp_dim', default=2048, type=int)
    parser.add_argument('--transformer_mask_dim', default=256, type=int)
    parser.add_argument('--transformer_fusion_layer_depth', default=1, type=int)
    parser.add_argument('--transformer_num_queries', default=200, type=int)
    parser.add_argument("--transformer_pre_norm", action="store_true", default=True)

    # evaluation
    parser.add_argument('--score_threshold', default=-1e9, type=float)
    parser.add_argument('--memory_decay_type', default='linear', type=str)
    parser.add_argument('--memory_decay_ratio', default=20, type=float)


    args = parser.parse_args()
    if args.fold_range:
        args.fold_range = eval(args.fold_range)

    print("{}".format(args).replace(', ', ',\n'))

    max_vid = 1000000000000000000000000
    if args.fast_eval:
        max_vid = 20

    assert args.output is not None

    """
    Data preparation
    """
    is_youtube = args.dataset.startswith('Y')
    is_davis = args.dataset.startswith('D')

    if is_youtube:
        out_path = path.join(args.output, 'Annotations')
    else:
        out_path = args.output

    if is_youtube:
        if args.dataset == 'Y18':
            yv_path = args.y18_path

        if args.split == 'val':
            args.split = 'valid'
            meta_dataset = YouTubeVOSTestDataset(data_root=yv_path, split='valid', size=args.size)
        elif args.split == 'test':
            meta_dataset = YouTubeVOSTestDataset(data_root=yv_path, split='test', size=args.size)
        elif args.split == 'train':
            meta_dataset = YouTubeVOSTestDataset(data_root=yv_path, split='train', size=args.size)
        else:
            raise NotImplementedError

    elif is_davis:
        if args.dataset == 'D16':
            if args.split == 'val':
                # Set up Dataset, a small hack to use the image set in the 2017 folder because the 2016 one is of a different format
                imset_root=os.path.split(os.path.abspath(__file__))[0].replace('/tools', '')
                meta_dataset = DAVISTestDataset(args.d16_path, imset=os.path.join(imset_root, 'datasets/vos/DAVIS17/trainval/ImageSets/2016/val.txt'), size=args.size)
            else:
                raise NotImplementedError
            palette = None
        elif args.dataset == 'D17':
            if args.split == 'val':
                if args.fold_range:
                    video_indices = args.fold_range
                else:
                    video_num = 30
                    fold_split = [math.ceil(30 / args.fold_num) * i for i in range(args.fold_num)] + [video_num]
                    video_indices = (fold_split[args.fold_index], fold_split[args.fold_index+1])
                meta_dataset = DAVISTestDataset(path.join(args.d17_path, 'trainval'), imset='2017/val.txt', size=args.size, indices=video_indices)
            elif args.split == 'test':
                meta_dataset = DAVISTestDataset(path.join(args.d17_path, 'test-dev'), imset='2017/test-dev.txt', size=args.size)
            else:
                raise NotImplementedError

    else:
        raise NotImplementedError

    meta_loader = meta_dataset.get_datasets()
    sampler_val = SequentialSampler(meta_dataset)
    data_loader_val = DataLoader(
        meta_dataset, batch_size=1, sampler=sampler_val, drop_last=False, num_workers=0,
        collate_fn=ddp_utils.dummy_collate_fn,
    )

    torch.autograd.set_grad_enabled(False)
    print('Model loaded.')
    device = torch.device("cuda")

    total_process_time = 0
    total_frames = 0

    J_score = []

    # # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    # Model initialization
    model = build_model(args)
    print(f"model: {model}")
    state_dict = torch.load(args.sine_weights, map_location="cpu")
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"msg: {msg}")
    model.to(device)
    model.eval()
    
    # Freeze randomness during testing for reproducibility
    fix_randseed(0)

    eval_video_names = None

    # Start eval
    for i, vid_reader in tqdm(zip(range(max_vid), data_loader_val), total=min(max_vid, len(data_loader_val)), disable=not ddp_utils.is_main_process()):
        vid_reader = vid_reader[0]
        loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=2)
        vid_name = vid_reader.vid_name
        vid_length = len(loader)
        
        if eval_video_names and vid_name not in eval_video_names:
            continue
        
        if args.resume:
            all_preds_exist = True                    
            for ti, data in enumerate(loader):
                info = data['info']
                frame = info['frame'][0]
                if (args.all_frame or info['save'][0]):
                    pred_path = os.path.join(out_path, vid_name, frame[:-4]+".png")
                    if not os.path.exists(pred_path):
                        all_preds_exist = False
                        break
            if all_preds_exist:
                os.system(f'echo "video {vid_name} results already exist, skip this video."')
                continue


        mapper = MaskMapper()
        prompt, prompt_target = None, None
        first_mask_loaded = False

        memory = {}
        n_obj = 0
        for ti, data in enumerate(loader):
            with torch.cuda.amp.autocast(enabled=True):
                rgb = data['rgb'].cuda()[0]
                msk = data.get('mask')
                info = data['info']
                frame = info['frame'][0]
                shape = info['shape']
                need_resize = info['need_resize'][0]
                
                frame_id = int(frame.strip(".jpg"))

                if not (args.all_frame or info['save'][0]):
                    continue

                """
                For timing see https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
                Seems to be very similar in testing as my previous timing method 
                with two cuda sync + time.time() in STCN though 
                """
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                if args.flip:
                    rgb = torch.flip(rgb, dims=[-1])
                    msk = torch.flip(msk, dims=[-1]) if msk is not None else None

                # Map possibly non-continuous labels to continuous ones
                if msk is not None:
                    msk, labels = mapper.convert_mask(msk[0].numpy(), exhaustive=args.split == 'train')
                    msk = torch.Tensor(msk).cuda()
                    msk_idx = (torch.tensor(labels, device=msk.device) - 1).to(torch.int64)
                    msk = msk[msk_idx]
                    if need_resize:
                        msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
                else:
                    labels = None

                # build memory for new ids
                if msk is not None:
                    for id_mask, id in zip(msk, labels):
                        support_img = rgb[None, None, ...]
                        support_mask = id_mask[None, None, ...]
                        class_id = torch.tensor([id], dtype=torch.int64, device=device)

                        batch = dict(
                            support_imgs=support_img,
                            support_masks=support_mask,
                            class_id=class_id
                        )

                        ref_dict = wrap_data_ref(batch, args)
                        assert id not in memory

                        memory[id] = Memory(
                            memory_len=args.num_frame,
                            fix_first_frame=args.fix_first_frame,
                            fix_last_frame=args.fix_last_frame,
                            memory_decay_type=args.memory_decay_type,
                            memory_decay_ratio=args.memory_decay_ratio
                        )

                        if memory[id].last_frame == None:
                            memory[id].update_memory(
                                Frame(
                                    obj=ref_dict,
                                    frame_id=frame_id,
                                    score=1.
                                )
                            )
                    n_obj += len(labels)

                query_img = rgb[None, ...]
                query_mask = None
                batch = dict(
                    query_img=query_img,
                    query_mask=query_mask,
                )
                tar_dict = wrap_data_tar(batch, args)

                ref_list = []
                for mem in memory.values():
                    id_list = [frame.obj for frame in mem.get_memory()]
                    ref_list.extend(id_list)

                tar_list = [tar_dict] + [None] * (len(ref_list) - 1)
                data = []
                for ref, tar in zip(ref_list, tar_list):
                    data.append({
                        'ref_dict': ref,
                        'tar_dict': tar
                    })
                
                # predict
                pred = model(data)

                pred_ids = pred['id_seg'].pred_ids
                pred_masks = torch.zeros_like(pred['id_seg'].pred_masks)
                pred_scores = [0] * pred['id_seg'].scores.shape[0]
                resize_size = (tar_dict['resize_height'], tar_dict['resize_width'])

                for i, object_id in enumerate(pred['id_seg'].pred_ids.tolist()):

                    pmask = pred['id_seg'].pred_masks[i]
                    pscore = pred['id_seg'].scores[i].item()

                    pred_masks[object_id-1] = pmask
                    pred_scores[object_id-1] = pscore

                    tar_dict_ = deepcopy(tar_dict)
                    tar_dict_['instances'] = wrap_instances(pmask[None,...], object_id, resize_size)

                    memory[object_id].update_memory(
                        Frame(
                            obj=tar_dict_,
                            frame_id=frame_id,
                            score=pscore
                        )
                    )

                # Upsample to original size if needed
                if need_resize:
                    pred_masks = F.interpolate(pred_masks[None, ...], shape, mode='nearest')[0]
                if args.flip:
                    pred_masks = torch.flip(pred_masks, dims=[-1])

                end.record()
                torch.cuda.synchronize()
                total_process_time += (start.elapsed_time(end)/1000)
                total_frames += 1

                # Probability mask -> index mask
                pred_masks_sem = pred_masks * torch.tensor(list(range(1, pred_masks.shape[-3] + 1)), device=pred_masks.device)[..., None, None]
                pred_masks_sem = pred_masks_sem.max(dim=0)[0]
                out_mask = pred_masks_sem.cpu().numpy().astype(np.uint8)

                # Save the mask
                if args.save_all or info['save'][0]:
                    this_out_path = path.join(out_path, vid_name)
                    os.makedirs(this_out_path, exist_ok=True)
                    out_mask = mapper.remap_index_mask(out_mask)
                    out_img = Image.fromarray(out_mask)
                    if vid_reader.get_palette() is not None:
                        out_img.putpalette(vid_reader.get_palette())
                    out_img.save(os.path.join(this_out_path, frame[:-4]+'.png'))

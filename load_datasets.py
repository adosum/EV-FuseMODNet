#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 16:40:32 2021

@author: user
"""
import os
from os.path import splitext
from os import listdir
import numpy as np
# from glob import glob
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# import logging
from PIL import Image
import torchvision.transforms as transforms
import random
from events_contrast_maximization.utils.event_utils import events_to_voxel_torch, \
    events_to_neg_pos_voxel_torch, binary_search_torch_tensor, events_to_image_torch, \
    binary_search_h5_dset, get_hot_event_mask, save_image
from data_augmentation import Flow_event_mask_Augmentor, center_crop
from glob import glob
import cv2
import frame_utils


class fusionDataset(Dataset):
    def __init__(self, root, data_split, crop_size=None, aug_params=None, split='training', dark=False):
        self.image_size = None
        self.data_split = data_split
        self.combined_voxel_channels = False
        self.sensor_resolution = (375, 1242)
        self.augmentor = None
        self.mask_list = []
        self.image_list = []
        self.dark_list = []
        self.event_list = []
        self.extra_info = []
        self.crop_size = crop_size
        self.dark = dark
        if aug_params is not None:
            self.augmentor = Flow_event_mask_Augmentor(**aug_params)
        root_image = os.path.join(root, split, 'images')
        root_dark = os.path.join(root, split, 'dark')
        root_mask = os.path.join(root, split, 'mask')
        for scene in os.listdir(root_image):
            # print(scene)
            image_list = sorted(glob(os.path.join(root_image, scene, '*.png')))
            dark_list = sorted(glob(os.path.join(root_dark, scene, '*.png')))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                self.dark_list += [[dark_list[i], dark_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id
            self.event_list += sorted(glob(os.path.join(root_image, scene, 'event_data', '*.npy')))
            self.event_list.pop(-1)
            mask_list_tmp = sorted(glob(os.path.join(root_mask, scene, '*.png')))
            mask_list_tmp.pop(0)
            self.mask_list += mask_list_tmp

    def get_empty_voxel_grid(self, combined_voxel_channels=True):
        """Return an empty voxel grid filled with zeros"""
        if combined_voxel_channels:
            size = (self.data_split, *self.sensor_resolution)
        else:
            size = (2 * self.data_split, *self.sensor_resolution)
        return torch.zeros(size, dtype=torch.float32)

    def get_voxel_grid(self, xs, ys, ts, ps, combined_voxel_channels=True):
        """
        Given events, return voxel grid
        :param xs: tensor containg x coords of events
        :param ys: tensor containg y coords of events
        :param ts: tensor containg t coords of events
        :param ps: tensor containg p coords of events
        :param combined_voxel_channels: if True, create voxel grid merging positive and
            negative events (resulting in NUM_BINS x H x W tensor). Otherwise, create
            voxel grid for positive and negative events separately
            (resulting in 2*NUM_BINS x H x W tensor)
        """
        if combined_voxel_channels:
            # generate voxel grid which has size self.num_bins x H x W
            voxel_grid = events_to_voxel_torch(xs, ys, ts, ps, self.data_split,
                                               sensor_size=self.sensor_resolution)
        else:
            # generate voxel grid which has size 2*self.num_bins x H x W
            voxel_grid = events_to_neg_pos_voxel_torch(xs, ys, ts, ps, self.data_split,
                                                       sensor_size=self.sensor_resolution)
            voxel_grid = torch.cat([voxel_grid[0], voxel_grid[1]], 0)  # voxel_pos, voxel_neg

        # voxel_grid = voxel_grid*self.hot_events_mask

        return voxel_grid

    def process_event(self, events):
        xs = events[:, 0]
        ys = events[:, 1]
        ts = events[:, 2]
        ps = events[:, 3]
        try:
            ts_0, ts_k = ts[0], ts[-1]
        except:
            ts_0, ts_k = 0, 0
        if len(xs) < 3:
            voxel = self.get_empty_voxel_grid(self.combined_voxel_channels)
        else:
            xs = torch.from_numpy(xs).float()
            ys = torch.from_numpy(ys).float()
            ts = torch.from_numpy((ts - ts_0)).float()
            ps = torch.from_numpy(ps).float()
            voxel = self.get_voxel_grid(xs, ys, ts, ps, combined_voxel_channels=self.combined_voxel_channels)
        return voxel

    def __getitem__(self, index):
        if self.dark == True:
            img1 = frame_utils.read_gen(self.dark_list[index][0])
            img2 = frame_utils.read_gen(self.dark_list[index][1])
        else:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        eventdata_raw = np.load(self.event_list[index])
        mask = frame_utils.read_gen(self.mask_list[index])
        mask = np.array(mask).astype(np.uint8)
        if np.max(mask) > 1:
            mask = mask / 255
        voxel = self.process_event(eventdata_raw)
        voxel = np.array(voxel.permute(1, 2, 0)).astype(np.float32)
        if self.augmentor is not None:
            img1, img2, flow, voxel, mask = self.augmentor(img1, img2, None, voxel, mask)
        else:
            img1, img2, flow, voxel, mask = center_crop(img1, img2, None, voxel, mask, self.crop_size)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        voxel = torch.from_numpy(voxel).permute(2, 0, 1).float()
        self.image_size = img1.size()
        eventdata = torch.zeros((4, int(self.data_split / 2), self.image_size[1], self.image_size[2]),
                                dtype=torch.float)
        eventdata[0, ...] = voxel[0:5, ...]
        eventdata[1, ...] = voxel[5:10, ...]
        eventdata[2, ...] = voxel[10:15, ...]
        eventdata[3, ...] = voxel[15:20, ...]
        assert img1.shape[1] == self.image_size[1], \
            f'The image shape should be {self.image_size}, ' \
            f'but loaded images have {img1.shape[1]}. Please check that ' \
            'the images are loaded correctly.'
        assert eventdata.shape[2] == self.image_size[1], \
            f'The eventdata shape should be {self.image_size}, ' \
            f'but loaded eventdata have {img1.shape[1]}. Please check that ' \
            'the images are loaded correctly.'

        return {
            'former_image': img1,
            'eventdata': eventdata,
            'mask': mask,
            'latter_image': img2
        }

    def __rmul__(self, v):
        # self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        self.event_list = v * self.event_list
        self.mask_list = v * self.mask_list
        return self

    def __len__(self):
        return len(self.image_list)


class FlowDataset(Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = Flow_event_mask_Augmentor(**aug_params)
            else:
                self.augmentor = Flow_event_mask_Augmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.event_list = []
        self.extra_info = []
        self.combined_voxel_channels = False
        self.data_split = 10
        self.crop_size = None

    def get_empty_voxel_grid(self, combined_voxel_channels=True):
        """Return an empty voxel grid filled with zeros"""
        if combined_voxel_channels:
            size = (self.data_split, *self.sensor_resolution)
        else:
            size = (2 * self.data_split, *self.sensor_resolution)
        return torch.zeros(size, dtype=torch.float32)

    def transform_voxel(self, voxel, seed):
        """
        Augment voxel and turn into tensor
        """
        if self.vox_transform:
            random.seed(seed)
            voxel = self.vox_transform(voxel)
        return voxel

    def get_voxel_grid(self, xs, ys, ts, ps, combined_voxel_channels=True):
        """
        Given events, return voxel grid
        :param xs: tensor containg x coords of events
        :param ys: tensor containg y coords of events
        :param ts: tensor containg t coords of events
        :param ps: tensor containg p coords of events
        :param combined_voxel_channels: if True, create voxel grid merging positive and
            negative events (resulting in NUM_BINS x H x W tensor). Otherwise, create
            voxel grid for positive and negative events separately
            (resulting in 2*NUM_BINS x H x W tensor)
        """
        if combined_voxel_channels:
            # generate voxel grid which has size self.num_bins x H x W
            voxel_grid = events_to_voxel_torch(xs, ys, ts, ps, self.data_split,
                                               sensor_size=self.sensor_resolution)
        else:
            # generate voxel grid which has size 2*self.num_bins x H x W
            voxel_grid = events_to_neg_pos_voxel_torch(xs, ys, ts, ps, self.data_split,
                                                       sensor_size=self.sensor_resolution)
            voxel_grid = torch.cat([voxel_grid[0], voxel_grid[1]], 0)  # voxel_pos, voxel_neg

        # voxel_grid = voxel_grid*self.hot_events_mask

        return voxel_grid

    def process_event(self, events):
        xs = events[:, 0]
        ys = events[:, 1]
        ts = events[:, 2]
        ps = events[:, 3]
        try:
            ts_0, ts_k = ts[0], ts[-1]
        except:
            ts_0, ts_k = 0, 0
        if len(xs) < 3:
            voxel = self.get_empty_voxel_grid(self.combined_voxel_channels)
        else:
            xs = torch.from_numpy(xs).float()
            ys = torch.from_numpy(ys).float()
            ts = torch.from_numpy((ts - ts_0)).float()
            ps = torch.from_numpy(ps).float()
            voxel = self.get_voxel_grid(xs, ys, ts, ps, combined_voxel_channels=self.combined_voxel_channels)
        return voxel

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            raw_event = np.load(self.event_list[index])
            voxel = self.process_event(raw_event)
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index], voxel

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        raw_event = np.load(self.event_list[index])
        voxel = self.process_event(raw_event)
        voxel = np.array(voxel.permute(1, 2, 0)).astype(np.float32)
        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, voxel, mask = self.augmentor(img1, img2, flow, voxel, mask=np.zeros_like(img1))
            else:
                img1, img2, flow, voxel, mask = self.augmentor(img1, img2, flow, voxel, mask=np.zeros_like(img1))
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        voxel = torch.from_numpy(voxel).permute(2, 0, 1).float()

        eventdata = torch.zeros((4, int(self.data_split / 2), self.crop_size[0], self.crop_size[1]),
                                dtype=torch.float)
        eventdata[0, ...] = voxel[0:5, ...]
        eventdata[1, ...] = voxel[5:10, ...]
        eventdata[2, ...] = voxel[10:15, ...]
        eventdata[3, ...] = voxel[15:20, ...]

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        return {
            'former_image': img1.float(),
            'eventdata': eventdata,
            'valid': valid,
            'latter_image': img2.float(),
            'flow': flow.float()
        }, raw_event

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        self.event_list = v * self.event_list
        return self

    def __len__(self):
        return len(self.image_list)


class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = os.path.join(root, split, 'flow')
        image_root = os.path.join(root, split, dstype)
        event_root = os.path.join(root, split, dstype)
        print(event_root)
        self.sensor_resolution = (436, 1024)
        if aug_params is not None:
            self.crop_size = aug_params['crop_size']
        else:
            self.crop_size = self.sensor_resolution

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(os.path.join(image_root, scene, '*.png')))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(os.path.join(flow_root, scene, '*.flo')))
            self.event_list += sorted(glob(os.path.join(event_root, scene, 'event_data', '*.npy')))
            self.event_list.pop()


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(os.path.join(root, '*.ppm')))
        flows = sorted(glob(os.path.join(root, '*.flo')))
        assert (len(images) // 2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == 'training' and xid == 1) or (split == 'validation' and xid == 2):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2 * i], images[2 * i + 1]]]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(os.path.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([os.path.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(os.path.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([os.path.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(os.path.join(idir, '*.png')))
                    flows = sorted(glob(os.path.join(fdir, '*.pfm')))
                    for i in range(len(flows) - 1):
                        if direction == 'into_future':
                            self.image_list += [[images[i], images[i + 1]]]
                            self.flow_list += [flows[i]]
                        elif direction == 'into_past':
                            self.image_list += [[images[i + 1], images[i]]]
                            self.flow_list += [flows[i + 1]]


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = os.path.join(root, split)
        images1 = sorted(glob(os.path.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(os.path.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if split == 'training':
            self.flow_list = sorted(glob(os.path.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows) - 1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i + 1]]]

            seq_ix += 1


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')

    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100 * sintel_clean + 100 * sintel_final + 200 * kitti + 5 * hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100 * sintel_clean + 100 * sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              pin_memory=False, shuffle=True, num_workers=2, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader


def fetch_dataloader_ev(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """
    # aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
    aug_params = None
    sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
    train_dataset = 10 * sintel_clean

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              pin_memory=False, shuffle=True, num_workers=2, drop_last=True, collate_fn=collate_fn)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader


def collate_fn(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    out = dict.fromkeys(['former_image', 'eventdata', 'valid', 'latter_image', 'flow'])
    former_image = []
    eventdata = []
    valid = []
    latter_image = []
    flow = []
    event = []
    # count = []
    for i, data in enumerate(batch):
        former_image.append(data[0]['former_image'])
        eventdata.append(data[0]['eventdata'])
        valid.append(data[0]['valid'])
        latter_image.append(data[0]['latter_image'])
        flow.append(data[0]['flow'])

        # count.append(len(data[1]))
        event.append(np.c_[i * np.ones(len(data[1])), data[1]])
    out['former_image'] = torch.stack(former_image, dim=0)
    out['eventdata'] = torch.stack(eventdata, dim=0)
    out['valid'] = torch.stack(valid, dim=0)
    out['latter_image'] = torch.stack(latter_image, dim=0)
    out['flow'] = torch.stack(flow, dim=0)
    event = np.concatenate(event)
    return out, event


# rgb pretrain
from torch.utils import data
from utils import recursive_glob


# from augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale
class cityscapesLoader(data.Dataset):
    """cityscapesLoader
    https://www.cityscapes-dataset.com
    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/
    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    mean_rgb = {
        "pascal": [103.939, 116.779, 123.68],
        "cityscapes": [0.0, 0.0, 0.0],
    }  # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(
            self,
            root,
            split="train",
            is_transform=False,
            img_size=(768, 368),
            augmentations=None,
            img_norm=True,
            version="cityscapes",
            test_mode=False,
    ):
        """__init__
        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 2
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array(self.mean_rgb[version])
        self.files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(self.root, "gtCoarse", self.split)

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 17, 19, 20, 21, 22, 23, 24,
                             29, 30, -1]
        self.valid_classes = [
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtCoarse_labelIds.png",
        )

        img = Image.open(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = Image.open(lbl_path)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        """transform
        :param img:
        :param lbl:
        """
        img = np.array(Image.fromarray(img).resize((self.img_size[0], self.img_size[1])))  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = np.array(Image.fromarray(lbl).resize((self.img_size[0], self.img_size[1])))
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = 0
        for _validc in self.valid_classes:
            mask[mask == _validc] = 1
        return mask

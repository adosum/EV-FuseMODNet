import numpy as np
from PIL import Image
import cv2
from model.models import FlowNet2
# file = '/home/user/ev-deepseg/KITTI_MOD_fixed/training/mask/sf_000127_10.png'
# image = Image.open(file)
# array = np.array(image)
# a = array
# cv2.imshow('', array)
# cv2.waitKey(-1)
from data_augmentation import FlowAugmentor, Flow_event_mask_Augmentor
import frame_utils
import cv2
from vis_utils import flow2img
import torch
from events_contrast_maximization.utils.event_utils import events_to_voxel_torch, \
    events_to_neg_pos_voxel_torch
from load_datasets import MpiSintel
dataset = MpiSintel(split='training', dstype='clean')

for i in range(len(dataset)):
    img1, img2, flow, _, voxel = dataset[i]

    img1 = np.array(img1.permute(1, 2, 0)).astype(np.uint8)
    img2 = np.array(img2.permute(1, 2, 0)).astype(np.uint8)
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.imshow('event1', np.array(voxel[0, ...]))
    cv2.imshow('event2', np.array(voxel[1, ...]))
    cv2.imshow('event3', np.array(voxel[2, ...]))
    cv2.imshow('event4', np.array(voxel[3, ...]))
    cv2.imshow('event5', np.array(voxel[4, ...]))
    flow_rgb = flow2img(np.array(flow[0, :, :]), np.array(flow[1, :, :]))
    cv2.imshow('flow', flow_rgb)
    cv2.waitKey(1)
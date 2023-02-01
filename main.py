#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 16:38:13 2021

@author: user
"""
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from model.models import evMODNet2
from load_datasets import fusionDataset
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
import os
import datetime
import argparse
import cv2
import numpy as np
from utils import AverageMeter, save_checkpoint
from data_augmentation import RandomCrop, RandomFlip, CenterCrop
from kornia.losses import focal_loss
from vis_utils import flow_viz_np, draw_mask
from torchmetrics import JaccardIndex

parser = argparse.ArgumentParser(description='Spike-FlowNet Training on several datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', type=str, metavar='DIR', default='./datasets/KITTI_MOD',
                    help='path to dataset')
parser.add_argument('--savedir', type=str, metavar='DATASET', default='3d_flownets',
                    help='results save dir')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--bias-decay', default=0, type=float,
                    metavar='B', help='bias decay')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')
parser.add_argument('--pretrained-ev', dest='pretrained_ev', default=None,
                    help='path to pre-trained model')
parser.add_argument('--pretrained-rgb', dest='pretrained_rgb', default=None,
                    help='path to pre-trained model')
parser.add_argument('--pretrained-seg', dest='pretrained_seg', default=None,
                    help='path to pre-trained model')
parser.add_argument('--render', dest='render', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--num_steps', type=int, default=100000)
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--image_size', type=int, nargs='+', default=[368, 768])
parser.add_argument('--device_id', type=int, nargs='+', default=[0])
# -----------------------------args for the RAFT model-----------------------
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--iters', type=int, default=12)

args = parser.parse_args()

# dir_img = '/home/user/Documents/train_seq_room1_obj1/images/'
# dir_mask = '/home/user/Documents/train_seq_room1_obj1/masks/masks_full/'
# dir_event = '/home/user/Documents/train_seq_room1_obj1/events/'
epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_split = 10
n_iter = 0
pos_weight = 2 * torch.ones(args.image_size)
pos_weight = pos_weight.to(device=device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def validate(test_loader, net, epoch):
    net.eval()
    jaccard = JaccardIndex(task="binary").to(device)
    loss_sum = 0
    MIoU = AverageMeter()
    test_loader_iter = iter(test_loader)
    for i in range(len(test_loader)):
        batch = next(test_loader_iter)
        former_gray = batch['former_image']
        latter_gray = batch['latter_image']
        true_masks = batch['mask']
        eventdata = batch['eventdata']
        true_masks = true_masks.to(device=device)
        former_gray = former_gray.to(device=device, dtype=torch.float32)
        latter_gray = latter_gray.to(device=device, dtype=torch.float32)
        eventdata = eventdata.to(device=device, dtype=torch.float32)
        eventframe_show = np.array(torch.sum(torch.sum(eventdata.squeeze(0), 0), 0).cpu())
        pred, flow_rgb = net(former_gray, latter_gray, eventdata)
        # loss_sum += focal_loss(pred, true_masks, alpha=0.5, gamma=2.0, reduction='mean')
        loss_sum += criterion(pred, true_masks)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5) * 1
        if true_masks.max() == 1:
            MIoU.update(jaccard(true_masks, pred), eventdata.size(0))
        # else:
        #     MIoU.update(1, eventdata.size(0))
        if args.render and epoch < 0:
            pred = pred.squeeze(0).squeeze(0).cpu().detach().numpy()
            true_masks = true_masks.squeeze(0).squeeze(0).cpu().detach().numpy()
            imgshow = np.array(latter_gray.squeeze(0).cpu().detach().permute(1, 2, 0)).copy()
            imgshow = draw_mask(imgshow / 255, pred)
            cv2.imshow('img', imgshow)
            cv2.imshow('pred', pred.astype('float'))
            cv2.imshow('events', eventframe_show)
            cv2.imshow('groundtruth', true_masks.astype('float'))
            flow_rgb = flow_viz_np(flow_rgb.cpu().numpy()[0, 0, :, :], flow_rgb.cpu().numpy()[0, 1, :, :])
            cv2.imshow('rgbflow', flow_rgb)

            cv2.waitKey(1)

    print('-------------------------------------------------------')
    print('BCEWithLogitsLoss and moving IoU is: {:.2f}-----and-------{:.4f}'
          .format(loss_sum, MIoU.avg))
    print('-------------------------------------------------------')
    return MIoU.avg


def main():
    # Data loading code
    Test_dataset = fusionDataset(args.data, data_split, crop_size=args.image_size, aug_params=None, split='val',
                                 dark=True)
    test_loader = DataLoader(dataset=Test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=2)
    net = evMODNet2(args).cuda()
    if args.pretrained:
        print(' =>use the pretrained model: {}'.format(args.pretrained))
        net.load_state_dict(torch.load(args.pretrained)['state_dict'], strict=True)
        for name, param in net.named_parameters():
            if 'ev_flow' in name or 'rgb_flow' in name:
                param.requires_grad = False
    if args.pretrained_rgb:
        print(' =>use the pretrained flow model: {}---{}'.format(args.pretrained_ev, args.pretrained_rgb))
        # pre_ev = torch.load(args.pretrained_ev, map_location=device)['state_dict']
        pre_rgb = torch.load(args.pretrained_rgb, map_location=device)
        pre_seg = torch.load(args.pretrained_seg, map_location=device)['state_dict']
        # pre_ev = {'ev_flow.' + str(key): val for key, val in pre_ev.items()}
        pre_rgb = {'rgb_flow.' + str(key)[7:]: val for key, val in pre_rgb.items()}
        pre_seg = {key: val for key, val in pre_seg.items() if 'Conv' in key}
        # net.load_state_dict(pre_ev, strict=False)
        net.load_state_dict(pre_rgb, strict=False)
        net.load_state_dict(pre_seg, strict=False)
        for name, param in net.named_parameters():
            if 'ev_flow' in name or 'rgb_flow' in name:
                param.requires_grad = False
    net = torch.nn.DataParallel(net, device_ids=args.device_id).cuda()
    cudnn.benchmark = True
    if args.evaluate:
        with torch.no_grad():
            best_EPE = validate(test_loader, net, -1)
        return


if __name__ == '__main__':
    main()

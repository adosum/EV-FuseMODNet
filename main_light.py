#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 16:38:13 2021

@author: user
"""
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from model.models import evMODNet_light
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


parser = argparse.ArgumentParser(description='Spike-FlowNet Training on several datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', type=str, metavar='DIR', default='/home/user/ev-deepseg/datasets/KITTI_MOD',
                    help='path to dataset')
parser.add_argument('--savedir', type=str, metavar='DATASET', default='3d_flownets',
                    help='results save dir')
parser.add_argument('--weight-decay', '--wd', default=5e-5, type=float,
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
parser.add_argument('--render', dest='render', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--num_steps', type=int, default=100000)
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--image_size', type=int, nargs='+', default=[368, 768])
# -----------------------------args for the RAFT model-----------------------
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--iters', type=int, default=12)

args = parser.parse_args()

# dir_img = '/home/user/Documents/train_seq_room1_obj1/images/'
# dir_mask = '/home/user/Documents/train_seq_room1_obj1/masks/masks_full/'
# dir_event = '/home/user/Documents/train_seq_room1_obj1/events/'
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_split = 10
n_iter = 0

criterion = torch.nn.CrossEntropyLoss()


# else:
#     criterion = torch.nn.BCEWithLogitsLoss()

def train(train_loader, net, optimizer, epoch, train_writer):
    global n_iter
    net.train()

    losses = AverageMeter()

    for batch in tqdm(train_loader):
        true_masks = batch['mask']
        eventdata = batch['eventdata']
        former_gray = batch['former_image']
        latter_gray = batch['latter_image']

        true_masks = true_masks.to(device=device, dtype=torch.long)
        true_masks = true_masks.squeeze(1)
        former_gray = former_gray.to(device=device, dtype=torch.float32)
        latter_gray = latter_gray.to(device=device, dtype=torch.float32)
        eventdata = eventdata.to(device=device, dtype=torch.float32)
        pred = net(former_gray, latter_gray, eventdata)
        # pred = torch.argmax(pred,1)
        loss = criterion(pred, true_masks)

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(net.parameters(), 0.1)
        optimizer.step()
        train_writer.add_scalar('train_loss', loss.item(), n_iter)
        losses.update(loss.item(), eventdata.size(0))
        n_iter += 1
    print('Epoch: [{0}]\t Loss {1}'.format(epoch, losses))
    return losses.avg


def validate(test_loader, net, epoch):
    net.eval()

    loss_sum = 0
    test_loader_iter = iter(test_loader)
    for i in range(len(test_loader)):
        batch = next(test_loader_iter)
        former_gray = batch['former_image']
        latter_gray = batch['latter_image']
        true_masks = batch['mask'].squeeze(1)
        eventdata = batch['eventdata']
        true_masks = true_masks.to(device=device, dtype=torch.long)
        former_gray = former_gray.to(device=device, dtype=torch.float32)
        latter_gray = latter_gray.to(device=device, dtype=torch.float32)
        eventdata = eventdata.to(device=device, dtype=torch.float32)
        eventframe_show = np.array(torch.sum(torch.sum(eventdata.squeeze(0), 0), 0).cpu())
        pred = net(former_gray, latter_gray, eventdata)
        loss_sum += criterion(pred, true_masks)
        pred = torch.argmax(pred, 1)
        pred = pred.squeeze(0).cpu().detach().numpy()
        true_masks = true_masks.squeeze(0).cpu().detach().numpy()
        if args.render and epoch < 0:
            imgshow = np.array(latter_gray.squeeze(0).cpu().detach().permute(1, 2, 0)).copy()
            cv2.imshow('img', imgshow/255)
            cv2.imshow('pred', pred.astype('float'))
            cv2.imshow('events', eventframe_show)
            cv2.imshow('groundtruth', true_masks.astype('float'))
            cv2.waitKey(1)

    print('-------------------------------------------------------')
    print('loss: {:.2f}'
          .format(loss_sum))
    print('-------------------------------------------------------')
    return loss_sum


def main():
    # Data loading code
    Test_dataset = fusionDataset(args.data, data_split, crop_size=args.image_size, aug_params=None, split='val')
    test_loader = DataLoader(dataset=Test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=2)
    net = evMODNet_light(args).cuda()
    if args.pretrained:
        print(' =>use the pretrained model: {}'.format(args.pretrained))
        net.load_state_dict(torch.load(args.pretrained)['state_dict'], strict=False)
    if args.pretrained_ev:
        print(' =>use the pretrained flow model: {}---{}'.format(args.pretrained_ev, args.pretrained_rgb))
        pre_ev = torch.load(args.pretrained_ev, map_location=device)['state_dict']
        pre_ev = {'ev_flow.' + str(key): val for key, val in pre_ev.items()}
        net.load_state_dict(pre_ev, strict=False)
        for name, param in net.named_parameters():
            if 'ev_flow' in name or 'rgb_flow' in name:
                param.requires_grad = False
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True
    if args.evaluate:
        with torch.no_grad():
            best_EPE = validate(test_loader, net, -1)
        return
    aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
    Train_dataset = fusionDataset(args.data, data_split, aug_params=aug_params, split='train')
    train_loader = DataLoader(dataset=10*Train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=2)

    save_path = 'b{},lr{}'.format(
        args.batch_size,
        args.lr)
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    save_path = os.path.join(timestamp, save_path)
    save_path = os.path.join(args.savedir, save_path)
    print('\n => Everything will be saved to {}'.format(save_path))
    # log_var_a = torch.zeros((1,), requires_grad=True)
    # log_var_b = torch.zeros((1,), requires_grad=True)
    # param_groups = [{'params': net.bias_parameters(), 'weight_decay': args.bias_decay},
    #                 {'params': net.weight_parameters(), 'weight_decay': args.weight_decay}]

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.epsilon)
    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    test_writer = SummaryWriter(os.path.join(save_path, 'test'))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=args.lr,
                                                    steps_per_epoch=len(train_loader), epochs=epochs,
                                                    pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    best_EPE = -1
    for name, param in net.module.named_parameters():
        if 'ev_flow' in name or 'rgb_flow' in name:
            param.requires_grad = False
    for epoch in range(epochs):

        print('=> training started')
        train_loss = train(train_loader, net, optimizer, epoch, train_writer)
        train_writer.add_scalar('mean loss', train_loss, epoch)

        if (epoch + 1) % 2 == 0:
            with torch.no_grad():
                EPE = validate(test_loader, net, epoch)
            test_writer.add_scalar('mean EPE', EPE, epoch)
            if best_EPE < 0:
                best_EPE = EPE

            test_writer.add_scalar('mean EPE', EPE, epoch)
            is_best = EPE < best_EPE
            best_EPE = min(EPE, best_EPE)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.module.state_dict(),
                'best_EPE': best_EPE,
            }, is_best, save_path)
            scheduler.step()


if __name__ == '__main__':
    main()
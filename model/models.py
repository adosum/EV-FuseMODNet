import torch.nn as nn
from torch.nn import init
import numpy as np
import torch.utils.data
import torch
# from torch.nn.init import kaiming_normal_, constant_
# import math
from model.modules import FlowNet3D, FlowNetS, FlowNetC, FlowNetSD, FlowNetFusion, raft
from functools import partial
# from mmcv.ops.deform_conv import DeformConv2dPack
import torch.nn.functional as F

from model.modules.netutils import crop_like, conv_ev
from model.modules.netutils import conv_3d


class ResBlock(nn.Module):
    kernel_size = 3
    padding = 1

    def __init__(self, n_channels, use_deform_conv=False):
        super().__init__()
        bn2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        conv2d = nn.Conv2d if not use_deform_conv else nn.Conv2d
        self.conv1 = conv2d(n_channels, n_channels, self.kernel_size, padding=self.padding, bias=False)
        self.norm1 = bn2d(n_channels)
        self.conv2 = conv2d(n_channels, n_channels, self.kernel_size, padding=self.padding, bias=False)
        self.norm2 = bn2d(n_channels)

    def forward(self, x):
        x_conv = F.relu(self.norm1(self.conv1(x)))
        x_conv = self.norm2(self.conv2(x_conv))
        x_conv = F.relu(x + x_conv)
        return x_conv


class UNet2D(nn.Module):
    def __init__(self, n_input_feat: int, cfg):
        super().__init__()
        norm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        ch_down = cfg.DOWN_CHANNELS
        ch_up = cfg.UP_CHANNELS
        self.n_output_feat = ch_up[-1]

        # ---
        # Downsampling
        # ---
        self.conv0 = nn.Sequential(nn.Conv2d(n_input_feat, ch_down[0], 7, padding=3, bias=False), norm2d(ch_down[0]))
        self.res0 = ResBlock(ch_down[0])

        self.conv1 = nn.Sequential(nn.Conv2d(ch_down[0], ch_down[1], 3, stride=2, padding=1, bias=False),
                                   norm2d(ch_down[1]))  # tot_stride = 2
        self.res1 = ResBlock(ch_down[1])

        self.conv2 = nn.Sequential(nn.Conv2d(ch_down[1], ch_down[2], 3, stride=2, padding=1, bias=False),
                                   norm2d(ch_down[2]))  # tot_stride = 4
        self.res2 = ResBlock(ch_down[2])

        self.conv3 = nn.Sequential(nn.Conv2d(ch_down[2], ch_down[3], 3, stride=2, padding=1, bias=False),
                                   norm2d(ch_down[3]))  # tot_stride = 8
        self.res3 = ResBlock(ch_down[3])

        # ---
        # Up
        # ---
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ch_down[3], ch_up[1], 3, stride=2, padding=1, output_padding=1, bias=False),
            norm2d(ch_up[1])
        )  # tot_stride = 4
        self.up_res1 = ResBlock(ch_up[1], cfg.UP_DEFORM_CONV[1])

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ch_up[1] + ch_down[2], ch_up[2], 3, stride=2, padding=1, output_padding=1, bias=False),
            norm2d(ch_up[2])
        )  # tot_stride = 2
        self.up_res2 = ResBlock(ch_up[2], cfg.UP_DEFORM_CONV[2])

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(ch_up[2] + ch_down[1], ch_up[3], 3, stride=2, padding=1, output_padding=1, bias=False),
            norm2d(ch_up[3])
        )  # tot_stride = 1
        self.up_res3 = ResBlock(ch_up[3], cfg.UP_DEFORM_CONV[3])

        self.up4 = nn.Sequential(
            nn.Conv2d(ch_up[3] + ch_down[0], ch_up[4], 1, bias=False) if not cfg.UP_DEFORM_CONV[4] else
            nn.Conv2d(ch_up[3] + ch_down[0], ch_up[4], 1, bias=False),
            norm2d(ch_up[4])
        )
        self.up5 = nn.Sequential(
            nn.Conv2d(ch_up[4], ch_up[5], 1, bias=False) if not cfg.UP_DEFORM_CONV[5] else
            nn.Conv2d(ch_up[4], ch_up[5], 1, bias=False),
            norm2d(ch_up[5])
        )

    def forward(self, x_in):
        # ---
        # Down
        # ---
        x_out = self.conv0(x_in)
        x_skip0 = torch.clone(x_out)
        x_out = self.res0(x_out)

        x_out = self.conv1(x_out)
        x_skip1 = torch.clone(x_out)  # stride = 2
        x_out = self.res1(x_out)

        x_out = self.conv2(x_out)
        x_skip2 = torch.clone(x_out)  # stride = 4
        x_out = self.res2(x_out)

        x_out = self.conv3(x_out)  # stride = 8
        x_out = self.res3(x_out)

        # ---
        # Up
        # ---
        x_out = self.up1(x_out)  # stride = 4
        x_out = self.up_res1(x_out)

        x_out = torch.cat([x_out, x_skip2], dim=1)
        x_out = self.up2(x_out)  # stride = 2
        x_out = self.up_res2(x_out)

        x_out = torch.cat([x_out, x_skip1], dim=1)
        x_out = self.up3(x_out)  # stride = 1
        x_out = self.up_res3(x_out)

        x_out = torch.cat([x_out, x_skip0], dim=1)
        x_out = self.up4(x_out)
        x_out = self.up5(x_out)
        return x_out


class ev_flow(FlowNet3D.FlowNetS_3d):
    def __init__(self, batchNorm=True, div_flow=1):
        super(ev_flow, self).__init__(batchNorm=batchNorm)
        self.div_flow = div_flow

    def forward(self, events):
        out_conv1 = self.conv1(events)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv3 = out_conv3.squeeze(2)
        out_conv4 = self.conv4(out_conv3)

        out_rconv11 = self.conv_r11(out_conv4)
        out_rconv12 = self.conv_r12(out_rconv11) + out_conv4
        out_rconv21 = self.conv_r21(out_rconv12)
        out_rconv22 = self.conv_r22(out_rconv21) + out_rconv12

        flow4 = self.predict_flow4(self.pre_predict4(self.upsampled_flow4_to_3(out_rconv22)))
        flow4_up = crop_like(flow4, out_conv3)
        out_deconv3 = crop_like(self.deconv3(out_rconv22), out_conv3)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(self.pre_predict3(self.upsampled_flow3_to_2(concat3)))
        out_conv2_flat = torch.flatten(out_conv2, start_dim=1, end_dim=2)
        flow3_up = crop_like(flow3, out_conv2_flat)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2_flat)

        concat2 = torch.cat((out_conv2_flat, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(self.pre_predict2(self.upsampled_flow2_to_1(concat2)))
        out_conv1_flat = torch.flatten(out_conv1, start_dim=1, end_dim=2)
        flow2_up = crop_like(flow2, out_conv1_flat)
        out_deconv1 = crop_like(self.deconv1(concat2), out_conv1_flat)

        concat1 = torch.cat((out_conv1_flat, out_deconv1, flow2_up), 1)
        flow1 = self.predict_flow1(self.pre_predict1(self.upsampled_flow1_to_0(concat1)))
        return torch.mul(flow1, self.div_flow)

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


class ev_flow_pre(FlowNet3D.FlowNetS_3d):
    def __init__(self, batchNorm=True, div_flow=1):
        super(ev_flow, self).__init__(batchNorm=batchNorm)
        self.div_flow = div_flow

    def forward(self, events):
        out_conv1 = self.conv1(events)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv3 = out_conv3.squeeze(2)
        out_conv4 = self.conv4(out_conv3)

        out_rconv11 = self.conv_r11(out_conv4)
        out_rconv12 = self.conv_r12(out_rconv11) + out_conv4
        out_rconv21 = self.conv_r21(out_rconv12)
        out_rconv22 = self.conv_r22(out_rconv21) + out_rconv12

        flow4 = self.predict_flow4(self.pre_predict4(self.upsampled_flow4_to_3(out_rconv22)))
        flow4_up = crop_like(flow4, out_conv3)
        out_deconv3 = crop_like(self.deconv3(out_rconv22), out_conv3)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(self.pre_predict3(self.upsampled_flow3_to_2(concat3)))
        out_conv2_flat = torch.flatten(out_conv2, start_dim=1, end_dim=2)
        flow3_up = crop_like(flow3, out_conv2_flat)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2_flat)

        concat2 = torch.cat((out_conv2_flat, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(self.pre_predict2(self.upsampled_flow2_to_1(concat2)))
        out_conv1_flat = torch.flatten(out_conv1, start_dim=1, end_dim=2)
        flow2_up = crop_like(flow2, out_conv1_flat)
        out_deconv1 = crop_like(self.deconv1(concat2), out_conv1_flat)

        concat1 = torch.cat((out_conv1_flat, out_deconv1, flow2_up), 1)
        flow1 = self.predict_flow1(self.pre_predict1(self.upsampled_flow1_to_0(concat1)))
        return torch.mul(flow1, self.div_flow)

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


class raft_rgb(raft.RAFT):
    def __init__(self, args):
        super(raft_rgb, self).__init__(args)


class evMODNet2(nn.Module):
    def __init__(self, args):
        super(evMODNet2, self).__init__()
        norm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        n1 = 64
        filters = [n1, n1, n1 * 2, n1 * 2]
        ch_up = [n1 * 2, n1 * 2, n1 * 2, n1 * 2, n1, n1]
        # the rbg flow parts
        self.rgb_flow = raft_rgb(args)
        self.iters = args.iters
        # rgb process

        self.Conv1_rgb_process = nn.Sequential(nn.Conv2d(6, filters[0], 7, padding=3, bias=False), norm2d(filters[0]))
        self.Conv1_rgb_process_res = ResBlock(filters[0])

        self.Conv2_rgb_process = nn.Sequential(nn.Conv2d(filters[0], filters[1], 3, stride=2, padding=1, bias=False),
                                               norm2d(filters[1]))  # tot_stride = 2
        self.Conv2_rgb_process_res = ResBlock(filters[1])

        self.Conv3_rgb_process = nn.Sequential(nn.Conv2d(filters[1], filters[2], 3, stride=2, padding=1, bias=False),
                                               norm2d(filters[2]))  # tot_stride = 4
        self.Conv3_rgb_process_res = ResBlock(filters[2])

        self.Conv4_rgb_process = nn.Sequential(nn.Conv2d(filters[2], filters[3], 3, stride=2, padding=1, bias=False),
                                               norm2d(filters[3]))  # tot_stride = 8
        self.Conv4_rgb_process_res = ResBlock(filters[3])

        # rgb flow process
        self.Conv1_rgbflow_process = nn.Sequential(nn.Conv2d(2, filters[0], 7, padding=3, bias=False),
                                                   norm2d(filters[0]))
        self.Conv1_rgbflow_process_res = ResBlock(filters[0])

        self.Conv2_rgbflow_process = nn.Sequential(
            nn.Conv2d(filters[0], filters[1], 3, stride=2, padding=1, bias=False),
            norm2d(filters[1]))  # tot_stride = 2
        self.Conv2_rgbflow_process_res = ResBlock(filters[1])

        self.Conv3_rgbflow_process = nn.Sequential(
            nn.Conv2d(filters[1], filters[2], 3, stride=2, padding=1, bias=False),
            norm2d(filters[2]))  # tot_stride = 4
        self.Conv3_rgbflow_process_res = ResBlock(filters[2])

        self.Conv4_rgbflow_process = nn.Sequential(
            nn.Conv2d(filters[2], filters[3], 3, stride=2, padding=1, bias=False),
            norm2d(filters[3]))  # tot_stride = 8
        self.Conv4_rgbflow_process_res = ResBlock(filters[3])
        # ev encoder
        batchNorm = True
        self.conv1 = conv_3d(batchNorm, 4, 64, kernel_size=3, stride=(2, 1, 1), padding=(2, 1, 1))  # 4
        self.conv2 = conv_3d(batchNorm, 64, 128, kernel_size=3, stride=2, padding=(2, 1, 1))  # 3
        self.conv3 = conv_3d(batchNorm, 128, 256, kernel_size=3, stride=2, padding=(0, 1, 1))  # 1
        self.conv4 = conv_ev(batchNorm, 256, 512, kernel_size=3, stride=2)

        self.conv_r11 = conv_ev(batchNorm, 512, 512, kernel_size=3, stride=1)
        self.conv_r12 = conv_ev(batchNorm, 512, 512, kernel_size=3, stride=1)
        self.conv_r21 = conv_ev(batchNorm, 512, 512, kernel_size=3, stride=1)
        self.conv_r22 = conv_ev(batchNorm, 512, 512, kernel_size=3, stride=1)

        # fusion decoder
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(filters[3] * 2 + 512, ch_up[1], 3, stride=2, padding=1, output_padding=1, bias=False),
            norm2d(ch_up[1])
        )  # tot_stride = 4
        self.up_res1 = ResBlock(ch_up[1])

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ch_up[1] + filters[2] * 2 + 256, ch_up[2], 3, stride=2, padding=1, output_padding=1,
                               bias=False),
            norm2d(ch_up[2])
        )  # tot_stride = 2
        self.up_res2 = ResBlock(ch_up[2])

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(ch_up[2] + filters[1] * 2 + 384, ch_up[3], 3, stride=2, padding=1, output_padding=1,
                               bias=False),
            norm2d(ch_up[3])
        )  # tot_stride = 1
        self.up_res3 = ResBlock(ch_up[3])

        self.up4 = nn.Sequential(
            nn.Conv2d(ch_up[3] + filters[0] * 2 + 256, ch_up[4], 1, bias=False),
            norm2d(ch_up[4])
        )
        self.up5 = nn.Sequential(
            nn.Conv2d(ch_up[4], ch_up[5], 1, bias=False),
            norm2d(ch_up[5])
        )

        self.out = nn.Sequential(
            # nn.Dropout(p=0.2), nn.Conv2d(ch_up[5], 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(ch_up[5], 1, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, former_rgb, latter_rgb, events):
        # rgb flow parts
        _, flow_rgb = self.rgb_flow(former_rgb, latter_rgb, iters=self.iters, test_mode=True)
        # ev flow process
        # flow_ev = self.ev_flow(events)
        # flow_ev = flow_rgb
        # rgb process
        input_data = torch.cat([former_rgb, latter_rgb], 1)
        rgb_mean = input_data.contiguous().view(input_data.size()[:2] + (-1,)).mean(dim=-1).view(
            input_data.size()[:2] + (1, 1))
        x = (input_data - rgb_mean) / 255.

        e1_process = self.Conv1_rgb_process(x)
        e1_process_skip = torch.clone(e1_process)
        e1_process = self.Conv1_rgb_process_res(e1_process)

        e2_process = self.Conv2_rgb_process(e1_process)
        e2_process_skip = torch.clone(e2_process)  # stride = 2
        e2_process = self.Conv2_rgb_process_res(e2_process)

        e3_process = self.Conv3_rgb_process(e2_process)
        e3_process_skip = torch.clone(e3_process)  # stride = 4
        e3_process = self.Conv3_rgb_process_res(e3_process)

        e4_process = self.Conv4_rgb_process(e3_process)  # stride = 8
        e4_process = self.Conv4_rgb_process_res(e4_process)

        # rgb flow process
        e1_rgbflowprocess = self.Conv1_rgbflow_process(flow_rgb)
        e1_rgbflowprocess_skip = torch.clone(e1_rgbflowprocess)
        e1_rgbflowprocess = self.Conv1_rgbflow_process_res(e1_rgbflowprocess)

        e2_rgbflowprocess = self.Conv2_rgbflow_process(e1_rgbflowprocess)
        e2_rgbflowprocess_skip = torch.clone(e2_rgbflowprocess)  # stride = 2
        e2_rgbflowprocess = self.Conv2_rgbflow_process_res(e2_rgbflowprocess)

        e3_rgbflowprocess = self.Conv3_rgbflow_process(e2_rgbflowprocess)
        e3_rgbflowprocess_skip = torch.clone(e3_rgbflowprocess)  # stride = 4
        e3_rgbflowprocess = self.Conv3_rgbflow_process_res(e3_rgbflowprocess)

        e4_rgbflowprocess = self.Conv4_rgbflow_process(e3_rgbflowprocess)  # stride = 8
        e4_rgbflowprocess = self.Conv4_rgbflow_process_res(e4_rgbflowprocess)

        # ev flow encoder
        out_conv1 = self.conv1(events)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv3 = out_conv3.squeeze(2)
        out_conv4 = self.conv4(out_conv3)

        out_rconv11 = self.conv_r11(out_conv4)
        out_rconv12 = self.conv_r12(out_rconv11) + out_conv4
        out_rconv21 = self.conv_r21(out_rconv12)
        out_rconv22 = self.conv_r22(out_rconv21) + out_rconv12
        # final fusion and decoder
        # first try the no skip connection version
        e4_final_fusion = torch.cat((e4_process, e4_rgbflowprocess, out_rconv22), dim=1)
        # e4_final_fusion = e4_process + e4_rgbflowprocess + e4_evflow_process
        x_out = self.up1(e4_final_fusion)  # stride = 4
        x_out = self.up_res1(x_out)

        x_out = torch.cat([x_out, e3_process_skip, e3_rgbflowprocess_skip, out_conv3], dim=1)
        x_out = self.up2(x_out)  # stride = 2
        x_out = self.up_res2(x_out)

        out_conv2_flat = torch.flatten(out_conv2, start_dim=1, end_dim=2)
        x_out = torch.cat([x_out, e2_process_skip, e2_rgbflowprocess_skip, out_conv2_flat], dim=1)
        x_out = self.up3(x_out)  # stride = 1
        x_out = self.up_res3(x_out)

        out_conv1_flat = torch.flatten(out_conv1, start_dim=1, end_dim=2)
        x_out = torch.cat([x_out, e1_process_skip, e1_rgbflowprocess_skip, out_conv1_flat], dim=1)
        x_out = self.up4(x_out)
        x_out = self.up5(x_out)
        x_out = self.out(x_out)

        return x_out, flow_rgb

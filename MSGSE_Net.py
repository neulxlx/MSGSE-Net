import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import init_weights
from layers import conv_block, up_conv


class LSEGA(nn.Module):
    def __init__(self, in_channels, inter_channels, pool_size):
        super(LSEGA, self).__init__()
        self.pool = nn.Sequential(
            # nn.AvgPool2d(pool_size),
            nn.Conv2d(64, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            # nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(pool_size)
            nn.AvgPool2d(pool_size)
        )
        self.conv = conv_block(in_channels*2, in_channels)
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            # nn.GroupNorm(32, inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            # nn.GroupNorm(32, in_channels),
            nn.Sigmoid()
        )
        self.spatial_attention = spatial_attention(in_channels, inter_channels)
        # self.up = UP(in_channels, in_channels, 2)

    def forward(self, x, multi_scale_feature):
        multi_scale_feature = self.pool(multi_scale_feature)
        mixed = torch.cat([x, multi_scale_feature], dim=1)
        mixed = self.conv(mixed)
        attention = self.attention(mixed)
        x = x * attention
        x = self.spatial_attention(x, mixed)
        return x, attention


class MSblock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MSblock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, dilation=1, padding=1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True)
                                  )
        self.conv2 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, dilation=3, padding=3),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_ch, out_ch, kernel_size=1, dilation=1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True),
                                  )
        self.conv3 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, dilation=1, padding=1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_ch, out_ch, kernel_size=3, dilation=3, padding=3),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_ch, out_ch, kernel_size=1, dilation=1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True),
                                  )
        self.conv4 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, dilation=1, padding=1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_ch, out_ch, kernel_size=3, dilation=3, padding=3),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_ch, out_ch, kernel_size=3, dilation=5, padding=5),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True),                                  
                                   nn.Conv2d(out_ch, out_ch, kernel_size=1, dilation=1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True),
                                  )

                             
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return x1 + x2 + x3 + x4


class MSGSE_Net_3_ds(nn.Module):

    def __init__(self):
        super(MSGSE_Net_3_ds, self).__init__()
        in_ch = 1
        num_classes = 7

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, 64)
        self.Conv2 = conv_block(64, 128)
        self.Conv3 = conv_block(128, 256)
        self.Conv4 = conv_block(256, 512)

        self.up4 = up_conv(512, 256)

        self.up_conv4 = conv_block(512, 256)

        self.up3 = up_conv(256, 128)

        self.up_conv3 = conv_block(256, 128)

        self.up2 = up_conv(128, 64)

        self.up_conv2 = conv_block(128, 64)

        self.up4_1 = up_conv(512, 64, 8)
        self.up3_1 = up_conv(256, 64, 4)
        self.up2_1 = up_conv(128, 64, 2)
        self.multi_scale_conv = conv_block(256, 64)
        self.msblock = MSblock(64, 64)

        self.lsega2 = LSEGA(64, 32, 1)
        self.lsega3 = LSEGA(128, 64, 2)
        self.lsega4 = LSEGA(256, 128, 4)

        self.final = nn.Conv2d(64, 7, kernel_size=1, stride=1, padding=0)
        
        self.ds = nn.Conv2d(64, 7, kernel_size=1, stride=1, padding=0)
        self.ds2 = nn.Conv2d(64, 7, kernel_size=1, stride=1, padding=0)

        self.ds3 = nn.Sequential(nn.Conv2d(128, 7, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor = 2))
        self.ds4 = nn.Sequential(nn.Conv2d(256, 7, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor = 4))


    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)

        up1 = e1
        up2 = self.up2_1(e2)
        up3 = self.up3_1(e3)
        up4 = self.up4_1(e4)

        multi_scale_feature = self.multi_scale_conv(torch.cat([up1, up2, up3, up4], dim=1))
        ds = self.ds(multi_scale_feature)
        multi_scale_feature = self.msblock(multi_scale_feature)


        d4 = self.up4(e4)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.up_conv4(d4)
        d4, att4 = self.lsega4(d4, multi_scale_feature)

        d3 = self.up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.up_conv3(d3)
        d3, att3 = self.lsega3(d3, multi_scale_feature)

        d2 = self.up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.up_conv2(d2)
        d2, att2 = self.lsega2(d2, multi_scale_feature)

        out = self.final(d2)
        
        att_ds2 = self.ds2(att2)
        att_ds3 = self.ds3(att3)
        att_ds4 = self.ds4(att4)
        
        if self.training:
            return out, att_ds2, att_ds3, att_ds4, ds
        else:
            return out


class MSGSE_4_ds(nn.Module):

    def __init__(self):
        super(MSGSE_4_ds, self).__init__()
        in_ch = 1
        num_classes = 7


        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, 64)
        self.Conv2 = conv_block(64, 128)
        self.Conv3 = conv_block(128, 256)
        self.Conv4 = conv_block(256, 512)
        self.center = conv_block(512, 512)

        self.up4 = up_conv(512, 512)
        self.up_conv4 = conv_block(1024, 512)

        self.up3 = up_conv(512, 256)
        self.up_conv3 = conv_block(512, 256)

        self.up2 = up_conv(256, 128)
        self.up_conv2 = conv_block(256, 128)

        self.up1 = up_conv(128, 64)
        self.up_conv1 = conv_block(128, 64)

        self.up_center = up_conv(512, 64, 16)
        self.up4_1 = up_conv(512, 64, 8)
        self.up3_1 = up_conv(256, 64, 4)
        self.up2_1 = up_conv(128, 64, 2)
        self.multi_scale_conv = conv_block(320, 64)
        self.msblock = MSblock(64, 64)
        self.ms_ds = nn.Conv2d(64, 7, kernel_size=1, stride=1, padding=0)

        self.lsega1 = LSEGA(64, 32, 1)
        self.lsega2 = LSEGA(128, 64, 2)
        self.lsega3 = LSEGA(256, 128, 4)
        self.lsega4 = LSEGA(512, 256, 8)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

        self.att_ds1 = nn.Conv2d(64, 7, kernel_size=1, stride=1, padding=0)
        self.att_ds2 = nn.Sequential(nn.Conv2d(128, 7, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor = 2))
        self.att_ds3 = nn.Sequential(nn.Conv2d(256, 7, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor = 4))
        self.att_ds4 = nn.Sequential(nn.Conv2d(512, 7, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor = 8))
    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)

        center = self.Maxpool(e4)
        center = self.center(center)

        up1 = e1
        up2 = self.up2_1(e2)
        up3 = self.up3_1(e3)
        up4 = self.up4_1(e4)
        up_center = self.up_center(center)

        multi_scale_feature = self.multi_scale_conv(torch.cat([up1, up2, up3, up4, up_center], dim=1))
        ms_ds = self.ms_ds(multi_scale_feature)
        multi_scale_feature = self.msblock(multi_scale_feature)

        d4 = self.up4(center)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.up_conv4(d4)
        d4, att4 = self.lsega4(d4, multi_scale_feature)

        d3 = self.up3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.up_conv3(d3)
        d3, att3 = self.lsega3(d3, multi_scale_feature)

        d2 = self.up2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.up_conv2(d2)
        d2, att2 = self.lsega2(d2, multi_scale_feature)

        d1 = self.up1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.up_conv1(d1)
        d1, att1 = self.lsega1(d1, multi_scale_feature)

        out = self.final(d1)
        att_ds1 = self.att_ds1(att1)
        att_ds2 = self.att_ds2(att2)
        att_ds3 = self.att_ds3(att3)
        att_ds4 = self.att_ds4(att4)

        if self.training:
            return out, att_ds1, att_ds2, att_ds3, att_ds4, ms_ds
        else:
            return out

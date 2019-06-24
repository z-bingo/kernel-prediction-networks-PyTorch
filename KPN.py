import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchsummary import summary
import torchvision.models as models

class Basic(nn.Module):
    def __init__(self, in_ch, out_ch, g=16, channel_att=False):
        super(Basic, self).__init__()
        self.channel_att = channel_att
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )
        self.att = nn.Sequential(
            nn.Linear(out_ch, out_ch//g),
            nn.ReLU(),
            nn.Linear(out_ch//g, out_ch),
            nn.Sigmoid()
        )

    def forward(self, data):
        fm = self.conv1(data)
        if self.channel_att:
            fm_pool = F.adaptive_max_pool2d(fm, (1, 1))
            att = self.att(fm_pool.squeeze())
            return fm*(att.unsqueeze(-1).unsqueeze(-1))
        else:
            return fm


class KPN(nn.Module):
    def __init__(self, in_channel, out_channel, channel_att=False):
        super(KPN, self).__init__()
        # 各个卷积层定义
        # 2~5层都是均值池化+3层卷积
        self.conv1 = Basic(in_channel, 64, channel_att=channel_att)
        self.conv2 = Basic(64, 128, channel_att=channel_att)
        self.conv3 = Basic(128, 256, channel_att=channel_att)
        self.conv4 = Basic(256, 512, channel_att=channel_att)
        self.conv5 = Basic(512, 512, channel_att=channel_att)
        # 6~8层要先上采样再卷积
        self.conv6 = Basic(512+512, 512, channel_att=channel_att)
        self.conv7 = Basic(256+512, 256, channel_att=channel_att)
        self.conv8 = Basic(256+128, out_channel, channel_att=False)
        self.outc = nn.Conv2d(out_channel, out_channel, 3, 1, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)


    # 前向传播函数
    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2))
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=2, stride=2))
        conv5 = self.conv5(F.avg_pool2d(conv4, kernel_size=2, stride=2))
        # 开始上采样  同时要进行skip connection
        conv6 = self.conv6(torch.cat([conv4, F.interpolate(conv5, scale_factor=2, mode='bilinear')], dim=1))
        conv7 = self.conv7(torch.cat([conv3, F.interpolate(conv6, scale_factor=2, mode='bilinear')], dim=1))
        conv8 = self.conv8(torch.cat([conv2, F.interpolate(conv7, scale_factor=2, mode='bilinear')], dim=1))
        # return channel K*K*N
        out = self.outc(F.interpolate(conv8, scale_factor=2, mode='bilinear'))
        return out

# if __name__ == '__main__':
    # kpn = KPN(6, 5*5*6).cuda()
    # vgg16 = models.vgg16(pretrained=True).cuda()
    # print(summary(vgg16, (3, 224, 224), batch_size=4))

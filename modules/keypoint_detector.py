import numpy as np
from torch import nn
import torch
from torchvision import models

from modules.test import imshow


class KPDetector(nn.Module):
    """
    Predict K*5 keypoints.
    """

    def __init__(self, num_bg, **kwargs):
        super(KPDetector, self).__init__()
        self.num_tps_bg = num_bg

        self.fg_encoder = models.resnet18(pretrained=False)
       # self.fg_encoder.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_features = self.fg_encoder.fc.in_features
        self.fg_encoder.fc = nn.Linear(num_features, num_bg*4*2)

    def forward(self, image):
        # 输入:源图像和驱动图像tensor(16,3,256,256)，16是batch_size的值
        # a = np.ones((16, 1, 256, 256))
        # a = torch.Tensor(a)
        # a=a.cuda()
        # image = torch.cat((a, image), 1)
        fg_kp = self.fg_encoder(image)
        bs, _, = fg_kp.shape

        fg_kp = torch.sigmoid(fg_kp)
        fg_kp = fg_kp * 2 - 1
        out = {'fg_kp': fg_kp.view(bs, self.num_tps_bg*4, -1)}

        # 输出：关键点坐标tensor(16,50,2),50个坐标
        return out

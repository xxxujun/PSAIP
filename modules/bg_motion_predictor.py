from torch import nn
import torch
from torchvision import models

class BGMotionPredictor(nn.Module):
    """
    Module for background estimation, return single transformation, parametrized as 3x3 matrix. The third row is [0 0 1]
    用于背景估计的模块，返回单个变换，参数化为3x3矩阵。第三行是[0 0 1]
    """

    def __init__(self):
        super(BGMotionPredictor, self).__init__()
        # 用resnet18来进行背景估计
        self.bg_encoder = models.resnet18(pretrained=False)
        self.bg_encoder.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_features = self.bg_encoder.fc.in_features
        self.bg_encoder.fc = nn.Linear(num_features, 6)
        self.bg_encoder.fc.weight.data.zero_() # weight置0
        self.bg_encoder.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)) #将bias置为[1.0.0.0.1.0]，偏置可以加速神经网络拟合

    def forward(self, source_image, driving_image):
        # 输入：S+D——tensor(16,3,256,256)+tensor(16,3,256,256)
        bs = source_image.shape[0]
        out = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).type(source_image.type())
        prediction = self.bg_encoder(torch.cat([source_image, driving_image], dim=1))
        # prediction(16,6):tensor[[1,0,0,0,1,0]×16]
        out[:, :2, :] = prediction.view(bs, 2, 3)
        # 输出：tensor(16,3,3)——tensor([[[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]]×16], device='cuda:0', grad_fn=<CopySlices>)
        return out

from torch import nn
import torch
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d, TPS
from torchvision import models
import numpy as np


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    VGG19用来计算感知损失
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        # 可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
        # 所以经过类型转换这个self.v变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        # 平均值、标准
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            x=self.parameters()
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        # 输入：
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        # 输出：
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    创建图像金字塔，用于计算金字塔感知损失。
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            # scales:[1,0.5,0.25,0.125],进行四次带限下采样，用于更好地保留输入信号。
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        # ModuleDict可以像常规Python字典一样索引，同样自动将每个 module 的 parameters 添加到网络之中的容器
        # 根据四次下采样组成金字塔
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, bg_predictor, dense_motion_network, inpainting_network, train_params, *kwargs):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.inpainting_network = inpainting_network
        self.dense_motion_network = dense_motion_network

        self.bg_predictor = None
        if bg_predictor:
            self.bg_predictor = bg_predictor
            self.bg_start = train_params['bg_start']

        self.train_params = train_params
        self.scales = train_params['scales']

        # 图像金字塔：对图像的多分辨率通过VGG-19进行感知损失计算
        self.pyramid = ImagePyramide(self.scales, inpainting_network.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']
        self.dropout_epoch = train_params['dropout_epoch']
        self.dropout_maxp = train_params['dropout_maxp']
        self.dropout_inc_epoch = train_params['dropout_inc_epoch']
        self.dropout_startp =train_params['dropout_startp']
        
        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()


    def forward(self, x, epoch):
        kp_source = self.kp_extractor(x['source']) #S关键点
        kp_driving = self.kp_extractor(x['driving'])# D关键点
        # print(kp_source_bg['fg_kp'].shape)
        # kp_source_face= x['source_keypoint']
        # kp_driving_face= x['driving_keypoint']
        # kp_source= {'fg_kp': torch.cat((kp_source_face['fg_kp'],kp_source_bg['fg_kp']),1)}
        # kp_driving= {'fg_kp': torch.cat((kp_driving_face['fg_kp'],kp_driving_bg['fg_kp']),1)}
        bg_param = None
        if self.bg_predictor:
            if(epoch>=self.bg_start): # epoch>10的时候才进行背景运动的估计
                bg_param = self.bg_predictor(x['source'], x['driving']) # 背景估计
          
        if(epoch>=self.dropout_epoch): # epoch<35的时候进行dropout,epoch>35就不再进行dropout
            dropout_flag = False
            dropout_p = 0
        else:
            # dropout_p will linearly increase from dropout_startp to dropout_maxp
            # dropout_p将从dropout_startp线性增加到dropout_maxp
            dropout_flag = True
            dropout_p = min(epoch/self.dropout_inc_epoch * self.dropout_maxp + self.dropout_startp, self.dropout_maxp)
        
        dense_motion = self.dense_motion_network(source_image=x['source'], kp_driving=kp_driving,
                                                    kp_source=kp_source, bg_param = bg_param, 
                                                    dropout_flag = dropout_flag, dropout_p = dropout_p)
        generated = self.inpainting_network(x['source'], dense_motion)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

        print(generated['prediction'])
        print(generated['warped_encoder_maps'])
        loss_values = {}

        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])

        # reconstruction loss  VGG计算重建损失——式（9）
        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
            loss_values['perceptual'] = value_total

        # equivariance loss  等方差损失——式（10），用来约束关键点检测器
        if self.loss_weights['equivariance_value'] != 0:
            transform_random = TPS(mode = 'random', bs = x['driving_mask'].shape[0], **self.train_params['transform_params'])
            transform_grid = transform_random.transform_frame(x['driving_mask'])
            transformed_frame = F.grid_sample(x['driving_mask'], transform_grid, padding_mode="reflection",align_corners=True)
            transformed_kp = self.kp_extractor(transformed_frame)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp
        
            warped = transform_random.warp_coordinates(transformed_kp['fg_kp'])
            kp_d = kp_driving_bg['fg_kp']
            value = torch.abs(kp_d - warped).mean()
            loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

        # warp loss——式（13）
        if self.loss_weights['warp_loss'] != 0:
            occlusion_map = generated['occlusion_map']
            encode_map = self.inpainting_network.get_encode(x['driving'], occlusion_map)
            decode_map = generated['warped_encoder_maps']
            value = 0
            for i in range(len(encode_map)):
                value += torch.abs(encode_map[i]-decode_map[-i-1]).mean()

            loss_values['warp_loss'] = self.loss_weights['warp_loss'] * value
        
        # bg loss——式（12）
        if self.bg_predictor and epoch >= self.bg_start and self.loss_weights['bg'] != 0:
            bg_param_reverse = self.bg_predictor(x['driving'], x['source'])
            value = torch.matmul(bg_param, bg_param_reverse)
            eye = torch.eye(3).view(1, 1, 3, 3).type(value.type())
            value = torch.abs(eye - value).mean()
            loss_values['bg'] = self.loss_weights['bg'] * value

        return loss_values, generated

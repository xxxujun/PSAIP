from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian
from modules.util import to_homogeneous, from_homogeneous, UpBlock2d, TPS
import math

class DenseMotionNetwork(nn.Module):
    """
    Module that estimating an optical flow and multi-resolution occlusion masks 
                        from K TPS transformations and an affine transformation.
    从K个TPS变换和仿射变换中估计光流和多分辨率遮挡掩模的模块。
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_tps,num_face,num_bg, num_channels,
                 scale_factor=0.25, bg = False, multi_mask = True, kp_variance=0.01):
        super(DenseMotionNetwork, self).__init__()

        if scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, scale_factor)
        self.scale_factor = scale_factor
        self.multi_mask = multi_mask

        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_channels * (num_tps+1) + num_tps*4+1),
                                   max_features=max_features, num_blocks=num_blocks)

        hourglass_output_size = self.hourglass.out_channels
        self.maps = nn.Conv2d(hourglass_output_size[-1], num_tps + 1, kernel_size=(7, 7), padding=(3, 3))

        if multi_mask: #多层分辨率遮罩层
            up = []
            self.up_nums = int(math.log(1/scale_factor, 2))
            self.occlusion_num = 4
            
            channel = [hourglass_output_size[-1]//(2**i) for i in range(self.up_nums)]
            for i in range(self.up_nums):
                up.append(UpBlock2d(channel[i], channel[i]//2, kernel_size=3, padding=1))
            self.up = nn.ModuleList(up)

            channel = [hourglass_output_size[-i-1] for i in range(self.occlusion_num-self.up_nums)[::-1]]
            for i in range(self.up_nums):
                channel.append(hourglass_output_size[-1]//(2**(i+1)))
            occlusion = []
            
            for i in range(self.occlusion_num):
                occlusion.append(nn.Conv2d(channel[i], 1, kernel_size=(7, 7), padding=(3, 3)))
            self.occlusion = nn.ModuleList(occlusion)
        else:
            occlusion = [nn.Conv2d(hourglass_output_size[-1], 1, kernel_size=(7, 7), padding=(3, 3))]
            self.occlusion = nn.ModuleList(occlusion)

        self.num_tps = num_tps
        self.num_face = num_face
        self.num_bg = num_bg
        self.bg = bg
        self.kp_variance = kp_variance

        
    def create_heatmap_representations(self, source_image, kp_driving, kp_source):

        # 计算 H_k用于表示关键点与邻域的变化
        # 每个关键点pk，我们额外计算热图Hk，指示每个转换发生的密集运动网络——FOMM中的公式6
        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving['fg_kp'], spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian(kp_source['fg_kp'], spatial_size=spatial_size, kp_variance=self.kp_variance)
        # 热图Hk=D的关键点及其邻域的高斯表示-S的关键点及其邻域的高斯表示
        heatmap = gaussian_driving - gaussian_source

        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type()).to(heatmap.device)
        heatmap = torch.cat([zeros, heatmap], dim=1)
# 输出：(16,51,64,64)
        return heatmap

    def create_transformations(self, source_image, kp_driving, kp_source, bg_param):
        # K TPS transformaions
        # 输入：source_image(16,3,64,64), kp_driving(16,50,2), kp_source(16,50,2), bg_param(16,3,3)
        bs, _, h, w = source_image.shape
        kp_1 = kp_driving['fg_kp']
        kp_2 = kp_source['fg_kp']
        # 将关键点分组，5个为一组做TPS变换
        kp_1 = kp_1.view(bs, -1, 4, 2)
        kp_2 = kp_2.view(bs, -1, 4, 2)
        # 定义对分组关键点进行的TPS变换
        trans = TPS(mode = 'kp', bs = bs, kp_1 = kp_1, kp_2 = kp_2)
        # 用TPS类内置的函数产生变换
        driving_to_source = trans.transform_frame(source_image)

        identity_grid = make_coordinate_grid((h, w), type=kp_1.type()).to(kp_1.device)
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)

        # affine background transformation 背景仿射变换
        if not (bg_param is None):            
            identity_grid = to_homogeneous(identity_grid)
            identity_grid = torch.matmul(bg_param.view(bs, 1, 1, 1, 3, 3), identity_grid.unsqueeze(-1)).squeeze(-1)
            identity_grid = from_homogeneous(identity_grid)

        transformations = torch.cat([identity_grid, driving_to_source], dim=1)
        # 返回TPS变换和背景变换的整合：(16,11,64,64,2)
        return transformations

    def create_deformed_source_image(self, source_image, transformations):
# 提取关键点变化并对齐到原图S
# 输入：源图像(16,3,64,64)、11个变换(16,11,64,64,2)
        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_tps + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_tps + 1), -1, h, w)
        transformations = transformations.view((bs * (self.num_tps + 1), h, w, -1))
        deformed = F.grid_sample(source_repeat, transformations, align_corners=True) # 执行空间转换
        deformed = deformed.view((bs, self.num_tps+1, -1, h, w))
#输出：变换后的图像（16,33,64,64）——每个通道变换有11个变换
        return deformed

    def dropout_softmax(self, X, P):
        '''
        Dropout for TPS transformations. Eq(7) and Eq(8) in the paper.
        TPS转换的退出。文中的方程(7)和方程(8)。
        对TPS转换进行dropout，防止网络过度依赖一些贡献图上像素值为0的TPS变换
        '''
        drop = (torch.rand(X.shape[0],X.shape[1]) < (1-P)).type(X.type()).to(X.device)
        drop[..., 0] = 1
        drop = drop.repeat(X.shape[2],X.shape[3],1,1).permute(2,3,0,1)

        maxx = X.max(1).values.unsqueeze_(1)
        X = X - maxx
        X_exp = X.exp()
        X[:,1:,...] /= (1-P)
        mask_bool =(drop == 0)
        X_exp = X_exp.masked_fill(mask_bool, 0)
        partition = X_exp.sum(dim=1, keepdim=True) + 1e-6
        return X_exp / partition  

    def forward(self, source_image, kp_driving, kp_source, bg_param = None, dropout_flag=False, dropout_p = 0):
        # 输入：源图像（16,3,256,256）、S关键点（16,50,2），D关键点（16,50,2），背景参数、dropout参数
        # 根据比例因子对图像进行适当的下采样（抗锯齿插值，用于更好的保留输入信号），源图像→(16,3,64,64)
        if self.scale_factor != 1:
            source_image = self.down(source_image)

        bs, _, h, w = source_image.shape

        out_dict = dict()
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)
        transformations = self.create_transformations(source_image, kp_driving, kp_source, bg_param)
        deformed_source = self.create_deformed_source_image(source_image, transformations)
        out_dict['deformed_source'] = deformed_source
        # out_dict['transformations'] = transformations
        deformed_source = deformed_source.view(bs,-1,h,w)
        input = torch.cat([heatmap_representation, deformed_source], dim=1)
        input = input.view(bs, -1, h, w)

        #预测运动光流
        prediction = self.hourglass(input, mode = 1)

        contribution_maps = self.maps(prediction[-1]) 
        if(dropout_flag):
            contribution_maps = self.dropout_softmax(contribution_maps, dropout_p)
        else:
            contribution_maps = F.softmax(contribution_maps, dim=1)
        out_dict['contribution_maps'] = contribution_maps

        # Combine the K+1 transformations
        # Eq(6) in the paper
        contribution_maps = contribution_maps.unsqueeze(2)
        transformations = transformations.permute(0, 1, 4, 2, 3)
        deformation = (transformations * contribution_maps).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)

        out_dict['deformation'] = deformation # Optical Flow

        occlusion_map = [] #多尺度遮挡层
        if self.multi_mask:
            for i in range(self.occlusion_num-self.up_nums):
                occlusion_map.append(torch.sigmoid(self.occlusion[i](prediction[self.up_nums-self.occlusion_num+i])))
            prediction = prediction[-1]
            for i in range(self.up_nums):
                prediction = self.up[i](prediction)
                occlusion_map.append(torch.sigmoid(self.occlusion[i+self.occlusion_num-self.up_nums](prediction)))
        else:
            occlusion_map.append(torch.sigmoid(self.occlusion[0](prediction[-1])))
                
        out_dict['occlusion_map'] = occlusion_map # Multi-resolution Occlusion Masks
        # 输出：扭曲图像(16,11,3,64,64)、dropout后的贡献图(16,11,64,64)、预测光流(16,64,64,2)、多尺度遮挡层((16,1,32,32),(16,1,64,64)，(16,1,128,128)，(16,1,256,256))
        return out_dict

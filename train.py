import cv2
import dlib
import numpy as np
from tqdm import trange
import torch
from torch.utils.data import DataLoader
from logger import Logger
from modules.model import GeneratorFullModel
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils import clip_grad_norm_
from frames_dataset import DatasetRepeater
import math
detector = dlib.get_frontal_face_detector()
# 下载人脸关键点检测模型： http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_path = './modules/shape_predictor_68_face_landmarks.dat'
# 检测人脸关键点
predictor = dlib.shape_predictor(predictor_path)

def keypoint_detector(image):
    bs, _, _, _, = image.shape
    for i in range(bs):
        img = image[i:i + 1]
        img = np.squeeze(img)
        img1 = img.transpose(0, 1)
        img2 = img1.transpose(2, 1)
        img = img2.cpu().numpy()
        img = (img * 255).astype(np.uint8)
        dets = detector(img, 1)
        global fg_kp1
        global mask1
        for k, d in enumerate(dets):
            shape = predictor(img, d)
            mask1 = img
            mask1[d.top():d.bottom(), d.left():d.right()] = 255
            mask1 = mask1.transpose(2, 0, 1)
            flog = -1
            a = [2, 3, 5, 6, 8, 10, 12, 13, 15, 16, 18, 20, 22, 23, 25, 27, 28, 30, 32, 34, 36, 37, 39, 40, 42, 43, 45,
                 46, 48, 50, 51, 53, 54, 56, 57, 59, 60, 62, 64, 66, 68]
            for p in shape.parts():
                flog = flog + 1
                if flog == 0:
                    fg_kp1 = np.array([[p.x, p.y]])
                elif flog not in a:
                    b = np.array([[p.x, p.y]])
                    fg_kp1 = np.append(fg_kp1, values=b, axis=0)
        if i == 0:
            fg_kp = np.array([fg_kp1])
            mask=np.array([mask1])
        else:
            b = np.array([fg_kp1])
            fg_kp = np.append(fg_kp, values=b, axis=0)
            c = np.array([mask1])
            mask = np.append(mask, values=c, axis=0)
    fg_kp = (fg_kp / 255).astype(np.float32)
    fg_kp = torch.tensor(fg_kp)
    fg_kp = fg_kp.cuda()
    fg_kp = fg_kp * 2 - 1
    out = {'fg_kp': fg_kp.view(bs, 28, -1)}
    mask = (mask / 255).astype(np.float32)
    mask = torch.tensor(mask)
    mask = mask.cuda()
    return out,mask

def train(config, inpainting_network, kp_detector, bg_predictor, dense_motion_network, checkpoint, log_dir, dataset):
    train_params = config['train_params']
    # 定义优化器：三个网络的参数
    # 对应三个损失计算：inpainting_network——感知损失，dense_motion_network——warp损失，kp_detector——等方差值（关键点计算的损失）
    optimizer = torch.optim.Adam(
        [{'params': list(inpainting_network.parameters()) +
                    list(dense_motion_network.parameters()) +
                    list(kp_detector.parameters()) , 'initial_lr': train_params['lr_generator']}],lr=train_params['lr_generator'], betas=(0.5, 0.999), weight_decay = 1e-4)
    # 单独定义背景估计的优化器（在FOMM的基础上新加的，不破坏以前的结构，单独写的优化器）
    optimizer_bg_predictor = None
    if bg_predictor:
        optimizer_bg_predictor = torch.optim.Adam(
            [{'params':bg_predictor.parameters(),'initial_lr': train_params['lr_generator']}], 
            lr=train_params['lr_generator'], betas=(0.5, 0.999), weight_decay = 1e-4)

    # 加载模型
    if checkpoint is not None:
        start_epoch = Logger.load_cpk(
            checkpoint, inpainting_network = inpainting_network, dense_motion_network = dense_motion_network,       
            kp_detector = kp_detector, bg_predictor = bg_predictor,
            optimizer = optimizer, optimizer_bg_predictor = optimizer_bg_predictor)
        print('load success:', start_epoch)
        start_epoch += 1
    else:
        start_epoch = 0

    # 动态调整学习率
    scheduler_optimizer = MultiStepLR(optimizer, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    if bg_predictor:
        scheduler_bg_predictor = MultiStepLR(optimizer_bg_predictor, train_params['epoch_milestones'],
                                              gamma=0.1, last_epoch=start_epoch - 1)

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats']) # 多次传递相同的数据集，以获得更好的i/o性能
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, 
                            num_workers=train_params['dataloader_workers'], drop_last=True)

    # 定义完整的生成模型
    generator_full = GeneratorFullModel(kp_detector, bg_predictor, dense_motion_network, inpainting_network, train_params)

    if torch.cuda.is_available():
        generator_full = torch.nn.DataParallel(generator_full).cuda()  
        
    # 第十次迭代才对背景进行单独的运动估计
    bg_start = train_params['bg_start']
    
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], 
                checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for x in dataloader:
                if(torch.cuda.is_available()):
                    x['driving'] = x['driving'].cuda()
                    x['source'] = x['source'].cuda()

                x['driving_keypoint'], x['driving_mask'] = keypoint_detector(x['driving'])
                x['source_keypoint'], x['source_mask'] = keypoint_detector(x['source'])
                losses_generator, generated = generator_full(x, epoch)
                print(losses_generator)
                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)
                loss.backward()

                clip_grad_norm_(kp_detector.parameters(), max_norm=10, norm_type = math.inf)
                clip_grad_norm_(dense_motion_network.parameters(), max_norm=10, norm_type = math.inf)
                if bg_predictor and epoch>=bg_start:
                    clip_grad_norm_(bg_predictor.parameters(), max_norm=10, norm_type = math.inf)
                
                optimizer.step()
                optimizer.zero_grad()
                if bg_predictor and epoch>=bg_start:
                    optimizer_bg_predictor.step()
                    optimizer_bg_predictor.zero_grad()
                
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)

            scheduler_optimizer.step()
            if bg_predictor:
                scheduler_bg_predictor.step()
            
            model_save = {
                'inpainting_network': inpainting_network,
                'dense_motion_network': dense_motion_network,
                'kp_detector': kp_detector,
                'optimizer': optimizer,
            }
            if bg_predictor and epoch>=bg_start:
                model_save['bg_predictor'] = bg_predictor
                model_save['optimizer_bg_predictor'] = optimizer_bg_predictor
            
            logger.log_epoch(epoch, model_save, inp=x, out=generated)


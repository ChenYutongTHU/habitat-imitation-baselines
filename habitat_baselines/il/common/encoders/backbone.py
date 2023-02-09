'''
Adapted from 
https://github.com/allenai/embodied-clip/blob/0ee02a194a75fb2d447f88e231ab6d8dbea79b8d/primitive_probing/generate_data/reachable_image_features.py
'''
import os
from glob import glob
from tqdm import tqdm
import argparse

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as T

import clip
from gym import spaces
from habitat_baselines.il.common.encoders.resnet_encoders import VlnResnetDepthEncoder
from habitat_baselines.utils.common import Flatten
from habitat import logger

 
def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if "BatchNorm" in type(module).__name__:
            module.momentum = 0.0
    model.eval()
    return model

def unnormalize(y, mean, std):
    mean = torch.tensor(mean)[None,None,:] #[1,1,1,3]
    std = torch.tensor(std)[None,None,:]
    x = std*y+mean 
    return x

class VisualPretrainedEncoder(nn.Module):
    def __init__(self, image_type, model_cfg, checkpoint=None):
        super().__init__()
        self.image_type = image_type
        assert self.image_type in ['rgb','depth']
        if self.image_type=='rgb':
            self.model_type = model_cfg.RGB_ENCODER.TYPE
            if self.model_type=='resnet50':
                resnet_model = torchvision.models.resnet50(pretrained=True)
                resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-2])
                self.visual_encoder = resnet_model
                self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1,1)),nn.Flatten())
                self.preprocess = T.Compose([
                    T.Resize(size=224, interpolation=Image.BICUBIC),
                    T.CenterCrop(size=(224, 224)),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                self.unnormalize = lambda y: unnormalize(y, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                #self.output_dim = np.prod() #TODO fixed
            elif 'CLIP' in self.model_type:
                clip_type = self.model_type.replace('CLIP_','')
                if clip_type=='ViT-B32':
                    clip_type='ViT-B/32'
                clip_model, self.preprocess = clip.load(clip_type, device=torch.device('cpu'))
                self.unnormalize = lambda y: unnormalize(y, 
                    mean=[0.48145466, 0.4578275, 0.40821073], 
                    std=[0.26862954, 0.26130258, 0.27577711])
                if self.model_type=='CLIP_RN50':
                    self.output_dim = 2048
                    self.visual_encoder = clip_model.visual
                    self.visual_encoder.attnpool = nn.Identity()
                    self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1,1)),nn.Flatten())
                elif self.model_type=='CLIP_ViT-B32':
                    self.output_dim = 512
                    self.visual_encoder = clip_model.visual
                    self.pool = nn.Identity()
            else:
                raise ValueError

        elif self.image_type=='depth':
            self.model_type = model_cfg.DEPTH_ENCODER.TYPE
            assert self.model_type=='resnet50', self.model_type
            logger.info('Load checkpoint from {}'.format(model_cfg.DEPTH_ENCODER.ddppo_checkpoint))
            self.visual_encoder = VlnResnetDepthEncoder(
                observation_space=spaces.Dict({"depth": spaces.Space(shape=[480,640,1])}), 
                checkpoint=model_cfg.DEPTH_ENCODER.ddppo_checkpoint,
                output_size=128, backbone='resnet50',
                trainable=False
            )
            self.visual_encoder.visual_fc = nn.Identity()
            self.pool = Flatten() #follow Ram81's code
            self.preprocess = None#TODO 
            self.output_dim = np.prod(self.visual_encoder.visual_encoder.output_shape)
        else:
            raise ValueError

        self.visual_encoder = freeze_model(self.visual_encoder)
        self.pool = freeze_model(self.pool)
        logger.info(f'Build model: {self.model_type} for {self.image_type} images')
    
    def forward(self, x):
        #x After preprocess (applied to the PIL.Image) and batching!
        if self.image_type=='depth':
            x = self.visual_encoder({'depth': x})
        else:
            x = self.visual_encoder(x)
        x = self.pool(x)
        return x
from src.model import *
import torch
import cv2
import numpy as np
import os
from config.configs import Config_Training
import logging
from .tpsV2 import *
config = Config_Training()
logger = logging.Logger()

def transform_im( im):
    im = im.transpose(2, 0, 1)
    im = torch.from_numpy(im).float()

    return im

def resize_im(im):
    im = cv2.resize(im, (992, 992), interpolation=cv2.INTER_LINEAR)
    # im = cv2.resize(im, (496, 496), interpolation=cv2.INTER_LINEAR)
    return im

if __name__ == "__main__":
    model = FiducialPoints(n_classes=config.n_classes, num_filter=32, architecture=DilatedResnetForFlatByFiducialPointsS2, BatchNorm='BN', in_channels=3)     #
    checkpoint = torch.load(config.pretrained)
    model.load_state_dict(checkpoint['model_state'])
    model.to(config.device)
    model.eval()
    logger.info("Load model successfully")
    createThinPlateSplineShapeTransformer(config.map_shape, fiducial_num=config.fiducial_num, device=config.device)
    image = cv2.imread("test.png",flags=cv2.IMREAD_COLOR)
    input_model = transform_im(resize_im(image))
    input_model = input_model.to(config.device).float().unsqueeze(0)
    
    
    
    
    
    
    
    
    
    
    
    

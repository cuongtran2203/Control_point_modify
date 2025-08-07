from src.model import *
import torch
import cv2
import numpy as np
import os
from config.configs import Config_Training
import logging
import time
from src.utils import *
from src.tpsV2 import *
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
    model.load_state_dict(checkpoint)
    model.to(config.device)
    model.eval()
    logger.info("Load model successfully")
    if config.tps:
        model_tps =createThinPlateSplineShapeTransformer(config.map_shape, fiducial_num=config.fiducial_num, device=config.device)
    #Load image
    image_path = "test.png"
    
    image = cv2.imread(image_path,flags=cv2.IMREAD_COLOR)
    input_model = transform_im(resize_im(image))
    input_model = input_model.to(config.device).float().unsqueeze(0)
    outputs, outputs_segment = model(input_model)
    # outputs, outputs_segment = self.input_model(images, is_softmax=True)

    pred_regress = outputs.data.cpu().numpy().transpose(0, 2, 3, 1)
    pred_segment = outputs_segment.data.round().int().cpu().numpy()
    perturbed_img_mark, flat_img = flatByfiducial_TPS(model_tps,config,pred_regress[0], pred_segment[0],image)
    
    cv2.imwrite("perturbed_img_mark.png",perturbed_img_mark)
    cv2.imwrite("flat_img.png",flat_img)
    logger.info("Done Task")
    

    
        
    
    
    
    
    
    
    
    
    
    
    

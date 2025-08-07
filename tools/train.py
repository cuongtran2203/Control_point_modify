import sys
sys.path.append("..")
from src.datasets import *
from src.loss import *
from src.model import *

import torch
import cv2
import os
import numpy as np
from config.configs import Config_Training
import time
from loguru import logger
from tqdm import tqdm
# logger = Logger('runs')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



if __name__ == "__main__":
    config = Config_Training()
    #Load model 
    model = FiducialPoints(n_classes=config.n_classes, num_filter=32, architecture=DilatedResnetForFlatByFiducialPointsS2, BatchNorm='BN', in_channels=3)     #
    logger.info("Load model successfully")
    
    if config.pretrained is not None:
        checkpoint = torch.load(config.pretrained)
        for key in list(checkpoint['model_state'].keys()):
            if 'module.' in key:
                checkpoint['model_state'][key.replace('module.', '')] = checkpoint['model_state'].pop(key)
        model.load_state_dict(checkpoint['model_state'])
        logger.info("Load checkpoints successfully")
    
    model.to(config.device)
    #Config loss and optimizer
    
    loss_fun_classes = Losses(classify_size_average=True, args_gpu=config.device)
    loss_fun = loss_fun_classes.loss_fn4_v5_r_4   # *
    # loss_fun = loss_fun_classes.loss_fn4_v5_r_3   # *
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=1e-10)
    loss_fun2 = loss_fun_classes.loss_fn_l1_loss
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 90, 150, 200], gamma=0.5)
    best_weights = 9999
    # Load datasets
    train_datasets = Train_CornerDetection_Datasets(root=config.DIR_TRAIN)
    val_datasets = Train_CornerDetection_Datasets(root=config.DIR_VAL)
    test_datasets = Testval_ConerDetection_Datasets(root=config.DIR_TEST)
    
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    valloader = torch.utils.data.DataLoader(val_datasets, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    losses = AverageMeter()
    train_time = AverageMeter()

    trainloader_len = len(trainloader)
    ### TRAINING LOOP ###
    logger.info("------------------Start training--------------")
    for epoch in range(config.NUM_EPOCHS):
        print('* lambda_loss :'+str(config.lambda_loss)+'\t'+'learning_rate :'+str(optimizer.param_groups[0]['lr']))
        print('* lambda_loss :'+str(config.lambda_loss)+'\t'+'learning_rate :'+str(optimizer.param_groups[0]['lr']))
        begin_train = time.time()
        loss_segment_list = 0
        loss_l1_list = 0
        loss_local_list = 0
        loss_edge_list = 0
        loss_rectangles_list = 0
        loss_list = []
        
        model.train()
        for i, (images, labels, segment) in tqdm(enumerate(trainloader)):
            images = images.to(config.device).float()
            labels = labels.to(config.device)
            segment = segment.to(config.device)
            optimizer.zero_grad()
            outputs, outputs_segment = model(images, is_softmax=False)
            
            loss_l1, loss_local, loss_edge, loss_rectangles = loss_fun(outputs, labels, size_average=True)
            loss_segment = loss_fun2(outputs_segment, segment)
            loss = config.lambda_loss*(loss_l1 + loss_local*config.lambda_loss_a + loss_edge*config.lambda_loss_b + loss_rectangles*config.lambda_loss_c) + config.lambda_loss_segment*loss_segment

            losses.update(loss.item())
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            loss_segment_list += loss_segment.item()
            loss_l1_list += loss_l1.item()
            loss_local_list += loss_local.item()
            
        list_len = len(loss_list)
        # Format training results in a more readable way
        logger.info(
            f'Epoch [{epoch + 1}][{i + 1}/{trainloader_len}] '
            f'Loss range: [{min(loss_list):.2f}, {sum(loss_list) / list_len:.4f}, {max(loss_list):.2f}] '
            f'Components: [L1: {loss_l1_list / list_len:.4f}, '
            f'Local: {loss_local_list / list_len:.4f}, '
            f'Edge: {loss_edge_list / list_len:.4f}, '
            f'Rect: {loss_rectangles_list / list_len:.4f}, '
            f'Segment: {loss_segment_list / list_len:.4f}] '
            f'Average: {losses.avg:.4f}'
        )
     
                
        del loss_list[:]
        loss_segment_list = 0
        loss_l1_list = 0
        loss_local_list = 0
        loss_edge_list = 0
        loss_rectangles_list = 0
        
        ### Eval model ####
        
        model.eval()
        trian_t = time.time()-begin_train
        losses.reset()
        logger.info("-----Starting Evaluate-------")
        train_time.update(trian_t)
        with torch.no_grad():
            for i, (images, labels, segment) in tqdm(enumerate(valloader)):
                images = images.to(config.device).float().to(config.device)
                labels = labels.to(config.device)
                segment = segment.to(config.device)
                optimizer.zero_grad()
                outputs, outputs_segment = model(images, is_softmax=False)
                
                loss_l1, loss_local, loss_edge, loss_rectangles = loss_fun(outputs, labels, size_average=True)
                loss_segment = loss_fun2(outputs_segment, segment)
                loss = config.lambda_loss*(loss_l1 + loss_local*config.lambda_loss_a + loss_edge*config.lambda_loss_b + loss_rectangles*config.lambda_loss_c) + config.lambda_loss_segment*loss_segment
                loss_list.append(loss.item())
                loss_segment_list += loss_segment.item()
                loss_l1_list += loss_l1.item()
                loss_local_list += loss_local.item()
                losses.update(loss.item())
                

        # Format validation results in a more readable way
        list_len = len(loss_list)
        logger.info(
            f'Epoch [{epoch + 1}][{i + 1}/{len(valloader)}] '
            f'Loss range: [{min(loss_list):.2f}, {sum(loss_list) / list_len:.4f}, {max(loss_list):.2f}] '
            f'Components: [L1: {loss_l1_list / list_len:.4f}, '
            f'Local: {loss_local_list / list_len:.4f}, '
            f'Edge: {loss_edge_list / list_len:.4f}, '
            f'Rect: {loss_rectangles_list / list_len:.4f}, '
            f'Segment: {loss_segment_list / list_len:.4f}] '
            f'Average: {losses.avg:.4f}'
        )

        
        scheduler.step()
        
        if losses.avg < best_weights:
            best_weights = losses.avg
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints",exist_ok=True)
            torch.save(model.state_dict(),"checkpoints/bestmodel.pth")
        
    m, s = divmod(train_time.sum, 60)
    h, m = divmod(m, 60)
    print("All Train Time : %02d:%02d:%02d\n" % (h, m, s))
    print("All Train Time : %02d:%02d:%02d\n" % (h, m, s))

        
        
        

                
            
                
            
            
            
        

        
        
        
        
        

    
    
    
    

    
    

    
    
    
 
    
    
    
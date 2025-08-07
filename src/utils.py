import torch
from torch.utils import data
from torch.autograd import Variable, Function
import numpy as np
import sys, os, math
import cv2
import time
import re
import random
from scipy.interpolate import griddata
from config.configs import Config_Training

config = Config_Training()


def adjust_position(x_min, y_min, x_max, y_max, new_shape):
    if (new_shape[0] - (x_max - x_min)) % 2 == 0:
        f_g_0_0 = (new_shape[0] - (x_max - x_min)) // 2
        f_g_0_1 = f_g_0_0
    else:
        f_g_0_0 = (new_shape[0] - (x_max - x_min)) // 2
        f_g_0_1 = f_g_0_0 + 1

    if (new_shape[1] - (y_max - y_min)) % 2 == 0:
        f_g_1_0 = (new_shape[1] - (y_max - y_min)) // 2
        f_g_1_1 = f_g_1_0
    else:
        f_g_1_0 = (new_shape[1] - (y_max - y_min)) // 2
        f_g_1_1 = f_g_1_0 + 1

    # return f_g_0_0, f_g_0_1, f_g_1_0, f_g_1_1
    return f_g_0_0, f_g_1_0, new_shape[0] - f_g_0_1, new_shape[1] - f_g_1_1

def get_matric_edge(matric):
    return np.concatenate((matric[:, 0, :], matric[:, -1, :], matric[0, 1:-1, :], matric[-1, 1:-1, :]), axis=0)

def location_mark(img, location, color=(0, 0, 255)):
    stepSize = 0
    for l in location.astype(np.int64).reshape(-1, 2):
        cv2.circle(img,
                    (l[0] + math.ceil(stepSize / 2), l[1] + math.ceil(stepSize / 2)), 3, color, -1)
    return img

def flatByfiducial_TPS(tps,config:Config_Training,fiducial_points, segment, perturbed_img=None, is_scaling=False):
    '''
    flat_shap controls the output image resolution
    '''
    perturbed_img = cv2.resize(perturbed_img, (960, 1024))
    
    fiducial_points = fiducial_points / [992, 992]
    perturbed_img_shape = perturbed_img.shape[:2]

    sshape = fiducial_points[::config.fiducial_point_gaps[config.row_gap], ::config.fiducial_point_gaps[config.col_gap], :]
    flat_shap = segment * [config.fiducial_point_gaps[config.col_gap], config.fiducial_point_gaps[config.row_gap]] * [config.fiducial_point_num[config.col_gap], config.fiducial_point_num[config.row_gap]]
    # flat_shap = perturbed_img_shape
    time_1 = time.time()
    perturbed_img_ = torch.tensor(perturbed_img.transpose(2,0,1)[None,:])

    fiducial_points_ = (torch.tensor(fiducial_points.transpose(1, 0,2).reshape(-1, 2))[None,:]-0.5)*2
    rectified = tps(perturbed_img_.double().to(config.device), fiducial_points_.to(config.device), list(flat_shap))
    time_2 = time.time()
    time_interval = time_2 - time_1
    print('TPS time: '+ str(time_interval))

    flat_img = rectified[0].cpu().numpy().transpose(1,2,0)

    '''save'''
    flat_img = flat_img.astype(np.uint8)
    perturbed_img_mark = location_mark(perturbed_img.copy(), sshape*perturbed_img_shape[::-1], (0, 0, 255))
    return perturbed_img_mark, flat_img


def flatByfiducial_interpolation(config:Config_Training,fiducial_points, segment, perturbed_img=None, is_scaling=False):
    perturbed_img = cv2.resize(perturbed_img, (960, 1024))
    fiducial_points = fiducial_points / [992, 992] * [960, 1024]
    col_gap = 2 #4
    row_gap = col_gap# col_gap + 1 if col_gap < 6 else col_gap
    # fiducial_point_gaps = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]  # POINTS NUM: 61, 31, 21, 16, 13, 11, 7, 6, 5, 4, 3, 2
    fiducial_point_gaps = [1, 2, 3, 5, 6, 10, 15, 30]        # POINTS NUM: 31, 16, 11, 7, 6, 4, 3, 2
    sshape = fiducial_points[::fiducial_point_gaps[row_gap], ::fiducial_point_gaps[col_gap], :]
    segment_h, segment_w = segment * [fiducial_point_gaps[col_gap], fiducial_point_gaps[row_gap]]
    fiducial_points_row, fiducial_points_col = sshape.shape[:2]

    im_x, im_y = np.mgrid[0:(fiducial_points_col - 1):complex(fiducial_points_col),
                    0:(fiducial_points_row - 1):complex(fiducial_points_row)]

    tshape = np.stack((im_x, im_y), axis=2) * [segment_w, segment_h]

    tshape = tshape.reshape(-1, 2)
    sshape = sshape.reshape(-1, 2)

    output_shape = (segment_h * (fiducial_points_col - 1), segment_w * (fiducial_points_row - 1))
    grid_x, grid_y = np.mgrid[0:output_shape[0] - 1:complex(output_shape[0]),
                        0:output_shape[1] - 1:complex(output_shape[1])]
    time_1 = time.time()
    # grid_z = griddata(tshape, sshape, (grid_y, grid_x), method='cubic').astype('float32')
    grid_ = griddata(tshape, sshape, (grid_y, grid_x), method='linear').astype('float32')
    flat_img = cv2.remap(perturbed_img, grid_[:, :, 0], grid_[:, :, 1], cv2.INTER_CUBIC)
    time_2 = time.time()
    time_interval = time_2 - time_1
    print('Interpolation time: '+ str(time_interval))
    flat_img = flat_img.astype(np.uint8)
    perturbed_img_mark = location_mark(perturbed_img.copy(), sshape, (0, 0, 255))

    shrink_paddig = 0   # 2 * edge_padding
    x_start, x_end, y_start, y_end = shrink_paddig, segment_h * (fiducial_points_col - 1) - shrink_paddig, shrink_paddig, segment_w * (fiducial_points_row - 1) - shrink_paddig

    x_ = (perturbed_img_mark.shape[0]-(x_end-x_start))//2
    y_ = (perturbed_img_mark.shape[1]-(y_end-y_start))//2

    flat_img_new = np.zeros_like(perturbed_img_mark)
    flat_img_new[x_:perturbed_img_mark.shape[0] - x_, y_:perturbed_img_mark.shape[1] - y_] = flat_img
    img_figure = np.concatenate(
        (perturbed_img_mark, flat_img_new), axis=1)
    
    return img_figure

    
    



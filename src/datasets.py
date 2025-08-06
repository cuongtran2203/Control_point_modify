import os
import pickle
import collections
import json
import torch
import numpy as np

import re
import cv2

from torch.utils import data
def get_data_path(name):
	"""Extract path to data from config file.

	Args:
		name (str): The name of the dataset.

	Returns:
		(str): The path to the root directory containing the dataset.
	"""
	with open('../xgw/segmentation/config.json') as f:
		js = f.read()
	# js = open('config.json').read()
	data = json.loads(js)
	return os.path.expanduser(data[name]['data_path'])
def getDatasets(dir):
	return os.listdir(dir)


def resize_image(origin_img, long_edge=1024, short_edge=960):
	# long_edge, short_edge = 2048, 1920
	# long_edge, short_edge = 1024, 960
	# long_edge, short_edge = 512, 480

	im_lr = origin_img.shape[0]
	im_ud = origin_img.shape[1]
	new_img = np.zeros([long_edge, short_edge, 3], dtype=np.uint8)
	new_shape = new_img.shape[:2]
	if im_lr > im_ud:
		img_shrink, base_img_shrink = long_edge, long_edge
		im_ud = int(im_ud / im_lr * base_img_shrink)
		im_ud += 32-im_ud%32
		im_ud = min(im_ud, short_edge)
		im_lr = img_shrink
		origin_img = cv2.resize(origin_img, (im_ud, im_lr), interpolation=cv2.INTER_CUBIC)
		new_img[:, (new_shape[1]-im_ud)//2:new_shape[1]-(new_shape[1]-im_ud)//2] = origin_img
		# mask = np.full(new_shape, 255, dtype='uint8')
		# mask[:, (new_shape[1] - im_ud) // 2:new_shape[1] - (new_shape[1] - im_ud) // 2] = 0
	else:
		img_shrink, base_img_shrink = short_edge, short_edge
		im_lr = int(im_lr / im_ud * base_img_shrink)
		im_lr += 32-im_lr%32
		im_lr = min(im_lr, long_edge)
		im_ud = img_shrink
		origin_img = cv2.resize(origin_img, (im_ud, im_lr), interpolation=cv2.INTER_CUBIC)
		new_img[(new_shape[0] - im_lr) // 2:new_shape[0] - (new_shape[0] - im_lr) // 2, :] = origin_img
	return new_img

class Testval_ConerDetection_Datasets(data.Dataset):
    def __init__(self,root=None,img_shrink=None,is_return_img_name=False,preproccess=False):
        self.root = root
        self.img_shrink = img_shrink
        self.is_return_img_name = is_return_img_name
        self.preproccess = preproccess
        # self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.images = collections.defaultdict(list)
        self.labels = collections.defaultdict(list)
        self.row_gap = 1  # value:0, 1, 2;  POINTS NUM: 61, 31, 21
        self.col_gap = 1

        self.img_file_list = getDatasets(os.path.join(self.root))

    def __len__(self):
        return len(self.img_file_list)
    
    def __getitem__(self, index):
        
       im_name = self.img_file_list[index]
       img_path = os.path.join(self.root, im_name)
       img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
       img = self.resize_im(img)
       img = self.transform_im(img)
       if self.is_return_img_name:
           return img, im_name
       return img
   
    def transform_im(self, im):
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).float()

        return im

    def resize_im(self, im):
        im = cv2.resize(im, (992, 992), interpolation=cv2.INTER_LINEAR)
        # im = cv2.resize(im, (496, 496), interpolation=cv2.INTER_LINEAR)
        return im
   
class Train_CornerDetection_Datasets(data.Dataset):
    def __init__(self,root=None,img_shrink=None,is_return_img_name=False,preproccess=False):
        self.root = root
        self.img_shrink = img_shrink
        self.is_return_img_name = is_return_img_name
        self.preproccess = preproccess
        # self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.images = collections.defaultdict(list)
        self.labels = collections.defaultdict(list)
        self.row_gap = 1  # value:0, 1, 2;  POINTS NUM: 61, 31, 21
        self.col_gap = 1

        self.img_file_list = getDatasets(os.path.join(self.root))

    def __len__(self):
        return len(self.img_file_list)
    
    
    def __getitem__(self, index):
        
        '''
        Load file .gw contain:
        image: image numpy array with 512x480
        fiducial_points : The Control point with size is 31x31
        segment: The mask of object

        '''
        gw_name = self.img_file_list[index]
        gw_path = os.path.join(self.root, gw_name)
        with open(gw_path, 'rb') as f:
            perturbed_data = pickle.load(f)
        im = perturbed_data.get('image')
        lbl = perturbed_data.get('fiducial_points')
        segment = perturbed_data.get('segment')

        im = self.resize_im(im)
        im = im.transpose(2, 0, 1)

        lbl = self.resize_lbl(lbl)
        lbl, segment = self.fiducal_points_lbl(lbl, segment)
        lbl = lbl.transpose(2, 0, 1)

        im = torch.from_numpy(im)
        lbl = torch.from_numpy(lbl).float()
        segment = torch.from_numpy(segment).float()

        if self.is_return_img_name:
            return im, lbl, segment, gw_name

        return im, lbl, segment
    def transform_im(self, im):
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).float()

        return im

    def resize_im(self, im):
        im = cv2.resize(im, (992, 992), interpolation=cv2.INTER_LINEAR)
        # im = cv2.resize(im, (496, 496), interpolation=cv2.INTER_LINEAR)
        return im

    def resize_lbl(self, lbl):
        lbl = lbl/[960, 1024]*[992, 992]
        # lbl = lbl/[960, 1024]*[496, 496]
        return lbl

    def fiducal_points_lbl(self, fiducial_points, segment):

        fiducial_point_gaps = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]  # POINTS NUM: 61, 31, 21, 16, 13, 11, 7, 6, 5, 4, 3, 2
        fiducial_points = fiducial_points[::fiducial_point_gaps[self.row_gap], ::fiducial_point_gaps[self.col_gap], :]
        segment = segment * [fiducial_point_gaps[self.col_gap], fiducial_point_gaps[self.row_gap]]
        return fiducial_points, segment


          


    





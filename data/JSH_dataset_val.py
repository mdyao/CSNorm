import random
import numpy as np
import cv2
import h5py
import torch
import torch.utils.data as data
import data.util as util
import glob
import os

class JSHDataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR), GT and noisy image pairs.
    If only GT and noisy images are provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(JSHDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.gtimglist = sorted(glob.glob(os.path.join(self.opt['dataroot_gt'], '*')))
        self.inputimglist = sorted(glob.glob(os.path.join(self.opt['dataroot_lq'], '*')))
        self.length = len(self.gtimglist)

    def __getitem__(self, index):
        self.input = cv2.imread(self.inputimglist[index])
        self.gt = cv2.imread(self.gtimglist[index])
        GT_size = self.opt['GT_size']

        # get GT image
        input_img = self.input/255.0
        gt_img = self.gt/255.0
        input_img = input_img.transpose(2,0,1)
        gt_img = gt_img.transpose(2,0,1)

        if self.opt['phase'] == 'train':
            C, H, W = input_img.shape
            x = random.randint(0, W - GT_size)
            y = random.randint(0, H - GT_size)
            # input_img = input_img[:, y:y + GT_size, x:x + GT_size]

            # augmentation - flip, rotate
            input_img, gt_img = util.augment([input_img, gt_img], self.opt['use_flip'],
                                          self.opt['use_rot'])
        # BGR to RGB, HWC to CHW, numpy to tensor
        input_img = torch.from_numpy(np.ascontiguousarray(input_img)).float()
        gt_img = torch.from_numpy(np.ascontiguousarray(gt_img)).float()

        return {'gt_img': gt_img,  'lq_img': input_img}

    def __len__(self):
        return self.length



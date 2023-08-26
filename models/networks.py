import torch
import logging
from models.modules.NAFNet.NAFNet import NAFNet

import math
logger = logging.getLogger('base')


####################
# define network
####################
def define_G(opt):
    img_channel = 3
    width = 32
    enc_blks= [2, 2, 4, 8]
    middle_blk_num= 6
    dec_blks= [2, 2, 2, 2]

    netG = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                   enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    return netG

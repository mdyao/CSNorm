import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import FFT_Loss
import numpy as np
import time
from models.modules.loss_new import SSIMLoss
import re

logger = logging.getLogger('base')


class CSNorm_Model(BaseModel):
    def __init__(self, opt):
        super(CSNorm_Model, self).__init__(opt)


        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt

        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)

        ######################### set parameters in CSNorm ###############################
        target_layer_patterns = re.compile(r'module\.(gate\.proj|CSN_\d+)\.')
        # target_layer_patterns = re.compile(r'(gate\.proj|CSN_\d+)\.')

        self.layer_aug = [
            name for name, param in self.netG.named_parameters()
            if target_layer_patterns.search(name)
        ]
        print('parameters in CSNorm:',self.layer_aug)
        ######################### set parameters in CSNorm ###############################

        # loss
        self.Back_rec = torch.nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.fft_loss = FFT_Loss()
        # self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            optim_params_aug = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            for k, v in self.netG.named_parameters():
                if k in self.layer_aug:
                    optim_params_aug.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize in aug.'.format(k))


            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))

            self.optimizer_G_aug = torch.optim.Adam(optim_params_aug, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_G_aug)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def amp_aug(self, x, y):
        x = x + 1e-8
        y = y + 1e-8
        x_freq= torch.fft.rfft2(x, norm='backward')
        x_amp = torch.abs(x_freq)
        x_phase = torch.angle(x_freq)

        y_freq= torch.fft.rfft2(y, norm='backward')
        y_amp = torch.abs(y_freq)
        y_phase = torch.angle(y_freq)

        mix_alpha = torch.rand(1).to(self.device)/0.5
        mix_alpha = torch.clip(mix_alpha, 0,0.5)
        y_amp = mix_alpha * y_amp + (1-mix_alpha) * x_amp

        real = y_amp * torch.cos(y_phase)
        imag = y_amp * torch.sin(y_phase)
        y_out = torch.complex(real, imag) + 1e-8
        y_out = torch.fft.irfft2(y_out) + 1e-8

        return y_out

    def feed_data(self, data):
        self.img_gt = data['gt_img'].to(self.device)  # GT
        self.img_input = data['lq_img'].to(self.device)  # Noisy
        self.img_input_aug = self.amp_aug(self.img_gt, self.img_input)  # Noisy

    def feed_data_test(self, data):
        # self.ref_L = data['LQ'].to(self.device)  # LQ
        self.img_gt = data['gt_img'].to(self.device)  # GT
        self.img_input = data['lq_img'].to(self.device)  # Noisy

    def loss_forward(self,img, gt):
        loss = 1 * self.Back_rec(img, gt)
        loss_ssim = self.ssim_loss(img, gt)

        return loss, loss_ssim


    def loss_forward_aug(self,img, gt):
        loss = 1 * self.Back_rec(img, gt)
        loss_ssim = self.ssim_loss(img, gt)

        l_amp, _ = self.fft_loss(img, gt)
        return loss, loss_ssim, l_amp

    def optimize_parameters(self, step):

        ############## optimizate parameters outside CSNorm ############################
        for k, v in self.netG.named_parameters():
            if k not in self.layer_aug:
                v.requires_grad = True
            else:
                v.requires_grad = False
        self.optimizer_G.zero_grad()

        # forward
        self.img_pred = self.netG(self.img_input, aug=True)
        loss, l_ssim = self.loss_forward(self.img_pred, self.img_gt)
        loss = loss + l_ssim

        # backward
        loss.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer_G.step()


        ############## optimizate parameters inside CSNorm ############################
        for k, v in self.netG.named_parameters():
            if k in self.layer_aug:
                v.requires_grad = True
            else:
                v.requires_grad = False

        self.optimizer_G_aug.zero_grad()

        # forward
        self.img_pred = self.netG(self.img_input, aug=True)
        loss_back, l_ssim, l_amp = self.loss_forward_aug(self.img_pred, self.img_gt)
        loss_aug = loss_back + l_ssim + l_amp
        # backward
        loss_aug.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer_G_aug.step()


        # set log
        self.log_dict['loss'] = loss.item()
        self.log_dict['l_amp'] = l_amp.item()
        self.log_dict['l_ssim'] = l_ssim.item()

    def test(self):

        self.netG.eval()
        with torch.no_grad():
            self.img_pred = self.netG(self.img_input, aug=True)

        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['img_pred'] = self.img_pred.detach()[0].float().cpu()
        out_dict['img_input'] = self.img_input.detach()[0].float().cpu()
        out_dict['img_gt'] = self.img_gt.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):

        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

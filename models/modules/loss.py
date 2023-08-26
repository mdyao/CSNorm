import torch
import torch.nn as nn
import numpy as np
from torchvision.models.vgg import vgg16
from torch.nn import functional as F
import torch.fft as fft

class ReconstructionLoss(nn.Module):
    def __init__(self, losstype='l2', eps=1e-3):
        super(ReconstructionLoss, self).__init__()
        self.losstype = losstype
        self.eps = eps

    def forward(self, x, target):
        if self.losstype == 'l2':
            return torch.mean(torch.sum((x - target)**2, (1, 2, 3)))
        elif self.losstype == 'l1':
            diff = x - target
            return torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3)))
        elif self.losstype == 'l_log':
            diff = x - target
            eps = 1e-6
            return torch.mean(torch.sum(-torch.log(1-diff.abs()+eps), (1, 2, 3)))
        else:
            print("reconstruction loss type error!")
            return 0


class FFT_Loss(nn.Module):
    def __init__(self, losstype='l2', eps=1e-3):
        super(FFT_Loss, self).__init__()
        # self.fpre =
    def forward(self, x, gt):
        x = x + 1e-8
        gt = gt + 1e-8
        x_freq= torch.fft.rfft2(x, norm='backward')
        x_amp = torch.abs(x_freq)
        x_phase = torch.angle(x_freq)

        gt_freq= torch.fft.rfft2(gt, norm='backward')
        gt_amp = torch.abs(gt_freq)
        gt_phase = torch.angle(gt_freq)

        loss_amp = torch.mean(torch.sum((x_amp - gt_amp) ** 2))
        loss_phase = torch.mean(torch.sum((x_phase - gt_phase) ** 2))
        return loss_amp, loss_phase

# Gradient Loss
class Gradient_Loss(nn.Module):
    def __init__(self, losstype='l2'):
        super(Gradient_Loss, self).__init__()
        a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
        a = torch.from_numpy(a).float().unsqueeze(0)
        a = torch.stack((a, a, a))
        conv1.weight = nn.Parameter(a, requires_grad=False)
        self.conv1 = conv1.cuda()

        b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
        b = torch.from_numpy(b).float().unsqueeze(0)
        b = torch.stack((b, b, b))
        conv2.weight = nn.Parameter(b, requires_grad=False)
        self.conv2 = conv2.cuda()

        # self.Loss_criterion = ReconstructionLoss(losstype)
        self.Loss_criterion = nn.L1Loss()

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        # x_total = torch.sqrt(torch.pow(x1, 2) + torch.pow(x2, 2))

        y1 = self.conv1(y)
        y2 = self.conv2(y)
        # y_total = torch.sqrt(torch.pow(y1, 2) + torch.pow(y2, 2))

        l_h = self.Loss_criterion(x1, y1)
        l_v = self.Loss_criterion(x2, y2)
        # l_total = self.Loss_criterion(x_total, y_total)
        return l_h + l_v #+ l_total


class SSIM_Loss(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM_Loss, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class TV_extractor(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TV_extractor, self).__init__()
        self.TVLoss_weight = TVLoss_weight
        self.fil = nn.Parameter(torch.ones(1, 1, 3, 3)/9, requires_grad=False)

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.abs((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]))
        w_tv = torch.abs((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]))
        h_tv = F.pad(h_tv, [0,0,0,1], "constant", 0)
        w_tv = F.pad(w_tv, [0,1,0,0], "constant", 0)

        h_tv = F.conv2d(h_tv, self.fil, stride=1, padding=1, groups=1)
        w_tv = F.conv2d(w_tv, self.fil, stride=1, padding=1, groups=1)

        # print(h_tv.shape, w_tv.shape)
        tv = torch.abs(h_tv)+torch.abs(w_tv)
        return tv

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class CL_Loss(nn.Module):
    def __init__(self, opt):
        super(CL_Loss, self).__init__()
        self.opt = opt
        self.d = nn.MSELoss(size_average=True)
        vgg = vgg16(pretrained=False).cuda()
        vgg.load_state_dict(torch.load(self.opt['vgg16_model']))
        self.loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in self.loss_network.parameters():
            param.requires_grad = False

    def forward(self, anchor, postive, negative):
        anchor_f = self.loss_network(anchor)
        positive_f = self.loss_network(postive)
        negative_f = self.loss_network(negative)

        loss = self.d(anchor_f, positive_f)/self.d(anchor_f, negative_f)
        return loss

class Percep_Loss(nn.Module):
    def __init__(self, opt):
        super(Percep_Loss, self).__init__()
        self.opt = opt
        self.d = nn.MSELoss(size_average=True)
        vgg = vgg16(pretrained=True).cuda()
        # vgg.load_state_dict(torch.load(self.opt['vgg16_model']))
        # self.loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        # for param in self.loss_network.parameters():
        #     param.requires_grad = False

        blocks = []
        blocks.append(vgg.features[:4].eval())
        blocks.append(vgg.features[4:9].eval())
        blocks.append(vgg.features[9:16].eval())
        blocks.append(vgg.features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target,feature_layers=[0, 1, 2, 3],weights=[1,1,1,1]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        # input = (input-self.mean) / self.std
        # target = (target-self.mean) / self.std
        loss = 0.0
        x = input
        y = target
        for i,block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += weights[i] * self.d(x, y)
        return loss

class SID_loss(nn.Module):
    def __init__(self):
        super(SID_loss).__init__()

        criterion = nn.KLDivLoss()

    def forward(self,x,y):
        p = torch.zeros_like(x).cuda()
        q = torch.zeros_like(x).cuda()
        Sid = 0
        # for i in range(len(x)):
        #     p[i] = x[i] / torch.sum(x)
        #     q[i] = y[i] / torch.sum(y)
            # print(p[i],q[i])
        for j in range(len(x)):
            Sid += p[j] * np.log10(p[j] / q[j]) + q[j] * np.log10(q[j] / p[j])
        return Sid

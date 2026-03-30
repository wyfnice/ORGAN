import random
import cv2
import lpips
from PIL import Image
from tqdm import tqdm
#import pycurvelab
import numpy as np
from torch import autograd
from torch import nn
from networks.blocks import *
from networks.loss import *
from utils import batched_index_select, batched_scatter
from torchvision import transforms
import os
import scipy.misc
from utils import  unloader
import math
import torch
def build_dct_kernel(k_size):
    """
    生成任意尺寸 k_size x k_size 的 DCT 变换核
    返回: (k_size*k_size, 1, k_size, k_size) 的 Tensor
    """
    filters = np.zeros((k_size * k_size, k_size, k_size))
    for u in range(k_size):
        for v in range(k_size):
            for i in range(k_size):
                for j in range(k_size):
                    alpha_u = math.sqrt(1/k_size) if u == 0 else math.sqrt(2/k_size)
                    alpha_v = math.sqrt(1/k_size) if v == 0 else math.sqrt(2/k_size)
                    basis = alpha_u * alpha_v * \
                            math.cos((2*i+1)*u*math.pi/(2*k_size)) * \
                            math.cos((2*j+1)*v*math.pi/(2*k_size))
                    filters[u * k_size + v, i, j] = basis
    return torch.from_numpy(filters).float().unsqueeze(1)

def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    # harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    # harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
   #  print(filter_LL.size())
   #  filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
   #  filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
    LL = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    # LH = net(in_channels, in_channels*2,
    #          kernel_size=2, stride=2, padding=0, bias=False,
    #          groups=in_channels)
    # HL = net(in_channels, in_channels*2,
    #          kernel_size=2, stride=2, padding=0, bias=False,
    #          groups=in_channels)
    HH = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    # LH.weight.requires_grad = False
    # HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    # LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    # HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)

    # return LL, LH, HL, HH
    return LL,HH

def get_dct_coefficients(in_channels, pool=True):
    """Generate LL, HL, LH and HH frequency components using DCT"""
    # Define DCT matrix for LL component
    w_LL = np.ones((2, 2))
    filter_LL = torch.from_numpy(w_LL).float().unsqueeze(0).unsqueeze(0)

    # Define DCT matrix for HL component
    w_HL = np.ones((2, 2))
    w_HL[1, :] = -1
    filter_HL = torch.from_numpy(w_HL).float().unsqueeze(0).unsqueeze(0)

    # Define DCT matrix for LH component
    w_LH = np.ones((2, 2))
    w_LH[:, 1] = -1
    filter_LH = torch.from_numpy(w_LH).float().unsqueeze(0).unsqueeze(0)

    # Define DCT matrix for HH component
    w_HH = np.ones((2, 2))
    w_HH[0, :] = -1
    w_HH[:, 0] = -1
    filter_HH = torch.from_numpy(w_HH).float().unsqueeze(0).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels*2, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    # HL = net(in_channels, in_channels*2, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    # LH = net(in_channels, in_channels*2, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    #HH = net(in_channels, in_channels*2, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)

    LL.weight.requires_grad = False
    # HL.weight.requires_grad = False
    # LH.weight.requires_grad = False
    #HH.weight.requires_grad = False

    LL.weight.data = filter_LL.expand(in_channels*2, -1, -1, -1)
    # # HL.weight.data = filter_HL.expand(in_channels*2, -1, -1, -1)
    # # LH.weight.data = filter_LH.expand(in_channels*2, -1, -1, -1)
    #HH.weight.data = filter_HH.expand(in_channels*2, -1, -1, -1)

    # return LL, HL, LH, HH
    return LL


class MultiScaleWavePool(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(MultiScaleWavePool, self).__init__()

        if out_channels is None:
            out_channels = in_channels  # 默认输出通道与输入一致，或者是原有逻辑的倍数

        # --- 分支 1: 2x2 DCT (原有逻辑，关注微观纹理) ---
        # 2x2 DCT 有 4 个分量 (LL, LH, HL, HH)
        self.k2 = 2
        dct_weights_2 = build_dct_kernel(self.k2)  # Shape: [4, 1, 2, 2]
        self.dct_conv_2 = nn.Conv2d(in_channels, in_channels * (self.k2 ** 2),
                                    kernel_size=2, stride=2, groups=in_channels, bias=False)
        self.dct_conv_2.weight.requires_grad = False
        # 将 DCT 核扩展到所有通道
        self.dct_conv_2.weight.data = dct_weights_2.repeat(in_channels, 1, 1, 1)

        # --- 分支 2: 4x4 DCT (新增逻辑，关注宏观结构) ---
        # 4x4 DCT 有 16 个分量
        self.k4 = 4
        dct_weights_4 = build_dct_kernel(self.k4)  # Shape: [16, 1, 4, 4]
        # 注意：为了保持输出尺寸一致(H/2, W/2)，这里 Stride=2, Padding=1
        self.dct_conv_4 = nn.Conv2d(in_channels, in_channels * (self.k4 ** 2),
                                    kernel_size=4, stride=2, padding=1, groups=in_channels, bias=False)
        self.dct_conv_4.weight.requires_grad = False
        self.dct_conv_4.weight.data = dct_weights_4.repeat(in_channels, 1, 1, 1)

        # --- 融合层 (Feature Fusion) ---
        # 输入通道数 = (in * 4) + (in * 16) = in * 20
        total_freq_channels = in_channels * (self.k2 ** 2 + self.k4 ** 2)

        # 使用 1x1 卷积学习如何融合不同尺度的频率信息
        self.fusion = nn.Sequential(
            nn.Conv2d(total_freq_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        # 1. 计算 2x2 频域特征 (Batch, In*4, H/2, W/2)
        feat_2x2 = self.dct_conv_2(x)

        # 2. 计算 4x4 频域特征 (Batch, In*16, H/2, W/2)
        feat_4x4 = self.dct_conv_4(x)

        # 3. 拼接
        feat_concat = torch.cat([feat_2x2, feat_4x4], dim=1)

        # 4. 融合并降维
        out = self.fusion(feat_concat)

        return out

class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        # self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)
        #self.LL,self.HH = get_dct_coefficients(in_channels)
        self.LL = get_dct_coefficients(in_channels)
    def forward(self, x):
        # return self.LL(x), self.LH(x), self.HL(x), self.HH(x)
        #return self.LL(x),self.HH(x)
        return self.LL(x)



def get_wav_two(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L

    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
   #  print(filter_LL.size())

    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False

    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, HH

def get_dct_coefficients_two(in_channels, pool=True):
    """Generate LL, HL, LH and HH frequency components using DCT"""
    # Define DCT matrix for LL component
    w_LL = np.ones((2, 2))
    filter_LL = torch.from_numpy(w_LL).float().unsqueeze(0).unsqueeze(0)

    # Define DCT matrix for HL component
    w_HL = np.ones((2, 2))
    w_HL[1, :] = -1
    filter_HL = torch.from_numpy(w_HL).float().unsqueeze(0).unsqueeze(0)

    # Define DCT matrix for LH component
    w_LH = np.ones((2, 2))
    w_LH[:, 1] = -1
    filter_LH = torch.from_numpy(w_LH).float().unsqueeze(0).unsqueeze(0)

    # Define DCT matrix for HH component
    w_HH = np.ones((2, 2))
    w_HH[0, :] = -1
    w_HH[:, 0] = -1
    filter_HH = torch.from_numpy(w_HH).float().unsqueeze(0).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    # HL = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    # LH = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    #HH = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)

    LL.weight.requires_grad = False
    # HL.weight.requires_grad = False
    # LH.weight.requires_grad = False
    #HH.weight.requires_grad = False

    LL.weight.data = filter_LL.expand(in_channels, -1, -1, -1)
    # # HL.weight.data = filter_HL.expand(in_channels, -1, -1, -1)
    # # LH.weight.data = filter_LH.expand(in_channels, -1, -1, -1)
    #HH.weight.data = filter_HH.expand(in_channels, -1, -1, -1)

    return LL
class WavePool2(nn.Module):
    def __init__(self, in_channels):
        super(WavePool2, self).__init__()
        #self.LL,self.HH = get_wav_two(in_channels)
        #self.LL, self.HH = get_dct_coefficients_two(in_channels)
        self.LL = get_dct_coefficients_two(in_channels)
    def forward(self, x):
        #return self.LL(x),self.HH(x)
        return self.LL(x)


class WaveUnpool(nn.Module):
    def __init__(self, in_channels, option_unpool='cat5'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        #self.LL,self.HH = get_wav_two(self.in_channels, pool=False)
        #self.LL, self.HH = get_dct_coefficients_two(self.in_channels, pool=False)
        self.LL = get_dct_coefficients_two(self.in_channels, pool=False)
    def forward(self, LL, HH, original=None):
        if self.option_unpool == 'sum':
            #return self.LL(LL) + self.HH(HH)
            return self.LL(LL)
        elif self.option_unpool == 'cat5' and original is not None:
            #return torch.cat([self.LL(LL), self.HH(HH), original], dim=1)
            return torch.cat([self.LL(LL), original], dim=1)
        elif self.option_unpool =='sumall':
            #return LL + HH
            return LL
        else:
            raise NotImplementedError

class LoFGAN(nn.Module):
    def __init__(self, config):
        super(LoFGAN, self).__init__()

        self.gen = Generator(config['gen'])
        self.dis = Discriminator(config['dis'])

        self.w_adv_g = config['w_adv_g']
        self.w_adv_d = config['w_adv_d']
        self.w_recon = config['w_recon']
        self.w_cls = config['w_cls']
        self.w_gp = config['w_gp']
        self.n_sample = config['n_sample_train']


    def forward(self, xs, y, mode):
        if mode == 'gen_update':
            fake_x, similarity, indices_feat, indices_ref, base_index = self.gen(xs)

            loss_recon = local_recon_criterion(xs, fake_x, similarity, indices_feat, indices_ref, base_index, s=8)

            feat_real, _, _ = self.dis(xs)
            feat_fake, logit_adv_fake, logit_c_fake = self.dis(fake_x)
            loss_adv_gen = torch.mean(-logit_adv_fake)
            loss_cls_gen = F.cross_entropy(logit_c_fake, y.squeeze())

            loss_recon = loss_recon * self.w_recon
            loss_adv_gen = loss_adv_gen * self.w_adv_g
            loss_cls_gen = loss_cls_gen * self.w_cls

            loss_total = loss_recon + loss_adv_gen + loss_cls_gen
            loss_total.backward()

            return {'loss_total': loss_total,
                    'loss_recon': loss_recon,
                    'loss_adv_gen': loss_adv_gen,
                    'loss_cls_gen': loss_cls_gen}

        elif mode == 'dis_update':
            xs.requires_grad_()

            _, logit_adv_real, logit_c_real = self.dis(xs)
            loss_adv_dis_real = torch.nn.ReLU()(1.0 - logit_adv_real).mean()
            loss_adv_dis_real = loss_adv_dis_real * self.w_adv_d
            loss_adv_dis_real.backward(retain_graph=True)

            y_extend = y.repeat(1, self.n_sample).view(-1).long()
            index = torch.LongTensor(range(y_extend.size(0))).cuda()
            # logit_c_real_forgp = logit_c_real[index, y_extend].unsqueeze(1)
            # loss_reg_dis = self.calc_grad2(logit_c_real_forgp, xs)
            #
            # loss_reg_dis = loss_reg_dis * self.w_gp
            # loss_reg_dis.backward(retain_graph=True)

            loss_cls_dis = F.cross_entropy(logit_c_real, y_extend)
            loss_cls_dis = loss_cls_dis * self.w_cls
            loss_cls_dis.backward()

            with torch.no_grad():
                fake_x = self.gen(xs)[0]

            _, logit_adv_fake, _ = self.dis(fake_x.detach())
            loss_adv_dis_fake = torch.nn.ReLU()(1.0 + logit_adv_fake).mean()
            loss_adv_dis_fake = loss_adv_dis_fake * self.w_adv_d
            loss_adv_dis_fake.backward()

            loss_total = loss_adv_dis_real + loss_adv_dis_fake + loss_cls_dis
            return {'loss_total': loss_total,
                    'loss_adv_dis': loss_adv_dis_fake + loss_adv_dis_real,
                    'loss_adv_dis_real': loss_adv_dis_real,
                    'loss_adv_dis_fake': loss_adv_dis_fake,
                    'loss_cls_dis': loss_cls_dis,
                    'loss_reg': loss_cls_dis}

        else:
            assert 0, 'Not support operation'

    def generate(self, xs):
        fake_x = self.gen(xs)[0]
        return fake_x

    def calc_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(outputs=d_out.mean(),
                                  inputs=x_in,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.sum()
        reg /= batch_size
        return reg


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.soft_label = False
        nf = config['nf']
        n_class = config['num_classes']
        n_res_blks = config['n_res_blks']

        cnn_f = [Conv2dBlock(3, nf, 5, 1, 2,
                             pad_type='reflect',
                             norm='sn',
                             activation='none')]
        for i in range(n_res_blks):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf_out, fhid=None, activation='lrelu', norm='sn')]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = np.min([nf * 2, 1024])

        nf_out = np.min([nf * 2, 1024])
        cnn_f += [ActFirstResBlock(nf, nf_out, fhid=None, activation='lrelu', norm='sn')]
        cnn_adv = [nn.AdaptiveAvgPool2d(1),
                   Conv2dBlock(nf_out, 1, 1, 1,
                               norm='none',
                               activation='none',
                               activation_first=False)]
        cnn_c = [nn.AdaptiveAvgPool2d(1),
                 Conv2dBlock(nf_out, n_class, 1, 1,
                             norm='none',
                             activation='none',
                             activation_first=False)]
        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_adv = nn.Sequential(*cnn_adv)
        self.cnn_c = nn.Sequential(*cnn_c)

    def forward(self, x):
        if len(x.size()) == 5:
            B, K, C, H, W = x.size()
            x = x.view(B * K, C, H, W)
        else:
            B, C, H, W = x.size()
            K = 1
        feat = self.cnn_f(x)
        logit_adv = self.cnn_adv(feat).view(B * K, -1)
        logit_c = self.cnn_c(feat).view(B * K, -1)
        return feat, logit_adv, logit_c


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fusion = LocalFusionModule(inplanes=128, rate=config['rate'])

    def forward(self, xs):
        b, k, C, H, W = xs.size()
        xs = xs.view(-1, C, H, W)
        # print("encoder")
        querys, skips = self.encoder(xs)
        c, h, w = querys.size()[-3:]
        querys = querys.view(b, k, c, h, w)

        similarity_total = torch.cat([torch.rand(b, 1) for _ in range(k)], dim=1).cuda()  # b*k
        similarity_sum = torch.sum(similarity_total, dim=1, keepdim=True).expand(b, k)  # b*k
        similarity = similarity_total / similarity_sum  # b*k

        base_index = random.choice(range(k))

        base_feat = querys[:, base_index, :, :, :]

        feat_gen, indices_feat, indices_ref = self.fusion(base_feat, querys, base_index, similarity)
       # feat_gen = querys.mean(dim=1)
        fake_x = self.decoder(feat_gen, skips, base_index)

        return fake_x, similarity, indices_feat, indices_ref, base_index


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # 在 Encoder 的 __init__ 中：

        self.conv1 = Conv2dBlock(3, 32, 5, 1, 2, norm='bn', activation='lrelu', pad_type='reflect')
        # 改进：输入32，内部多尺度提取后，投影到64维，以便在 forward 中与 conv2 的结果相加
        self.pool1 = MultiScaleWavePool(32, out_channels=64).cuda()

        self.conv2 = Conv2dBlock(32, 64, 3, 2, 1, norm='bn', activation='lrelu', pad_type='reflect')
        # 改进：输入64 -> 投影到128
        self.pool2 = MultiScaleWavePool(64, out_channels=128).cuda()

        self.conv3 = Conv2dBlock(64, 128, 3, 2, 1, norm='bn', activation='lrelu', pad_type='reflect')
        # 改进：输入128 -> 保持128
        self.pool3 = MultiScaleWavePool(128, out_channels=128).cuda()

        self.conv4 = Conv2dBlock(128, 128, 3, 2, 1, norm='bn', activation='lrelu', pad_type='reflect')
        # 改进：输入128 -> 保持128
        self.pool4 = MultiScaleWavePool(128, out_channels=128).cuda()

        self.conv5 = Conv2dBlock(128, 128, 3, 2, 1, norm='bn', activation='lrelu', pad_type='reflect')

    def forward(self, x):
        skips = {}
        x = self.conv1(x)  # (B, 32, 128, 128)
        skips['conv1_1'] = x

        LL1 = self.pool1(x)  # (B, 32, 64, 64) -> 融合了2x2和4x4频率特征

        x = self.conv2(x)  # (B, 64, 64, 64) - 注意：原代码这里 conv2 是 stride 2，可能会再次降采样？
        # 原代码逻辑：pool1 只是用来计算 LL1 给下一层做加法的，还是直接传给 conv2？
        # 仔细看原代码：x = self.conv3(x + LL1)。
        # 这意味着 LL1 的尺寸必须和 x (经过 conv2 后的 x) 一致。
        # 原代码 conv2 stride=2，输入128 -> 输出64。
        # 我们的 pool1 stride=2，输入128 -> 输出64。尺寸匹配！

        # ... (后续 forward 逻辑保持不变，因为我们控制了 pool 的输出尺寸和通道) ...

        skips['conv2_1'] = x
        LL2 = self.pool2(x)

        # 这里的残差连接 (x + LL1) 蕴含了频域增强的思想
        # 注意：原代码中 conv3 输入是 (x + LL1)。
        # x 是 conv2 输出 (64ch)，LL1 是 pool1 输出。
        # 如果 conv2 输出 64ch，pool1 也必须输出 64ch 才能相加。
        # 请根据这一行调整上面的 out_channels。
        # 如果 conv2(32->64)，那么 x 是 64ch。
        # 如果 pool1(32->out)，那么 LL1 是 out ch。
        # 所以 pool1 的 out_channels 应该设为 64，而不是 32！

        # --- 修正后的参数初始化 ---
        # 请回到上面的 __init__ 修改 out_channels 参数：
        # self.pool1 = MultiScaleWavePool(32, out_channels=64).cuda()
        # self.pool2 = MultiScaleWavePool(64, out_channels=128).cuda()
        # self.pool3 = MultiScaleWavePool(128, out_channels=128).cuda() # conv4 out is 128
        # self.pool4 = MultiScaleWavePool(128, out_channels=128).cuda()

        x = self.conv3(x + LL1)
        skips['conv3_1'] = x
        LL3 = self.pool3(x)

        x = self.conv4(x + LL2)
        skips['conv4_1'] = x
        LL4 = self.pool4(x)

        x = self.conv5(x + LL3)

        return x, skips

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.Upsample = nn.Upsample(scale_factor=2)
        self.Conv1 = Conv2dBlock(128, 128, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.recon_block1 = WaveUnpool(128,"sum").cuda()
        self.Conv2 = Conv2dBlock(128, 128, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.recon_block2 = WaveUnpool(128, "sum").cuda()
        self.Conv3 = Conv2dBlock(128, 64, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.recon_block3 = WaveUnpool(64, "sumall").cuda()
        self.Conv4 = Conv2dBlock(64, 32, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.recon_block4 = WaveUnpool(32, "sum").cuda()
        self.Conv5 = Conv2dBlock(32, 3, 5, 1, 2,
                             norm='none',
                             activation='tanh',
                             pad_type='reflect')



    def forward(self, x, skips,base_index):
        x1 = self.Upsample(x)
        x2 = self.Conv1(x1)
        #HH1 = skips['pool4']
        #c, h, w = HH1.size()[-3:]
        # Mean_index
        # LH1, HL1, HH1 = LH1.view(8,3,c, h, w).mean(dim=1), HL1.view(8,3,c, h, w).mean(dim=1), HH1.view(8,3,c, h, w).mean(dim=1)
        # Base_index
        # HH1 = HH1.view(8, 3,c, h, w)
        # HH1 = HH1[:,base_index,:,:,:]
        original1 = skips['conv4_1']
        # x_deconv = self.recon_block1(x, HH1, original1)
        x_deconv = self.recon_block1(x, original1)
        x2 = x_deconv + x2

        x3 = self.Upsample(x2)
        x4 = self.Conv2(x3)
        #HH2 = skips['pool3']
        original2 = skips['conv3_1']
        #c, h, w = HH2.size()[-3:]
        #Mean_index
        #LH2, HL2, HH2 = LH2.view(8, 3, c, h, w).mean(dim=1), HL2.view(8, 3, c, h, w).mean(dim=1), HH2.view(8, 3, c, h,w).mean(dim=1)
        # Base_index
        # HH2 = HH2.view(8, 3, c, h, w)
        # HH2 = HH2[:, base_index, :, :, :]
        #x_deconv2 = self.recon_block1(x1, HH2, original2)
        x_deconv2 = self.recon_block1(x1,original2)
        #HH3 = skips['pool2']
        c, h, w = skips['conv2_1'].size()[-3:]
        original3 = skips['conv2_1']
        #c, h, w = HH3.size()[-3:]
        #Mean_index
        #LH3, HL3, HH3 = LH3.view(8, 3, c, h, w).mean(dim=1), HL3.view(8, 3, c, h, w).mean(dim=1), HH3.view(8, 3, c, h,w).mean(dim=1)
        # Base_index
        # HH3 = HH3.view(8, 3, c, h, w)
        # HH3 = HH3[:, base_index, :, :, :]
        #x_deconv3 = self.recon_block1(x3, HH3, original2)
        x_deconv3 = self.recon_block1(x3,original2)
        x5 = self.Upsample(x4+x_deconv2)
        #x5 = self.Upsample(x4)
        x6 = self.Conv3(x5+x_deconv3)
        #x6 = self.Conv3(x5)
        #HH4 = skips['pool1']
        original4 = skips['conv1_1']
        #c, h, w = HH4.size()[-3:]
        # Mean_index
        #LH4, HL4, HH4 = LH4.view(8, 3, c, h, w).mean(dim=1), HL4.view(8, 3, c, h, w).mean(dim=1), HH4.view(8, 3, c, h,w).mean(dim=1)
        # Base_index
        # HH4 = HH4.view(8, 3, c, h, w)
        # HH4 = HH4[:, base_index, :, :, :]
        #x_deconv4 = self.recon_block3(x6,HH4, original3)
        x_deconv4 = self.recon_block3(x6,original3)
        x7 = self.Upsample(x6+x_deconv4)
        #x7 = self.Upsample(x6)
        x8 = self.Conv4(x7)

        x9 = self.Conv5(x8)

        return x9

class LocalFusionModule(nn.Module):
    def __init__(self, inplanes, rate):
        super(LocalFusionModule, self).__init__()

        self.W = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(inplanes)
        )
        self.rate = rate

    def forward(self, feat, refs, index, similarity):
        refs = torch.cat([refs[:, :index, :, :, :], refs[:, (index + 1):, :, :, :]], dim=1)
        base_similarity = similarity[:, index]
        ref_similarities = torch.cat([similarity[:, :index], similarity[:, (index + 1):]], dim=1)

        # take ref:(32, 2, 128, 8, 8) for example
        b, n, c, h, w = refs.size()
        refs = refs.view(b * n, c, h, w)

        w_feat = feat.view(b, c, -1)
        w_feat = w_feat.permute(0, 2, 1).contiguous()
        w_feat = F.normalize(w_feat, dim=2)  # (32*64*128)

        w_refs = refs.view(b, n, c, -1)
        w_refs = w_refs.permute(0, 2, 1, 3).contiguous().view(b, c, -1)
        w_refs = F.normalize(w_refs, dim=1)  # (32*128*128)

        # local selection
        rate = self.rate
        num = int(rate * h * w)
        feat_indices = torch.cat([torch.LongTensor(random.sample(range(h * w), num)).unsqueeze(0) for _ in range(b)],
                                 dim=0).cuda()  # B*num

        feat = feat.view(b, c, -1)  # (32*128*64)
        feat_select = batched_index_select(feat, dim=2, index=feat_indices)  # (32*128*12)

        # local matching
        w_feat_select = batched_index_select(w_feat, dim=1, index=feat_indices)  # (32*12*128)
        w_feat_select = F.normalize(w_feat_select, dim=2)  # (32*12*128)

        refs = refs.view(b, n, c, h * w)
        ref_indices = []
        ref_selects = []
        for j in range(n):
            ref = refs[:, j, :, :]  # (32*128*64)
            w_ref = w_refs.view(b, c, n, h * w)[:, :, j, :]  # (32*128*64)
            fx = torch.matmul(w_feat_select, w_ref)  # (32*12*64)
            _, indice = torch.topk(fx, dim=2, k=1)
            indice = indice.squeeze(0).squeeze(-1)  # (32*10)
            select = batched_index_select(ref, dim=2, index=indice)  # (32*128*12)
            ref_indices.append(indice)
            ref_selects.append(select)
        ref_indices = torch.cat([item.unsqueeze(1) for item in ref_indices], dim=1)  # (32*2*12)
        ref_selects = torch.cat([item.unsqueeze(1) for item in ref_selects], dim=1)  # (32*2*128*12)

        # local replacement
        base_similarity = base_similarity.view(b, 1, 1)  # (32*1*1)
        ref_similarities = ref_similarities.view(b, 1, n)  # (32*1*2)
        feat_select = feat_select.view(b, 1, -1)  # (32*1*(128*12))
        ref_selects = ref_selects.view(b, n, -1)  # (32*2*(128*12))

        feat_fused = torch.matmul(base_similarity, feat_select) \
                     + torch.matmul(ref_similarities, ref_selects)  # (32*1*(128*12))
        feat_fused = feat_fused.view(b, c, num)  # (32*128*12)

        feat = batched_scatter(feat, dim=2, index=feat_indices, src=feat_fused)
        feat = feat.view(b, c, h, w)  # (32*128*8*8)

        return feat, feat_indices, ref_indices  # (32*128*8*8), (32*12), (32*2*12)



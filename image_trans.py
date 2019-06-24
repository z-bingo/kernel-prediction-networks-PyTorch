import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
import math

# sRGB transform
# 单纯的mask操作会造成梯度nan
# 原因分析：即使通过分段来避免梯度nan的情况，但是，仍然是所有X都参与了所有的计算
def gamma_inv(X):
    b = 0.0031308
    gamma = 1. / 2.4
    a = 1./(1./(b**gamma*(1.-gamma))-1.)
    k0 = (1 + a) * gamma * b ** (gamma - 1.)
    srgb = torch.FloatTensor(X.size()).type_as(X)
    mask = X < b
    srgb[mask] = k0 * X[mask]
    mask = 1 - mask
    srgb[mask] = (1+a)*torch.pow(X[mask], gamma)-a
    mask = X > 1
    srgb[mask] = (1+a)*gamma*X[mask] - (1+a)*gamma + 1
    return srgb

# 计算图像的一阶导数
# 采用两个方向梯度绝对值相加的方式
def gradient_image_L1(img):
    l = F.pad(img, [1, 0, 0, 0])
    r = F.pad(img, [0, 1, 0, 0])
    u = F.pad(img, [0, 0, 1, 0])
    d = F.pad(img, [0, 0, 0, 1])
    if len(img.size()) == 3:
        return torch.abs((l-r)[:, 0:img.size(1), 0:img.size(2)]) + torch.abs((u-d)[:, 0:img.size(1), 0:img.size(2)])
    elif len(img.size()) == 4:
        return torch.abs((l - r)[:, :, 0:img.size(1), 0:img.size(2)]) + torch.abs((u - d)[:, :, 0:img.size(1), 0:img.size(2)])

# L2形式的图像梯度计算
def gradient_image_L2(img):
    l = F.pad(img, [1, 0, 0, 0])
    r = F.pad(img, [0, 1, 0, 0])
    u = F.pad(img, [0, 0, 1, 0])
    d = F.pad(img, [0, 0, 0, 1])
    return (torch.sqrt(torch.pow(l-r, 2) + torch.pow(u-d, 2)))

# 基本的loss函数
def loss_basic(Y, Y_truth):
    loss_mse = nn.MSELoss()
    loss_l1 = nn.L1Loss()
    return loss_mse(Y, Y_truth) + loss_l1(gradient_image_L1(Y), gradient_image_L1(Y_truth))

# 含有anneal的loss函数
# 论文中最终的loss函数  用于backward
def loss_annealed(Y_truth, burst, burst_size, core_filts, white_level, final_K, global_step, beta, alpha):
    # batch * burst * color * height * width
    Yi = convolve(burst, burst_size, core_filts, final_K)
    Yi = Yi / white_level
    # print(torch.max(Yi), torch.min(Yi))
    Y = Yi.mean(dim=1, keepdim=False)
    loss1 = loss_basic(gamma_inv(Y), gamma_inv(Y_truth))
    loss2 = []
    for i in range(burst_size):
        loss2.append(loss_basic(gamma_inv(Yi[:, i, ...]), gamma_inv(Y_truth)))
    loss2 = beta * (alpha ** global_step) * torch.stack(loss2).mean()
    return loss1+loss2, loss1, loss2

# 对最后的输出结果进行卷积计算
# burst: batch * burst * (in_channel) * H * W
# core_filt: batch * (K*K*burst) * H * W
# return: batch * burst * final_Channel * H * W
def convolve(burst, burst_size, core_filts, final_K):
    """

    :param burst:
    :param burst_size:
    :param core_filts:
    :param final_K: must be a list or a tuple
    :return:
    """
    K_total = np.sum(np.array(final_K) ** 2)

    # 得到尺寸
    batch_size, _, channel, height, width = burst.size()

    final_channel = core_filts.size(1) // burst_size

    if K_total != final_channel:
        core_dic = sep_conv_core(core_filts, final_K, batch_size, burst_size, height, width)
    else:
        core_dic = {}
        core_dic[final_K[0]] = core_filts.view(batch_size, burst_size, final_K[0]**2, height, width)

    pred_img = torch.zeros(len(final_K), batch_size, burst_size, channel, height, width, device=burst.device)
    for index, K in enumerate(final_K):
        padding = K // 2
        burst_pad = F.pad(burst, [padding, padding, padding, padding])  # 形成final_K*final_K个burst
        core = core_dic[K].view(batch_size, burst_size, K*K, 1, height, width).expand(
            batch_size, burst_size, K*K, channel, height, width
        )
        img_stack = []
        for i in range(K):
            for j in range(K):
                img_stack.append(burst_pad[:, :, :, i:i+height, j:j+width])
        # 组成新的stack  用于计算卷积
        img_stack = torch.stack(img_stack, dim=2)
        img_stack = img_stack.view(batch_size, burst_size, K*K, channel, height, width)

        pred_img[index, ...] = torch.sum(img_stack.mul(core), dim=2)

    return torch.mean(pred_img, dim=0, keepdim=False)

# 把2p形式的卷积核转换为p^2
def sep_conv_core(core_filts, final_K, batch_size, burst_size, height, width):
    K_total = sum(sorted(final_K))
    core_filts = core_filts.view(batch_size, burst_size, 2*K_total, height, width)
    core1 = core_filts[:, :, 0:K_total, ...]
    core2 = core_filts[:, :, K_total:, ...]
    core = {}
    cur = 0
    for K in final_K:
        t1 = core1[:, :, cur:cur+K, ...].view(batch_size, burst_size, K, 1, height, width)
        t2 = core2[:, :, cur:cur+K, ...].view(batch_size, burst_size, 1, K, height, width)
        core[K] = torch.einsum('ijklno,ijlmno->ijkmno', [t1, t2]).view(batch_size, burst_size, K*K, height, width)
        cur += K
    return core

if __name__ == '__main__':
    core_filt = torch.randn(8, 200, 128, 128)
    burst = torch.randn(8, 8, 3, 128, 128)
    t = convolve(burst, 8, core_filt, [5])
    print('over')
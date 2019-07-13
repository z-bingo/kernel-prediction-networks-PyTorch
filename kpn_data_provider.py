import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
from skimage.color import rgb2xyz
import inspect
from utils.training_util import read_config
from data_generation.data_utils import *
import torch.nn.functional as F

def sRGBGamma(tensor):
    threshold = 0.0031308
    a = 0.055
    mult = 12.92
    gamma = 2.4
    res = torch.zeros_like(tensor)
    mask = tensor > threshold
    # image_lo = tensor * mult
    # 0.001 is to avoid funny thing at 0.
    # image_hi = (1 + a) * torch.pow(tensor + 0.001, 1.0 / gamma) - a
    res[mask] = (1 + a) * torch.pow(tensor[mask] + 0.001, 1.0 / gamma) - a

    res[1-mask] = tensor[1-mask] * mult
    # return mask * image_hi + (1 - mask) * image_lo
    return res


def UndosRGBGamma(tensor):
    threshold = 0.0031308
    a = 0.055
    mult = 12.92
    gamma = 2.4
    res = torch.zeros_like(tensor)
    mask = tensor > threshold
    # image_lo = tensor / mult
    # image_hi = torch.pow(tensor + a, gamma) / (1 + a)
    res[1-mask] = tensor[1-mask] / mult
    res[mask] = torch.pow(tensor[mask] + a, gamma) / (1 + a)
    # return mask * image_hi + (1 - mask) * image_lo
    return res

class Random_Horizontal_Flip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, tensor):
        if np.random.rand() < self.p:
            return torch.flip(tensor, dims=[-1])
        return tensor

class Random_Vertical_Flip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, tensor):
        if np.random.rand() < self.p:
            return torch.flip(tensor, dims=[-2])
        return tensor

class TrainDataSet(torch.utils.data.Dataset):
    def __init__(self, config_file,
                 config_spec=None, img_format='.bmp', degamma=True, color=True, blind=False, train=True):
        super(TrainDataSet, self).__init__()
        if config_spec is None:
            config_spec = self._configspec_path()
        config = read_config(config_file, config_spec)
        self.dataset_config = config['dataset_configs']
        self.dataset_dir = self.dataset_config['dataset_dir']
        self.images = list(filter(lambda x: True if img_format in x else False, os.listdir(self.dataset_dir)))
        self.burst_size = self.dataset_config['burst_length']
        self.patch_size = self.dataset_config['patch_size']

        self.upscale = self.dataset_config['down_sample']
        self.big_jitter = self.dataset_config['big_jitter']
        self.small_jitter = self.dataset_config['small_jitter']
        # 对应下采样之前图像的最大偏移量
        self.jitter_upscale = self.big_jitter * self.upscale
        # 对应下采样之前的图像的patch尺寸
        self.size_upscale = self.patch_size * self.upscale + 2 * self.jitter_upscale
        # 产生大jitter和小jitter之间的delta  在下采样之前的尺度上
        self.delta_upscale = (self.big_jitter - self.small_jitter) * self.upscale
        # 对应到原图的patch的尺寸
        self.patch_size_upscale = self.patch_size * self.upscale
        # 去伽马效应
        self.degamma = degamma
        # 是否用彩色图像进行处理
        self.color = color
        # 是否盲估计  盲估计即估计的噪声方差不会作为网络的输入
        self.blind = blind
        self.train = train

        self.vertical_flip = Random_Vertical_Flip(p=0.5)
        self.horizontal_flip = Random_Horizontal_Flip(p=0.5)

    @staticmethod
    def _configspec_path():
        current_dir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        )
        return os.path.join(current_dir,
                            'dataset_specs/data_configspec.conf')

    @staticmethod
    def crop_random(tensor, patch_size):
        return random_crop(tensor, 1, patch_size)[0]

    # get一个item  根据index检索
    def __getitem__(self, index):
        # print(index)
        image = Image.open(os.path.join(self.dataset_dir, self.images[index])).convert('RGB')

        # 先转换为Tensor进行degamma
        image = transforms.ToTensor()(image)
        # if self.degamma:
        #     image = UndosRGBGamma(tensor=image)
        image_crop = self.crop_random(image, self.size_upscale)
        # 3*H*W  对应于较小jitter下
        image_crop_small = image_crop[:, self.delta_upscale:-self.delta_upscale,
                                        self.delta_upscale:-self.delta_upscale]

        # 进一步进行random_crop所需的transform

        # burst中的第一个不做偏移  后期作为target
        # output shape: N*3*H*W
        img_burst = []
        for i in range(self.burst_size):
            if i == 0:
                img_burst.append(
                    image_crop[:, self.jitter_upscale:-self.jitter_upscale, self.jitter_upscale:-self.jitter_upscale]
                )
            else:
                if np.random.binomial(1, min(1.0, np.random.poisson(lam=1.5) / self.burst_size)) == 0:
                    img_burst.append(
                        self.crop_random(
                            image_crop_small, self.patch_size_upscale
                        )
                    )
                else:  #big
                    img_burst.append(
                        self.crop_random(image_crop, self.patch_size_upscale)
                    )
        image_burst = torch.stack(img_burst, dim=0)
        image_burst = F.adaptive_avg_pool2d(image_burst, (self.patch_size, self.patch_size))

        # label为patch中burst的第一个
        if not self.color:
            image_burst = 0.2989*image_burst[:, 0, ...] + 0.5870 * image_burst[:, 1, ...] + 0.1140*image_burst[:, 2, ...]
            image_burst = torch.clamp(image_burst, 0.0, 1.0)

        if self.degamma:
            UndosRGBGamma(image_burst)

        if self.train:
            # data augment
            image_burst = self.horizontal_flip(image_burst)
            image_burst = self.vertical_flip(image_burst)

        gt = image_burst[0, ...]


        # 以上得到的patch size为burst*（3）*size*size
        """
        数据加噪声等一系列处理  全部基于rgb图像做
        """
        # 要产生[log10(0.1), log10(1.0)]之间的均匀分布随机数  也就是[0,1加负号即可]
        # 产生pred之后  再除以white_level恢复原来的亮度
        # batch中的每一个burst  产生一个white_level
        white_level = torch.from_numpy(np.power(10, -np.random.rand(1, 1, 1))).type_as(image_burst)
        # 论文中对图像亮度赋值进行线性缩放[0.1, 1]
        image_burst = white_level * image_burst

        # gray image
        if not self.color:
            # 生成随机的read和shot噪声方差
            sigma_read = torch.from_numpy(
                np.power(10, np.random.uniform(-3.0, -1.5, (1, 1, 1)))).type_as(image_burst)
            sigma_shot = torch.from_numpy(
                np.power(10, np.random.uniform(-4.0, -2.0, (1, 1, 1)))).type_as(image_burst)

            # sigma_read = torch.from_numpy(2*np.power(10, np.array([[[-2.0]]]))).type_as(image_burst)
            # sigma_shot = torch.from_numpy(6.4 * np.power(10, np.array([[[-3.0]]]))).type_as(image_burst)

            # 产生噪声  依据论文中公式产生
            sigma_read_com = sigma_read.expand_as(image_burst)
            sigma_shot_com = sigma_shot.expand_as(image_burst)

            # generate noise
            burst_noise = torch.normal(image_burst, torch.sqrt(sigma_read_com**2 + image_burst * sigma_shot_com)).type_as(image_burst)

            # burst_noise 恢复到[0,1] 截去外面的值
            burst_noise = torch.clamp(burst_noise, 0.0, 1.0)

            # 非盲估计  就要估计噪声的方差
            if not self.blind:
                # 接下来就是根据两个sigma  将估计的噪声标准差也作为输入  用burst中的第一个进行估计
                # estimation shape: H*W
                sigma_read_est = sigma_read.view(1, 1).expand_as(gt)
                sigma_shot_est = sigma_shot.view(1, 1).expand_as(gt)
                sigma_estimate = torch.sqrt(sigma_read_est ** 2 + sigma_shot_est.mul(
                    torch.max(torch.stack([burst_noise[0, ...], torch.zeros_like(burst_noise[0, ...])], dim=0), dim=0)[0]))

                # 把噪声的估计和burst图像连接在一起
                burst_noise = torch.cat([burst_noise, sigma_estimate.unsqueeze(0)], dim=0)

            # 按照文章中的 ref Image作为target进行了训练  输出结果和ref很相似  没能起到太大的去噪作用
            # return patches_with_noise, patches_with_noise[:, 0, ...], white_level
            # 不含噪声的ref作为target进行测试

            return burst_noise, gt, white_level
        # color image
        else:
            # 生成随机的read和shot噪声方差
            sigma_read = torch.from_numpy(
                np.power(10, np.random.uniform(-3.0, -1.5, (1, 1, 1, 1)))).type_as(image_burst)
            sigma_shot = torch.from_numpy(
                np.power(10, np.random.uniform(-4.0, -2.0, (1, 1, 1, 1)))).type_as(image_burst)

            # 产生噪声  依据论文中公式产生
            sigma_read_com = sigma_read.expand_as(image_burst)
            sigma_shot_com = sigma_shot.expand_as(image_burst)

            # generate noise
            burst_noise = torch.normal(image_burst,
                                       torch.sqrt(sigma_read_com ** 2 + image_burst * sigma_shot_com)).type_as(image_burst)

            # burst_noise 恢复到[0,1] 截去外面的值
            burst_noise = torch.clamp(burst_noise, 0.0, 1.0)

            # 非盲估计  就要估计噪声的方差
            if not self.blind:
                # 接下来就是根据两个sigma  将估计的噪声标准差也作为输入  用burst中的第一个进行估计
                # estimation shape: H*W
                sigma_read_est = sigma_read.view(1, 1, 1).expand_as(gt)
                sigma_shot_est = sigma_shot.view(1, 1, 1).expand_as(gt)
                sigma_estimate = torch.sqrt(sigma_read_est ** 2 + sigma_shot_est.mul(
                    torch.max(torch.stack([burst_noise[0, ...], torch.zeros_like(burst_noise[0, ...])], dim=0), dim=0)[0]))

                # 把噪声的估计和burst图像连接在一起
                burst_noise = torch.cat([burst_noise, sigma_estimate.unsqueeze(0)], dim=0)

            white_level = white_level.unsqueeze(0)
            return burst_noise, gt, white_level

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    # path = 'F:/BinZhang/Codes/deep-burst-denoising/data/train'
    # dataset = TrainDataSet(path, '.jpg', 8, 128, 4, 16, 2, color=False)
    # dataloader = DataLoader(dataset,
    #                         batch_size=4,
    #                         shuffle=True,
    #                         num_workers=4)
    # dataloader = iter(dataloader)
    # a, b, c = next(dataloader)
    # print(a.size(), b.size(), c.size())

    hf = Random_Horizontal_Flip(0.5)
    a = torch.randint(0, 10, (2, 2))
    print(a, hf(a))


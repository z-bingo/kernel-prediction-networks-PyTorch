import torch
import torch.tensor
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np

class TrainDataSet(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, img_format='.jpg', burst_size=8, patch_size=64, upscale=4,
                 big_jitter=16, small_jitter=2, degamma=1.0, color=True, blind=False):
        super(TrainDataSet, self).__init__()
        self.dataset_dir = dataset_dir
        self.images = list(filter(lambda x: True if img_format in x else False, os.listdir(self.dataset_dir)))
        self.burst_size = burst_size
        self.patch_size = patch_size
        # self.upscale = upscale
        # self.big_jitter = big_jitter
        # self.small_jitter = small_jitter
        # 对应下采样之前图像的最大偏移量
        self.jitter_upscale = big_jitter * upscale
        # 对应下采样之前的图像的patch尺寸
        self.size_upscale = patch_size * upscale + 2 * self.jitter_upscale
        # 产生大jitter和小jitter之间的delta  在下采样之前的尺度上
        self.delta_upscale = (big_jitter - small_jitter) * upscale
        # 对应到原图的patch的尺寸
        self.patch_size_upscale = patch_size * upscale
        # 去伽马效应
        self.degamma = degamma
        # 是否用彩色图像进行处理
        self.color = color
        # 是否盲估计  盲估计即估计的噪声方差不会作为网络的输入
        self.blind = blind

    # get一个item  根据index检索
    def __getitem__(self, index):
        # print(index)
        image = Image.open(os.path.join(self.dataset_dir, self.images[index])).convert('RGB')
        width, height = image.size
        # 进行padding
        h_padding = max((self.size_upscale - width + 1) // 2, 0)
        v_padding = max((self.size_upscale - height + 1) // 2, 0)
        # 先转换为Tensor进行degamma
        image = transforms.ToTensor()(image)
        image = image ** self.degamma
        # 首先进行padding,再进行RandomCrop，再将结果转换为Tensor
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(self.size_upscale, padding=(h_padding, v_padding, h_padding, v_padding)),
            transforms.ToTensor()
        ])
        # 3*H*W
        image_crop = transform(image)
        # 3*H*W  对应于较小jitter下
        image_crop_small = image_crop[:, self.delta_upscale:-self.delta_upscale,
                                        self.delta_upscale:-self.delta_upscale]

        # 进一步进行random_crop所需的transform
        # 先转为PIL图像  再进行RandomCrop  之后进行4倍下采样  再转换为Tensor
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(self.patch_size_upscale),
            transforms.Resize(self.patch_size, Image.BOX),
            transforms.ToTensor()
        ])
        transform_target = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.patch_size, Image.BOX),
            transforms.ToTensor()
        ])

        # 产生big_jitter的概率  服从泊松分布  lambda参数为1.5
        prob_thresh = torch.from_numpy(np.random.poisson(1.5, [1, 1])).view(-1).expand(self.burst_size).float() / self.burst_size
        # 对于burst中的每个元素  通过filp the coin产生
        prob = torch.rand(self.burst_size)
        # 概率大于这个值得花  是较小的偏移
        small_jitter_index = (prob >= prob_thresh).view(-1)

        # burst中的第一个不做偏移  后期作为target
        # output shape: N*3*H*W
        image_burst = torch.stack([transform_target(image_crop[:, self.jitter_upscale:-self.jitter_upscale,
                                                     self.jitter_upscale:-self.jitter_upscale])]
                                   + [transform(image_crop_small) if small_jitter_index[i] == 1
                                             else transform(image_crop) for i in range(1, self.burst_size)], dim=0)

        # label为patch中burst的第一个
        gt = image_burst[0, ...]
        if not self.color:
            gt = torch.mean(gt, dim=0, keepdim=True)
            image_burst = torch.mean(image_burst, dim=1, keepdim=True)


        # 以上得到的patch size为burst*3*size*size
        """
        数据加噪声等一系列处理  全部基于rgb图像做
        """
        # 要产生[log10(0.1), log10(1.0)]之间的均匀分布随机数  也就是[0,1加负号即可]
        # 产生pred之后  再除以white_level恢复原来的亮度
        # batch中的每一个burst  产生一个white_level
        white_level = torch.from_numpy(np.power(10, -np.random.rand(1, 1, 1, 1)))
        white_level = white_level.type_as(image_burst).expand_as(image_burst)
        # 论文中对图像亮度赋值进行线性缩放[0.1, 1]
        image_burst = image_burst.mul(white_level)

        # 生成随机的read和shot噪声方差
        sigma_read = torch.from_numpy(
            np.power(10, np.random.uniform(-3.0, -1.5, (1, 1, 1, 1)))).type_as(image_burst)
        sigma_shot = torch.from_numpy(
            np.power(10, np.random.uniform(-4.0, -2.0, (1, 1, 1, 1)))).type_as(image_burst)

        # 产生噪声  依据论文中公式产生
        sigma_read_com = sigma_read.expand_as(image_burst)
        sigma_shot_com = sigma_shot.expand_as(image_burst)

        # generate noise
        # 论文中的噪声生成方式好像有点问题
        # noise = torch.randn(patches_out.size()).mul(
        #     (sigma_read_com ** 2 + sigma_shot_com.mul(patches_out)) ** 0.5)
        # noise = torch.randn(patches_out.size()).mul(sigma_read_com)
        #         # + torch.randn(patches_out.size()).mul(torch.sqrt(patches_out*sigma_shot_com))
        # burst_noise = torch.normal(image_burst, torch.sqrt(sigma_read_com**2 + image_burst * sigma_shot_com)).type_as(image_burst)
        burst_noise = torch.randn_like(image_burst)
        burst_noise = burst_noise * torch.sqrt(sigma_read_com**2 + image_burst * sigma_shot_com) + image_burst

        # burst_noise 恢复到[0,1] 截去外面的值
        burst_noise = torch.clamp(burst_noise, 0.0, 1.0)

        # 非盲估计  就要估计噪声的方差
        if not self.blind:
            # 接下来就是根据两个sigma  将估计的噪声标准差也作为输入  用burst中的第一个进行估计
            # estimation shape: 3*H*W
            sigma_read_est = sigma_read.view(1, 1, 1).expand_as(gt)
            sigma_shot_est = sigma_shot.view(1, 1, 1).expand_as(gt)
            sigma_estimate = torch.sqrt(sigma_read_est ** 2 + sigma_shot_est.mul(
                torch.max(torch.stack([burst_noise[0, ...], torch.zeros_like(burst_noise[0, ...])], dim=0), dim=0)[0]))

            # 按照最后一个维度  进行合并
            if not self.color:
                sigma_estimate = sigma_estimate.view(1, 1, self.patch_size, self.patch_size)

            # 把噪声的估计和burst图像连接在一起
            burst_noise = torch.cat([burst_noise, sigma_estimate], dim=0)

        # burst_noise shape: N/(N+1) * 3 * H * W
        # 训练图像采用灰度图  则将彩色图转换为灰度图
        if not self.color:
            burst_noise = torch.mean(burst_noise, dim=1, keepdim=True)
            gt = torch.mean(gt, dim=0, keepdim=True)

        # 按照文章中的 ref Image作为target进行了训练  输出结果和ref很相似  没能起到太大的去噪作用
        # return patches_with_noise, patches_with_noise[:, 0, ...], white_level
        # 不含噪声的ref作为target进行测试
        return burst_noise, gt, white_level

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    path = 'G:\\BinZhang\\DataSets\\burst-denoising\\test'
    dataset = TrainDataSet(path, '.jpg', 8, 128, 4, 16, 2, color=False)
    dataloader = DataLoader(dataset,
                            batch_size=4,
                            shuffle=True,
                            num_workers=4)
    dataloader = iter(dataloader)
    a, b, c = next(dataloader)
    print(a.size(), b.size(), c.size())


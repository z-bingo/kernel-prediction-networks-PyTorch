import torch
import torch.tensor
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np

class TrainDataSet_RAW(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, img_format='.jpg', burst_size=8, patches_per_image=4, patch_size=64, upscale=2,
                 big_jitter=16, small_jitter=2, degamma=1.0):
        super(TrainDataSet_RAW, self).__init__()
        self.dataset_dir = dataset_dir
        self.images = list(filter(lambda x: True if img_format in x else False, os.listdir(self.dataset_dir)))
        self.burst_size = burst_size
        self.patches_per_image = patches_per_image
        self.patch_size = 2*patch_size  # 此时产生的patch size为设定的2倍  下一步  四个像素合为一个bayer pattern  即可
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

    def __getitem__(self, index):
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
        patches = []
        for i in range(self.patches_per_image):
            patches.append(transform(image))
            # 产生了big_jitter的patches
            # patched_per_image * 3 * height * width
        patches = torch.stack([patch for patch in patches], dim=0)
        # small jitter对应的patches
        patches_small = patches[:, :, self.delta_upscale:-self.delta_upscale,
                        self.delta_upscale:-self.delta_upscale]
        # 进一步进行random_crop所需的transform
        # 先转为PIL图像  再进行RandomCrop  之后进行J倍下采样  再转换为Tensor
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(self.patch_size_upscale),
            transforms.Resize(self.patch_size, Image.BOX),
            transforms.ToTensor()
        ])
        patches_out = []
        # 不需要进行偏移的transform  仅进行resize  即下采样
        transform_target = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.patch_size, Image.BOX),
            transforms.ToTensor()
        ])
        # burst中的第一个不做偏移  后期作为target
        for num_patch in range(0, self.patches_per_image):
            # 产生big_jitter的概率  服从泊松分布  lambda参数为1.5
            prob_thresh = torch.from_numpy(np.random.poisson(1.5, [1, 1])).view(-1).expand(
                self.burst_size).float() / self.burst_size
            # 对于burst中的每个元素  通过filp the coin产生
            prob = torch.rand(self.burst_size)
            small_jitter_index = (prob >= prob_thresh).view(-1)
            # burst中的第一个是没有偏移的
            # 3 * patch_size * patch_size * burst
            patch_bursts = torch.stack([transform_target(patches[num_patch, :, self.jitter_upscale:-self.jitter_upscale,
                                                         self.jitter_upscale:-self.jitter_upscale])]
                                       + [transform(patches_small[num_patch, ...]) if small_jitter_index[i] == 1
                                          else transform(patches[num_patch, ...]) for i in range(1, self.burst_size)],
                                       dim=3)
            patches_out.append(patch_bursts)
        # batch*3*size*size*burst
        patches_out = torch.stack([patch for patch in patches_out], dim=0)
        # 四个像素看做一个bayer pattern  忽略每个像素中的两个color
        # batch*size*size*burst
        r = patches_out[:, 0, ::2, ::2, :]
        g1 = patches_out[:, 1, ::2, 1::2, :]
        g2 = patches_out[:, 1, 1::2, ::2, :]
        b = patches_out[:, 2, 1::2, 1::2, :]
        # batch*4*size*size*burst
        patches_out = torch.stack([r, g1, g2, b], dim=1)
        # (batch*4*size*size*burst) --> (batch*burst*4*size*size)
        # 此时patch_out的格式为bayer pattern
        patches_out = patches_out.permute(0, 4, 1, 2, 3).contiguous()
        # label为patch中burst的第一个  格式也是bayer pattern
        patches_label = patches_out[:, 0, ...]

        # 对于RAW域的数据  就不再区分color image和grayscale了
        """
        以下，加噪声等操作  加噪声参考Unprocessing Image for Learned RAW Denoising
        文中，给出了read噪声和shot噪声的线性关系
        """
        # compresses the histogram of intensities to more closely match our real data
        white_level = torch.from_numpy(np.power(10, -np.random.rand(self.patches_per_image)))
        white_level = white_level.view(self.patches_per_image, 1, 1, 1, 1).expand_as(patches_out).type_as(patches_out)
        #
        patches_out = patches_out.mul(white_level)
        # 生成噪声的lambda参数
        lambda_read = torch.from_numpy(np.power(10, np.random.uniform(np.log10(0.0001), np.log10(0.012), self.patches_per_image))).type_as(patches_out)
        lambda_shot = torch.from_numpy(np.power(10, np.random.normal(2.18*torch.log10(lambda_read)+1.2, 0.26, self.patches_per_image))).type_as(patches_out)
        # expand the shape
        lambda_read_com = lambda_read.view(self.patches_per_image,1,1,1,1).expand_as(patches_out)
        lambda_shot_com = lambda_shot.view(self.patches_per_image,1,1,1,1).expand_as(patches_out)
        # 含有噪声的patches
        burst_noise = torch.normal(patches_out, torch.sqrt(lambda_read_com + lambda_shot_com.mul(patches_out))).type_as(patches_out)

        # burst_noise 恢复到[0,1] 截去外面的值
        burst_noise[burst_noise > 1] = 1.0
        burst_noise[burst_noise < 0] = 0.0

        # 接下来就是根据两个sigma  将估计的噪声标准差也作为输入  用burst中的第一个进行估计
        sigma_read_est = lambda_read.view(self.patches_per_image,1,1,1).expand_as(patches_label)
        sigma_shot_est = lambda_shot.view(self.patches_per_image,1,1,1).expand_as(patches_label)
        sigma_estimate = torch.sqrt(sigma_read_est ** 2 + sigma_shot_est.mul(
            torch.max(torch.stack([burst_noise[:, 0, ...], torch.zeros_like(burst_noise[:, 0, ...])], dim=1), dim=1)[0]))
        # 按照最后一个维度  进行合并
        sigma_estimate = sigma_estimate.view(self.patches_per_image, 1, 4, self.patch_size//2, self.patch_size//2)
        patches_with_noise = torch.cat([burst_noise, sigma_estimate], dim=1)

        return patches_with_noise, patches_label, white_level

    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    path = 'D:\\BinZhang\\Codes\\BurstDenosing\\burst-denoising-master\\dataset\\train'
    dataset = TrainDataSet_RAW(
        dataset_dir=path,
        img_format='.jpg',
        burst_size=8,
        patches_per_image=4,
        patch_size=128,
        upscale=2,
        big_jitter=16,
        small_jitter=2,
        degamma=1.0
    )
    dataloader = DataLoader(dataset,
                            batch_size=4,
                            shuffle=False,
                            num_workers=4)
    print(len(dataset))
    for data in enumerate(dataloader):
        print([data[1][i].size() for i in range(3)])
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
from kpn_data_provider import TrainDataSet
from kpn_raw_data_provider import TrainDataSet_RAW
from torch.utils.data import Dataset, DataLoader
from KPN import KPN
from image_trans import loss_annealed, convolve, loss_basic, gradient_image_L1, gamma_inv
import image_trans
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import image_evaluate

# 相关参数定义
flags = {}
flags['batch_size'] = 16
flags['patch_size'] = 128
flags['burst_length'] = 8
flags['train_log_dir'] = './kpn_logs/'
flags['dataset_dir'] = 'G:/BinZhang/DataSets/burst-denoising'
flags['learning_rate'] = 0.0001
flags['anneal'] = 0.9998
flags['max_number_of_epochs'] = 100
flags['use_noise'] = True
flags['real_data'] = False

class sampler(torch.utils.data.Sampler):
    def __init__(self, data_source, num_samples):
        self.num_samples = num_samples
        self.total_num = len(data_source)

    def __iter__(self):
        if self.total_num % self.num_samples != 0:
            return iter(torch.randperm(self.total_num).tolist() + torch.randperm(self.total_num).tolist()[0:(self.total_num//self.num_samples+1)*self.num_samples - self.total_num])
        else:
            return iter(torch.randperm(self.total_num).tolist())

def learning_rate_half(optimizer, lr):
    """
    Decay the learning rate by one half.
    :param optimizer:
    :param lr:
    :return:
    """
    lr = lr / 2
    for param in optimizer.param_groups:
        param['lr'] = lr
    return lr

def train_simple(lr_decay=10):
    batch_size = flags['batch_size']
    burst_size = flags['burst_length']
    patch_size = flags['patch_size']
    global_step = 0
    #
    for file in os.listdir('.\\train_logs'):
        os.remove(os.path.join('.\\train_logs', file))
    loss_writer = SummaryWriter('.\\train_logs', 'Loss')

    # final_kernel = [1, 3, 5, 7, 9]
    final_kernel = [5]

    # 是否彩色图像
    color = False
    # 训练数据是合成的

    # kpn = KPN((burst_size + 1), 2*sum(final_kernel)*burst_size).cuda()
    kpn = KPN((burst_size + 1), 25 * burst_size, channel_att=True).cuda()

    # 加载训练到中间的模型  但是加载不了loss了
    start_epoch = 0
    resume = False
    if resume:
        kpn.load_state_dict(torch.load('./models/mkpn_newest_rgb.pkl'))

    dataset_dir = os.path.join(flags['dataset_dir'], 'train')
    dataset = TrainDataSet(
        dataset_dir=dataset_dir,
        img_format='.jpg',
        burst_size=burst_size,
        patch_size=patch_size,
        upscale=4,
        big_jitter=16,
        small_jitter=2,
        degamma=2.4,
        color=color,
        blind=False
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler(dataset.images, batch_size),
        num_workers=4
    )

    lr = flags['learning_rate']
    optimizer = optim.Adam(kpn.parameters(), lr=lr)
    # 创建一个data_loader的迭代器
    # 返回数据：batch_size*patches_per_iamge*size*size*color_channel*burst_size
    max_epoch = flags['max_number_of_epochs']
    global_step = 0
    min_loss = 10**9+7
    # train mode
    kpn.train()
    #
    from torch.optim import lr_scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)
    for i in range(0, max_epoch):
        # if global_step > 0 and global_step % 35000 == 0:
        #     lr = learning_rate_half(optimizer, lr=lr)
        scheduler.step()
        print('===========Epoch:{}, lr:{}.============'.format(i, lr))
        loss_sum = 0
        # burst: batch_size * patches * burst * (3) * size * size
        for step, (burst_noise, target, white_level) in enumerate(data_loader):
            # (batch*patch) * size * size
            burst_noise_feed = burst_noise.view(batch_size, -1, patch_size, patch_size).cuda()
            burst_noise = burst_noise[:, 0:burst_size, ...].cuda()
            target = target.cuda()
            white_level = white_level.cuda()

            # feed into the KPN network
            core_filt = kpn(burst_noise_feed)

            # 计算误差
            _, loss1, loss2 = loss_annealed(Y_truth=target, burst=burst_noise, burst_size=burst_size,
                                      core_filts=core_filt, white_level=white_level, final_K=final_kernel, global_step=global_step,
                                      beta=100, alpha=flags['anneal'])
            loss_temp = loss1*5+loss2
            # print(kpn.conv8[0].bias.grad)
            # TensorBoardX 可视化loss
            loss_writer.add_scalar('loss_basic', loss1, global_step)
            loss_writer.add_scalar('loss_anneal', loss2, global_step)
            loss_writer.add_scalar('loss', loss1 + loss2, global_step)
            global_step += 1
            # loss_sum += loss_temp
            # loss_avg = loss_sum / ((step + 1) * 1.0)
            print(
                'Epoch:{}. Step:{}. Loss_basic:{:.4f}. Loss_anneal:{:.4f}. Loss:{:.4f}.'.format(
                    i, step, loss1, loss2, loss_temp))
            optimizer.zero_grad()
            loss_temp.backward()
            optimizer.step()

            # save the model and validation
            if step % 500 == 0:
                torch.save(kpn.state_dict(), './models/mkpn_newest_L25_wt.pkl')
                eval_simple('./models/mkpn_min_loss_L25_wt.pkl')

            if loss_temp < min_loss:
                min_loss = loss_temp
                torch.save(kpn.state_dict(), './models/mkpn_min_loss_L25_wt.pkl')

def eval_simple(model):
    burst_size = 8
    # final_kernel = [1, 3, 5, 7, 9]
    final_kernel = [5]
    with torch.no_grad():
        kpn = KPN((burst_size + 1), sum(final_kernel)**2 * burst_size, channel_att=True).cuda()
        kpn.load_state_dict(torch.load(model))
    # kpn = kpn.cuda()
    color = False
    dataset_dir = 'G:\\BinZhang\\DataSets\\burst-denoising\\test'
    dataset = TrainDataSet(
        dataset_dir=dataset_dir,
        img_format='.jpg',
        burst_size=8,
        patch_size=128,
        upscale=4,
        big_jitter=16,
        small_jitter=2,
        degamma=2.4,
        color=color
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        sampler=sampler(dataset.images, 1),
        num_workers=4
    )
    kpn.eval()
    data_loader = iter(data_loader)

    # 删除原有的eval image
    for file in os.listdir('.\\eval_image'):
        if '.' in file and not 'raw' in file:
            os.remove(os.path.join('.\\eval_image', file))
    psnr_total = 0
    # tranforms
    trans = transforms.ToPILImage()
    for i in range(100):
        data, target, white_level = next(data_loader)
        data_feed = data.view(1, -1, 128, 128).cuda()
        data = data[:, :8, ...].cuda()
        target = target.cuda()
        white_level = white_level.cuda()
        with torch.no_grad():
            core_filt = kpn(data_feed)
        Yi = convolve(data, 8, core_filt, final_kernel)
        Yi = Yi / white_level
        Y = Yi.mean(dim=1, keepdim=False).squeeze()

        target = target.squeeze()
        # 保存到本地
        rgb_pred = Y.cpu().detach()
        rgb_target = target.cpu().detach()
        rgb_ref = (data[:, 0, ...] / white_level[:, 0, ...]).cpu().squeeze()
        rgb_pred[rgb_pred > 1] = 1.0
        rgb_pred[rgb_pred < 0] = 0.0

        # rgb_pred[j, ...] = (rgb_pred[j, ...] - torch.min(rgb_pred[j, ...])) / (
        #         torch.max(rgb_pred[j, ...]) - torch.min(rgb_pred[j, ...]))
        img_pred = trans(rgb_pred)

        # rgb_target[j, ...] = (rgb_target[j, ...] - torch.min(rgb_target[j, ...])) / (
        #         torch.max(rgb_target[j, ...]) - torch.min(rgb_target[j, ...]))
        img_target = trans(rgb_target)

        # rgb_ref = (rgb_ref - torch.min(rgb_ref[j, ...])) / (
        #         torch.max(rgb_ref[j, ...]) - torch.min(rgb_ref[j, ...]))
        img_ref = trans(rgb_ref)

        psnr = image_evaluate.PSNR(rgb_pred, rgb_target)
        psnr_total += psnr

        img_pred.save('./eval_image/{}_pred_{:.2f}dB.png'.format(i, psnr), 'png')
        img_target.save('./eval_image/{}_target.png'.format(i), 'png')
        img_ref.save('./eval_image/{}_ref.png'.format(i), 'png')

    print('Average PSNR: {:.2f}dB'.format(psnr_total/100))
    print('Over!')

# 将burst和core_filts计算得到去噪结果
def get_eval_results(burst, burst_size, core_filts, final_K, final_Channel, white_level):
    Yi = convolve(burst, burst_size, core_filts, final_K, final_Channel)
    Yi = Yi / white_level
    # print(torch.max(Yi), torch.min(Yi))
    Y = Yi.mean(dim=1).squeeze()
    return Y

# RAW stack to RGB
# shape of RAW: patch*4*height*width
# return: patch*3*height*width
def raw_stack_to_rgb(raw):
    return torch.stack([raw[:, 0, ...], (raw[:, 1, ...] + raw[:, 2, ...])/2, raw[:, 3, ...]], dim=1)


if __name__ == '__main__':
    # train_simple(True)
    eval_simple('./models/mkpn_min_loss_L25_wt.pkl')
    # show_patches()


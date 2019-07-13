import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import argparse

import os, sys, time, shutil

from data_provider import OnTheFlyDataset, _configspec_path
from kpn_data_provider import TrainDataSet, UndosRGBGamma, sRGBGamma
from KPN import KPN, LossFunc
from utils.training_util import MovingAverage, save_checkpoint, load_checkpoint, read_config
from utils.training_util import calculate_psnr, calculate_ssim

from tensorboardX import SummaryWriter
from PIL import Image
from torchvision.transforms import transforms

def train(config, num_workers, num_threads, cuda, restart_train, mGPU):
    # torch.set_num_threads(num_threads)

    train_config = config['training']
    arch_config = config['architecture']

    batch_size = train_config['batch_size']
    lr = train_config['learning_rate']
    weight_decay = train_config['weight_decay']
    decay_step = train_config['decay_steps']
    lr_decay = train_config['lr_decay']

    n_epoch = train_config['num_epochs']
    use_cache = train_config['use_cache']

    print('Configs:', config)
    # checkpoint path
    checkpoint_dir = train_config['checkpoint_dir']
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # logs path
    logs_dir = train_config['logs_dir']
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    shutil.rmtree(logs_dir)
    log_writer = SummaryWriter(logs_dir)

    # dataset and dataloader
    data_set = TrainDataSet(
        train_config['dataset_configs'],
        img_format='.bmp',
        degamma=True,
        color=False,
        blind=arch_config['blind_est']
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    dataset_config = read_config(train_config['dataset_configs'], _configspec_path())['dataset_configs']

    # model here
    model = KPN(
        color=False,
        burst_length=dataset_config['burst_length'],
        blind_est=arch_config['blind_est'],
        kernel_size=list(map(int, arch_config['kernel_size'].split())),
        sep_conv=arch_config['sep_conv'],
        channel_att=arch_config['channel_att'],
        spatial_att=arch_config['spatial_att'],
        upMode=arch_config['upMode'],
        core_bias=arch_config['core_bias']
    )
    if cuda:
        model = model.cuda()

    if mGPU:
        model = nn.DataParallel(model)
    model.train()

    # loss function here
    loss_func = LossFunc(
        coeff_basic=1.0,
        coeff_anneal=1.0,
        gradient_L1=True,
        alpha=arch_config['alpha'],
        beta=arch_config['beta']
    )

    # Optimizer here
    if train_config['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr
        )
    elif train_config['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError("Optimizer must be 'sgd' or 'adam', but received {}.".format(train_config['optimizer']))
    optimizer.zero_grad()

    # learning rate scheduler here
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=lr_decay)

    average_loss = MovingAverage(train_config['save_freq'])
    if not restart_train:
        try:
            checkpoint = load_checkpoint(checkpoint_dir, 'best')
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_iter']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
        except:
            start_epoch = 0
            global_step = 0
            best_loss = np.inf
            print('=> no checkpoint file to be loaded.')
    else:
        start_epoch = 0
        global_step = 0
        best_loss = np.inf
        if os.path.exists(checkpoint_dir):
            pass
            # files = os.listdir(checkpoint_dir)
            # for f in files:
            #     os.remove(os.path.join(checkpoint_dir, f))
        else:
            os.mkdir(checkpoint_dir)
        print('=> training')

    burst_length = dataset_config['burst_length']
    data_length = burst_length if arch_config['blind_est'] else burst_length+1
    patch_size = dataset_config['patch_size']

    for epoch in range(start_epoch, n_epoch):
        epoch_start_time = time.time()
        # decay the learning rate
        lr_cur = [param['lr'] for param in optimizer.param_groups]
        if lr_cur[0] > 5e-6:
            scheduler.step()
        else:
            for param in optimizer.param_groups:
                param['lr'] = 5e-6
        print('='*20, 'lr={}'.format([param['lr'] for param in optimizer.param_groups]), '='*20)
        t1 = time.time()
        for step, (burst_noise, gt, white_level) in enumerate(data_loader):
            if cuda:
                burst_noise = burst_noise.cuda()
                gt = gt.cuda()
            # print('white_level', white_level, white_level.size())

            #
            pred_i, pred = model(burst_noise, burst_noise[:, 0:burst_length, ...], white_level)

            #
            loss_basic, loss_anneal = loss_func(sRGBGamma(pred_i), sRGBGamma(pred), sRGBGamma(gt), global_step)
            loss = loss_basic + loss_anneal
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update the average loss
            average_loss.update(loss)
            # calculate PSNR
            psnr = calculate_psnr(pred.unsqueeze(1), gt.unsqueeze(1))
            ssim = calculate_ssim(pred.unsqueeze(1), gt.unsqueeze(1))

            # add scalars to tensorboardX
            log_writer.add_scalar('loss_basic', loss_basic, global_step)
            log_writer.add_scalar('loss_anneal', loss_anneal, global_step)
            log_writer.add_scalar('loss_total', loss, global_step)
            log_writer.add_scalar('psnr', psnr, global_step)
            log_writer.add_scalar('ssim', ssim, global_step)

            # print
            print('{:-4d}\t| epoch {:2d}\t| step {:4d}\t| loss_basic: {:.4f}\t| loss_anneal: {:.4f}\t|'
                  ' loss: {:.4f}\t| PSNR: {:.2f}dB\t| SSIM: {:.4f}\t| time:{:.2f} seconds.'
                  .format(global_step, epoch, step, loss_basic, loss_anneal, loss, psnr, ssim, time.time()-t1))
            t1 = time.time()
            # global_step
            global_step += 1

            if global_step % train_config['save_freq'] == 0:
                if average_loss.get_value() < best_loss:
                    is_best = True
                    best_loss = average_loss.get_value()
                else:
                    is_best = False

                save_dict = {
                    'epoch': epoch,
                    'global_iter': global_step,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': scheduler.state_dict()
                }
                save_checkpoint(
                    save_dict, is_best, checkpoint_dir, global_step, max_keep=train_config['ckpt_to_keep']
                )

        print('Epoch {} is finished, time elapsed {:.2f} seconds.'.format(epoch, time.time()-epoch_start_time))


def eval(config, args):
    train_config = config['training']
    arch_config = config['architecture']

    use_cache = train_config['use_cache']

    print('Eval Process......')

    checkpoint_dir = train_config['checkpoint_dir']
    if not os.path.exists(checkpoint_dir) or len(os.listdir(checkpoint_dir)) == 0:
        print('There is no any checkpoint file in path:{}'.format(checkpoint_dir))
    # the path for saving eval images
    eval_dir = train_config['eval_dir']
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)
    files = os.listdir(eval_dir)
    for f in files:
        os.remove(os.path.join(eval_dir, f))

    # dataset and dataloader
    data_set = TrainDataSet(
        train_config['dataset_configs'],
        img_format='.bmp',
        degamma=True,
        color=False,
        blind=arch_config['blind_est'],
        train=False
    )
    data_loader = DataLoader(
        data_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )

    dataset_config = read_config(train_config['dataset_configs'], _configspec_path())['dataset_configs']

    # model here
    model = KPN(
        color=False,
        burst_length=dataset_config['burst_length'],
        blind_est=arch_config['blind_est'],
        kernel_size=list(map(int, arch_config['kernel_size'].split())),
        sep_conv=arch_config['sep_conv'],
        channel_att=arch_config['channel_att'],
        spatial_att=arch_config['spatial_att'],
        upMode=arch_config['upMode'],
        core_bias=arch_config['core_bias']
    )
    if args.cuda:
        model = model.cuda()

    if args.mGPU:
        model = nn.DataParallel(model)
    # load trained model
    ckpt = load_checkpoint(checkpoint_dir, args.checkpoint)
    model.load_state_dict(ckpt['state_dict'])
    print('The model has been loaded from epoch {}, n_iter {}.'.format(ckpt['epoch'], ckpt['global_iter']))
    # switch the eval mode
    model.eval()

    # data_loader = iter(data_loader)
    burst_length = dataset_config['burst_length']
    data_length = burst_length if arch_config['blind_est'] else burst_length + 1
    patch_size = dataset_config['patch_size']

    trans = transforms.ToPILImage()

    with torch.no_grad():
        psnr = 0.0
        ssim = 0.0
        for i, (burst_noise, gt, white_level) in enumerate(data_loader):
            if i < 100:
                # data = next(data_loader)
                if args.cuda:
                    burst_noise = burst_noise.cuda()
                    gt = gt.cuda()
                    white_level = white_level.cuda()

                pred_i, pred = model(burst_noise, burst_noise[:, 0:burst_length, ...], white_level)

                pred_i = sRGBGamma(pred_i)
                pred = sRGBGamma(pred)
                gt = sRGBGamma(gt)
                burst_noise = sRGBGamma(burst_noise / white_level)

                psnr_t = calculate_psnr(pred.unsqueeze(1), gt.unsqueeze(1))
                ssim_t = calculate_ssim(pred.unsqueeze(1), gt.unsqueeze(1))
                psnr_noisy = calculate_psnr(burst_noise[:, 0, ...].unsqueeze(1), gt.unsqueeze(1))
                psnr += psnr_t
                ssim += ssim_t

                pred = torch.clamp(pred, 0.0, 1.0)

                if args.cuda:
                    pred = pred.cpu()
                    gt = gt.cpu()
                    burst_noise = burst_noise.cpu()

                trans(burst_noise[0, 0, ...].squeeze()).save(os.path.join(eval_dir, '{}_noisy_{:.2f}dB.png'.format(i, psnr_noisy)), quality=100)
                trans(pred.squeeze()).save(os.path.join(eval_dir, '{}_pred_{:.2f}dB.png'.format(i, psnr_t)), quality=100)
                trans(gt.squeeze()).save(os.path.join(eval_dir, '{}_gt.png'.format(i)), quality=100)

                print('{}-th image is OK, with PSNR: {:.2f}dB, SSIM: {:.4f}'.format(i, psnr_t, ssim_t))
            else:
                break
        print('All images are OK, average PSNR: {:.2f}dB, SSIM: {:.4f}'.format(psnr/100, ssim/100))


if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--config_file', dest='config_file', default='kpn_specs/kpn_config.conf', help='path to config file')
    parser.add_argument('--config_spec', dest='config_spec', default='kpn_specs/configspec.conf', help='path to config spec file')
    parser.add_argument('--restart', action='store_true', help='Whether to remove all old files and restart the training process')
    parser.add_argument('--num_workers', '-nw', default=4, type=int, help='number of workers in data loader')
    parser.add_argument('--num_threads', '-nt', default=8, type=int, help='number of threads in data loader')
    parser.add_argument('--cuda', '-c', action='store_true', help='whether to train on the GPU')
    parser.add_argument('--mGPU', '-m', action='store_true', help='whether to train on multiple GPUs')
    parser.add_argument('--eval', action='store_true', help='whether to work on the evaluation mode')
    parser.add_argument('--checkpoint', '-ckpt', dest='checkpoint', type=str, default='best',
                        help='the checkpoint to eval')
    args = parser.parse_args()
    #
    config = read_config(args.config_file, args.config_spec)
    if args.eval:
        eval(config, args)
    else:
        train(config, args.num_workers, args.num_threads, args.cuda, args.restart, args.mGPU)

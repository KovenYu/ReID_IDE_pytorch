from __future__ import division
from utils import AverageMeter, Recorder, TransOptions, Logger, time_string, convert_secs2time, eval_cmc_map, \
    reset_state_dict, extract_features, create_stat_string, save_checkpoint, adjust_learning_rate, accuracy, \
    partition_params
from ReIDdatasets import Market, FullTraining
from gan_net import Discriminator, BottleneckGenerator
from resnet import resnet56

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from itertools import izip

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from scipy.spatial.distance import cdist
from skimage.transform import resize

cudnn.benchmark = True


def main():
    opts = TransOptions()
    args = opts.parse()
    logger = Logger(args.save_path)
    opts.print_options(logger)

    source_data = FullTraining('data/JSTL.mat') if args.source == 'JSTL' else \
        Market('data/{}.mat'.format(args.source), state='train', require_views=False)

    mean = np.array([101, 102, 101]) / 255.0
    std = np.array([63, 62, 62]) / 255.0
    num_classes = source_data.return_num_class()

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop((144, 56), padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    source_data.turn_on_transform(transform=train_transform)
    target_data = Market('data/{}.mat'.format(args.target), state='train', transform=train_transform)
    gallery_data = Market('data/{}.mat'.format(args.target), state='gallery', transform=test_transform)
    probe_data = Market('data/{}.mat'.format(args.target), state='probe', transform=test_transform)

    source_loader = torch.utils.data.DataLoader(source_data, batch_size=args.batch_size, shuffle=True,
                                                num_workers=2, pin_memory=True, drop_last=True)
    target_loader = torch.utils.data.DataLoader(target_data, batch_size=args.batch_size, shuffle=True,
                                                num_workers=2, pin_memory=True, drop_last=True)
    gallery_loader = torch.utils.data.DataLoader(gallery_data, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=2, pin_memory=True)
    probe_loader = torch.utils.data.DataLoader(probe_data, batch_size=args.batch_size, shuffle=False,
                                               num_workers=2, pin_memory=True)

    net_s = resnet56(num_classes).cuda()
    checkpoint = torch.load(args.pretrain_path)
    if args.source != 'JSTL':
        fixed_layers = ('fc_final', 'bn_final')
        state_dict = reset_state_dict(checkpoint['state_dict'], net_s, *fixed_layers)
    else:
        state_dict = checkpoint['state_dict']
    net_s.load_state_dict(state_dict)
    net_t = copy.deepcopy(net_s).cuda()
    logger.print_log('loaded pre-trained feature net')
    generator = BottleneckGenerator(is_transfer_net=False).cuda()
    discriminator = Discriminator().cuda()

    criterion_CE = nn.CrossEntropyLoss().cuda()
    criterion_MSE = nn.MSELoss().cuda()
    criterion_L1 = nn.L1Loss().cuda()

    high_params_T, _ = partition_params(net_t, 'specified', *('stage_3', 'fc_final', 'bn_final'))

    optimizer_Th = torch.optim.Adam(high_params_T, lr=args.lr_Th, weight_decay=args.wd_Th, betas=(0.5, 0.999))
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_G, weight_decay=0, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr_D, weight_decay=0, betas=(0.5, 0.999))

    train_stats = ('acc/r1', 'var', 'D_real', 'D_fake')
    val_stats = ('acc/r1', 'var', 'D_real', 'r1_cat')
    recorder = Recorder(args.epochs, val_stats[0], train_stats, val_stats)
    logger.print_log('observing training stats: {} \nvalidation stats: {}'.format(train_stats, val_stats))

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.print_log("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            start_epoch = checkpoint['epoch']
            net_t.load_state_dict(checkpoint['state_dict_T'])
            net_s.load_state_dict(checkpoint['state_dict_S'])
            generator.load_state_dict(checkpoint['state_dict_G'])
            discriminator.load_state_dict(checkpoint['state_dict_D'])
            optimizer_Th.load_state_dict(checkpoint['optimizer_Th'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            logger.print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.print_log("=> no checkpoint found at '{}'".format(args.resume))

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()

    gallery_features_o, _, _ = extract_features(gallery_loader, net_s, index_feature=1, require_views=True)
    probe_features_o, _, _ = extract_features(probe_loader, net_s, index_feature=1, require_views=True)

    for epoch in range(start_epoch, args.epochs):

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        logger.print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s}'.format(time_string(), epoch, args.epochs, need_time))

        lr_D, = adjust_learning_rate(optimizer_D, (args.lr_D,), epoch, args.epochs, args.lr_strategy_GAN)
        lr_G, = adjust_learning_rate(optimizer_G, (args.lr_G,), epoch, args.epochs, args.lr_strategy_GAN)
        lr_Th, = adjust_learning_rate(optimizer_Th, (args.lr_Th,), epoch, args.epochs, args.lr_strategy_Th)
        print("   lr_D:{}, lr_G:{}, lr_Th:{}".format(epoch, lr_D, lr_G, lr_Th))

        if epoch < args.pretrain_epochs:
            pretrain(target_loader, net_t, generator,
                     criterion_L1,
                     optimizer_G, epoch, logger)
            is_best = False
        else:
            train(source_loader, target_loader, net_s, net_t, generator, discriminator,
                  criterion_CE, criterion_L1, criterion_MSE,
                  optimizer_Th, optimizer_G, optimizer_D, epoch, recorder, logger, args)

            is_best = evaluate(gallery_loader, probe_loader, net_s, net_t, generator, discriminator,
                               epoch, recorder, logger, args,
                               gallery_features_o, probe_features_o)

        if (epoch+1) % args.plot_freq == 0:
            net_s.eval()
            net_t.eval()
            generator.eval()
            img_t, _, _ = target_loader.dataset[epoch]
            img_s, _ = source_loader.dataset[epoch]
            img_t_var = torch.autograd.Variable(img_t.cuda())
            img_s_var = torch.autograd.Variable(img_s.cuda())
            _, _, _, mid_maps_t = net_t(img_t_var.view(1, 3, 144, 56))
            _, _, _, mid_maps_s = net_s(img_s_var.view(1, 3, 144, 56))
            fake_maps, _ = generator(mid_maps_s, is_target=False)
            _, ax = plt.subplots(3, 33, figsize=(14, 6))
            img_t_back = img_t.permute(1, 2, 0).cpu().numpy() * std + mean
            ax[0, 0].imshow(img_t_back)
            ax[0, 0].axis('off')
            for i in range(1, 33):
                figure = mid_maps_t.data[0, i - 1]
                figure = resize(figure.cpu().numpy(), (144, 56), preserve_range=True)
                ax[0, i].imshow(figure)
                ax[0, i].axis('off')
            img_s_back = img_s.permute(1, 2, 0).cpu().numpy() * std + mean
            ax[1, 0].imshow(img_s_back)
            ax[1, 0].axis('off')
            for i in range(1, 33):
                figure = mid_maps_s.data[0, i - 1]
                figure = resize(figure.cpu().numpy(), (144, 56), preserve_range=True)
                ax[1, i].imshow(figure)
                ax[1, i].axis('off')
            img_s_back = img_s.permute(1, 2, 0).cpu().numpy() * std + mean
            ax[2, 0].imshow(img_s_back)
            ax[2, 0].axis('off')
            for i in range(1, 33):
                figure = fake_maps.data[0, i - 1]
                figure = resize(figure.cpu().numpy(), (144, 56), preserve_range=True)
                ax[2, i].imshow(figure)
                ax[2, i].axis('off')
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig('{}/{}.png'.format(args.save_path, epoch))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict_S': net_s.state_dict(),
            'state_dict_D': discriminator.state_dict(),
            'state_dict_G': generator.state_dict(),
            'state_dict_T': net_t.state_dict(),
            'recorder': recorder,
            'optimizer_D': optimizer_D.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_Th': optimizer_Th.state_dict(),
        }, is_best, args.save_path, 'checkpoint.pth.tar')
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()


def train(source_loader, target_loader, net_s, net_t, generator, discriminator,
          criterion_CE, criterion_L1, criterion_MSE,
          optimizer_Th, optimizer_G, optimizer_D, epoch, recorder, logger, args):

    batch_time_meter = AverageMeter()
    stats = recorder.train_stats
    meters = {stat: AverageMeter() for stat in stats}

    net_s.train()
    net_t.train()
    generator.train()
    discriminator.train()
    gradient_weight = torch.FloatTensor([1]).cuda()

    end = time.time()
    is_real = torch.Tensor(args.batch_size).cuda()
    joint_loader = izip(target_loader, source_loader)
    for i, ((imgs_t, _, _), (imgs_s, labels_s)) in enumerate(joint_loader):
        labels_t = labels_s
        imgs_t_var = torch.autograd.Variable(imgs_t.cuda())
        imgs_s_var = torch.autograd.Variable(imgs_s.cuda())
        labels_t_var = torch.autograd.Variable(labels_t.cuda())

        _, _, _, mid_maps_s = net_s(imgs_s_var)
        _, _, _, mid_maps_t = net_t(imgs_t_var)

        fake_maps, _ = generator(mid_maps_s.detach(), is_target=False)

        _, _, predictions_s, _ = net_t(mid_maps_s.detach(), is_mid_maps=True)
        maps_t, _, _, _ = net_t(mid_maps_t.detach(), is_mid_maps=True)
        maps_fake, _, predictions_fake, _ = net_t(fake_maps.detach(), is_mid_maps=True)
        fake_prob = discriminator(maps_fake)
        real_prob = discriminator(maps_t)

        optimizer_D.zero_grad()
        optimizer_Th.zero_grad()
        # backward two GAN loss
        is_real_var = torch.autograd.Variable(is_real.resize_(imgs_t.size(0), 1).fill_(1))
        loss_D_real = criterion_MSE(real_prob, is_real_var)
        loss_D_real.backward(gradient=gradient_weight * args.R_GAN)
        is_real_var = torch.autograd.Variable(is_real.resize_(imgs_s.size(0), 1).fill_(0))
        loss_D_fake = criterion_MSE(fake_prob, is_real_var)
        loss_D_fake.backward(retain_graph=True, gradient=gradient_weight * args.R_GAN)
        # backward two softmax loss
        softmax_s = criterion_CE(predictions_s, labels_t_var)
        softmax_s.backward()
        softmax_ada = criterion_CE(predictions_fake, labels_t_var)
        softmax_ada.backward(gradient=gradient_weight * args.R_AdaSoftmax)
        acc = accuracy(predictions_fake.data, labels_t.cuda(), topk=(1,))
        optimizer_D.step()
        optimizer_Th.step()

        recons, _ = generator(mid_maps_t.detach(), is_target=True)
        maps_fake, _, predictions_fake, _ = net_t(fake_maps, is_mid_maps=True)
        fake_prob = discriminator(maps_fake)

        optimizer_G.zero_grad()
        # backward softmax loss
        if args.G_joint:
            softmax_ada = criterion_CE(predictions_fake, labels_t_var)
            softmax_ada.backward(retain_graph=True, gradient=gradient_weight * args.R_AdaSoftmax)
        # backward GAN loss
        is_real_var = torch.autograd.Variable(is_real.resize_(imgs_s.size(0), 1).fill_(1))
        loss_G = criterion_MSE(fake_prob, is_real_var)
        loss_G.backward(gradient=gradient_weight * args.R_GAN)
        # backward rec loss
        rec_loss = criterion_L1(recons, mid_maps_t.detach())
        rec_loss.backward(gradient=gradient_weight * args.R_rec * args.R_GAN)
        optimizer_G.step()

        # update meters
        meters['acc/r1'].update(acc[0][0], args.batch_size)
        meters['D_fake'].update(fake_prob.data.mean(), args.batch_size)
        meters['D_real'].update(real_prob.data.mean(), args.batch_size)
        meters['var'].update(torch.var(fake_maps).data[0], args.batch_size)

        # measure elapsed time
        batch_time_meter.update(time.time() - end)
        freq = args.batch_size / batch_time_meter.avg
        end = time.time()

        if i % args.print_freq == 0:
            logger.print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   Freq {:.1f}   '.format(
                epoch, i, min(len(target_loader), len(source_loader)), freq) + create_stat_string(meters) + time_string())

    logger.print_log('  **Train**  ' + create_stat_string(meters))

    recorder.update(epoch=epoch, is_train=True, meters=meters)


def evaluate(gallery_loader, probe_loader, net_s, net_t, generator, discriminator,
             epoch, recorder, logger, args,
             gallery_features_o, probe_features_o):

    stats = recorder.val_stats
    meters = {stat: AverageMeter() for stat in stats}
    net_s.eval()
    net_t.eval()
    generator.eval()
    discriminator.eval()

    for i, (imgs_t, _, _) in enumerate(probe_loader):
        imgs_t = imgs_t.cuda()
        imgs_t_var = torch.autograd.Variable(imgs_t)
        maps_t, _, _, mid_maps_t = net_t(imgs_t_var)
        real_prob = discriminator(maps_t)

        meters['D_real'].update(real_prob.data.mean(), args.batch_size)
        meters['var'].update(torch.var(mid_maps_t).data[0], args.batch_size)

    gallery_features, gallery_labels, gallery_views = extract_features(gallery_loader, net_t, index_feature=1, require_views=True)
    probe_features, probe_labels, probe_views = extract_features(probe_loader, net_t, index_feature=1, require_views=True)
    gallery_features_cat = np.concatenate((gallery_features, gallery_features_o), axis=1)
    probe_features_cat = np.concatenate((probe_features, probe_features_o), axis=1)
    dist = cdist(gallery_features, probe_features, metric='euclidean')
    dist_cat = cdist(gallery_features_cat, probe_features_cat, metric='euclidean')
    CMC, MAP = eval_cmc_map(dist, gallery_labels, probe_labels, gallery_views, probe_views)
    CMC_cat, MAP_cat = eval_cmc_map(dist_cat, gallery_labels, probe_labels, gallery_views, probe_views)
    rank1 = CMC[0]
    rank1_cat = CMC_cat[0]
    meters['acc/r1'].update(rank1, 1)
    meters['r1_cat'].update(rank1_cat, 1)

    logger.print_log('  **Test**  ' + create_stat_string(meters))
    is_best = recorder.update(epoch=epoch, is_train=False, meters=meters)
    return is_best


def pretrain(target_loader, net_t, generator,
             criterion_L1,
             optimizer_G, epoch, logger):
    net_t.train()
    generator.train()
    logger.print_log('  **Pretrain**    Epoch: [{:03d}]   '.format(epoch))
    for i, (imgs_t, _, _) in enumerate(target_loader):
        imgs_t_var = torch.autograd.Variable(imgs_t.cuda())
        _, _, _, mid_maps_t = net_t(imgs_t_var)

        optimizer_G.zero_grad()
        recons, _ = generator(mid_maps_t.detach(), is_target=True)
        L1_t = criterion_L1(recons, mid_maps_t.detach())
        L1_t.backward()  # passing generator
        optimizer_G.step()


if __name__ == '__main__':
    main()

from __future__ import division
from utils import AverageMeter, BaseOptions, Recorder, Logger, time_string, convert_secs2time, eval_cmc_map, \
    reset_state_dict, extract_features, create_stat_string, save_checkpoint, adjust_learning_rate, accuracy, \
    partition_params
from ReIDdatasets import Market

import os
import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
import resnet
from scipy.spatial.distance import cdist

cudnn.benchmark = True


def main():
    opts = BaseOptions()
    args = opts.parse()
    logger = Logger(args.save_path)
    opts.print_options(logger)

    mean = np.array([0.485, 0.406, 0.456])
    std = np.array([0.229, 0.224, 0.225])

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop((224, 224), padding=7), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_data = Market('data/{}.mat'.format(args.dataset), state='train', transform=train_transform)
    gallery_data = Market('data/{}.mat'.format(args.dataset), state='gallery', transform=test_transform)
    probe_data = Market('data/{}.mat'.format(args.dataset), state='probe', transform=test_transform)
    num_classes = train_data.return_num_class()

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=2, pin_memory=True, drop_last=True)
    gallery_loader = torch.utils.data.DataLoader(gallery_data, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=2, pin_memory=True)
    probe_loader = torch.utils.data.DataLoader(probe_data, batch_size=args.batch_size, shuffle=False,
                                               num_workers=2, pin_memory=True)

    net = resnet.resnet50(pretrained=False, num_classes=num_classes).cuda()
    checkpoint = torch.load(args.pretrain_path)
    fixed_layers = ('fc',)
    state_dict = reset_state_dict(checkpoint, net, *fixed_layers)
    net.load_state_dict(state_dict)
    logger.print_log('loaded pre-trained feature net')

    criterion_CE = nn.CrossEntropyLoss().cuda()

    bn_params, conv_params = partition_params(net, 'bn')

    optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0},
                                 {'params': conv_params}], lr=args.lr, momentum=0.9, weight_decay=args.wd)

    train_stats = ('acc', 'loss')
    val_stats = ('acc',)
    recorder = Recorder(args.epochs, val_stats[0], train_stats, val_stats)
    logger.print_log('observing training stats: {} \nvalidation stats: {}'.format(train_stats, val_stats))

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.print_log("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.print_log("=> no checkpoint found at '{}'".format(args.resume))

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()

    for epoch in range(start_epoch, args.epochs):

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        logger.print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s}'.format(time_string(), epoch, args.epochs, need_time))

        lr, _ = adjust_learning_rate(optimizer, (args.lr, args.lr), epoch, args.epochs, args.lr_strategy)
        print("   lr:{}".format(lr))

        train(train_loader, net,
              criterion_CE,
              optimizer, epoch, recorder, logger, args)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'recorder': recorder,
            'optimizer': optimizer.state_dict(),
        }, False, args.save_path, 'checkpoint.pth.tar')
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    evaluate(gallery_loader, probe_loader, net,
             args.epochs - 1, recorder, logger)


def train(train_loader, net,
          criterion_CE,
          optimizer, epoch, recorder, logger, args):

    batch_time_meter = AverageMeter()
    stats = recorder.train_stats
    meters = {stat: AverageMeter() for stat in stats}

    net.train()

    end = time.time()
    for i, (imgs, labels, views) in enumerate(train_loader):
        imgs_var = torch.autograd.Variable(imgs.cuda())
        labels_var = torch.autograd.Variable(labels.cuda())

        _, predictions = net(imgs_var)

        optimizer.zero_grad()
        softmax = criterion_CE(predictions, labels_var)
        softmax.backward()
        acc = accuracy(predictions.data, labels.cuda(), topk=(1,))
        optimizer.step()

        # update meters
        meters['acc'].update(acc[0][0], args.batch_size)
        meters['loss'].update(softmax.data.mean(), args.batch_size)

        # measure elapsed time
        batch_time_meter.update(time.time() - end)
        freq = args.batch_size / batch_time_meter.avg
        end = time.time()

        if i % args.print_freq == 0:
            logger.print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   Freq {:.1f}   '.format(
                epoch, i, len(train_loader), freq) + create_stat_string(meters) + time_string())

    logger.print_log('  **Train**  ' + create_stat_string(meters))

    recorder.update(epoch=epoch, is_train=True, meters=meters)


def evaluate(gallery_loader, probe_loader, net,
             epoch, recorder, logger):

    stats = recorder.val_stats
    meters = {stat: AverageMeter() for stat in stats}
    net.eval()

    gallery_features, gallery_labels, gallery_views = extract_features(gallery_loader, net, index_feature=0, require_views=True)
    probe_features, probe_labels, probe_views = extract_features(probe_loader, net, index_feature=0, require_views=True)
    dist = cdist(gallery_features, probe_features, metric='euclidean')
    CMC, MAP = eval_cmc_map(dist, gallery_labels, probe_labels, gallery_views, probe_views)
    rank1 = CMC[0]
    meters['acc'].update(rank1, 1)

    logger.print_log('  **Test**  ' + create_stat_string(meters))
    recorder.update(epoch=epoch, is_train=False, meters=meters)


if __name__ == '__main__':
    main()

from __future__ import division
from utils import AverageMeter, Recorder, time_string, convert_secs2time, eval_cmc_map, occupy_gpu_memory
from ReIDdatasets import Market
from gan_net import Discriminator, BottleneckGenerator, LongneckGenerator
from resnet import resnet56

import argparse
import os
import shutil
import sys
import time
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from scipy.spatial.distance import cdist
from skimage.transform import resize

parser = argparse.ArgumentParser(description='Re-ID transfer net',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Optimization options
parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--decrease_lr_Th', action='store_true')
parser.add_argument('--decrease_lr_GAN', action='store_true')
parser.add_argument('--lr_D', type=float, default=0.0001)
parser.add_argument('--lr_G', type=float, default=0.0002)
parser.add_argument('--lr_Th', type=float, default=0.0001)
parser.add_argument('--wd_D', type=float, default=0)
parser.add_argument('--wd_G', type=float, default=0)
parser.add_argument('--wd_Th', type=float, default=0)
parser.add_argument('--weight_L1', type=float, default=1, help='weight for L1 loss in generator')
# Checkpoints
parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./debug', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--model_path', default='data/pretrain_model.pth.tar', type=str)
parser.add_argument('--plot_freq', default=10, type=int)
parser.add_argument('--target', default='Market', type=str, choices=['Market', 'Duke'])
parser.add_argument('--source', default='', type=str)
# model options
parser.add_argument('--is_transfer_net', action='store_true')
parser.add_argument('--G_structure', default='Bottleneck', type=str, choices=['Bottleneck', 'Longneck'])
# Acceleration
parser.add_argument('--gpu', type=str, default='0', help='gpu used.')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.use_cuda = torch.cuda.is_available()
if not args.source:
    if args.target == 'Market':
        args.source = 'Duke'
    else:
        args.source = 'Market'

random_seed = 0
torch.manual_seed(random_seed)
if args.use_cuda:
    torch.cuda.manual_seed_all(random_seed)
cudnn.benchmark = True


def main():
    # Init logger
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path, 'log_{}.txt'.format(time_string())), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(random_seed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)

    # prepare data
    source_data = Market('data/{}.mat'.format(args.source), state='train')

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

    net_s = resnet56(num_classes)
    checkpoint = torch.load(args.model_path)
    checkpoint['state_dict']['fc_final.weight'] = net_s.fc_final.weight.data
    checkpoint['state_dict']['bn_final.bias'] = net_s.bn_final.bias.data
    checkpoint['state_dict']['bn_final.running_mean'] = net_s.bn_final.running_mean
    checkpoint['state_dict']['bn_final.running_var'] = net_s.bn_final.running_var
    checkpoint['state_dict']['bn_final.weight'] = net_s.bn_final.weight.data
    net_s.load_state_dict(checkpoint['state_dict'])
    net_t = copy.deepcopy(net_s)
    print_log('loaded pre-trained feature net', log)
    if args.G_structure == 'Bottleneck':
        generator = BottleneckGenerator(args.is_transfer_net)
    else:
        generator = LongneckGenerator(args.is_transfer_net)
    discriminator = Discriminator()

    criterion_CE = nn.CrossEntropyLoss()
    criterion_MSE = nn.MSELoss()
    criterion_L1 = nn.L1Loss()

    high_params_T, _ = partition_high_params(net_t)

    optimizer_Th = torch.optim.Adam(high_params_T, lr=args.lr_Th, weight_decay=args.wd_Th, betas=(0.5, 0.999))

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_G, weight_decay=args.wd_G, betas=(0.5, 0.999))

    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr_D, weight_decay=args.wd_D, betas=(0.5, 0.999))

    if args.use_cuda:
        print('=> moving model to gpu')
        net_s.cuda()
        net_t.cuda()
        discriminator.cuda()
        generator.cuda()
        criterion_CE.cuda()
        criterion_MSE.cuda()
        criterion_L1.cuda()
        print('=> model has been moved to gpu')

    stats = ('r1', 'acc', 'var', 'L1', 'D_real_after_G_updated', 'D_fake')
    recorder = Recorder(args.epochs, stats[0], *stats)

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
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
            print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint", log)

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()

    gallery_features_o, _, _ = extract_features(gallery_loader, net_s)
    probe_features_o, _, _ = extract_features(probe_loader, net_s)

    for epoch in range(start_epoch, args.epochs):

        if args.decrease_lr_GAN:
            adjust_learning_rate(optimizer_D, args.lr_D, epoch)
            adjust_learning_rate(optimizer_G, args.lr_G, epoch)
        if args.decrease_lr_Th:
            adjust_learning_rate(optimizer_Th, args.lr_Th, epoch)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s}'.format(time_string(), epoch, args.epochs, need_time), log)

        train(source_loader, target_loader, net_s, net_t, generator, discriminator,
              criterion_CE, criterion_L1, criterion_MSE,
              optimizer_Th, optimizer_G, optimizer_D, epoch, recorder, log)

        gallery_features, gallery_labels, gallery_views = extract_features(gallery_loader, net_t)
        probe_features, probe_labels, probe_views = extract_features(probe_loader, net_t)
        gallery_features_cat = np.concatenate((gallery_features, gallery_features_o), axis=1)
        probe_features_cat = np.concatenate((probe_features, probe_features_o), axis=1)
        dist = cdist(gallery_features, probe_features, metric='euclidean')
        dist_cat = cdist(gallery_features_cat, probe_features_cat, metric='euclidean')
        CMC, MAP = eval_cmc_map(dist, gallery_labels, probe_labels, gallery_views, probe_views)
        CMC_cat, MAP_cat = eval_cmc_map(dist_cat, gallery_labels, probe_labels, gallery_views, probe_views)
        rank1 = CMC[0]
        rank1_cat = CMC_cat[0]
        print_log('  **Test** rank1 {:.2f}, rank1_cat {:.2f}  '.format(rank1, rank1_cat), log)
        recorder.update(epoch, is_train=True, r1=rank1)
        is_best = recorder.update(epoch, is_train=False, r1=rank1_cat)

        evaluate(probe_loader, net_t, generator, discriminator, criterion_L1, epoch, recorder, log)

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

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

    log.close()


def train(source_loader, target_loader, net_s, net_t, generator, discriminator,
          criterion_CE, criterion_L1, criterion_MSE,
          optimizer_Th, optimizer_G, optimizer_D, epoch, recorder, log):

    batch_time_meter = AverageMeter()
    softmax_meter = AverageMeter()
    acc_meter = AverageMeter()
    L1_meter = AverageMeter()
    loss_G_meter = AverageMeter()
    loss_D_meter = AverageMeter()
    D_real_meter = AverageMeter()
    D_fake_meter = AverageMeter()
    D_fake_after_G_updated_meter = AverageMeter()
    var_meter = AverageMeter()

    net_s.train()
    net_t.train()
    generator.train()
    discriminator.train()
    gradient_weight = torch.FloatTensor([1])

    end = time.time()
    source_iter = iter(source_loader)
    is_real = torch.Tensor(args.batch_size)
    for i, (imgs_t, _, _) in enumerate(target_loader):
        imgs_s, labels_s, _ = next(source_iter)
        labels_t = labels_s

        if args.use_cuda:
            labels_t = labels_t.cuda(async=True)
            imgs_t = imgs_t.cuda()
            imgs_s = imgs_s.cuda()
            is_real = is_real.cuda()
            gradient_weight = gradient_weight.cuda()
        imgs_t_var = torch.autograd.Variable(imgs_t)
        imgs_s_var = torch.autograd.Variable(imgs_s)
        labels_t_var = torch.autograd.Variable(labels_t)

        _, _, _, mid_maps_s = net_s(imgs_s_var)
        _, _, _, mid_maps_t = net_t(imgs_t_var)

        optimizer_D.zero_grad()
        # train D with real
        is_real_var = torch.autograd.Variable(is_real.resize_(imgs_t.size(0), 1).fill_(1))
        prob = discriminator(mid_maps_t.detach())
        loss_D_real = criterion_MSE(prob, is_real_var)
        loss_D_real.backward()  # passing discriminator
        D_real = prob.data.mean()

        # train D with fake
        is_real_var = torch.autograd.Variable(is_real.resize_(imgs_s.size(0), 1).fill_(0))
        fake_maps, _ = generator(mid_maps_s.detach(), is_target=False)
        prob = discriminator(fake_maps.detach())
        loss_D_fake = criterion_MSE(prob, is_real_var)
        loss_D_fake.backward()  # only passing discriminator
        D_fake_after_G_updated = prob.data.mean()
        optimizer_D.step()
        loss_D = loss_D_real + loss_D_fake

        optimizer_G.zero_grad()
        # train G with real
        recons, _ = generator(mid_maps_t.detach(), is_target=True)
        L1_t = criterion_L1(recons, mid_maps_t.detach())
        L1_t.backward(gradient=gradient_weight * args.weight_L1)  # passing generator
        L1 = L1_t.data[0]

        # train G with fake
        is_real_var = torch.autograd.Variable(is_real.resize_(imgs_s.size(0), 1).fill_(1))
        prob = discriminator(fake_maps)
        loss_G = criterion_MSE(prob, is_real_var)
        loss_G.backward(retain_graph=True)  # passing G and D
        D_fake = prob.data.mean()
        optimizer_G.step()

        _, _, predictions, _ = net_t(fake_maps.detach(), is_mid_maps=True)  # update Th using previous fake_maps
        softmax = criterion_CE(predictions, labels_t_var)
        acc = accuracy(predictions.data, labels_t, topk=(1,))

        optimizer_Th.zero_grad()
        softmax.backward(gradient=gradient_weight)  # only passing net
        optimizer_Th.step()

        # update meters
        acc_meter.update(acc[0][0], args.batch_size)
        softmax_meter.update(softmax.data[0], args.batch_size)
        loss_D_meter.update(loss_D.data[0], args.batch_size)
        loss_G_meter.update(loss_G.data[0], args.batch_size)
        D_fake_meter.update(D_fake, args.batch_size)
        D_fake_after_G_updated_meter.update(D_fake_after_G_updated, args.batch_size)
        D_real_meter.update(D_real, args.batch_size)
        L1_meter.update(L1, args.batch_size)
        var_meter.update(torch.var(fake_maps).data[0], args.batch_size)

        # measure elapsed time
        batch_time_meter.update(time.time() - end)
        freq = args.batch_size / batch_time_meter.avg
        end = time.time()

        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                      'Freq {:.3f}   '
                      'var {var:.3f}   '
                      'Acc {acc:.3f}   '
                      'softmax {softmax:.4f}   '
                      'loss_G {loss_G:.4f}   '
                      'loss_D {loss_D:.4f}   '
                      'L1 {L1:.4f}   '
                      'D_real_after_G_updated {D_real:.4f}   '
                      'D_fake_after_D_updated {D_fake:.4f}   '.format(
                       epoch, i, len(target_loader), freq, var=var_meter.avg, acc=acc_meter.avg,
                       softmax=softmax_meter.avg, loss_G=loss_G_meter.avg, loss_D=loss_D_meter.avg, L1=L1_meter.avg,
                       D_real=D_real_meter.avg, D_fake=D_fake_meter.avg) + time_string(), log)
    print_log('  **Train**  '
              'Acc {acc:.3f}   '
              'var {var:.2f}   '
              'loss_G {loss_G:.4f}   '
              'loss_D {loss_D:.4f}   '
              'L1 {L1:.4f}   '
              'D_real_after_G_updated {D_real:.4f}   '
              'D_fake_after_D_updated {D_fake:.4f}   '.format(
               var=var_meter.avg, acc=acc_meter.avg,
               loss_G=loss_G_meter.avg, loss_D=loss_D_meter.avg, L1=L1_meter.avg,
               D_real=D_real_meter.avg, D_fake=D_fake_meter.avg), log)
    recorder.update(epoch=epoch, is_train=True, acc=acc_meter.avg, L1=L1_meter.avg, D_real_after_G_updated=D_real_meter.avg,
                    D_fake=D_fake_meter.avg, var=var_meter.avg)
    recorder.update(epoch=epoch, is_train=False, D_fake=D_fake_after_G_updated_meter.avg)

    if epoch % args.plot_freq == 0:  # plot mid maps
        net_s.eval()
        net_t.eval()
        generator.eval()
        mean = np.array([101, 102, 101]) / 255.0
        std = np.array([63, 62, 62]) / 255.0
        img_t, _, _ = target_loader.dataset[epoch]
        img_s, _, _ = source_loader.dataset[epoch]
        if args.use_cuda:
            img_t = img_t.cuda()
            img_s = img_s.cuda()
        img_t_var = torch.autograd.Variable(img_t)
        img_s_var = torch.autograd.Variable(img_s)
        _, _, _, mid_maps_t = net_t(img_t_var.view(1, 3, 144, 56))
        _, _, _, mid_maps_s = net_s(img_s_var.view(1, 3, 144, 56))
        fake_maps, _ = generator(mid_maps_s, is_target=False)
        _, ax = plt.subplots(3, 33, figsize=(14, 6))
        img_t_back = img_t.permute(1, 2, 0).cpu().numpy() * std + mean
        ax[0, 0].imshow(img_t_back)
        ax[0, 0].axis('off')
        for i in range(1, 33):
            figure = mid_maps_t.data[0, i-1]
            figure = resize(figure.cpu().numpy(), (144, 56), preserve_range=True)
            ax[0, i].imshow(figure)
            ax[0, i].axis('off')
        img_s_back = img_s.permute(1, 2, 0).cpu().numpy() * std + mean
        ax[1, 0].imshow(img_s_back)
        ax[1, 0].axis('off')
        for i in range(1, 33):
            figure = mid_maps_s.data[0, i-1]
            figure = resize(figure.cpu().numpy(), (144, 56), preserve_range=True)
            ax[1, i].imshow(figure)
            ax[1, i].axis('off')
        img_s_back = img_s.permute(1, 2, 0).cpu().numpy() * std + mean
        ax[2, 0].imshow(img_s_back)
        ax[2, 0].axis('off')
        for i in range(1, 33):
            figure = fake_maps.data[0, i-1]
            figure = resize(figure.cpu().numpy(), (144, 56), preserve_range=True)
            ax[2, i].imshow(figure)
            ax[2, i].axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig('{}/{}.png'.format(args.save_path, epoch))


def evaluate(probe_loader, net_t, generator, discriminator, criterion_L1, epoch, recorder, log):
    D_real_meter = AverageMeter()
    L1_meter = AverageMeter()
    var_meter = AverageMeter()
    generator.eval()
    discriminator.eval()
    net_t.eval()
    for i, (imgs_t, _, _) in enumerate(probe_loader):
        if args.use_cuda:
            imgs_t = imgs_t.cuda()
        imgs_t_var = torch.autograd.Variable(imgs_t)
        _, _, _, mid_maps_t = net_t(imgs_t_var)
        prob = discriminator(mid_maps_t)
        D_real = prob.data.mean()
        recons_maps, _ = generator(mid_maps_t)
        L1 = criterion_L1(recons_maps, mid_maps_t.detach())

        D_real_meter.update(D_real, args.batch_size)
        L1_meter.update(L1.data[0], args.batch_size)
        var_meter.update(torch.var(mid_maps_t).data[0], args.batch_size)
    print_log('  **Test**  '
              'D_real {D_real.avg:.4f}   '
              'var {var.avg:.2f}   '
              'L1 {L1.avg:.4f}   '.format(D_real=D_real_meter, var=var_meter, L1=L1_meter),
              log)
    recorder.update(epoch=epoch, is_train=False, D_real_after_G_updated=D_real_meter.avg, L1=L1_meter.avg, var=var_meter.avg)


def extract_features(loader, model):
    # switch to evaluate mode
    model.eval()

    labels = torch.zeros((len(loader.dataset),))
    views = torch.zeros((len(loader.dataset),))

    idx = 0
    for i, (imgs, label_batch, view_batch) in enumerate(loader):
        if args.use_cuda:
            imgs = imgs.cuda()
        imgs_var = torch.autograd.Variable(imgs, volatile=True)
        _, feature_batch, _, mid_maps = model(imgs_var)
        feature_batch = feature_batch.data.cpu()

        if i == 0:
            feature_dim = feature_batch.shape[1]
            features = torch.zeros((len(loader.dataset), feature_dim))

        batch_size = label_batch.shape[0]
        # noinspection PyUnboundLocalVariable
        features[idx: idx + batch_size, :] = feature_batch
        labels[idx: idx + batch_size] = label_batch
        views[idx: idx + batch_size] = view_batch
        idx += batch_size

    features_np = features.numpy()
    labels_np = labels.numpy()
    views_np = views.numpy()
    return features_np, labels_np, views_np


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)


def adjust_learning_rate(optimizer, lr, epoch):
    schedule = [i for i in range(int(args.epochs/2), args.epochs)]
    gammas = [1 - x/float(args.epochs/2) for x in range(int(args.epochs/2))]
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    new_lr = lr
    for (gamma, step) in zip(gammas, schedule):
        if epoch >= step:
            new_lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


def accuracy(output, labels, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = labels.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def partition_bn_params(module):
    bn_params_set = set()
    for m in module.modules():
        if (isinstance(m, torch.nn.BatchNorm1d) or
                isinstance(m, torch.nn.BatchNorm2d) or
                isinstance(m, torch.nn.BatchNorm3d)):
            bn_params_set.update(set(m.parameters()))
    all_params_set = set(module.parameters())
    other_params_set = all_params_set.difference(bn_params_set)
    bn_params = (p for p in bn_params_set)
    other_params = (p for p in other_params_set)
    return bn_params, other_params


def partition_high_params(net_t):
    high_params_set = set()
    for m in net_t.stage_3.modules():
        high_params_set.update(set(m.parameters()))
    high_params_set.update(set(net_t.fc_final.parameters()))
    high_params_set.update(set(net_t.bn_final.parameters()))
    all_params_set = set(net_t.parameters())
    other_params_set = all_params_set.difference(high_params_set)
    high_params = (p for p in high_params_set)
    other_params = (p for p in other_params_set)
    return high_params, other_params


if __name__ == '__main__':
    main()

import os, sys, time, shutil
import numpy as np
import matplotlib
import subprocess
import torch
import argparse
matplotlib.use('agg')
import matplotlib.pyplot as plt


class BaseOptions(object):
    """
    base options for deep learning for Re-ID.
    parse basic arguments by parse(), print all the arguments by print_options()
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.args = None

        self.parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
        self.parser.add_argument('--batch_size', type=int, default=16)
        self.parser.add_argument('--print_freq', default=100, type=int, help='print after several batches')
        self.parser.add_argument('--save_path', type=str, default='./debug', help='Folder to save checkpoints and log.')
        self.parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
        self.parser.add_argument('--gpu', type=str, default='0', help='gpu used.')
        self.parser.add_argument('--pretrain_path', default='data/resnet50-19c8e357.pth', type=str)
        self.parser.add_argument('--lr_strategy', type=str, default='finetune_style')
        self.parser.add_argument('--lr', type=float, default=0.001)
        self.parser.add_argument('--wd', type=float, default=0.0001)
        self.parser.add_argument('--dataset', default='Market', type=str, choices=['Market', 'Duke'])

    def parse(self):
        self.args = self.parser.parse_args()
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        return self.args

    def print_options(self, logger):
        logger.print_log("")
        logger.print_log("----- options -----".center(120, '-'))
        args = vars(self.args)
        string = ''
        for i, (k, v) in enumerate(sorted(args.items())):
            string += "{}: {}".format(k, v).center(40, ' ')
            if i % 3 == 2 or i == len(args.items())-1:
                logger.print_log(string)
                string = ''
        logger.print_log("".center(120, '-'))


class Logger(object):
    def __init__(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        self.file = open(os.path.join(save_path, 'log_{}.txt'.format(time_string())), 'w')
        self.print_log("python version : {}".format(sys.version.replace('\n', ' ')))
        self.print_log("torch  version : {}".format(torch.__version__))

    def print_log(self, string):
        self.file.write("{}\n".format(string))
        self.file.flush()
        print(string)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Recorder(object):
    """
    record specified stats in every epoch for train and val,
    and compute the mean of them.
    For any stat, you might update it by calling Recorder.update() at each epoch,
    and finally plot it in a subplot with both train (blue) and val (green).
    """

    def __init__(self, total_epoch, metric, train_stats, val_stats):
        """
        :param total_epoch:
        :param metric: string. Metric to judge a model. Must appear in args.
        :param train_stats: tuple of strings. Stats to record in the training phase.
        :param val_stats: tuple of strings. Stats to record in the validation phase.
        You can specify any stats here, like ('loss', 'acc', ...)
        """
        assert total_epoch > 0
        assert metric in val_stats, 'metric {} must be specified in val_stats.'.format(metric)
        self.total_epoch = total_epoch
        self.metric = metric
        self.current_epoch = -1
        self.stat = {}
        self.train_stats = train_stats
        self.val_stats = val_stats
        self.all_stats = set(train_stats) | set(val_stats)
        for stat in self.all_stats:
            self.stat[stat] = np.zeros((self.total_epoch, 2), dtype=np.float32)

    def update(self, epoch, is_train, meters):
        """
        :param epoch: current epoch.
        :param is_train: bool
        :param meters: dict of meters, {str(stat_name): AverageMeter}
        :return: if not is_train,
        return a bool indicating whether current metric is the highest in history in validation set.
        """
        self.current_epoch = epoch
        if is_train:
            for stat, meter in meters.items():
                self.stat[stat][epoch, 0] = meter.avg
            return
        else:
            for stat, meter in meters.items():
                self.stat[stat][epoch, 1] = meter.avg
            assert self.metric in meters.keys()
            metric_val = meters[self.metric].avg
            _, metric_highest = self.retrieve_stat(self.metric, 'max')
            return metric_val == metric_highest

    def specific_update(self, epoch, is_train, **kwargs):
        """
        update specific stats of interest. this is rarely used, as a complementary to standard meters update.
        :param epoch: current epoch.
        :param is_train: bool
        :param kwargs: {stat: value}
        :return:
        """
        if is_train:
            for stat, value in kwargs.items():
                self.stat[stat][epoch, 0] = value
        else:
            for stat, value in kwargs.items():
                self.stat[stat][epoch, 1] = value

    def retrieve_stat(self, stat, criterion):
        """
        retrieve the specified criterion of a stat.
        :param stat: string. some stat.
        :param criterion: string. 'max', 'min', 'mean', 'var' or 'std'
        :return:
        """
        if self.current_epoch == -1:
            return 0, 0
        target = self.stat[stat]
        method = getattr(np, criterion)
        results = method(target[:self.current_epoch + 1, :], axis=0)
        trn_result = results[0]
        val_result = results[1]
        return trn_result, val_result

    def plot_curve(self, save_path, stabilize=False):
        """
        :param save_path:
        :param stabilize: whether stabilize the plot. If True, the values in the specified subplot(s)
        are stabilized, i.e., extreme values are set 0. Typically, this function is used when the stat
        is extreme in the very beginning; otherwise in the plot the details might be hidden because
        the scale of the y-axis is determined by the extreme values.

        Note that this argument can be specified either as a bool or a list which must have the same
        size as the stat.
        :return:
        """
        fig = plt.figure(figsize=(17, 6))
        x_axis = np.array([i for i in range(self.current_epoch + 1)])
        if type(stabilize) == bool:
            stabilize = [stabilize for i in range(len(self.stat))]
        else:
            assert len(stabilize) == len(self.stat)

        for i, stat in enumerate(self.stat):
            i += 1
            ax = fig.add_subplot(1, len(self.stat), i)
            y_axis = self.stat[stat][:self.current_epoch + 1, 0].copy()
            if stabilize[i-1]:
                y_axis[np.abs(y_axis) > 10 * np.mean(np.abs(y_axis))] = 0
            ax.plot(x_axis, y_axis, color='b', label='train')
            y_axis = self.stat[stat][:self.current_epoch + 1, 1]
            if stabilize[i-1]:
                y_axis[np.abs(y_axis) > 10 * np.mean(np.abs(y_axis))] = 0
            ax.plot(x_axis, y_axis, color='g', label='val')
            plt.xlabel('training epoch')
            plt.xlim(0, self.current_epoch + 1)
            ax.grid()
            ax.set_title(stat)

        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')
            print ('---- save learning curve into {}'.format(save_path))
        plt.close(fig)


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime(time.time())))
    return string


def extract_features(loader, model, index_feature=1, require_views=True):
    """
    extract features for the given loader using the given model
    :param loader: must return (imgs, labels, views)
    :param model: returns a tuple, containing the feature
    :param index_feature: in the tuple returned by model, the index of the feature
    :param require_views: if True, also return view information
    :return: features, labels, (if required) views, all as n-by-d numpy array
    """
    # switch to evaluate mode
    model.eval()

    labels = torch.zeros((len(loader.dataset),))
    views = torch.zeros((len(loader.dataset),))

    idx = 0
    assert loader.dataset.require_views == require_views, 'require_views not consistent in loader and specified option'
    for i, data in enumerate(loader):
        imgs = data[0].cuda()
        label_batch = data[1]
        imgs_var = torch.autograd.Variable(imgs, volatile=True)
        output_tuple = model(imgs_var)
        feature_batch = output_tuple[index_feature]
        feature_batch = feature_batch.data.cpu()

        if i == 0:
            feature_dim = feature_batch.shape[1]
            features = torch.zeros((len(loader.dataset), feature_dim))

        batch_size = label_batch.shape[0]
        features[idx: idx + batch_size, :] = feature_batch
        labels[idx: idx + batch_size] = label_batch
        if require_views:
            view_batch = data[2]
            views[idx: idx + batch_size] = view_batch
        idx += batch_size

    features_np = features.numpy()
    labels_np = labels.numpy()
    views_np = views.numpy()
    if require_views:
        return features_np, labels_np, views_np
    else:
        return features_np, labels_np


def create_stat_string(meters):
    stat_string = ''
    for stat, meter in meters.items():
        stat_string += '{} {:.3f}   '.format(stat, meter.avg)
    return stat_string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def eval_cmc_map(dist, gallery_labels, probe_labels, gallery_views=None, probe_views=None):
    """
    I shall note that the MAP evaluated by this function is different from Zheng Liang's code.
    basically, this one is lower.
    although I believe my code is the correct one (an example is, in the original paper which Market-1501 is published,
    the toy example is wrongly evaluated BOTH in Zheng's code and the paper. XD),
    you might want a higher performance XD.
    so, if you want to compare your result with the published ones, please use Zheng's code in MATLAB XD.
    :param dist: 2-d np array, shape=(num_gallery, num_probe), distance matrix.
    :param gallery_labels: np array, shape=(num_gallery,)
    :param probe_labels:
    :param gallery_views: np array, shape=(num_gallery,) if specified, for any probe image,
    the gallery correct matches from the same view are ignored.
    :param probe_views: must be specified if gallery_views are specified.
    :return:
    CMC: np array, shape=(num_gallery,). Measured by percentage
    MAP: np array, shape=(1,). Measured by percentage
    """
    is_view_sensitive = False
    num_gallery = gallery_labels.shape[0]
    num_probe = probe_labels.shape[0]
    if gallery_views is not None or probe_views is not None:
        assert gallery_views is not None and probe_views is not None, \
            'gallery_views and probe_views must be specified together. \n'
        is_view_sensitive = True
    cmc = np.zeros((num_gallery, num_probe))
    map = np.zeros((num_probe,))
    for i in range(num_probe):
        cmc_ = np.zeros((num_gallery,))
        dist_ = dist[:, i]
        probe_label = probe_labels[i]
        if is_view_sensitive:
            probe_view = probe_views[i]
            is_from_same_view = gallery_views == probe_view
            is_correct = gallery_labels == probe_label
            should_be_excluded = is_from_same_view & is_correct
            dist_ = dist_[~should_be_excluded]
        ranking_list = np.argsort(dist_)
        inference_list = gallery_labels[ranking_list]
        positions_correct_tuple = np.nonzero(probe_label == inference_list)
        positions_correct = positions_correct_tuple[0]
        pos_first_correct = positions_correct[0]
        cmc_[pos_first_correct:] = 1
        num_correct = np.arange(positions_correct.shape[0]) + 1  # [1, 2, 3, ..., n]
        map[i] = np.mean(num_correct.astype(float) / (positions_correct + 1))
        cmc[:, i] = cmc_

    CMC = np.mean(cmc, axis=1)
    MAP = np.mean(map)
    return CMC*100, MAP*100


def occupy_gpu_memory(gpu_id, reserved_memory=1000):
    """
    When you are annoyed by others who regularly put their experiments on the gpu that you have run yours
    on just because you have not taken up most gpu memory, you may want to use this function.
    My friends, please find some idle gpus; you can judge by calling gpustat or nvidia-smi and looking at the
    instant power and average power consumption (gpu temperature).
    :param gpu_id: int
    :param reserved_memory: int, measured in MB
    :return:
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'])
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    available_memory = gpu_memory_map[gpu_id]
    if available_memory < reserved_memory+1000:
        print('Gpu memory is mostly occupied (although maybe not by you)!')
        return None
    else:
        dim = int((available_memory - reserved_memory) * 1024 * 1024 * 8 / 32)
        x = torch.zeros(dim)
        x.pin_memory()
        print('Occupied {}MB extra gpu memory.'.format(available_memory - reserved_memory))
        return x.cuda()


def reset_state_dict(state_dict, model, *fixed_layers):
    """
    :param state_dict: to be modified
    :param model: must be initialized
    :param fixed_layers: must be in both state_dict and model.
    :return:
    """
    for k in state_dict.keys():
        for l in fixed_layers:
            if k.startswith(l):
                if k.endswith('bias'):
                    state_dict[k] = model.__getattr__(l).bias.data
                elif k.endswith('weight'):
                    state_dict[k] = model.__getattr__(l).weight.data
                elif k.endswith('running_mean'):
                    state_dict[k] = model.__getattr__(l).running_mean
                elif k.endswith('running_var'):
                    state_dict[k] = model.__getattr__(l).running_var
                else:
                    assert False, 'Not specified param type: {}'.format(k)
    return state_dict


def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        best_name = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, best_name)


def adjust_learning_rate(optimizer, init_lr, epoch, total_epoch, strategy, lr_list=None):
    """
    :param optimizer:
    :param init_lr: tuple of float, has len(param_groups) elements. each element corresponds to a param_group
    :param epoch: int, current epoch index
    :param total_epoch: int
    :param strategy: choices are: ['constant', 'resnet_style', 'cyclegan_style', 'specified', 'finetune_style'],
    'constant': keep learning rate unchanged through training
    'resnet_style': divide lr by 10 in total_epoch/2, by 100 in total_epoch*(3/4)
    'cyclegan_style': linearly decrease lr to 0 after total_epoch/2
    'finetune_style': divide lr by 10 in total_epoch*(3/4)
    'specified': according to the given lr_list
    :param lr_list: numpy array, shape=(n_groups, total_epoch), only required when strategy == 'specified'
    :return: new_lr, tuple, has the same shape as init_lr
    """

    n_group = len(init_lr)
    if strategy == 'constant':
        new_lr = init_lr
        return new_lr
    elif strategy == 'resnet_style':
        lr_list = np.ones((n_group, total_epoch), dtype=float)
        for i in range(n_group):
            lr_list[i, :] *= init_lr[i]
        factors = np.ones(total_epoch,)
        factors[int(total_epoch/2):] *= 0.1
        factors[int(3*total_epoch/4):] *= 0.1
        lr_list *= factors
        new_lr = lr_list[:, epoch]
    elif strategy == 'finetune_style':
        lr_list = np.ones((n_group, total_epoch), dtype=float)
        for i in range(n_group):
            lr_list[i, :] *= init_lr[i]
        factors = np.ones(total_epoch,)
        factors[int(total_epoch*3/4):] *= 0.1
        lr_list *= factors
        new_lr = lr_list[:, epoch]
    elif strategy == 'cyclegan_style':
        lr_list = np.ones((n_group, total_epoch), dtype=float)
        for i in range(n_group):
            lr_list[i, :] *= init_lr[i]
        factors = np.ones(total_epoch,)
        n_elements = len(factors[int(total_epoch/2):])
        factors[int(total_epoch / 2):] = np.linspace(1, 0, n_elements)
        lr_list *= factors
        new_lr = lr_list[:, epoch]
    elif strategy == 'specified':
        assert lr_list is not None, 'if strategy is "specified", must provide lr_list'
        new_lr = lr_list[:, epoch]
    else:
        assert False, 'unknown strategy: {}'.format(strategy)

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = new_lr[i]
    return tuple(new_lr)


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


def partition_params(module, strategy, *desired_modules):
    """
    partition params into desired part and the residual
    :param module:
    :param strategy: choices are: ['bn', 'specified'].
    'bn': desired_params = bn_params
    'specified': desired_params = all params within desired_modules
    :param desired_modules: tuple of strings, each corresponds to a specific module
    :return: two iterators
    """
    if strategy == 'bn':
        desired_params_set = set()
        for m in module.modules():
            if (isinstance(m, torch.nn.BatchNorm1d) or
                    isinstance(m, torch.nn.BatchNorm2d) or
                    isinstance(m, torch.nn.BatchNorm3d)):
                desired_params_set.update(set(m.parameters()))
    elif strategy == 'specified':
        desired_params_set = set()
        for module_name in desired_modules:
            sub_module = module.__getattr__(module_name)
            for m in sub_module.modules():
                desired_params_set.update(set(m.parameters()))
    else:
        assert False, 'unknown strategy: {}'.format(strategy)
    all_params_set = set(module.parameters())
    other_params_set = all_params_set.difference(desired_params_set)
    desired_params = (p for p in desired_params_set)
    other_params = (p for p in other_params_set)
    return desired_params, other_params


if __name__ == '__main__':

    opts = TransOptions()
    args = opts.parse()
    logger = Logger(args.save_path)
    opts.print_options(logger)
    metric = 'acc'
    stats = (metric, 'loss')
    recorder = Recorder(2, metric, stats, stats)
    recorder.specific_update(epoch=0, is_train=True, acc=1, loss=1)
    recorder.specific_update(epoch=0, is_train=False, acc=2, loss=2)
    recorder.specific_update(epoch=1, is_train=True, acc=3, loss=3)
    recorder.current_epoch = 1
    trn, val = recorder.retrieve_stat(metric, 'max')
    print(trn)
    print(val)
    pass

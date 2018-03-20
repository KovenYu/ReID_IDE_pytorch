import os, sys, time
import numpy as np
import matplotlib
import subprocess
import torch
matplotlib.use('agg')
import matplotlib.pyplot as plt


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

    def __init__(self, total_epoch, metric, *args):
        """
        :param total_epoch:
        :param metric: string. Metric to judge a model. Must appear in args.
        :param args: string. Stats to record.
        You can specify any stats here, like ('loss', 'acc', ...)
        """
        assert total_epoch > 0
        assert metric in args, 'metric {} must be specified in args.'.format(metric)
        self.total_epoch = total_epoch
        self.metric = metric
        self.current_epoch = 0
        self.stat = {}
        for stat in args:
            self.stat[stat] = np.zeros((self.total_epoch, 2), dtype=np.float32)

    def update(self, epoch, is_train, **kwargs):
        """
        :param epoch: current epoch.
        :param is_train: bool
        :param kwargs: {stat: value}
        :return: if not is_train and metric in the kwargs,
        return a bool indicating whether current metric is the highest in history in validation set.
        """
        if is_train:
            self.current_epoch = epoch + 1
            for stat, value in kwargs.items():
                current_stat = self.stat[stat]
                current_stat[epoch, 0] = value
            return
        else:
            epoch = self.current_epoch - 1
            for stat, value in kwargs.items():
                current_stat = self.stat[stat]
                current_stat[epoch, 1] = value
            if self.metric in kwargs:
                metric_val = kwargs[self.metric]
                _, metric_highest = self.retrieve_stat(self.metric, 'max')
                return metric_val == metric_highest

    def retrieve_stat(self, stat, criterion):
        """
        retrieve the specified criterion of a stat.
        :param stat: string. some stat.
        :param criterion: string. 'max', 'min', 'mean', 'var' or 'std'
        :return:
        """
        if self.current_epoch == 0:
            return 0, 0
        target = self.stat[stat]
        method = getattr(np, criterion)
        results = method(target[:self.current_epoch, :], axis=0)
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
        x_axis = np.array([i for i in range(self.current_epoch)])
        if type(stabilize) == bool:
            stabilize = [stabilize for i in range(len(self.stat))]
        else:
            assert len(stabilize) == len(self.stat)

        for i, stat in enumerate(self.stat):
            i += 1
            ax = fig.add_subplot(1, len(self.stat), i)
            y_axis = self.stat[stat][:self.current_epoch, 0].copy()
            if stabilize[i-1]:
                y_axis[np.abs(y_axis) > 10 * np.mean(np.abs(y_axis))] = 0
            ax.plot(x_axis, y_axis, color='b', label='train')
            y_axis = self.stat[stat][:self.current_epoch, 1]
            if stabilize[i-1]:
                y_axis[np.abs(y_axis) > 10 * np.mean(np.abs(y_axis))] = 0
            ax.plot(x_axis, y_axis, color='g', label='val')
            plt.xlabel('training epoch')
            plt.xlim(0, self.current_epoch)
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


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def eval_cmc_map(dist, gallery_labels, probe_labels, gallery_views=None, probe_views=None):
    """
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


def occupy_gpu_memory(gpu_id):
    """
    When you are annoyed by others who regularly put their experiments on the gpu that you have run yours
    on just because you have not taken up most gpu memory, you may want to use this function.
    My friends, please find some idle gpus; you can judge by calling gpustat or nvidia-smi and looking at the
    instant power and average power consumption (gpu temperature).
    :param gpu_id: int
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
    if available_memory < 3000:
        print('Gpu memory is mostly occupied (although maybe not by you)!')
        return None
    else:
        dim = int((available_memory - 1000) * 1024 * 1024 * 8 / 32)
        x = torch.zeros(dim)
        x.pin_memory()
        print('Occupied {}MB extra gpu memory.'.format(available_memory - 1000))
        return x.cuda()


def main():
    metric = 'acc'
    stats = (metric, 'loss')
    recorder = Recorder(20, metric, *stats)
    recorder.update(epoch=0, is_train=True, acc=1, loss=1)
    recorder.update(epoch=0, is_train=False, acc=2, loss=2)
    recorder.update(epoch=1, is_train=True, acc=3, loss=3)
    is_best = recorder.update(epoch=1, is_train=False, acc=4, loss=4)
    trn, val = recorder.retrieve_stat(metric, 'max')
    print(trn)
    print(val)
    print(is_best)
    pass


if __name__ == '__main__':
    main()

import os
import os.path as osp
from datetime import datetime
import shutil
from tensorboardX import SummaryWriter
import torch
#from utils import write_param


def _check_mk_path(path):
    if not osp.exists(path):
        os.makedirs(path)
import numpy as np
from functools import partial

def write_param(net, writer, epoch, lp=1):
    def _write_single_param(writer, key, param, lp, epoch):
        if param is not None:
            view_channels = param.view(param.size(0), -1)
            mags = lp(view_channels, dim=1)
            writer.add_histogram(key, mags.clone().cpu().data.numpy(), epoch)
            return mags.clone().cpu().data.numpy()
        return np.array([])
    lp_magnitude = partial(torch.norm, p=lp)
    mag = np.array([])
    for key, module in net.named_modules():
        if isinstance(module, nn.modules.conv._ConvNd):
            f_mags = _write_single_param(writer, 'CONV/' + key + '.weight', module.weight, lp_magnitude, epoch)
            mag = np.append(mag, f_mags)
            f_mags = _write_single_param(writer, 'CONV/' + key + '.bias', module.bias, lp_magnitude, epoch)
            mag = np.append(mag, f_mags)
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            _write_single_param(writer, 'BN/' + key + '.weight', module.weight, lp_magnitude, epoch)
            _write_single_param(writer, 'BN/' + key + '.bias', module.bias, lp_magnitude, epoch)
    writer.add_histogram('all-param', mag, epoch)

class Reporter:
    def __init__(self, log_dir,exp_name):
        now = datetime.now().strftime("-%Y-%m-%d-%H:%M:%S")

        self.log_dir = osp.join(log_dir, exp_name + now)
        _check_mk_path(self.log_dir)

        self.writer = SummaryWriter(self.log_dir)

        self.ckpt_log_dir = osp.join(self.log_dir, "checkpoints")
        _check_mk_path(self.ckpt_log_dir)

        self.config_log_dir = osp.join(self.log_dir, "config")
        _check_mk_path(self.config_log_dir)

    def log_config(self, path):
        target = osp.join(self.config_log_dir, path.split("/")[-1])
        shutil.copyfile(path, target)

    def get_writer(self):
        return self.writer

    def log_metric(self, key, value, step):
        self.writer.add_scalar("data/" + key, value, step)

    def log_text(self, msg):
        print(msg)
    
    def log_param(self, model, step, lp=1):
        write_param(model, self.writer, step, lp=lp)

    def save_checkpoint(self, state_dict, ckpt_name, epoch=0):
        checkpoint = {"state_dict": state_dict, "epoch": epoch}
        torch.save(checkpoint, osp.join(self.ckpt_log_dir, ckpt_name))


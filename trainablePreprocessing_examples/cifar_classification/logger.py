#   Copyright (c) 2020, Xilinx, Inc.
#   All rights reserved.
# 
#   Redistribution and use in source and binary forms, with or without 
#   modification, are permitted provided that the following conditions are met:
#
#   1.  Redistributions of source code must retain the above copyright notice, 
#       this list of conditions and the following disclaimer.
#
#   2.  Redistributions in binary form must reproduce the above copyright 
#       notice, this list of conditions and the following disclaimer in the 
#       documentation and/or other materials provided with the distribution.
#
#   3.  Neither the name of the copyright holder nor the names of its 
#       contributors may be used to endorse or promote products derived from 
#       this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
#   THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
#   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
#   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
#   OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
#   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
#   OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
#   ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import logging
import sys
import os
import csv
import git

from collections import OrderedDict

from tensorboardX import SummaryWriter

from brevitas.utils.logging import LogWeightBitWidth, LogActivationBitWidth

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TrainingEpochMeters(object):
    def __init__(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()


class EvalEpochMeters(object):
    def __init__(self):
        self.model_time = AverageMeter()
        self.loss_time = AverageMeter()
        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()


class Logger(object):

    def __init__(self, args, config, output_dir_path):
        self.args = args
        self.config = config
        self.output_dir_path = output_dir_path
        self.log = logging.getLogger('log')
        self.log.setLevel(logging.INFO)

        # Stout logging
        out_hdlr = logging.StreamHandler(sys.stdout)
        out_hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        out_hdlr.setLevel(logging.INFO)
        self.log.addHandler(out_hdlr)

        # Txt and tensorboard logging
        if not args.dry_run:
            file_hdlr = logging.FileHandler(os.path.join(self.output_dir_path, 'log.txt'))
            file_hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
            file_hdlr.setLevel(logging.INFO)
            self.log.addHandler(file_hdlr)
            self.log.propagate = False
            self.writer = SummaryWriter(self.output_dir_path)

    def info(self, arg):
        self.log.info(arg)

    def tb_scalar(self, name, scalar, iter):
        if not self.args.dry_run:
            self.writer.add_scalar(name, scalar, iter)

    def tb_histogram(self, name, tensor, iter):
        if not self.args.dry_run:
            self.writer.add_histogram(name, tensor, iter)

    def log_config(self):
        self.info(self.config)
        if not self.args.dry_run:
            self.writer.add_text('Config', str(self.config))
            with open(os.path.join(self.output_dir_path, 'config.json'), 'w') as fp:
                json.dump(self.config, fp, indent=4)
            if self.config.config_csv_path is not None:
                sorted_config = OrderedDict(sorted(self.config.items()))
                if os.path.isfile(self.config.config_csv_path):
                    with open(self.config.config_csv_path, 'a') as f:
                        w = csv.DictWriter(f, sorted_config.keys())
                        w.writerow(sorted_config)
                else:
                    with open(self.config.config_csv_path, 'w') as f:
                        w = csv.DictWriter(f, sorted_config.keys())
                        w.writeheader()
                        w.writerow(sorted_config)

    def log_cmd_args(self):
        self.info(self.args.__dict__)
        if not self.args.dry_run:
            self.writer.add_text('CMD Args', str(self.args))
            with open(os.path.join(self.output_dir_path, 'cmd_args.txt'), 'a') as fp:
                json.dump(self.args.__dict__, fp)


    def eval_batch_cli_log(self, epoch_meters, batch, tot_batches):
        self.info('Test: [{0}/{1}]\t'
                  'Model Time {model_time.val:.3f} ({model_time.avg:.3f})\t'
                  'Loss Time {loss_time.val:.3f} ({loss_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  .format(batch, tot_batches,
                          model_time=epoch_meters.model_time,
                          loss_time=epoch_meters.loss_time,
                          loss=epoch_meters.losses,
                          top1=epoch_meters.top1,
                          top5=epoch_meters.top5))

    def eval_epoch_tensorboard_log(self, epoch_meters, epoch):
        self.tb_scalar('Test/EpochLossAvg', epoch_meters.losses.avg, epoch)
        self.tb_scalar('Test/EpochTop1Avg', epoch_meters.top1.avg, epoch)
        self.tb_scalar('Test/EpochTop5Avg', epoch_meters.top5.avg, epoch)

    def training_batch_cli_log(self, epoch_meters, epoch, batch, tot_batches):
        self.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                         .format(epoch, batch, tot_batches,
                                 batch_time=epoch_meters.batch_time,
                                 data_time=epoch_meters.data_time,
                                 loss=epoch_meters.losses,
                                 top1=epoch_meters.top1,
                                 top5=epoch_meters.top5))

    def training_batch_tensorboard_log(self, epoch_meters, niter):
        self.tb_scalar('Train/AvgLoss', epoch_meters.losses.avg, niter)
        self.tb_scalar('Train/AvgPrec@1', epoch_meters.top1.avg, niter)
        self.tb_scalar('Train/AvgPrec@5', epoch_meters.top5.avg, niter)



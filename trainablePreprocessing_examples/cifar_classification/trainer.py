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

import shutil
import time
import random
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn.utils.prune as prune

from torch import nn
from torch.optim.lr_scheduler import MultiStepLR

from logger import *
from models import *
import cifar_dataloader
import utils


def get_model(config, num_classes, input_size):
    if config.network == "VGG11":
        return VGG('VGG11', num_classes, config)
    elif config.network == "VGG11_preproc":
        return VGG_Preproc(num_classes, config)
    elif config.network == 'CNV':
        return CNV(config=config,
                    num_classes=num_classes)
    else:
        raise Exception("Model not recognized")


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Trainer(object):
    def __init__(self, args, config):

        # Init arguments
        self.args = args
        self.config = config
        self.config.experiment_start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.config.experiment_name = self.experiment_name
        if not self.args.dry_run:
            self.checkpoints_dir_path = os.path.join(self.output_dir_path, 'checkpoints')
            self.setup_experiment_output()
        self.logger = Logger(args, config, self.output_dir_path)
        self.logger.log_config()

        # Randomness
        random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)

        # Datasets
        if config.dataset == 'CIFAR10' or config.dataset == 'CIFAR100':
            self.input_size = 32
            train_loader, test_loader, num_classes = cifar_dataloader.get_loaders(config.dataset,
                                                                                  args.datadir,
                                                                                  args.batch_size,
                                                                                  args.num_workers)
            self.train_loader = train_loader
            self.test_loader = test_loader
            self.num_classes = num_classes

        elif config.dataset == 'GTSRB':
            self.input_size = 32
            train_loader, val_loader, test_loader, num_classes = gtsrb_dataloader.cf_gtsrb.get_loaders(args.datadir,
                                                                                  args.batch_size,
                                                                                  args.num_workers)

            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader
            self.num_classes = num_classes

        else:
            raise Exception("Dataset not supported: {}".format(config.dataset))

        # Init starting values
        self.starting_epoch = 1
        self.best_val_acc = 0

        # Setup device
        if self.args.gpus is not None:
            self.args.gpus = [int(i) for i in self.args.gpus.split(',')]
            self.device = 'cuda:' + str(args.gpus[0])
            torch.backends.cudnn.benchmark = True
        else:
            self.device = 'cpu'
        self.device = torch.device(self.device)

        # Setup model
        model = get_model(self.config, self.num_classes, self.input_size)

        # Resume model, if any
        if args.resume:
            print('Loading model checkpoint at: {}'.format(args.resume))
            package = torch.load(args.resume, map_location=self.device)
            model_state_dict = package['state_dict']
            #model_state_dict = utils.state_dict_retrocompatibility(model_state_dict)
            model.load_state_dict(model_state_dict, strict=args.strict)

        if args.pruned_retrain:
            for name, module in model.named_modules():
                # prune 20% of connections in all 2D-conv layers
                if isinstance(module, torch.nn.Conv2d):
                    prune.ln_structured(module, name='weight', amount=0.7, n=1, dim=0)

        self.model = model.to(device=self.device)
        if self.args.gpus is not None and len(self.args.gpus) > 1:
            self.model = nn.DataParallel(self.model, self.args.gpus)

        #Loss function
        self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(device=self.device)

        # Init optimizer
        self.optimizer = self.model  # setter syntax

        # Resume optimizer, if any
        if args.resume and not args.evaluate and not args.pruned_retrain:
            self.logger.log.info("Loading optimizer checkpoint")
            if 'optim_dict' in package.keys():
                self.optimizer.load_state_dict(package['optim_dict'])
            if 'epoch' in package.keys():
                self.starting_epoch = package['epoch']

        # LR scheduler
        self.scheduler = self.optimizer  # setter syntax

        # Resume scheduler, if any
        if args.resume \
                and not args.evaluate \
                and self.scheduler is not None and 'epoch' in package.keys():
            self.scheduler.last_epoch = package['epoch'] - 1

        # Recap
        self.logger.log_cmd_args()

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def scheduler(self):
        return self._scheduler

    @optimizer.setter
    def optimizer(self, model):
        if self.config.optim == 'SGD':
            nesterov = False
            self._optimizer = optim.SGD([{'params': model.preproc_features.parameters(), 'lr': self.config.preproc_lr, 'weight_decay': self.config.preproc_wd},
                                         {'params': model.conv_features.parameters()},
                                         #{'params': model.linear_features.parameters()},
                                         {'params': model.classifier.parameters()}],
                                        lr=self.config.lr,
                                        momentum=self.config.momentum,
                                        weight_decay=self.config.weight_decay,
                                        nesterov=nesterov)
        else:
            raise Exception("Unrecognized optimizer {}".format(self.config.optim))

    @scheduler.setter
    def scheduler(self, optimizer):
        if self.config.scheduler == 'STEP':
            milestones =  [int(i) for i in self.config.milestones.split(',')]
            self._scheduler = MultiStepLR(optimizer=optimizer,
                                          milestones=milestones,
                                          gamma=0.1)
        elif self.config.scheduler == 'FIXED':
            self._scheduler = None
        else:
            raise Exception("Unrecognized scheduler {}".format(self.config.scheduler))

    @property
    def experiment_name(self):
        if self.args.resume and not self.args.pruned_retrain:
            checkpoints_path = os.path.dirname(self.args.resume)
            experiment_path, checkpoints_dir_name = os.path.split(checkpoints_path)
            assert(checkpoints_dir_name == 'checkpoints')
            name = os.path.basename(experiment_path)
            return name
        else:
            postfix = '{}_{}_{}'.format(self.config.network, self.config.dataset, self.config.experiment_start_time)
            if self.config.name_prefix is not None:
                return '{}_{}'.format(self.config.name_prefix, postfix)
            else:
                return postfix

    @property
    def output_dir_path(self):
        return os.path.join(self.args.experiments, self.experiment_name)

    def setup_experiment_output(self):
        if not self.args.resume or self.args.resume and self.args.pruned_retrain:
            os.mkdir(self.output_dir_path)
            os.mkdir(self.checkpoints_dir_path)

    def checkpoint(self, epoch, best=False):
        file_path = os.path.join(self.checkpoints_dir_path, 'epoch_{}.tar'.format(epoch))
        self.logger.info("Saving checkpoint model to {}".format(file_path))
        torch.save({
            'state_dict': self.model.state_dict(),
            'optim_dict': self.optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_val_acc': self.best_val_acc,
            'config': self.config,
        }, file_path)
        if best:
            shutil.copyfile(file_path, os.path.join(self.checkpoints_dir_path, "best.tar"))

    def compute_loss(self, output, target):
        losses = []
        loss = self.criterion(output, target)
        losses.append(loss)

        loss = sum(losses)
        return loss, output

    def train_model(self):

        # training starts
        if self.config.detect_nan:
            torch.autograd.set_detect_anomaly(True)

        for epoch in range(self.starting_epoch, self.config.epochs):
            # Epoch starts

            # Set current mode to training
            self.model.train()
            self.criterion.train()

            # Init metrics
            epoch_meters = TrainingEpochMeters()
            start_data_loading = time.time()

            # Set the learning rate at the beginning
            if self.scheduler is not None:
                if self.scheduler == 'STEP':
                    self.scheduler.step(epoch)

            for i, data in enumerate(self.train_loader):

                (input, target) = data

                input = input.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                # measure data loading time
                epoch_meters.data_time.update(time.time() - start_data_loading)

                # Training batch starts
                start_batch = time.time()
                output = self.model(input)

                loss, output = self.compute_loss(output, target)

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                epoch_meters.batch_time.update(time.time() - start_batch)

                if i % int(self.args.log_freq) == 0 or i == len(self.train_loader) - 1:
                    niter = epoch * len(self.train_loader) + i
                    prec1, prec5 = accuracy(output.detach(), target, topk=(1, 5))
                    epoch_meters.losses.update(loss.item(), input.size(0))
                    epoch_meters.top1.update(prec1.item(), input.size(0))
                    epoch_meters.top5.update(prec5.item(), input.size(0))
                    self.logger.training_batch_cli_log(epoch_meters, epoch, i, len(self.train_loader))
                    self.logger.training_batch_tensorboard_log(epoch_meters, niter)

                # training batch ends
                start_data_loading = time.time()

            # training epoch ends

            # Perform eval
            with torch.no_grad():
                top1avg = self.eval_model(epoch)

            # checkpoint
            if top1avg > self.best_val_acc and not self.args.dry_run:
                self.best_val_acc = top1avg
                if not self.args.no_checkpoints:
                    self.checkpoint(epoch, best=True)

            '''if (epoch % int(self.config.checkpoint_freq) == 0 or epoch == self.config.epochs - 1) \
                    and not self.args.dry_run \
                    and not self.args.no_checkpoints:
                self.checkpoint(epoch)'''

            # keep track of best accuracy
            if not self.args.dry_run:
                niter = epoch * len(self.train_loader) + i
                self.logger.writer.add_scalar('Test/BestTop1', self.best_val_acc, niter)

            # Epoch ends

        # training ends
        if not self.args.dry_run and not self.args.no_checkpoints:
            return os.path.join(self.checkpoints_dir_path, "best.tar")

    def eval_model(self, epoch=None):
        eval_meters = EvalEpochMeters()

        # switch to evaluate mode
        self.model.eval()
        self.criterion.eval()


        for i, data in enumerate(self.test_loader):

            end = time.time()
            (input, target) = data

            #print(input.device)

            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            # compute output
            output = self.model(input)

            # measure model elapsed time
            eval_meters.model_time.update(time.time() - end)
            end = time.time()

            #compute loss
            loss, output = self.compute_loss(output, target)
            eval_meters.loss_time.update(time.time() - end)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            eval_meters.losses.update(loss.item(), input.size(0))
            eval_meters.top1.update(prec1.item(), input.size(0))
            eval_meters.top5.update(prec5.item(), input.size(0))

            #Eval batch ends
            self.logger.eval_batch_cli_log(eval_meters, i, len(self.test_loader))

        # Eval epoch ends
        self.logger.eval_epoch_tensorboard_log(eval_meters, epoch)

        return eval_meters.top1.avg

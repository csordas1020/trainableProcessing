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

import argparse
import os
import json

import torch

from trainer import Trainer

from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.stats import StatsOp
from brevitas.core.scaling import ScalingImplType
from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType

parser = argparse.ArgumentParser(description="PyTorch CIFAR10/100 Training")

# Util method to add mutually exclusive boolean
def add_bool_arg(parser, name, default):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=name, action="store_true")
    group.add_argument("--no_" + name, dest=name, action="store_false")
    parser.set_defaults(**{name: default})


# Util method to pass None as a string and be recognized as None value
def none_or_str(value):
    if value == "None":
        return None
    return value


# I/O
parser.add_argument("--datadir", help="Read config from file", default="./data/")
parser.add_argument("--config", help="Config file path. Command line will be ignored", default=None)
parser.add_argument("--dataset", help="Dataset", type=none_or_str, default="CIFAR10")
parser.add_argument("--log_freq", help="Print every n-th batch", default=1)
parser.add_argument("--checkpoint_freq", help="Checkpoint every n-th epoch", default=1)
parser.add_argument("--no_checkpoints", action="store_true", help="Force disable all checkpoints")
parser.add_argument("--experiments", default="./experiments", help="Path to experiments folder")
parser.add_argument("--dry_run", action="store_true", help="Disable output files generation")
parser.add_argument("--config_csv_path", type=none_or_str, default=None, help="Append config to .csv passed as path")
parser.add_argument("--name_prefix", type=none_or_str, default=None, help="Prefix to add to the name")

# Execution modes
parser.add_argument("--resume", type=none_or_str, help="resume from checkpoint")
parser.add_argument("--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set")
add_bool_arg(parser, "strict", default=True)

# Compute resources
parser.add_argument("--num_workers", default=4, type=int, help="Number of workers")
parser.add_argument("--gpus", type=none_or_str, default=None, help="Comma separated GPUs")

# Optimizer hyperparams
parser.add_argument("--batch_size", default=128, type=int, help="batch size")
parser.add_argument("--optim", default="SGD", type=none_or_str)
parser.add_argument("--lr", default=0.1, type=float, help="Learning rate")
parser.add_argument("--scheduler", default="STEP", type=none_or_str, help="LR Scheduler")
parser.add_argument("--milestones", type=none_or_str, default='150,250,350', help="Scheduler milestones")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum")
parser.add_argument("--weight_decay", default=5e-5, type=float, help="Weight decay")
parser.add_argument("--epochs", default=400, type=int, help="Number of epochs")
parser.add_argument("--random_seed", default=123456, type=int, help="Random seed")
add_bool_arg(parser, "detect_nan", default=False)

# Neural network Architecture
parser.add_argument("--network", default="VGG11", type=none_or_str, help="neural network")
parser.add_argument("--pretrained_path", type=none_or_str, default=None)
add_bool_arg(parser, "download_pretrained", default=False)

# Preprocessing
parser.add_argument("--input_bit_width", type=int, default=8)
parser.add_argument("--preproc_mode", type=none_or_str, default=None)
parser.add_argument("--preproc_lr", type=float, default=0.1)
parser.add_argument("--preproc_wd", type=float, default=0.0)
add_bool_arg(parser, "grayscale", default=False)
add_bool_arg(parser, "colortrans", default=False)


# Weight quantization
parser.add_argument("--weight_bit_width", type=int, default=None)

# Activation quantization
parser.add_argument("--activation_bit_width", type=int, default=None)

# Pytorch precision
torch.set_printoptions(precision=10)

class objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


if __name__ == "__main__":
    args = parser.parse_args()

    config = {}

    # Set relative paths relative to main.py
    path_args = ["config", "datadir", "resume", "experiments"]
    for path_arg in path_args:
        path = getattr(args, path_arg)
        if path is not None and not os.path.isabs(path):
            abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
            setattr(args, path_arg, abs_path)

    # Read config from checkpoint
    if args.config:
        with open(args.config) as handle:
            config = json.loads(handle.read())
    # else from command line
    else:
        for arg in args.__dict__.keys():
            config[arg] = getattr(args, arg)

    # Access config as an object
    config = objdict(config)

    # Avoid creating new folders etc.
    if args.evaluate:
        args.dry_run = True

    # Init trainer
    trainer = Trainer(args, config)

    if args.evaluate:
        with torch.no_grad():
            trainer.eval_model()
    else:
        trainer.train_model()

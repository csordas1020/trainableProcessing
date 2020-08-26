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

import torch
from torch.nn import ModuleList, Module
from brevitas_examples.imagenet_classification import model_with_cfg
from brevitas_examples.imagenet_classification.models.mobilenetv1 import MobileNet

from trainablePreprocessing.preprocessing import TrainedDithering, FixedDithering
from trainablePreprocessing.preprocessing import Quantization, ColorSpaceTransformation


class Normalize(Module):
    def __init__(self, mean, std, channels):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
        self.channels = channels

    def forward(self, x):
        channels = self.channels
        x = x - torch.tensor(self.mean, device=x.device).reshape(1, channels, 1, 1)
        x = x / torch.tensor(self.std, device=x.device).reshape(1, channels, 1, 1)
        return x


class PreprocMobileNetV1(MobileNet):

    def __init__(
            self,
            channels,
            first_stage_stride,
            input_norm_mean,
            input_norm_std,
            input_channels,
            first_layer_bit_width,
            bit_width,
            preproc_mode='quant', colortrans=False, input_bit_width=8):
        super(PreprocMobileNetV1, self).__init__(
            channels,
            first_stage_stride,
            first_layer_bit_width,
            bit_width)
        self.preproc_features = ModuleList()
        if colortrans:
            self.preproc_features.append(ColorSpaceTransformation(3, 3))
        if preproc_mode == 'trained_dithering':
            self.preproc_features.append(TrainedDithering(input_bit_width, 3, 3))
        elif preproc_mode == 'fixed_dithering':
            self.preproc_features.append(FixedDithering(input_bit_width, 3))
        elif preproc_mode == 'quant':
            self.preproc_features.append(Quantization(input_bit_width))
        self.preproc_features.append(Normalize(input_norm_mean, input_norm_std, input_channels))

    def forward(self, x):
        for mod in self.preproc_features:
            x = mod(x)
        x = super(PreprocMobileNetV1, self).forward(x)
        return x


def _preproc_mobilenet_v1(pretrained, first_layer_bit_width, bit_width, **kwargs):
    first_stage_stride = False
    channels = [[32], [64], [128, 128], [256, 256], [512, 512, 512, 512, 512, 512], [1024, 1024]]
    orig_quant_mobilenet_v1_4b, cfg = model_with_cfg('quant_mobilenet_v1_4b', pretrained=pretrained)
    width_scale = float(cfg.get('MODEL', 'WIDTH_SCALE'))
    input_channels = 3
    mean = [float(cfg.get('PREPROCESS', 'MEAN_0')),
            float(cfg.get('PREPROCESS', 'MEAN_1')),
            float(cfg.get('PREPROCESS', 'MEAN_2'))]
    std = [float(cfg.get('PREPROCESS', 'STD_0')),
           float(cfg.get('PREPROCESS', 'STD_1')),
           float(cfg.get('PREPROCESS', 'STD_2'))]
    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
    net = PreprocMobileNetV1(
        channels, first_stage_stride, mean, std, input_channels, first_layer_bit_width, bit_width, **kwargs)
    if pretrained:
        orig_state_dict = orig_quant_mobilenet_v1_4b.state_dict()
        net.load_state_dict(orig_state_dict, strict=False)
    return net


def preproc_mobilenet_v1_4b(pretrained=True, **kwargs):
    return _preproc_mobilenet_v1(pretrained, 4, 4, **kwargs)


def preproc_mobilenet_v1_3b(pretrained=True, **kwargs):
    return _preproc_mobilenet_v1(pretrained, 3, 3, **kwargs)
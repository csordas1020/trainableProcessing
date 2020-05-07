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

'''VGG11/13/16/19 in Pytorch.'''
import torch.nn as nn
from models.pre_quantized_layers import PreQuantizedReLU, PreQuantizedConv2d, PreQuantizedLinear
from preprocessing import TrainedDithering, FixedDithering, GammaRescaling, Quantization


cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']


class VGG(nn.Module):
    def __init__(self, vgg_name, n_class, config):
        super(VGG, self).__init__()
        self.preproc_features = nn.ModuleList()
        if config.preproc_mode == 'trained_dithering':
            if config.grayscale:
                self.preproc_features.append(TrainedDithering(config.input_bit_width, 3, 1))
            else:
                self.preproc_features.append(TrainedDithering(config.input_bit_width, 3, 3))
        elif config.preproc_mode == 'quant':
            self.preproc_features.append(Quantization(config.input_bit_width))
        self.grayscale = config.grayscale
        self.conv_features = self._make_layers(cfg, config)
        self.classifier = PreQuantizedLinear(512, n_class, bias=False, config=config)

    def forward(self, x):
        if self.grayscale:
            x = x[:,0,:,:].reshape(x.shape[0],1,32,32)
        out = 2.0 * x - 1.0
        for mod in self.preproc_features:
            out = mod(out)
        out = self.conv_features(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, config):
        layers = []

        if config.grayscale:
            in_channels = 1
        else:
            in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [PreQuantizedConv2d(in_channels, x,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False,
                                              config=config),
                           nn.BatchNorm2d(x),
                           PreQuantizedReLU(config)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
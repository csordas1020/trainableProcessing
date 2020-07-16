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

from torch.nn import Module, ModuleList, BatchNorm2d, MaxPool2d, BatchNorm1d, Sequential

from .common import get_quant_conv2d, get_quant_linear, get_act_quant, get_quant_type, get_stats_op

from trainablePreprocessing.preprocessing import TrainedDithering, FixedDithering, GammaRescaling, Quantization, ColorSpaceTransformation

from brevitas.core.quant import QuantType

# QuantConv2d configuration
CNV_OUT_CH_POOL = [(0, 64, False), (1, 64, True), (2, 128, False), (3, 128, True), (4,256, False), (5, 256, False)]


# Intermediate QuantLinear configuration
INTERMEDIATE_FC_PER_OUT_CH_SCALING = True
INTERMEDIATE_FC_FEATURES = [(256, 512), (512, 512)]


# Last QuantLinear configuration
LAST_FC_IN_FEATURES = 512

LAST_FC_PER_OUT_CH_SCALING = False

# MaxPool2d configuration
POOL_SIZE = 2

class CNV(Module):

    def __init__(self, config, num_classes=10):
        super(CNV, self).__init__()

        self.config = config
        weight_quant_type = get_quant_type(config.weight_bit_width)
        act_quant_type = get_quant_type(config.activation_bit_width)
        in_quant_type = get_quant_type(config.input_bit_width)
        stats_op = get_stats_op(weight_quant_type)

        self.preproc_features = ModuleList()
        self.conv_features = ModuleList()
        self.linear_features = ModuleList()

        if config.colortrans:
            self.preproc_features.append(ColorSpaceTransformation(3, 3))

        if config.preproc_mode == 'trained_dithering':
            self.preproc_features.append(TrainedDithering(config.input_bit_width, 3, 3))
        elif config.preproc_mode == 'fixed_dithering':
            self.preproc_features.append(FixedDithering(config.input_bit_width, 3))
        elif config.preproc_mode == 'quant':
            self.preproc_features.append(Quantization(config.input_bit_width))

        in_ch = 3
        for i, out_ch, is_pool_enabled in CNV_OUT_CH_POOL:
            self.conv_features.append(get_quant_conv2d(in_ch=in_ch,
                                                       out_ch=out_ch,
                                                       bit_width=config.weight_bit_width,
                                                       quant_type=weight_quant_type,
                                                       stats_op=stats_op))
            in_ch = out_ch
            if is_pool_enabled:
                self.conv_features.append(MaxPool2d(kernel_size=2))
            if i == 5:
                self.conv_features.append(Sequential())
            self.conv_features.append(BatchNorm2d(in_ch))
            self.conv_features.append(get_act_quant(config.activation_bit_width, act_quant_type))
            

        for in_features, out_features in INTERMEDIATE_FC_FEATURES:
            self.linear_features.append(get_quant_linear(in_features=in_features,
                                                         out_features=out_features,
                                                         per_out_ch_scaling=INTERMEDIATE_FC_PER_OUT_CH_SCALING,
                                                         bit_width=config.weight_bit_width,
                                                         quant_type=weight_quant_type,
                                                         stats_op=stats_op))
            self.linear_features.append(BatchNorm1d(out_features))
            self.linear_features.append(get_act_quant(config.activation_bit_width, act_quant_type))
        self.classifier = get_quant_linear(in_features=LAST_FC_IN_FEATURES,
                                   out_features=num_classes,
                                   per_out_ch_scaling=LAST_FC_PER_OUT_CH_SCALING,
                                   bit_width=config.weight_bit_width,
                                   quant_type=weight_quant_type,
                                   stats_op=stats_op)

    def forward(self, x):
        x = 2.0 * x - 1.0
        for mod in self.preproc_features:
            x = mod(x)
        for mod in self.conv_features:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.linear_features:
            x = mod(x)
        out = self.classifier(x)
        return out

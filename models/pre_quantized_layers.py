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

from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU
from brevitas.core.quant import QuantType



def PreQuantizedLinear(in_features,
                       out_features,
                       config,
                       bias=True):
    return QuantLinear(in_features,
                       out_features,
                       bias,
                       weight_quant_type=QuantType.INT,
                       weight_narrow_range=True,
                       weight_bit_width=config.weight_bit_width,
                       weight_scaling_per_output_channel=False)



def PreQuantizedConv2d(in_channels,
                       out_channels,
                       kernel_size,
                       config,
                       stride=1,
                       padding=0,
                       dilation=1,
                       groups=1,
                       bias=True):
    return QuantConv2d(in_channels=in_channels,
                       out_channels=out_channels,
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=padding,
                       dilation=dilation,
                       groups=groups,
                       bias=bias,
                       weight_quant_type=QuantType.INT,
                       weight_narrow_range=True,
                       weight_bit_width=config.weight_bit_width,
                       weight_scaling_per_output_channel=True)


def PreQuantizedReLU(config):
    return QuantReLU(bit_width=config.activation_bit_width,
                     max_val=6.0,
                     quant_type=QuantType.INT)
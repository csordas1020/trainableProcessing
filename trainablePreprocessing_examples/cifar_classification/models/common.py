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

from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.stats import StatsOp
from brevitas.nn import QuantConv2d, QuantHardTanh, QuantLinear
# Quant common
BIT_WIDTH_IMPL_TYPE = BitWidthImplType.CONST
SCALING_VALUE_TYPE = RestrictValueType.LOG_FP
SCALING_IMPL_TYPE = ScalingImplType.PARAMETER
NARROW_RANGE_ENABLED = True

# Weight quant common
STATS_OP = StatsOp.MEAN_LEARN_SIGMA_STD
BIAS_ENABLED = False
WEIGHT_SCALING_IMPL_TYPE = ScalingImplType.STATS
SIGMA = 0.001

# QuantHardTanh configuration
HARD_TANH_MIN = -1.0
HARD_TANH_MAX = 1.0
ACT_PER_OUT_CH_SCALING = False

# QuantConv2d configuration
KERNEL_SIZE = 3
CONV_PER_OUT_CH_SCALING = True


def get_stats_op(quant_type):
    if quant_type == QuantType.BINARY:
        return StatsOp.AVE
    else:
        return StatsOp.MAX


def get_quant_type(bit_width):
    if bit_width is None:
        return QuantType.FP
    elif bit_width == 1:
        return QuantType.BINARY
    else:
        return QuantType.INT


def get_act_quant(act_bit_width, act_quant_type):
    if act_quant_type == QuantType.INT:
        act_scaling_impl_type = ScalingImplType.PARAMETER
    else:
        act_scaling_impl_type = ScalingImplType.CONST
    return QuantHardTanh(quant_type=act_quant_type,
                         bit_width=act_bit_width,
                         bit_width_impl_type=BIT_WIDTH_IMPL_TYPE,
                         min_val=HARD_TANH_MIN,
                         max_val=HARD_TANH_MAX,
                         scaling_impl_type=act_scaling_impl_type,
                         restrict_scaling_type=RestrictValueType.LOG_FP,
                         scaling_per_channel=ACT_PER_OUT_CH_SCALING,
                         narrow_range=NARROW_RANGE_ENABLED)


def get_quant_linear(in_features, out_features, per_out_ch_scaling, bit_width, quant_type, stats_op):
    return QuantLinear(bias=BIAS_ENABLED,
                       in_features=in_features,
                       out_features=out_features,
                       weight_quant_type=quant_type,
                       weight_bit_width=bit_width,
                       weight_bit_width_impl_type=BIT_WIDTH_IMPL_TYPE,
                       weight_scaling_per_output_channel=per_out_ch_scaling,
                       weight_scaling_stats_op=stats_op,
                       weight_scaling_stats_sigma=SIGMA)


def get_quant_conv2d(in_ch, out_ch, bit_width, quant_type, stats_op):
    return QuantConv2d(in_channels=in_ch,
                       kernel_size=KERNEL_SIZE,
                       out_channels=out_ch,
                       weight_quant_type=quant_type,
                       weight_bit_width=bit_width,
                       weight_narrow_range=NARROW_RANGE_ENABLED,
                       weight_scaling_impl_type=WEIGHT_SCALING_IMPL_TYPE,
                       weight_scaling_stats_op=stats_op,
                       weight_scaling_stats_sigma=SIGMA,
                       weight_scaling_per_output_channel=CONV_PER_OUT_CH_SCALING,
                       weight_restrict_scaling_type=SCALING_VALUE_TYPE,
                       weight_bit_width_impl_type=BIT_WIDTH_IMPL_TYPE,
                       bias=BIAS_ENABLED)

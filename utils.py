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

from collections import OrderedDict

def state_dict_retrocompatibility(state_dict):
    new_state_dict = OrderedDict()

    weight_bit_width_offset_update = lambda key: key.replace('weight_quantization.tensor_quantization.bit_width_logic.bit_width_offset',
                                                      'weight_quant.tensor_quant.msb_clamp_bit_width_impl.bit_width_offset') \
        if key.endswith('bit_width_logic.bit_width_offset') else key

    act_bit_width_offset_update = lambda key: key.replace('tensor_quantization.bit_width_logic.bit_width_offset',
                                                      'activation_quant.fused_activation_quant.tensor_quant.msb_clamp_bit_width_impl.bit_width_offset') \
        if key.endswith('bit_width_logic.bit_width_offset') else key

    scaling_update = lambda key: key.replace('tensor_quantization.alphas_logic.value',
                                                      'activation_quant.fused_activation_quant.tensor_quant.scaling_impl.value') \
        if key.endswith('alphas_logic.value') else key

    for key, value in state_dict.items():
        new_key = weight_bit_width_offset_update(key)
        new_key = act_bit_width_offset_update(new_key)
        new_key = scaling_update(new_key)
        new_state_dict[new_key] = value
    return new_state_dict

def state_dict_update(state_dict, config):

    for key, value in state_dict.items():
        if config.override_bit_width:
            if key.endswith('bit_width_offset'):
                del state_dict[key]
    return state_dict
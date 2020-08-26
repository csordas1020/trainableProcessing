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

#!/bin/bash

input_bitwidths=(4 3 2 1)
preproc_bitwidths=(8 4 3 2 1)

# Baseline float params.
for b in ${input_bitwidths[@]}; do
    CUDA_VISIBLE_DEVICES=0 python main.py -a preproc_resnet18 --preproc_mode trained_dithering --input_bit_width ${b} -e --resume trained_dithering/${b}bit/model_best.pth.tar ../../imagenet_symlink | tee trained_dithering/${b}bit/model_best.log
done

# Shift-based params.
for b in ${input_bitwidths[@]}; do
    CUDA_VISIBLE_DEVICES=0 python main.py -a preproc_resnet18 --preproc_mode trained_dithering --input_bit_width ${b} -e --resume trained_dithering/${b}bit/model_best-shifted.pth.tar ../../imagenet_symlink | tee trained_dithering/${b}bit/model_best-shifted.log
done

for b in ${input_bitwidths[@]}; do
    for pb in ${preproc_bitwidths[@]}; do
        CUDA_VISIBLE_DEVICES=0 python main.py -a preproc_resnet18 --preproc_mode trained_dithering --input_bit_width ${b} -e --resume trained_dithering/${b}bit/model_best-${pb}bit.pth.tar ../../imagenet_symlink | tee trained_dithering/${b}bit/model_best-${pb}bit.log
    done
done


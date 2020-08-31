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

input_bitwidths=(2)

for b in ${input_bitwidths[@]}; do
    CUDA_VISIBLE_DEVICES=0 python preproc_image.py -a preproc_resnet18 --preproc-mode trained_dithering --input-bit-width ${b} --input-checkpoint trained_dithering/${b}bit/model_best.pth.tar --input-image ../../imagenet_symlink/val/n02086646/ILSVRC2012_val_00002988.JPEG --output-proc-image trained_dithering_val_00002988.jpeg --output-input-image input_val_00002988.jpeg
    CUDA_VISIBLE_DEVICES=0 python preproc_image.py -a preproc_resnet18 --preproc-mode quant --input-bit-width ${b} --input-checkpoint direct_quant/${b}bit/model_best.pth.tar --input-image ../../imagenet_symlink/val/n02086646/ILSVRC2012_val_00002988.JPEG --output-proc-image direct_quant_val_00002988.jpeg --output-input-image input_val_00002988.jpeg
    CUDA_VISIBLE_DEVICES=0 python preproc_image.py -a preproc_resnet18 --preproc-mode quant --input-bit-width ${b} --colortrans --input-checkpoint colour_trans/${b}bit/model_best.pth.tar --input-image ../../imagenet_symlink/val/n02086646/ILSVRC2012_val_00002988.JPEG --output-proc-image colour_trans_val_00002988.jpeg --output-input-image input_val_00002988.jpeg
    CUDA_VISIBLE_DEVICES=0 python preproc_image.py -a preproc_resnet18 --preproc-mode fixed_dithering --input-bit-width ${b} --input-checkpoint fixed_dithering/${b}bit/model_best.pth.tar --input-image ../../imagenet_symlink/val/n02086646/ILSVRC2012_val_00002988.JPEG --output-proc-image fixed_dithering_val_00002988.jpeg --output-input-image input_val_00002988.jpeg
done

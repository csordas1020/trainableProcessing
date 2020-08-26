#!/bin/bash

input_bitwidths=(2)

for b in ${input_bitwidths[@]}; do
    CUDA_VISIBLE_DEVICES=0 python preproc_image.py -a preproc_resnet18 --preproc-mode trained_dithering --input-bit-width ${b} --input-checkpoint trained_dithering/${b}bit/model_best.pth.tar --input-image ../../imagenet_symlink/val/n02086646/ILSVRC2012_val_00002988.JPEG --output-proc-image trained_dithering_val_00002988.jpeg --output-input-image input_val_00002988.jpeg
    CUDA_VISIBLE_DEVICES=0 python preproc_image.py -a preproc_resnet18 --preproc-mode quant --input-bit-width ${b} --input-checkpoint direct_quant/${b}bit/model_best.pth.tar --input-image ../../imagenet_symlink/val/n02086646/ILSVRC2012_val_00002988.JPEG --output-proc-image direct_quant_val_00002988.jpeg --output-input-image input_val_00002988.jpeg
    CUDA_VISIBLE_DEVICES=0 python preproc_image.py -a preproc_resnet18 --preproc-mode quant --input-bit-width ${b} --colortrans --input-checkpoint colour_trans/${b}bit/model_best.pth.tar --input-image ../../imagenet_symlink/val/n02086646/ILSVRC2012_val_00002988.JPEG --output-proc-image colour_trans_val_00002988.jpeg --output-input-image input_val_00002988.jpeg
    CUDA_VISIBLE_DEVICES=0 python preproc_image.py -a preproc_resnet18 --preproc-mode fixed_dithering --input-bit-width ${b} --input-checkpoint fixed_dithering/${b}bit/model_best.pth.tar --input-image ../../imagenet_symlink/val/n02086646/ILSVRC2012_val_00002988.JPEG --output-proc-image fixed_dithering_val_00002988.jpeg --output-input-image input_val_00002988.jpeg
done

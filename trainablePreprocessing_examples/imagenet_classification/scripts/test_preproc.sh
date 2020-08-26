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


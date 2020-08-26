#!/bin/bash

input_bitwidths=(4 3 2 1)
preproc_bitwidths=(8 4 3 2 1)

for b in ${input_bitwidths[@]}; do
    python quantize_preproc_kernels.py --input-checkpoint trained_dithering/${b}bit/model_best.pth.tar --output-checkpoint trained_dithering/${b}bit/model_best-shifted.pth.tar --shift-based --verbose
done

for b in ${input_bitwidths[@]}; do
    for pb in ${preproc_bitwidths[@]}; do
        python quantize_preproc_kernels.py --input-checkpoint trained_dithering/${b}bit/model_best.pth.tar --output-checkpoint trained_dithering/${b}bit/model_best-${pb}bit.pth.tar --bit-width ${pb} --verbose
    done
done


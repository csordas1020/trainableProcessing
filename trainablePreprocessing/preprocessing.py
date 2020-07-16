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


import torch
import torch.autograd
import torch.nn as nn
import preproc_cpp

CUDA_EXT = True

GAMMA_EPS = 2**(-16)

def quant(x, bit_width):
    y = (x * (2 ** (bit_width)-1)).round() / (2 ** (bit_width) - 1)
    return y

class Quantization(nn.Module):
    def __init__(self, bit_width):
        super(Quantization, self).__init__()
        self.bit_width = bit_width

    def forward(self, input):
        input = input.clamp(-1.0, 1.0)
        input = (input + 1.0) / 2.0
        output = input + (quant(input, self.bit_width)-input).detach()
        output = output * 2.0 - 1.0
        return output

class DitheringFunction2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, kernel, bit_width):
        mini_batch_size = input.shape[0]
        channels = input.shape[1]
        rows = input.shape[2]
        cols = input.shape[3]
        krows = kernel.shape[1]
        kcols = kernel.shape[2]
        rpad = int((krows - 1) / 2)
        cpad = int((kcols - 1) / 2)

        img_dit = torch.zeros((mini_batch_size, channels, rows + 2 * rpad, cols + 2 * cpad))
        img_dit = img_dit.to(input.device)
        img_dit[:, :, 1:rows + 1, 1:cols + 1] = input.clone()

        for r in range(rpad, rows + rpad):
            for c in range(cpad, cols + cpad):
                prods = (img_dit[:, :, r - rpad:r + rpad + 1, c - cpad:c + cpad + 1] -
                         quant(img_dit[:, :, r - rpad:r + rpad + 1, c - cpad:c + cpad + 1], bit_width)) * kernel.clamp_min(0.0)
                img_dit[:, :, r, c] = (img_dit[:, :, r, c] + prods.sum((2, 3))).clamp(0.0,1.0)
        output = img_dit[:, :, rpad:rows + rpad, cpad:cols + cpad]
        output = quant(output, bit_width)

        ctx.save_for_backward(img_dit, kernel)
        ctx.rows = rows
        ctx.cols = cols
        ctx.krows = krows
        ctx.kcols = kcols
        ctx.bit_width = bit_width
        ctx.channels = channels

        return output

    @staticmethod
    def backward(ctx, grad_output):
        img_dit, kernel = ctx.saved_tensors
        err = img_dit - quant(img_dit, ctx.bit_width)

        grad_kernel = torch.zeros((ctx.channels, ctx.krows, ctx.kcols))
        grad_kernel = grad_kernel.to(kernel.device)
        for r in range(ctx.krows):
            for c in range(ctx.kcols):
                if r < int((ctx.krows - 1) / 2) or (r == int((ctx.krows - 1) / 2) and c < int((ctx.kcols - 1) / 2)):
                    grad_kernel[:, r, c] = (grad_output * err[:, :, r:ctx.rows + r, c:ctx.cols + c]).sum((0, 2, 3))
        return grad_output, grad_kernel, None

class FixedDithering(nn.Module):

    def __init__(self, bit_width, in_ch):
        super(FixedDithering, self).__init__()
        self.bit_width = bit_width
        self.in_ch = in_ch

    def forward(self, input):
        if CUDA_EXT:
            dither_ker = torch.tensor([[0.0,  0.0,  0.0 ],
                                       [0.0,  0.0,  7/16],
                                       [3/16, 5/16, 1/16]], dtype=input.dtype, device=input.device).expand(self.in_ch, 3, 3)
        else:
            dither_ker = torch.tensor([[1/16,  5/16,  3/16 ],
                                       [7/16,  0.0,  0.0],
                                       [0.0, 0.0, 0.0]], dtype=input.dtype, device=input.device).expand(self.in_ch, 3, 3)
        input = input.clamp(-1.0, 1.0)
        input = (input + 1.0) / 2.0
        if CUDA_EXT:
            output = preproc_cpp.dither2d_ste(input, dither_ker, self.bit_width)
        else:
            output = DitheringFunction2d.apply(input, dither_ker, self.bit_width)
        output = output * 2.0 - 1.0
        return output

class TrainedDithering(nn.Module):

    def __init__(self, bit_width, kernel_size, in_ch):
        super(TrainedDithering, self).__init__()
        self.bit_width = bit_width
        self.kernel_size = kernel_size
        self.kernel = nn.Parameter(torch.zeros((in_ch, kernel_size, kernel_size)))

    def forward(self, input):
        input = input.clamp(-1.0, 1.0)
        input = (input + 1.0) / 2.0
        if CUDA_EXT:
            output = preproc_cpp.dither2d_ste(input, self.kernel, self.bit_width)
        else:
            output = DitheringFunction2d.apply(input, self.kernel, self.bit_width)
        output = output * 2.0 - 1.0
        return output


class GammaCorrectionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, gamma):
        clamped_gamma = torch.clamp(gamma, GAMMA_EPS, 2 - GAMMA_EPS)
        clamped_gamma = torch.where(clamped_gamma < 1.0, clamped_gamma, 1 / (2 - clamped_gamma))
        ctx.clamped_gamma = clamped_gamma
        ctx.save_for_backward(input)
        output = input ** (clamped_gamma)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        clamped_gamma = ctx.clamped_gamma
        grad_gamma = torch.where(input == 0, torch.zeros_like(input),
                                 (input ** clamped_gamma) * input.log() * grad_output)
        grad_input = torch.where(input == 0, torch.zeros_like(input),
                                 clamped_gamma * (input ** clamped_gamma) / input * grad_output)
        return grad_input, grad_gamma


class GammaRescaling(nn.Module):

    def __init__(self, parameter_shape):
        super(GammaRescaling, self).__init__()
        self.learned_value = nn.Parameter(torch.full(parameter_shape, 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma_rescale = self.learned_value
        y = x.clamp(-1.0 + GAMMA_EPS, 1.0 - GAMMA_EPS)
        y = (y + 1.0) / 2.0
        y = GammaCorrectionFunction.apply(y, gamma_rescale)
        y = y * 2.0 - 1.0
        return y

class ColorSpaceTransformation(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ColorSpaceTransformation, self).__init__()
        self.kernel = nn.Parameter(torch.zeros(in_channels,out_channels))
        self.kernel.data[:in_channels,:in_channels] = torch.eye(in_channels)
        self.bias = nn.Parameter(torch.zeros(in_channels))
        self.out_channels = out_channels

    def forward(self, input):
        output = (input.permute(0,3,2,1).reshape(-1, input.shape[1]).mm(self.kernel)).reshape(input.shape[0], input.shape[2], input.shape[3], self.out_channels).permute(0,3,2,1)
        return output.clamp(-1.0,1.0)

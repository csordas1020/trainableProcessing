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

import argparse

import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-checkpoint', type=str,
                        help='Checkpoint which needs to be modified')
    parser.add_argument('--output-checkpoint', type=str,
                        help='New checkpoint with the preprocessing parameters quantized')
    parser.add_argument('--shift-based', help="Use shift-based quantization, otherwise fixed point is used (default: %(default)s)",
                        action='store_true', default=False)
    parser.add_argument('--bit-width', type=int, help="Bit-width to use for fixed point quantization (default: %(default)s)",
                        default=8)
    parser.add_argument('--verbose', help="Output some diagnostic information (default: %(default)s)",
                        action='store_true', default=False)
    args = parser.parse_args()
    verbose = args.verbose
    m = torch.load(args.input_checkpoint, map_location='cpu')
    with torch.no_grad():
        kernel = m['state_dict']['module.preproc_features.0.kernel']
        if verbose:
            print("Kernel before quantization")
            print(kernel)
        if args.shift_based:
            kernel = torch.where(kernel <= 0.0, -torch.ones(1), kernel)
            if verbose:
                print("Adjusted kernel values to be negative, to avoid NaN")
                print(kernel)
            kernel_sign = torch.sign(kernel)
            kernel = kernel_sign * torch.pow(2, torch.round(torch.log2(kernel.abs())))
            kernel = torch.where(kernel <= 0.0, torch.zeros(1), kernel)
        else:
            kernel = torch.where(kernel < 0.0, torch.zeros(1), kernel)
            if verbose:
                print("Kernel after clamping")
                print(kernel)
            scale_factor = kernel.max() / (2**args.bit_width - 1)
            if verbose:
                print("Scale factor")
                print(scale_factor)
            kernel = scale_factor * torch.round(kernel / scale_factor)
        if verbose:
            print("Quantized kernel")
            print(kernel)
        m['state_dict']['module.preproc_features.0.kernel'] = kernel
    torch.save(m, args.output_checkpoint)


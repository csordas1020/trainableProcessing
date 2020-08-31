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

import numpy as np
import PIL
import torch
import torchvision.models as models
import torchvision.transforms as transforms

import preproc_models

if __name__ == '__main__':
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
    
    preproc_model_names = sorted(name for name in preproc_models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(preproc_models.__dict__[name]))
    
    model_names += preproc_model_names
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('--preproc-mode', default=None, type=str)
    parser.add_argument('--colortrans', dest='colortrans', action='store_true')
    parser.add_argument('--input-bit-width', default=8, type=int)
    parser.add_argument('--input-checkpoint', type=str,
                        help='Checkpoint to test', required=True)
    parser.add_argument('--input-image', type=str,
                        help='Input image to process', required=True)
    parser.add_argument('--output-proc-image', type=str,
                        help='Output processed image', required=True)
    parser.add_argument('--output-input-image', type=str,
                        help='Output the image before preprocessing', required=True)
    args = parser.parse_args()

    model = preproc_models.__dict__[args.arch](preproc_mode=args.preproc_mode, colortrans=args.colortrans,
                                                input_bit_width=args.input_bit_width)

    state = torch.load(args.input_checkpoint, map_location='cpu')['state_dict']
    new_state = {}

    for k in state.keys():
        if 'module' in k:
            nk = '.'.join(k.split('.')[1:])
            new_state[nk] = state[k]
        else:
            new_state[k] = state[k]

    model.load_state_dict(new_state)
    model.float()
    model.cuda()
    model.eval()

    resize = transforms.Resize(256)
    crop = transforms.CenterCrop(224)
    convert = transforms.ToTensor()
    topil = transforms.ToPILImage(mode='RGB')

    image = PIL.Image.open(args.input_image)
    image = resize(image)
    image = crop(image)
    image = convert(image)
    image = image.view(1,3,224,224)
    input_image = topil(image.view(3,224,224))
    input_image.save(args.output_input_image)

    image = 2.*image.cuda() - 1.
    for mod in model.preproc_features:
        image = mod(image)

    image = (image + 1.0) / 2.
    image = topil(image.view(3,224,224).cpu())
    image.save(args.output_proc_image)


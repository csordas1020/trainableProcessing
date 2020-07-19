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

import torchvision.models.resnet as resnet

from torch.nn import ModuleList
from trainablePreprocessing.preprocessing import TrainedDithering, FixedDithering, Quantization, ColorSpaceTransformation


class PreprocResNet(resnet.ResNet):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, preproc_mode='quant', colortrans=False, input_bit_width=8):
        super(PreprocResNet, self).__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group,
                                            replace_stride_with_dilation, norm_layer)
        self.preproc_features = ModuleList()
        if colortrans:
            self.preproc_features.append(ColorSpaceTransformation(3, 3))
        if preproc_mode == 'trained_dithering':
            self.preproc_features.append(TrainedDithering(input_bit_width, 3, 3))
        elif preproc_mode == 'fixed_dithering':
            self.preproc_features.append(FixedDithering(input_bit_width, 3))
        elif preproc_mode == 'quant':
            self.preproc_features.append(Quantization(input_bit_width))

    def forward(self, x):
        for mod in self.preproc_features:
            x = mod(x)
        x = super(PreprocResNet, self).forward(x)
        return x


def preproc_resnet18(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return PreprocResNet(resnet.BasicBlock, [2, 2, 2, 2], **kwargs)


def preproc_resnet34(**kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_"""
    return PreprocResNet(resnet.BasicBlock, [3, 4, 6, 3], **kwargs)


def preproc_resnet50(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_"""
    return PreprocResNet(resnet.Bottleneck, [3, 4, 6, 3], **kwargs)


def preproc_resnet101(**kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_"""
    return PreprocResNet(resnet.Bottleneck, [3, 4, 23, 3], **kwargs)


def preproc_resnet152(**kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_"""
    return PreprocResNet(resnet.Bottleneck, [3, 8, 36, 3], **kwargs)


def preproc_resnext50_32x4d(**kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_"""
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return PreprocResNet(resnet.Bottleneck, [3, 4, 6, 3], **kwargs)


def preproc_resnext101_32x8d(**kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_"""
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return PreprocResNet(resnet.Bottleneck, [3, 4, 23, 3], **kwargs)


def preproc_wide_resnet50_2(**kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048."""
    kwargs['width_per_group'] = 64 * 2
    return PreprocResNet(resnet.Bottleneck, [3, 4, 6, 3], **kwargs)


def preproc_wide_resnet101_2(**kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048."""
    kwargs['width_per_group'] = 64 * 2
    return PreprocResNet(resnet.Bottleneck, [3, 4, 23, 3], **kwargs)

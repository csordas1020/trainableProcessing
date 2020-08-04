# ImageNet training in PyTorch with Trainable Preprocessing

This implements training of popular model architectures, such as ResNet, AlexNet, and VGG on the ImageNet dataset with trainable preprocessing such as trained color conversion and trained dithering.

## Prerequisites

* PyTorch (last tested with v1.4.0)
* Torchvision (last tested with v0.5.0)

## Training

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
python main.py -a preproc_resnet18 --preproc_mode trained_dithering --input_bit_width 4 [imagenet-folder with train and val folders]
```

## Usage

```
usage: main.py [-h] [--arch ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N]
               [--lr LR] [--momentum M] [--weight-decay W] [--print-freq N]
               [--resume PATH] [-e] [--pretrained] [--seed SEED] [--gpu GPU]
               [--preperoc_mode] [--colortrans] [--input_bit_width]
               DIR

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH, -a ARCH  model architecture: alexnet | densenet121 |
                        densenet161 | densenet169 | densenet201 |
                        resnet101 | resnet152 | resnet18 | resnet34 |
                        resnet50 | squeezenet1_0 | squeezenet1_1 | vgg11 |
                        vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19
                        | vgg19_bn (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --preproc_mode        preprocessing mode: quant, fixed_dithering, trained dithering
  --colortrans          added trained color conversion
  --input_bit_width     input bit width
```

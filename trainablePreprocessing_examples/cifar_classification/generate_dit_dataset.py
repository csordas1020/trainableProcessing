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
import torchvision
import numpy as np
import pickle
import matplotlib.pyplot as plt

import os
import shutil

import cifar_dataloader

import preprocessing

def norm_img(img):
    return img * 2.0 - 1.0

def denorm_img(img):
    return (img + 1.0) / 2.0

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.axis('off')

def generate_dit_dataset(input_bit_width_array, im_size):

    for input_bit_width in input_bit_width_array:

        load_path = './data/'
        save_path = './data/'

        if input_bit_width < 8:
            save_path = save_path + 'fs' + str(input_bit_width) + '/'
        load_path += 'cifar-10-batches-py/'
        save_path += 'cifar-10-batches-py/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        meta_files = ['batches.meta', 'readme.html']
        for file in meta_files:
            if not os.path.isfile(save_path+file):
                shutil.copyfile(load_path+file, save_path+file)

        for batch in range(1,7):
            if batch < 6:
                load_file_name = load_path + 'data_batch_' + str(batch)
                save_file_name = save_path + 'data_batch_' + str(batch)
            else:
                load_file_name = load_path + 'test_batch'
                save_file_name = save_path + 'test_batch'

            load_file = open(load_file_name, 'rb')
            dictionary = pickle.load(load_file, encoding='latin1')
            images = dictionary['data']
            images = torch.tensor(images.reshape(images.shape[0],3,im_size,im_size), dtype=torch.float32)
            images = images / 255
            if input_bit_width < 8:
                fs_dithering = preprocessing.FixedDithering(input_bit_width, 3)
                images = denorm_img(fs_dithering(norm_img(images)))
            np_images = images.numpy().reshape(images.shape[0],3*im_size*im_size)
            np_images = np_images * 255
            np_images = np_images.astype(np.uint8)
            new_dict = {'data': np_images}
            dictionary.update(new_dict)
            save_file = open(save_file_name, 'wb')
            pickle.dump(dictionary, save_file, pickle.HIGHEST_PROTOCOL)

def generate_gray_dataset(input_bit_width_array, im_size):

    for input_bit_width in input_bit_width_array:

        load_path = './data/'
        save_path = './data/'

        if input_bit_width < 8:
            save_path = save_path + 'gray_fs' + str(input_bit_width) + '/'
        else:
            save_path = save_path + 'gray/'
        load_path += 'cifar-10-batches-py/'
        save_path += 'cifar-10-batches-py/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        meta_files = ['batches.meta', 'readme.html']
        for file in meta_files:
            if not os.path.isfile(save_path+file):
                shutil.copyfile(load_path+file, save_path+file)

        for batch in range(1,7):
            if batch < 6:
                load_file_name = load_path + 'data_batch_' + str(batch)
                save_file_name = save_path + 'data_batch_' + str(batch)
            else:
                load_file_name = load_path + 'test_batch'
                save_file_name = save_path + 'test_batch'

            load_file = open(load_file_name, 'rb')
            dictionary = pickle.load(load_file, encoding='latin1')
            images = dictionary['data']
            images = torch.tensor(images.reshape(images.shape[0],3,im_size,im_size), dtype=torch.float32)
            images = images / 255
            grayscale_images = images
            grayscale_images[:,0,:,:] = images[:,0,:,:] * 0.299 + images[:,1,:,:] * 0.587 + images[:,2,:,:] * 0.144
            grayscale_images[:,1,:,:] = images[:,0,:,:] * 0.299 + images[:,1,:,:] * 0.587 + images[:,2,:,:] * 0.144
            grayscale_images[:,2,:,:] = images[:,0,:,:] * 0.299 + images[:,1,:,:] * 0.587 + images[:,2,:,:] * 0.144
            grayscale_images = grayscale_images.clamp(0.0,1.0)
            if input_bit_width < 8:
                fs_dithering = preprocessing.FixedDithering(input_bit_width, 3)
                grayscale_images = denorm_img(fs_dithering(norm_img(grayscale_images)))
            np_images = grayscale_images.numpy().reshape(images.shape[0],3*im_size*im_size)
            np_images = np_images * 255
            np_images = np_images.astype(np.uint8)
            new_dict = {'data': np_images}
            dictionary.update(new_dict)
            save_file = open(save_file_name, 'wb')
            pickle.dump(dictionary, save_file, pickle.HIGHEST_PROTOCOL)

generate_gray_dataset([8,4,3,2,1],32)
generate_dit_dataset([4,3,2,1],32)

cifar10_org_train_loader, cifar10_org_test_loader, cifar10_org_num_classes = cifar_dataloader.get_loaders('CIFAR10','./data/',1, 4)
cifar10_gray_train_loader, cifar10_gray_test_loader, cifar10_gray_num_classes = cifar_dataloader.get_loaders('CIFAR10','./data/gray',1, 4)

cifar10_gray_fs4_train_loader, cifar10_gray_fs4_test_loader, cifar10_gray_fs4_num_classes = cifar_dataloader.get_loaders('CIFAR10','./data/gray_fs4/',1, 4)
cifar10_gray_fs3_train_loader, cifar10_gray_fs3_test_loader, cifar10_gray_fs3_num_classes = cifar_dataloader.get_loaders('CIFAR10','./data/gray_fs3/',1, 4)
cifar10_gray_fs2_train_loader, cifar10_gray_fs2_test_loader, cifar10_gray_fs2_num_classes = cifar_dataloader.get_loaders('CIFAR10','./data/gray_fs2/',1, 4)
cifar10_gray_fs1_train_loader, cifar10_gray_fs1_test_loader, cifar10_gray_fs1_num_classes = cifar_dataloader.get_loaders('CIFAR10','./data/gray_fs1/',1, 4)

cifar10_fs4_train_loader, cifar10_fs4_test_loader, cifar10_fs14_num_classes = cifar_dataloader.get_loaders('CIFAR10','./data/fs4/',1, 4)
cifar10_fs3_train_loader, cifar10_fs3_test_loader, cifar10_fs3_num_classes = cifar_dataloader.get_loaders('CIFAR10','./data/fs3/',1, 4)
cifar10_fs2_train_loader, cifar10_fs2_test_loader, cifar10_fs2_num_classes = cifar_dataloader.get_loaders('CIFAR10','./data/fs2/',1, 4)
cifar10_fs1_train_loader, cifar10_fs1_test_loader, cifar10_fs1_num_classes = cifar_dataloader.get_loaders('CIFAR10','./data/fs1/',1, 4)


cifar10_org_images, cifar10_org_labels = iter(cifar10_org_test_loader).next()
cifar10_gray_images, cifar10_gray_labels = iter(cifar10_gray_test_loader).next()

cifar10_gray_fs4_images, cifar10_gray_fs4_labels = iter(cifar10_gray_fs4_test_loader).next()
cifar10_gray_fs3_images, cifar10_gray_fs3_labels = iter(cifar10_gray_fs3_test_loader).next()
cifar10_gray_fs2_images, cifar10_gray_fs2_labels = iter(cifar10_gray_fs2_test_loader).next()
cifar10_gray_fs1_images, cifar10_gray_fs1_labels = iter(cifar10_gray_fs1_test_loader).next()

cifar10_fs4_images, cifar10_fs4_labels = iter(cifar10_fs4_test_loader).next()
cifar10_fs3_images, cifar10_fs3_labels = iter(cifar10_fs3_test_loader).next()
cifar10_fs2_images, cifar10_fs2_labels = iter(cifar10_fs2_test_loader).next()
cifar10_fs1_images, cifar10_fs1_labels = iter(cifar10_fs1_test_loader).next()


cifar10_gray_quant4_images = preprocessing.quant(cifar10_gray_images, 4)
cifar10_gray_quant3_images = preprocessing.quant(cifar10_gray_images, 3)
cifar10_gray_quant2_images = preprocessing.quant(cifar10_gray_images, 2)
cifar10_gray_quant1_images = preprocessing.quant(cifar10_gray_images, 1)

cifar10_quant4_images = preprocessing.quant(cifar10_org_images, 4)
cifar10_quant3_images = preprocessing.quant(cifar10_org_images, 3)
cifar10_quant2_images = preprocessing.quant(cifar10_org_images, 2)
cifar10_quant1_images = preprocessing.quant(cifar10_org_images, 1)

im_size = 32

plt.figure(1)
plt.subplot(3,4,1)
imshow(torchvision.utils.make_grid(cifar10_org_images))
plt.subplot(3,4,2)
imshow(torchvision.utils.make_grid(cifar10_gray_images[:,0,:,:].reshape(cifar10_org_images.shape[0],1,im_size,im_size)))
plt.subplot(3,4,5)
imshow(torchvision.utils.make_grid(cifar10_gray_quant4_images[:,0,:,:].reshape(cifar10_org_images.shape[0],1,im_size,im_size)))
plt.subplot(3,4,6)
imshow(torchvision.utils.make_grid(cifar10_gray_quant3_images[:,0,:,:].reshape(cifar10_org_images.shape[0],1,im_size,im_size)))
plt.subplot(3,4,7)
imshow(torchvision.utils.make_grid(cifar10_gray_quant2_images[:,0,:,:].reshape(cifar10_org_images.shape[0],1,im_size,im_size)))
plt.subplot(3,4,8)
imshow(torchvision.utils.make_grid(cifar10_gray_quant1_images[:,0,:,:].reshape(cifar10_org_images.shape[0],1,im_size,im_size)))
plt.subplot(3,4,9)
imshow(torchvision.utils.make_grid(cifar10_gray_fs4_images[:,0,:,:].reshape(cifar10_org_images.shape[0],1,im_size,im_size)))
plt.subplot(3,4,10)
imshow(torchvision.utils.make_grid(cifar10_gray_fs3_images[:,0,:,:].reshape(cifar10_org_images.shape[0],1,im_size,im_size)))
plt.subplot(3,4,11)
imshow(torchvision.utils.make_grid(cifar10_gray_fs2_images[:,0,:,:].reshape(cifar10_org_images.shape[0],1,im_size,im_size)))
plt.subplot(3,4,12)
imshow(torchvision.utils.make_grid(cifar10_gray_fs1_images[:,0,:,:].reshape(cifar10_org_images.shape[0],1,im_size,im_size)))

plt.figure(2)
plt.subplot(3,4,1)
imshow(torchvision.utils.make_grid(cifar10_org_images))
plt.subplot(3,4,5)
imshow(torchvision.utils.make_grid(cifar10_quant4_images))
plt.subplot(3,4,6)
imshow(torchvision.utils.make_grid(cifar10_quant3_images))
plt.subplot(3,4,7)
imshow(torchvision.utils.make_grid(cifar10_quant2_images))
plt.subplot(3,4,8)
imshow(torchvision.utils.make_grid(cifar10_quant1_images))
plt.subplot(3,4,9)
imshow(torchvision.utils.make_grid(cifar10_fs4_images))
plt.subplot(3,4,10)
imshow(torchvision.utils.make_grid(cifar10_fs3_images))
plt.subplot(3,4,11)
imshow(torchvision.utils.make_grid(cifar10_fs2_images))
plt.subplot(3,4,12)
imshow(torchvision.utils.make_grid(cifar10_fs1_images))
plt.show()

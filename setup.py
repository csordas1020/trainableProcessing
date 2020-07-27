from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


setup(name='trainablePreprocessing',
      version='1.0',
      packages=find_packages(),
      ext_modules=[
        CUDAExtension('preproc_cpp', [
            'trainablePreprocessing/csrc/preproc.cpp',
            'trainablePreprocessing/csrc/preproc_cuda_kernel.cu',
        ])
      ],
      cmdclass={
        'build_ext': BuildExtension
      }
      )
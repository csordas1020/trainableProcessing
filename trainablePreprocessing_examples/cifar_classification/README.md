# trainable (pre) processing for neural networks

# Prerequisites:
The models use brevitas:
1. Clone brevitas somewhere on your machine from: git clone github.com/Xilinx/brevitas
2. Set path for it: export PYTHONPATH=/path/to/brevitas
3. Conda environments:
put yml here
  

# Run code:

PYTORCH_JIT=1 python main.py --network VGG_preproc --dataset CIFAR10 --experiments ./results/ --preproc_mode trained_dithering --input_bit_width 1

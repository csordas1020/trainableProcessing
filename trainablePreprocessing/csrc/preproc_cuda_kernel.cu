#include <torch/extension.h>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

using torch::Tensor;

__global__ void preproc_cuda_forward_kernel(
    torch::PackedTensorAccessor32<float, 4> input,
    torch::PackedTensorAccessor32<float, 3> dither_ker,
    torch::PackedTensorAccessor32<float, 4> quant_input,
    const int n_row,
    const int n_col,
    const int bit_width) {
    const int b = blockIdx.x ;
    const int ch = threadIdx.x;
    for(int r = 0; r < n_row; r++){
        for(int c = 0; c < n_col; c++){
            if(input[b][ch][r][c] < 0.0){
                input[b][ch][r][c] = 0.0;
            }
            if(input[b][ch][r][c] > 1.0){
                input[b][ch][r][c] = 1.0;
            }
            auto quant_pix = input[b][ch][r][c] * (pow(2,bit_width) - 1);
            quant_pix = round(quant_pix) / (pow(2,bit_width) - 1);
            quant_input[b][ch][r][c] = quant_pix;
            auto quant_err = input[b][ch][r][c] - quant_pix;
            if(c < n_col - 1){
                input[b][ch][r][c + 1] += quant_err *  dither_ker[ch][1][2];
            }
            if(r < n_row - 1){
                if(c > 0){
                    input[b][ch][r + 1][c - 1] += quant_err * dither_ker[ch][2][0];
                }
                input[b][ch][r + 1][c] += quant_err * dither_ker[ch][2][1];
                if(c < n_col - 1){
                    input[b][ch][r + 1][c + 1] += quant_err * dither_ker[ch][2][2];
                }
            }
        }
    }
}

__global__ void preproc_cuda_backward_kernel(
    torch::PackedTensorAccessor32<float, 4> grad_output,
    torch::PackedTensorAccessor32<float, 4> quant_err,
    torch::PackedTensorAccessor32<float, 3> grad_dither_ker,
    const int n_batch,
    const int n_row,
    const int n_col) {

    const int kx = blockIdx.x;
    const int ky = blockIdx.y;
    const int ch = threadIdx.x;

    float grad = 0;

    if(kx == 1 && ky == 2){
        for(int b = 0; b < n_batch; b++){
            for(int r = 0; r < n_row; r++){
                for(int c = 0; c < n_col-1; c++){
                    grad += grad_output[b][ch][r][c+1] * quant_err[b][ch][r][c];
                }
            }
        }
        grad_dither_ker[ch][1][2] = grad;
    } else if (kx == 2 && ky == 0){
        for(int b = 0; b < n_batch; b++){
            for(int r = 0; r < n_row-1; r++){
                for(int c = 1; c < n_col; c++){
                    grad += grad_output[b][ch][r+1][c-1] * quant_err[b][ch][r][c];
                }
            }
        }
        grad_dither_ker[ch][2][0] = grad;
    } else if (kx == 2 && ky == 1){
        for(int b = 0; b < n_batch; b++){
            for(int r = 0; r < n_row-1; r++){
                for(int c = 0; c < n_col; c++){
                    grad += grad_output[b][ch][r+1][c] * quant_err[b][ch][r][c];
                }
            }
        }
        grad_dither_ker[ch][2][1] = grad;
    } else if (kx == 2 && ky == 2){
        for(int b = 0; b < n_batch; b++){
            for(int r = 0; r < n_row-1; r++){
                for(int c = 0; c < n_col-1; c++){
                    grad += grad_output[b][ch][r+1][c+1] * quant_err[b][ch][r][c];
                }
            }
        }
        grad_dither_ker[ch][2][2] = grad;
    }
}

Tensor preproc_forward(
    Tensor& input,
    Tensor& dither_ker,
    Tensor& quant_input,
    const int n_batch,
    const int n_ch,
    const int n_row,
    const int n_col,
    const int bit_width){
    const int blocks = n_batch;
    const int threads = n_ch;

    preproc_cuda_forward_kernel<<<blocks, threads>>>(
        input.packed_accessor32<float,4>(),
        dither_ker.packed_accessor32<float,3>(),
        quant_input.packed_accessor32<float,4>(),
        n_row,
        n_col,
        bit_width
    );

    return quant_input;
}

Tensor preproc_backward(
    Tensor& grad_output,
    Tensor& quant_err,
    Tensor& grad_dither_ker,
    const int n_batch,
    const int n_ch,
    const int n_row,
    const int n_col){

    const dim3 blocks(3,3);
    const int threads = n_ch;

    preproc_cuda_backward_kernel<<<blocks, threads>>>(
        grad_output.packed_accessor32<float,4>(),
        quant_err.packed_accessor32<float,4>(),
        grad_dither_ker.packed_accessor32<float,3>(),
        n_batch,
        n_row,
        n_col
    );

    return grad_dither_ker;
}

#include <torch/script.h>
#include <torch/extension.h>
#include <ATen/TensorUtils.h>
#include <cmath>
#include <chrono>

using namespace at;
using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

Tensor preproc_forward(Tensor& input,
                       Tensor& dither_ker,
                       Tensor& quant_input,
                       const int n_batch,
                       const int n_ch,
                       const int n_row,
                       const int n_col,
                       const int bit_width);

Tensor preproc_backward(Tensor& grad_output,
                        Tensor& quant_err,
                        Tensor& grad_dither_ker,
                        const int n_batch,
                        const int n_ch,
                        const int n_row,
                        const int n_col);*/

class Dither2dSteFn : public torch::autograd::Function<Dither2dSteFn> {
    public:
        static variable_list forward(AutogradContext* ctx,
                                     Variable input,
                                     Variable dither_ker,
                                     const long bit_width){
            const long n_batch = input.sizes()[0];
            const long n_ch = input.sizes()[1];
            const long n_row = input.sizes()[2];
            const long n_col = input.sizes()[3];

            dither_ker = at::clamp_min(dither_ker, 0.0);
            auto quant_input = torch::zeros_like(input);
            preproc_forward(input, dither_ker, quant_input, n_batch, n_ch, n_row, n_col, bit_width);
            auto quant_err = input - quant_input;
            ctx->save_for_backward({quant_err, dither_ker});
            ctx->saved_data["n_batch"] = n_batch;
            ctx->saved_data["n_ch"] = n_ch;
            ctx->saved_data["n_row"] = n_row;
            ctx->saved_data["n_col"] = n_col;

            return {quant_input};
        }

        static variable_list backward(AutogradContext* ctx,
                                      variable_list grad_output){
            auto quant_err = ctx->get_saved_variables()[0];
            auto dither_ker = ctx->get_saved_variables()[1];
            auto grad_dither_ker = torch::zeros_like(dither_ker);

            auto n_batch = ctx->saved_data["n_batch"].toInt();
            auto n_ch = ctx->saved_data["n_ch"].toInt();
            auto n_row = ctx->saved_data["n_row"].toInt();
            auto n_col = ctx->saved_data["n_col"].toInt();

            auto grad_input = grad_output[0];
            preproc_backward(grad_input, quant_err, grad_dither_ker, n_batch, n_ch, n_row, n_col);

            return {grad_input, grad_dither_ker, Variable()};
        }
};

Tensor dither2d_ste(const Tensor& input,
                    const Tensor& dither_ker,
                    const long bit_width){
    return Dither2dSteFn::apply(input, dither_ker, bit_width)[0];
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dither2d_ste", &dither2d_ste, "CUDA 2D dithering");
}
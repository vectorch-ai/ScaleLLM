/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <ATen/core/TensorBody.h>
#include <torch/torch.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/numeric_types.h>
#include <torch/types.h>

#include "flash.h"
#include "static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")


void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *alibi_slopes_d,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.is_bf16 = q.dtype() == torch::kBFloat16;

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
        params.o_batch_stride = out.stride(0);
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.alibi_slopes = static_cast<float *>(alibi_slopes_d);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    TORCH_CHECK(p_dropout < 1.f);

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;

    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    params.is_seqlens_k_cumulative = true;
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        FWD_HEADDIM_SWITCH(params.d, [&] {
            run_mha_fwd_<elem_type, kHeadDim>(params, stream);
        });
    });
}

std::vector<at::Tensor>
mha_varlen_fwd(const at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               c10::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               c10::optional<at::Tensor> alibi_slopes,  // [num_heads]
               int max_seqlen_q,
               int max_seqlen_k,
               float p_dropout,
               float softmax_scale,
               bool zero_tensors,
               bool is_causal,
               int window_size_left,
               int window_size_right,
               bool return_softmax,
               c10::optional<at::Generator> gen_) {

    if (is_causal) { window_size_right = 0; }
    auto dprops = at::cuda::getCurrentDeviceProperties();
    // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");
    // We will support Turing in the near future
    // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports Turing GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    if (q_dtype == torch::kBFloat16) {
        TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
    }
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_DEVICE(cu_seqlens_k);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    const auto sizes = q.sizes();

    const int total_q = sizes[0];
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int num_heads = sizes[1];
    const int head_size_og = sizes[2];
    const int total_k = k.size(0);
    const int num_heads_k = k.size(1);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    CHECK_SHAPE(q, total_q, num_heads, head_size_og);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size_og);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size_og);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    
    if (alibi_slopes) {
        at::Tensor slopes = alibi_slopes.value();
        CHECK_DEVICE(slopes);
        CHECK_CONTIGUOUS(slopes);
        CHECK_SHAPE(slopes, num_heads);
        TORCH_CHECK(slopes.dtype() == torch::kFloat32, "alibi_slopes must have dtype float32");
    }

    at::Tensor q_padded, k_padded, v_padded;
    if (head_size_og % 8 != 0) {
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        q_padded = q;
        k_padded = k;
        v_padded = v;
    }

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, total_q, num_heads, head_size_og);
        if (head_size_og % 8 != 0) { out = torch::empty_like(q_padded); }
    } else {
        out = torch::empty_like(q_padded);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();

    auto softmax_lse = torch::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor p;
    // Only return softmax if there's dropout to reduce compilation time
    if (return_softmax) {
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = torch::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, opts);
    }

    if (zero_tensors) {
        out.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
        if (return_softmax) {p.zero_();}
    }

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, k_padded, v_padded, out,
                     cu_seqlens_q.data_ptr(),
                     cu_seqlens_k.data_ptr(),
                     alibi_slopes ? alibi_slopes->data_ptr() : nullptr,
                     return_softmax ? p.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right);

    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
    // Forward kernel will populate memory with the seed and offset.
    params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

    if (p_dropout > 0.0)  {
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_mha_fwd(params, stream);

    at::Tensor out_padded = out;
    if (head_size_og % 8 != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (out_.has_value()) { out_.value().copy_(out); }
    }

    return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p, rng_state};
}

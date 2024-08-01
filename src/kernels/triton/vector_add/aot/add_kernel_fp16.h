#include <cuda.h>

CUresult add_kernel_fp16_16_warps1xstages3(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements);
void load_add_kernel_fp16_16_warps1xstages3();
void unload_add_kernel_fp16_16_warps1xstages3();
    
int add_kernel_fp16_get_num_algos(void);

CUresult add_kernel_fp16_default(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements);
CUresult add_kernel_fp16(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements, int algo_id);
void load_add_kernel_fp16();
void unload_add_kernel_fp16();
    
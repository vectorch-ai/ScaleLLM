#include <cuda.h>

CUresult add_kernel_fp32_16_warps1xstages3(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements);
void load_add_kernel_fp32_16_warps1xstages3();
void unload_add_kernel_fp32_16_warps1xstages3();
    
int add_kernel_fp32_get_num_algos(void);

CUresult add_kernel_fp32_default(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements);
CUresult add_kernel_fp32(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements, int algo_id);
void load_add_kernel_fp32();
void unload_add_kernel_fp32();
    
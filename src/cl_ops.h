#ifndef CL_OPS_H
#define CL_OPS_H

#include <stddef.h>

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;

    cl_kernel k_matmul;
    cl_kernel k_matmul_at;
    cl_kernel k_matmul_bt;
    cl_kernel k_add_bias_cols;
    cl_kernel k_relu;
    cl_kernel k_relu_backward;
    cl_kernel k_softmax_rows;
    cl_kernel k_col_sum;
    cl_kernel k_vec_subtract;
    cl_kernel k_sgd_update;
} GpuContext;

// Initializes the OpenCL context, compiles kernels.cl
int gpu_init(GpuContext * ctx, const char * kernel_path);

// Releases all OpenCL resources held by ctx
void gpu_shutdown(GpuContext * ctx);

// Allocates a device buffer holding count floats
cl_mem gpu_alloc(GpuContext * ctx, size_t count);

void gpu_write(GpuContext * ctx, cl_mem buf, const float * host, size_t count);
void gpu_read(GpuContext * ctx, cl_mem buf, float * host, size_t count);

// C(MxN) = A(MxK) * B(KxN)
void gpu_matmul(GpuContext * ctx, cl_mem A, cl_mem B, cl_mem C, int M, int K, int N);
// C(MxN) = A^T * B, A stored as KxM
void gpu_matmul_at(GpuContext * ctx, cl_mem A, cl_mem B, cl_mem C, int M, int K, int N);
// C(MxN) = A * B^T, B stored as NxK
void gpu_matmul_bt(GpuContext * ctx, cl_mem A, cl_mem B, cl_mem C, int M, int K, int N);

// Z(rows x cols) += bias(cols), broadcast per row
void gpu_add_bias_cols(GpuContext * ctx, cl_mem Z, cl_mem bias, int rows, int cols);

void gpu_relu(GpuContext * ctx, cl_mem in, cl_mem out, int n);
// out = (z >= 0) ? da : 0
void gpu_relu_backward(GpuContext * ctx, cl_mem da, cl_mem z, cl_mem out, int n);

void gpu_softmax_rows(GpuContext * ctx, cl_mem Z, cl_mem A, int rows, int cols);

// out[col] = sum over rows of M[row][col]
void gpu_col_sum(GpuContext * ctx, cl_mem M, cl_mem out, int rows, int cols);

void gpu_vec_subtract(GpuContext * ctx, cl_mem a, cl_mem b, cl_mem out, int n);

// param -= scaled_lr * grad
void gpu_sgd_update(GpuContext * ctx, cl_mem param, cl_mem grad, float scaled_lr, int n);

#endif

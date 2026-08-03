#include "cl_ops.h"

#include <stdio.h>
#include <stdlib.h>

#define CL_CHECK(err, msg) \
    do { \
        if ((err) != CL_SUCCESS) { \
            fprintf(stderr, "\nOpenCL error %d: %s (%s:%d)\n", (int)(err), (msg), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

static char * read_file(const char * path, size_t * out_len) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "\nFailed to open kernel source: %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len < 0) {
        fclose(f);
        return NULL;
    }
    char * buf = (char *)malloc((size_t)len + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    size_t read = fread(buf, 1, (size_t)len, f);
    fclose(f);
    buf[read] = '\0';
    if (out_len) *out_len = read;
    return buf;
}

static cl_kernel make_kernel(cl_program program, const char * name) {
    cl_int err;
    cl_kernel k = clCreateKernel(program, name, &err);
    CL_CHECK(err, name);
    return k;
}

int gpu_init(GpuContext * ctx, const char * kernel_path) {
    cl_int err;

    err = clGetPlatformIDs(1, &ctx->platform, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "\nNo OpenCL platform found (error %d)\n", (int)err);
        return 1;
    }

    err = clGetDeviceIDs(ctx->platform, CL_DEVICE_TYPE_GPU, 1, &ctx->device, NULL);
    if (err != CL_SUCCESS) {
        // Fall back to any available device (e.g. CPU) if no GPU is present.
        err = clGetDeviceIDs(ctx->platform, CL_DEVICE_TYPE_ALL, 1, &ctx->device, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "\nNo OpenCL device found (error %d)\n", (int)err);
            return 1;
        }
    }

    char name[256] = {0};
    clGetDeviceInfo(ctx->device, CL_DEVICE_NAME, sizeof(name) - 1, name, NULL);
    printf("\nOpenCL device: %s\n", name);

    ctx->context = clCreateContext(NULL, 1, &ctx->device, NULL, NULL, &err);
    CL_CHECK(err, "clCreateContext");

    ctx->queue = clCreateCommandQueueWithProperties(ctx->context, ctx->device, NULL, &err);
    CL_CHECK(err, "clCreateCommandQueueWithProperties");

    size_t src_len = 0;
    char * source = read_file(kernel_path, &src_len);
    if (!source) return 1;

    ctx->program = clCreateProgramWithSource(ctx->context, 1, (const char **)&source, &src_len, &err);
    free(source);
    CL_CHECK(err, "clCreateProgramWithSource");

    err = clBuildProgram(ctx->program, 1, &ctx->device, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_len = 0;
        clGetProgramBuildInfo(ctx->program, ctx->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_len);
        char * log = (char *)malloc(log_len + 1);
        if (log) {
            clGetProgramBuildInfo(ctx->program, ctx->device, CL_PROGRAM_BUILD_LOG, log_len, log, NULL);
            log[log_len] = '\0';
            fprintf(stderr, "\nOpenCL build failed:\n%s\n", log);
            free(log);
        }
        return 1;
    }

    ctx->k_matmul = make_kernel(ctx->program, "matmul");
    ctx->k_matmul_at = make_kernel(ctx->program, "matmul_at");
    ctx->k_matmul_bt = make_kernel(ctx->program, "matmul_bt");
    ctx->k_add_bias_cols = make_kernel(ctx->program, "add_bias_cols");
    ctx->k_relu = make_kernel(ctx->program, "relu");
    ctx->k_relu_backward = make_kernel(ctx->program, "relu_backward");
    ctx->k_softmax_rows = make_kernel(ctx->program, "softmax_rows");
    ctx->k_col_sum = make_kernel(ctx->program, "col_sum");
    ctx->k_vec_subtract = make_kernel(ctx->program, "vec_subtract");
    ctx->k_sgd_update = make_kernel(ctx->program, "sgd_update");

    return 0;
}

void gpu_shutdown(GpuContext * ctx) {
    clReleaseKernel(ctx->k_matmul);
    clReleaseKernel(ctx->k_matmul_at);
    clReleaseKernel(ctx->k_matmul_bt);
    clReleaseKernel(ctx->k_add_bias_cols);
    clReleaseKernel(ctx->k_relu);
    clReleaseKernel(ctx->k_relu_backward);
    clReleaseKernel(ctx->k_softmax_rows);
    clReleaseKernel(ctx->k_col_sum);
    clReleaseKernel(ctx->k_vec_subtract);
    clReleaseKernel(ctx->k_sgd_update);
    clReleaseProgram(ctx->program);
    clReleaseCommandQueue(ctx->queue);
    clReleaseContext(ctx->context);
}

cl_mem gpu_alloc(GpuContext * ctx, size_t count) {
    cl_int err;
    cl_mem buf = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE, count * sizeof(float), NULL, &err);
    CL_CHECK(err, "clCreateBuffer");
    return buf;
}

void gpu_write(GpuContext * ctx, cl_mem buf, const float * host, size_t count) {
    cl_int err = clEnqueueWriteBuffer(ctx->queue, buf, CL_TRUE, 0, count * sizeof(float), host, 0, NULL, NULL);
    CL_CHECK(err, "clEnqueueWriteBuffer");
}

void gpu_read(GpuContext * ctx, cl_mem buf, float * host, size_t count) {
    cl_int err = clEnqueueReadBuffer(ctx->queue, buf, CL_TRUE, 0, count * sizeof(float), host, 0, NULL, NULL);
    CL_CHECK(err, "clEnqueueReadBuffer");
}

static void run_2d(GpuContext * ctx, cl_kernel k, int M, int N) {
    size_t global[2] = { (size_t)M, (size_t)N };
    cl_int err = clEnqueueNDRangeKernel(ctx->queue, k, 2, NULL, global, NULL, 0, NULL, NULL);
    CL_CHECK(err, "clEnqueueNDRangeKernel(2D)");
}

static void run_1d(GpuContext * ctx, cl_kernel k, int n) {
    size_t global = (size_t)n;
    cl_int err = clEnqueueNDRangeKernel(ctx->queue, k, 1, NULL, &global, NULL, 0, NULL, NULL);
    CL_CHECK(err, "clEnqueueNDRangeKernel(1D)");
}

static void set_matmul_args(cl_kernel k, cl_mem A, cl_mem B, cl_mem C, int M, int K, int N) {
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k, 0, sizeof(cl_mem), &A);
    err |= clSetKernelArg(k, 1, sizeof(cl_mem), &B);
    err |= clSetKernelArg(k, 2, sizeof(cl_mem), &C);
    err |= clSetKernelArg(k, 3, sizeof(int), &M);
    err |= clSetKernelArg(k, 4, sizeof(int), &K);
    err |= clSetKernelArg(k, 5, sizeof(int), &N);
    CL_CHECK(err, "clSetKernelArg(matmul family)");
}

void gpu_matmul(GpuContext * ctx, cl_mem A, cl_mem B, cl_mem C, int M, int K, int N) {
    set_matmul_args(ctx->k_matmul, A, B, C, M, K, N);
    run_2d(ctx, ctx->k_matmul, M, N);
}

void gpu_matmul_at(GpuContext * ctx, cl_mem A, cl_mem B, cl_mem C, int M, int K, int N) {
    set_matmul_args(ctx->k_matmul_at, A, B, C, M, K, N);
    run_2d(ctx, ctx->k_matmul_at, M, N);
}

void gpu_matmul_bt(GpuContext * ctx, cl_mem A, cl_mem B, cl_mem C, int M, int K, int N) {
    set_matmul_args(ctx->k_matmul_bt, A, B, C, M, K, N);
    run_2d(ctx, ctx->k_matmul_bt, M, N);
}

void gpu_add_bias_cols(GpuContext * ctx, cl_mem Z, cl_mem bias, int rows, int cols) {
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(ctx->k_add_bias_cols, 0, sizeof(cl_mem), &Z);
    err |= clSetKernelArg(ctx->k_add_bias_cols, 1, sizeof(cl_mem), &bias);
    err |= clSetKernelArg(ctx->k_add_bias_cols, 2, sizeof(int), &rows);
    err |= clSetKernelArg(ctx->k_add_bias_cols, 3, sizeof(int), &cols);
    CL_CHECK(err, "clSetKernelArg(add_bias_cols)");
    run_2d(ctx, ctx->k_add_bias_cols, rows, cols);
}

void gpu_relu(GpuContext * ctx, cl_mem in, cl_mem out, int n) {
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(ctx->k_relu, 0, sizeof(cl_mem), &in);
    err |= clSetKernelArg(ctx->k_relu, 1, sizeof(cl_mem), &out);
    err |= clSetKernelArg(ctx->k_relu, 2, sizeof(int), &n);
    CL_CHECK(err, "clSetKernelArg(relu)");
    run_1d(ctx, ctx->k_relu, n);
}

void gpu_relu_backward(GpuContext * ctx, cl_mem da, cl_mem z, cl_mem out, int n) {
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(ctx->k_relu_backward, 0, sizeof(cl_mem), &da);
    err |= clSetKernelArg(ctx->k_relu_backward, 1, sizeof(cl_mem), &z);
    err |= clSetKernelArg(ctx->k_relu_backward, 2, sizeof(cl_mem), &out);
    err |= clSetKernelArg(ctx->k_relu_backward, 3, sizeof(int), &n);
    CL_CHECK(err, "clSetKernelArg(relu_backward)");
    run_1d(ctx, ctx->k_relu_backward, n);
}

void gpu_softmax_rows(GpuContext * ctx, cl_mem Z, cl_mem A, int rows, int cols) {
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(ctx->k_softmax_rows, 0, sizeof(cl_mem), &Z);
    err |= clSetKernelArg(ctx->k_softmax_rows, 1, sizeof(cl_mem), &A);
    err |= clSetKernelArg(ctx->k_softmax_rows, 2, sizeof(int), &rows);
    err |= clSetKernelArg(ctx->k_softmax_rows, 3, sizeof(int), &cols);
    CL_CHECK(err, "clSetKernelArg(softmax_rows)");
    run_1d(ctx, ctx->k_softmax_rows, rows);
}

void gpu_col_sum(GpuContext * ctx, cl_mem M, cl_mem out, int rows, int cols) {
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(ctx->k_col_sum, 0, sizeof(cl_mem), &M);
    err |= clSetKernelArg(ctx->k_col_sum, 1, sizeof(cl_mem), &out);
    err |= clSetKernelArg(ctx->k_col_sum, 2, sizeof(int), &rows);
    err |= clSetKernelArg(ctx->k_col_sum, 3, sizeof(int), &cols);
    CL_CHECK(err, "clSetKernelArg(col_sum)");
    run_1d(ctx, ctx->k_col_sum, cols);
}

void gpu_vec_subtract(GpuContext * ctx, cl_mem a, cl_mem b, cl_mem out, int n) {
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(ctx->k_vec_subtract, 0, sizeof(cl_mem), &a);
    err |= clSetKernelArg(ctx->k_vec_subtract, 1, sizeof(cl_mem), &b);
    err |= clSetKernelArg(ctx->k_vec_subtract, 2, sizeof(cl_mem), &out);
    err |= clSetKernelArg(ctx->k_vec_subtract, 3, sizeof(int), &n);
    CL_CHECK(err, "clSetKernelArg(vec_subtract)");
    run_1d(ctx, ctx->k_vec_subtract, n);
}

void gpu_sgd_update(GpuContext * ctx, cl_mem param, cl_mem grad, float scaled_lr, int n) {
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(ctx->k_sgd_update, 0, sizeof(cl_mem), &param);
    err |= clSetKernelArg(ctx->k_sgd_update, 1, sizeof(cl_mem), &grad);
    err |= clSetKernelArg(ctx->k_sgd_update, 2, sizeof(float), &scaled_lr);
    err |= clSetKernelArg(ctx->k_sgd_update, 3, sizeof(int), &n);
    CL_CHECK(err, "clSetKernelArg(sgd_update)");
    run_1d(ctx, ctx->k_sgd_update, n);
}

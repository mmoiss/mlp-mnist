#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include "cl_ops.h"

#define A_0 784
#define L_0 128
#define L_1 64
#define L_2 10

#define LEARNING_RATE 0.3f
#define BATCH_SIZE 100
#define EPOCHS 10
#define OUTPUT_ITER 100
#define KERNEL_SOURCE_PATH "kernels.cl"

// Helper method for swapping big-endian 32-bit to host endianness
uint32_t swap_uint32(uint32_t val) {
    return ((val << 24) & 0xFF000000) |
           ((val << 8)  & 0x00FF0000) |
           ((val >> 8)  & 0x0000FF00) |
           ((val >> 24) & 0x000000FF);
}

int run_training(
    GpuContext * ctx,
    float * w_0, float * b_0,
    float * w_1, float * b_1,
    float * w_2, float * b_2)
{
    int status = 1;
    // Allocate Memory
    size_t i = 0;
    //# Data buffers and file handles
    float * all_images = NULL;
    uint8_t * temp_disk_buffer = NULL;
    uint8_t * all_labels = NULL;
    FILE * trn_f = NULL;
    FILE * lbl_f = NULL;
    //# Loss
    float loss_sum = 0.0f;
    //# Host-side scratch for one batch
    float * y_batch = (float *)malloc(BATCH_SIZE * L_2 * sizeof(float));
    float * a3_batch = (float *)malloc(BATCH_SIZE * L_2 * sizeof(float));

    //# Device buffers: parameters
    cl_mem dev_w0 = gpu_alloc(ctx, L_0 * A_0);
    cl_mem dev_b0 = gpu_alloc(ctx, L_0);
    cl_mem dev_w1 = gpu_alloc(ctx, L_1 * L_0);
    cl_mem dev_b1 = gpu_alloc(ctx, L_1);
    cl_mem dev_w2 = gpu_alloc(ctx, L_2 * L_1);
    cl_mem dev_b2 = gpu_alloc(ctx, L_2);
    //# Device buffers: forward pass (batched, rows = BATCH_SIZE)
    cl_mem dev_a0 = gpu_alloc(ctx, BATCH_SIZE * A_0);
    cl_mem dev_z0 = gpu_alloc(ctx, BATCH_SIZE * L_0);
    cl_mem dev_a1 = gpu_alloc(ctx, BATCH_SIZE * L_0);
    cl_mem dev_z1 = gpu_alloc(ctx, BATCH_SIZE * L_1);
    cl_mem dev_a2 = gpu_alloc(ctx, BATCH_SIZE * L_1);
    cl_mem dev_z2 = gpu_alloc(ctx, BATCH_SIZE * L_2);
    cl_mem dev_a3 = gpu_alloc(ctx, BATCH_SIZE * L_2);
    cl_mem dev_y = gpu_alloc(ctx, BATCH_SIZE * L_2);
    //# Device buffers: backpropagation
    cl_mem dev_dz2 = gpu_alloc(ctx, BATCH_SIZE * L_2);
    cl_mem dev_dw2 = gpu_alloc(ctx, L_2 * L_1);
    cl_mem dev_db2 = gpu_alloc(ctx, L_2);
    cl_mem dev_da2 = gpu_alloc(ctx, BATCH_SIZE * L_1);
    cl_mem dev_dz1 = gpu_alloc(ctx, BATCH_SIZE * L_1);
    cl_mem dev_dw1 = gpu_alloc(ctx, L_1 * L_0);
    cl_mem dev_db1 = gpu_alloc(ctx, L_1);
    cl_mem dev_da1 = gpu_alloc(ctx, BATCH_SIZE * L_0);
    cl_mem dev_dz0 = gpu_alloc(ctx, BATCH_SIZE * L_0);
    cl_mem dev_dw0 = gpu_alloc(ctx, L_0 * A_0);
    cl_mem dev_db0 = gpu_alloc(ctx, L_0);

    if (!y_batch || !a3_batch) {
        fprintf(stderr, "\nERROR: Out of memory\n");
        goto cleanup;
    }

    // Importing the Data
    const char * train_filename = "data/train-images.idx3-ubyte";
    const char * label_filename = "data/train-labels.idx1-ubyte";
    trn_f = fopen(train_filename, "rb");
    lbl_f = fopen(label_filename, "rb");
    if (!trn_f || !lbl_f) {
        perror("Failed to open file");
        goto cleanup;
    }

    uint32_t train_magic = 0;
    uint32_t num_images = 0;
    uint32_t train_rows = 0;
    uint32_t train_cols = 0;

    uint32_t label_magic = 0;
    uint32_t num_labels = 0;

    //# Read header values
    if (fread(&train_magic, sizeof(train_magic), 1, trn_f) != 1 ||
        fread(&num_images, sizeof(num_images), 1, trn_f) != 1 ||
        fread(&train_rows, sizeof(train_rows), 1, trn_f) != 1 ||
        fread(&train_cols, sizeof(train_cols), 1, trn_f) != 1) {
        fprintf(stderr, "\nFailed to read image file header\n");
        goto cleanup;
    }

    if (fread(&label_magic, sizeof(label_magic), 1, lbl_f) != 1 ||
        fread(&num_labels, sizeof(num_labels), 1, lbl_f) != 1) {
        fprintf(stderr, "\nFailed to read label file header\n");
        goto cleanup;
    }

    //# Convert from big-endian
    train_magic = swap_uint32(train_magic);
    num_images = swap_uint32(num_images);
    train_rows = swap_uint32(train_rows);
    train_cols = swap_uint32(train_cols);

    label_magic = swap_uint32(label_magic);
    num_labels = swap_uint32(num_labels);

    printf("\nTraining Data:\n\tMagic: 0x%X\n\tImages: %u\n\tRows: %u\n\tCols: %u\n", train_magic, num_images, train_rows, train_cols);
    printf("\nLabel Data:\n\tMagic: 0x%X\n\tLabels: %u\n", label_magic, num_labels);

    //# Magic number asserts
    if (train_magic != 2051) {
        fprintf(stderr, "\nInvalid IDX3 magic number!\n");
        goto cleanup;
    }
    if (label_magic != 2049) {
        fprintf(stderr, "\nInvalid IDX1 magic number!\n");
        goto cleanup;
    }
    if (num_labels != num_images) {
        fprintf(stderr, "\nWARNING: Label count (%u) does not match image count (%u)\n",
                num_labels, num_images);
        goto cleanup;
    }
    //# Allocating memory for pixel data
    size_t num_pixels = train_rows * train_cols;
    size_t total_elements = num_images * num_pixels;
    all_images = (float *)malloc(total_elements * sizeof(float));
    temp_disk_buffer = (uint8_t *)malloc(total_elements);

    //# Allocating memory for label data
    all_labels = (uint8_t *)malloc(num_labels);

    if (!all_images || !temp_disk_buffer || !all_labels) {
        fprintf(stderr, "\nOut of memory\n");
        goto cleanup;
    }

    //# Read all pixel data from disk at once
    if (fread(temp_disk_buffer, 1, total_elements, trn_f) != total_elements) {
        fprintf(stderr, "\nImage file truncated\n");
        goto cleanup;
    }
    //# Read all label data from disk at once
    if (fread(all_labels, 1, num_labels, lbl_f) != num_labels) {
        fprintf(stderr, "\nLabel file truncated\n");
        goto cleanup;
    }

    //# Cast and normalize the entire dataset
    for (size_t i = 0; i < total_elements; ++i) {
        all_images[i] = (float)temp_disk_buffer[i] / 255.0f;
    }

    //# Teardown temporary memory
    free(temp_disk_buffer);
    temp_disk_buffer = NULL;
    fclose(trn_f);
    trn_f = NULL;
    fclose(lbl_f);
    lbl_f = NULL;

    // Seeding random variables
    srand(time(NULL));
    //# First Layer
    for (i = 0; i < L_0; ++i) {
        b_0[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        for (size_t j = 0; j < A_0; ++j) {
            w_0[i * A_0 + j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    }
    //# Second Layer
    for (i = 0; i < L_1; ++i) {
        b_1[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        for (size_t j = 0; j < L_0; ++j) {
            w_1[i * L_0 + j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    }
    //# Third Layer
    for (i = 0; i < L_2; ++i) {
        b_2[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        for (size_t j = 0; j < L_1; ++j) {
            w_2[i * L_1 + j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }

    }

    gpu_write(ctx, dev_w0, w_0, L_0 * A_0);
    gpu_write(ctx, dev_b0, b_0, L_0);
    gpu_write(ctx, dev_w1, w_1, L_1 * L_0);
    gpu_write(ctx, dev_b1, b_1, L_1);
    gpu_write(ctx, dev_w2, w_2, L_2 * L_1);
    gpu_write(ctx, dev_b2, b_2, L_2);

    if (num_images % BATCH_SIZE != 0) {
        fprintf(stderr, "\nWARNING: %u images not evenly divisible by BATCH_SIZE=%d, dropping remainder\n",
                num_images, BATCH_SIZE);
    }
    size_t num_batches = num_images / BATCH_SIZE;
    size_t total_batches = num_batches * EPOCHS;

    printf("\nSimulation Start:\n\tSTEP\tAVG LOSS\n");
    for (i = 0; i < total_batches; ++i) {
        size_t batch_start = (i % num_batches) * BATCH_SIZE;

        // Defining inputs
        gpu_write(ctx, dev_a0, &all_images[batch_start * num_pixels], BATCH_SIZE * A_0);

        // Defining the one-hot label matrix
        memset(y_batch, 0, BATCH_SIZE * L_2 * sizeof(float));
        for (size_t b = 0; b < BATCH_SIZE; ++b) {
            y_batch[b * L_2 + all_labels[batch_start + b]] = 1.0f;
        }
        gpu_write(ctx, dev_y, y_batch, BATCH_SIZE * L_2);

        // Forward Pass
        //# First Layer
        gpu_matmul_bt(ctx, dev_a0, dev_w0, dev_z0, BATCH_SIZE, A_0, L_0);
        gpu_add_bias_cols(ctx, dev_z0, dev_b0, BATCH_SIZE, L_0);
        //# Second Layer
        gpu_relu(ctx, dev_z0, dev_a1, BATCH_SIZE * L_0);
        gpu_matmul_bt(ctx, dev_a1, dev_w1, dev_z1, BATCH_SIZE, L_0, L_1);
        gpu_add_bias_cols(ctx, dev_z1, dev_b1, BATCH_SIZE, L_1);
        //# Third Layer
        gpu_relu(ctx, dev_z1, dev_a2, BATCH_SIZE * L_1);
        gpu_matmul_bt(ctx, dev_a2, dev_w2, dev_z2, BATCH_SIZE, L_1, L_2);
        gpu_add_bias_cols(ctx, dev_z2, dev_b2, BATCH_SIZE, L_2);
        gpu_softmax_rows(ctx, dev_z2, dev_a3, BATCH_SIZE, L_2);

        //# Single batch's average loss
        if ((i + 1) % OUTPUT_ITER == 0) {
            gpu_read(ctx, dev_a3, a3_batch, BATCH_SIZE * L_2);
            loss_sum = 0.0f;
            for (size_t b = 0; b < BATCH_SIZE; ++b) {
                loss_sum -= logf(a3_batch[b * L_2 + all_labels[batch_start + b]] + 1e-7f);
            }
            unsigned int samples_seen = (unsigned int)((i + 1) * BATCH_SIZE);
            printf("\t%u\t%.4f\n", samples_seen, loss_sum / BATCH_SIZE);
        }

        // Back Propagation
        //# Third Layer
        gpu_vec_subtract(ctx, dev_a3, dev_y, dev_dz2, BATCH_SIZE * L_2);
        gpu_matmul_at(ctx, dev_dz2, dev_a2, dev_dw2, L_2, BATCH_SIZE, L_1);
        gpu_col_sum(ctx, dev_dz2, dev_db2, BATCH_SIZE, L_2);
        gpu_matmul(ctx, dev_dz2, dev_w2, dev_da2, BATCH_SIZE, L_2, L_1);
        //# Second Layer
        gpu_relu_backward(ctx, dev_da2, dev_z1, dev_dz1, BATCH_SIZE * L_1);
        gpu_matmul_at(ctx, dev_dz1, dev_a1, dev_dw1, L_1, BATCH_SIZE, L_0);
        gpu_col_sum(ctx, dev_dz1, dev_db1, BATCH_SIZE, L_1);
        gpu_matmul(ctx, dev_dz1, dev_w1, dev_da1, BATCH_SIZE, L_1, L_0);
        //# First Layer
        gpu_relu_backward(ctx, dev_da1, dev_z0, dev_dz0, BATCH_SIZE * L_0);
        gpu_matmul_at(ctx, dev_dz0, dev_a0, dev_dw0, L_0, BATCH_SIZE, A_0);
        gpu_col_sum(ctx, dev_dz0, dev_db0, BATCH_SIZE, L_0);

        // Updates
        float scaled_lr = LEARNING_RATE / (float)BATCH_SIZE;
        //# First Layer
        gpu_sgd_update(ctx, dev_w0, dev_dw0, scaled_lr, L_0 * A_0);
        gpu_sgd_update(ctx, dev_b0, dev_db0, scaled_lr, L_0);
        //# Second Layer
        gpu_sgd_update(ctx, dev_w1, dev_dw1, scaled_lr, L_1 * L_0);
        gpu_sgd_update(ctx, dev_b1, dev_db1, scaled_lr, L_1);
        //# Third Layer
        gpu_sgd_update(ctx, dev_w2, dev_dw2, scaled_lr, L_2 * L_1);
        gpu_sgd_update(ctx, dev_b2, dev_db2, scaled_lr, L_2);
    }

    // Copy trained parameters back to the host
    gpu_read(ctx, dev_w0, w_0, L_0 * A_0);
    gpu_read(ctx, dev_b0, b_0, L_0);
    gpu_read(ctx, dev_w1, w_1, L_1 * L_0);
    gpu_read(ctx, dev_b1, b_1, L_1);
    gpu_read(ctx, dev_w2, w_2, L_2 * L_1);
    gpu_read(ctx, dev_b2, b_2, L_2);

    status = 0;

    // Memory Teardown
cleanup:
    if (trn_f) fclose(trn_f);
    if (lbl_f) fclose(lbl_f);
    free(all_images);
    free(all_labels);
    free(temp_disk_buffer);
    free(y_batch);
    free(a3_batch);
    //# Device buffers: parameters
    clReleaseMemObject(dev_w0);
    clReleaseMemObject(dev_b0);
    clReleaseMemObject(dev_w1);
    clReleaseMemObject(dev_b1);
    clReleaseMemObject(dev_w2);
    clReleaseMemObject(dev_b2);
    //# Device buffers: forward pass
    clReleaseMemObject(dev_a0);
    clReleaseMemObject(dev_z0);
    clReleaseMemObject(dev_a1);
    clReleaseMemObject(dev_z1);
    clReleaseMemObject(dev_a2);
    clReleaseMemObject(dev_z2);
    clReleaseMemObject(dev_a3);
    clReleaseMemObject(dev_y);
    //# Device buffers: backpropagation
    clReleaseMemObject(dev_dz2);
    clReleaseMemObject(dev_dw2);
    clReleaseMemObject(dev_db2);
    clReleaseMemObject(dev_da2);
    clReleaseMemObject(dev_dz1);
    clReleaseMemObject(dev_dw1);
    clReleaseMemObject(dev_db1);
    clReleaseMemObject(dev_da1);
    clReleaseMemObject(dev_dz0);
    clReleaseMemObject(dev_dw0);
    clReleaseMemObject(dev_db0);

    return status;
}

int run_validation(
    GpuContext * ctx,
    float * w_0, float * b_0,
    float * w_1, float * b_1,
    float * w_2, float * b_2)
{
    int status = 1;
    // Allocate Memory
    size_t i = 0;
    //# Data buffers and file handles
    float * all_images = NULL;
    uint8_t * temp_disk_buffer = NULL;
    uint8_t * all_labels = NULL;
    FILE * evl_f = NULL;
    FILE * lbl_f = NULL;
    //# Host-side scratch for one batch
    float * a3_batch = (float *)malloc(BATCH_SIZE * L_2 * sizeof(float));

    //# Device buffers for parameters
    cl_mem dev_w0 = gpu_alloc(ctx, L_0 * A_0);
    cl_mem dev_b0 = gpu_alloc(ctx, L_0);
    cl_mem dev_w1 = gpu_alloc(ctx, L_1 * L_0);
    cl_mem dev_b1 = gpu_alloc(ctx, L_1);
    cl_mem dev_w2 = gpu_alloc(ctx, L_2 * L_1);
    cl_mem dev_b2 = gpu_alloc(ctx, L_2);
    //# Device buffers for forward pass
    cl_mem dev_a0 = gpu_alloc(ctx, BATCH_SIZE * A_0);
    cl_mem dev_z0 = gpu_alloc(ctx, BATCH_SIZE * L_0);
    cl_mem dev_a1 = gpu_alloc(ctx, BATCH_SIZE * L_0);
    cl_mem dev_z1 = gpu_alloc(ctx, BATCH_SIZE * L_1);
    cl_mem dev_a2 = gpu_alloc(ctx, BATCH_SIZE * L_1);
    cl_mem dev_z2 = gpu_alloc(ctx, BATCH_SIZE * L_2);
    cl_mem dev_a3 = gpu_alloc(ctx, BATCH_SIZE * L_2);

    if (!a3_batch) {
        fprintf(stderr, "\nERROR: Out of memory\n");
        goto cleanup;
    }

    gpu_write(ctx, dev_w0, w_0, L_0 * A_0);
    gpu_write(ctx, dev_b0, b_0, L_0);
    gpu_write(ctx, dev_w1, w_1, L_1 * L_0);
    gpu_write(ctx, dev_b1, b_1, L_1);
    gpu_write(ctx, dev_w2, w_2, L_2 * L_1);
    gpu_write(ctx, dev_b2, b_2, L_2);

    // Importing the Data
    const char * eval_filename = "data/t10k-images.idx3-ubyte";
    const char * label_filename = "data/t10k-labels.idx1-ubyte";
    evl_f = fopen(eval_filename, "rb");
    lbl_f = fopen(label_filename, "rb");
    if (!evl_f || !lbl_f) {
        perror("Failed to open file");
        goto cleanup;
    }

    uint32_t eval_magic = 0;
    uint32_t num_images = 0;
    uint32_t eval_rows = 0;
    uint32_t eval_cols = 0;

    uint32_t label_magic = 0;
    uint32_t num_labels = 0;

    //# Read header values
    if (fread(&eval_magic, sizeof(eval_magic), 1, evl_f) != 1 ||
        fread(&num_images, sizeof(num_images), 1, evl_f) != 1 ||
        fread(&eval_rows, sizeof(eval_rows), 1, evl_f) != 1 ||
        fread(&eval_cols, sizeof(eval_cols), 1, evl_f) != 1) {
        fprintf(stderr, "\nFailed to read image file header\n");
        goto cleanup;
    }

    if (fread(&label_magic, sizeof(label_magic), 1, lbl_f) != 1 ||
        fread(&num_labels, sizeof(num_labels), 1, lbl_f) != 1) {
        fprintf(stderr, "\nFailed to read label file header\n");
        goto cleanup;
    }

    //# Convert from big-endian
    eval_magic = swap_uint32(eval_magic);
    num_images = swap_uint32(num_images);
    eval_rows = swap_uint32(eval_rows);
    eval_cols = swap_uint32(eval_cols);

    label_magic = swap_uint32(label_magic);
    num_labels = swap_uint32(num_labels);

    printf("\nEvaluation Data:\n\tMagic: 0x%X\n\tImages: %u\n\tRows: %u\n\tCols: %u\n", eval_magic, num_images, eval_rows, eval_cols);
    printf("\nLabel Data:\n\tMagic: 0x%X\n\tLabels: %u\n", label_magic, num_labels);

    //# Magic number asserts
    if (eval_magic != 2051) {
        fprintf(stderr, "\nInvalid IDX3 magic number!\n");
        goto cleanup;
    }
    if (label_magic != 2049) {
        fprintf(stderr, "\nInvalid IDX1 magic number!\n");
        goto cleanup;
    }
    if (num_labels != num_images) {
        fprintf(stderr, "\nWARNING: Label count (%u) does not match image count (%u)\n",
                num_labels, num_images);
        goto cleanup;
    }
    //# Allocating memory for pixel data
    size_t num_pixels = eval_rows * eval_cols;
    size_t total_elements = num_images * num_pixels;
    all_images = (float *)malloc(total_elements * sizeof(float));
    temp_disk_buffer = (uint8_t *)malloc(total_elements);

    //# Allocating memory for label data
    all_labels = (uint8_t *)malloc(num_labels);

    if (!all_images || !temp_disk_buffer || !all_labels) {
        fprintf(stderr, "\nOut of memory\n");
        goto cleanup;
    }

    //# Read all pixel data from disk at once
    if (fread(temp_disk_buffer, 1, total_elements, evl_f) != total_elements) {
        fprintf(stderr, "\nImage file truncated\n");
        goto cleanup;
    }
    //# Read all label data from disk at once
    if (fread(all_labels, 1, num_labels, lbl_f) != num_labels) {
        fprintf(stderr, "\nLabel file truncated\n");
        goto cleanup;
    }

    //# Cast and normalize the entire dataset
    for (size_t i = 0; i < total_elements; ++i) {
        all_images[i] = (float)temp_disk_buffer[i] / 255.0f;
    }

    //# Teardown temporary memory
    free(temp_disk_buffer);
    temp_disk_buffer = NULL;
    fclose(evl_f);
    evl_f = NULL;
    fclose(lbl_f);
    lbl_f = NULL;

    if (num_images % BATCH_SIZE != 0) {
        fprintf(stderr, "\nWARNING: %u images not evenly divisible by BATCH_SIZE=%d, dropping remainder\n",
                num_images, BATCH_SIZE);
    }
    size_t num_batches = num_images / BATCH_SIZE;

    uint32_t count_correct = 0;
    for (i = 0; i < num_batches; ++i) {
        size_t batch_start = i * BATCH_SIZE;

        // Defining input values
        gpu_write(ctx, dev_a0, &all_images[batch_start * num_pixels], BATCH_SIZE * A_0);

        // Forward Pass
        //# First Layer
        gpu_matmul_bt(ctx, dev_a0, dev_w0, dev_z0, BATCH_SIZE, A_0, L_0);
        gpu_add_bias_cols(ctx, dev_z0, dev_b0, BATCH_SIZE, L_0);
        //# Second Layer
        gpu_relu(ctx, dev_z0, dev_a1, BATCH_SIZE * L_0);
        gpu_matmul_bt(ctx, dev_a1, dev_w1, dev_z1, BATCH_SIZE, L_0, L_1);
        gpu_add_bias_cols(ctx, dev_z1, dev_b1, BATCH_SIZE, L_1);
        //# Third Layer
        gpu_relu(ctx, dev_z1, dev_a2, BATCH_SIZE * L_1);
        gpu_matmul_bt(ctx, dev_a2, dev_w2, dev_z2, BATCH_SIZE, L_1, L_2);
        gpu_add_bias_cols(ctx, dev_z2, dev_b2, BATCH_SIZE, L_2);
        gpu_softmax_rows(ctx, dev_z2, dev_a3, BATCH_SIZE, L_2);

        gpu_read(ctx, dev_a3, a3_batch, BATCH_SIZE * L_2);

        // Validate Result
        for (size_t b = 0; b < BATCH_SIZE; ++b) {
            float * a3_row = &a3_batch[b * L_2];
            size_t max_id = 0;
            for (size_t j = 1; j < L_2; ++j) {
                if (a3_row[j] > a3_row[max_id]) max_id = j;
            }
            if (all_labels[batch_start + b] == max_id) {
                ++count_correct;
            }
        }
    }
    printf("\nValidation Results: %u / %u correct\n", count_correct, (unsigned int)(num_batches * BATCH_SIZE));

    status = 0;
        // Memory Teardown
cleanup:
    if (evl_f) fclose(evl_f);
    if (lbl_f) fclose(lbl_f);
    free(all_images);
    free(all_labels);
    free(temp_disk_buffer);
    free(a3_batch);
    //# Device buffers: parameters
    clReleaseMemObject(dev_w0);
    clReleaseMemObject(dev_b0);
    clReleaseMemObject(dev_w1);
    clReleaseMemObject(dev_b1);
    clReleaseMemObject(dev_w2);
    clReleaseMemObject(dev_b2);
    //# Device buffers: forward pass
    clReleaseMemObject(dev_a0);
    clReleaseMemObject(dev_z0);
    clReleaseMemObject(dev_a1);
    clReleaseMemObject(dev_z1);
    clReleaseMemObject(dev_a2);
    clReleaseMemObject(dev_z2);
    clReleaseMemObject(dev_a3);

    return status;
}

int main() {
    int err;

    GpuContext ctx;
    if (gpu_init(&ctx, KERNEL_SOURCE_PATH) != 0) {
        fprintf(stderr, "\nFailed to initialize OpenCL\n");
        return 1;
    }

    float * b_0 = (float *)malloc(L_0 * sizeof(float));
    float * w_0 = (float *)malloc(L_0 * A_0 * sizeof(float));
    float * b_1 = (float *)malloc(L_1 * sizeof(float));
    float * w_1 = (float *)malloc(L_1 * L_0 * sizeof(float));
    float * b_2 = (float *)malloc(L_2 * sizeof(float));
    float * w_2 = (float *)malloc(L_2 * L_1 * sizeof(float));

    err = run_training(&ctx, w_0, b_0, w_1, b_1, w_2, b_2);
    err |= run_validation(&ctx, w_0, b_0, w_1, b_1, w_2, b_2);

    free(w_0);
    free(b_0);
    free(w_1);
    free(b_1);
    free(w_2);
    free(b_2);

    gpu_shutdown(&ctx);

    return err;
}

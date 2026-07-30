#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define A_0 784
#define L_0 128
#define L_1 64
#define L_2 10

#define LEARNING_RATE 0.01f
#define OUTPUT_ITER 1000

// Helper method for swapping big-endian 32-bit to host endianness
uint32_t swap_uint32(uint32_t val) {
    return ((val << 24) & 0xFF000000) |
           ((val << 8)  & 0x00FF0000) |
           ((val >> 8)  & 0x0000FF00) |
           ((val >> 24) & 0x000000FF);
}

void ReLU(float * v, float * out, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        out[i] = (v[i] >= 0) ? v[i] : 0;
    }
}

void d_ReLU(float * v, float * out, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        out[i] = (v[i] >= 0) ? 1 : 0;
    }
}

void mat_vec_mult(float * m, float * v, float * out, size_t num_rows, size_t num_cols) {
    for (size_t i = 0; i < num_rows; ++i) {
        float temp_sum = 0.0f;
        for (size_t j = 0; j < num_cols; ++j) {
            temp_sum += m[i * num_cols + j] * v[j];
        }
        out[i] = temp_sum; 
    }
}

void mat_scalar_mult(float * m, float scalar, float * out, size_t num_rows, size_t num_cols) {
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < num_cols; ++j) {
            out[i * num_cols + j] = m[i * num_cols + j] * scalar;
        }
    }
}

void mat_mat_subtract(float * m_1, float * m_2, float * out, size_t num_rows, size_t num_cols) {
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < num_cols; ++j) {
            out[i * num_cols + j] = m_1[i * num_cols + j] - m_2[i * num_cols + j];
        }
    }
}

void vec_vec_mult(float * v_1, float * v_2, float * out, size_t v_1_len, size_t v_2_len) {
    for (size_t i = 0; i < v_1_len; ++i) {
        for (size_t j = 0; j < v_2_len; ++j) {
           out[i * v_2_len + j] = v_1[i] * v_2[j];
        }
    }
}

void vec_vec_sum(float * v_1, float *v_2, float * out, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        out[i] = v_1[i] + v_2[i];
    }
}

void vec_vec_subtract(float * v_1, float * v_2, float * out, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        out[i] = v_1[i] - v_2[i];
    }
}

void vec_scalar_mult(float * v, float scalar, float * out, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        out[i] = v[i] * scalar; 
    }
}

void transpose_mat(float * mat, float * out, size_t num_rows, size_t num_cols) {
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < num_cols; ++j) {
            out[j * num_rows + i] = mat[i * num_cols + j];
        }
    }
}

void hadamard(float * v_1, float * v_2, float * out, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        out[i] = v_1[i] * v_2[i]; 
    }
}

void softmax(float * v, float * out, size_t length) {
    float max_v = v[0];
    for (size_t i = 1; i < length; ++i) {
        if (v[i] > max_v) max_v = v[i];
    }
    float total = 0.0f;
    for (size_t j = 0; j < length; ++j) {
        total += exp(v[j] - max_v);
    }
    for (size_t i = 0; i < length; ++i) {
        out[i] = exp(v[i] - max_v) / total;
    }
}

int run_training(
    float * w_0, float * b_0,
    float * w_1, float * b_1,
    float * w_2, float * b_2)
{
    int status = 1;
    // Allocate Memory
    size_t i, j = 0;
    float temp_arr[A_0 * L_0];
    //# Data buffers and file handles
    float * all_images = NULL;
    uint8_t * temp_disk_buffer = NULL;
    uint8_t * all_labels = NULL;
    FILE * trn_f = NULL;
    FILE * lbl_f = NULL;
    //# Loss
    float loss_sum = 0.0f;
    //# Forward pass
    //## First Layer
    float * a_0 = (float *)malloc(A_0 * sizeof(float));
    float * z_0 = (float *)malloc(L_0 * sizeof(float));
    //## Second Layer
    float * a_1 = (float *)malloc(L_0 * sizeof(float));
    float * z_1 = (float *)malloc(L_1 * sizeof(float));
    //## Third Layer
    float * a_2 = (float *)malloc(L_1 * sizeof(float));
    float * z_2 = (float *)malloc(L_2 * sizeof(float));
    //## Output
    float * a_3 = (float *)malloc(L_2 * sizeof(float));

    //# Backpropagation
    float * y = (float *)malloc(L_2 * sizeof(float));
    //## Third Layer
    float * dz_2 = (float *)malloc(L_2 * sizeof(float));
    float * dw_2 = (float *)malloc(L_2 * L_1 * sizeof(float));
    float * db_2 = dz_2;
    float * da_2 = (float *)malloc(L_1 * sizeof(float));
    //## Second Layer
    float * dz_1 = (float *)malloc(L_1 * sizeof(float));
    float * dw_1 = (float *)malloc(L_1 * L_0 * sizeof(float));
    float * db_1 = dz_1;
    float * da_1 = (float *)malloc(L_0 * sizeof(float));
    //## First Layer
    float * dz_0 = (float *)malloc(L_0 * sizeof(float));
    float * dw_0 = (float *)malloc(L_0 * A_0 * sizeof(float));
    float * db_0 = dz_0;

    //# Verify layer allocations
    if (!a_0 || !b_0 || !w_0 || !z_0 ||
        !a_1 || !b_1 || !w_1 || !z_1 ||
        !a_2 || !b_2 || !w_2 || !z_2 ||
        !a_3 || !y ||
        !dz_2 || !dw_2 || !da_2 ||
        !dz_1 || !dw_1 || !da_1 ||
        !dz_0 || !dw_0) {
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
        for (j = 0; j < A_0; ++j) {
            w_0[i * A_0 + j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    }
    //# Second Layer
    for (i = 0; i < L_1; ++i) {
        b_1[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        for (j = 0; j < L_0; ++j) {
            w_1[i * L_0 + j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    }
    //# Third Layer
    for (i = 0; i < L_2; ++i) {
        b_2[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        for (j = 0; j < L_1; ++j) {
            w_2[i * L_1 + j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }

    }

    printf("\nSimulation Start:\n\tSTEP\tAVG LOSS\n");
    for (i = 0; i < num_images; ++i) {

        // Defining Inputs
        memcpy(a_0, &all_images[i * num_pixels], A_0 * sizeof(float));

        // Defining the label vector
        memset(y, 0, L_2 * sizeof(float));
        y[all_labels[i]] = 1.0f;

        // Forward Pass
        //# First Layer
        mat_vec_mult(w_0, a_0, temp_arr, L_0, A_0);
        vec_vec_sum(temp_arr, b_0, z_0, L_0);
        //# Second Layer
        ReLU(z_0, a_1, L_0);
        mat_vec_mult(w_1, a_1, temp_arr, L_1, L_0);
        vec_vec_sum(temp_arr, b_1, z_1, L_1);
        //# Third Layer
        ReLU(z_1, a_2, L_1);
        mat_vec_mult(w_2, a_2, temp_arr, L_2, L_1);
        vec_vec_sum(temp_arr, b_2, z_2, L_2);
        softmax(z_2, a_3, L_2);

        //# Calculate loss
        loss_sum -= logf(a_3[all_labels[i]] + 1e-7f);
        if ((i + 1)%OUTPUT_ITER == 0) {
            printf("\t%lu\t%.4f\n", i + 1, loss_sum / OUTPUT_ITER);
            loss_sum = 0.0f;
        }

        
        // Back Propagation
        //# Third Layer
        vec_vec_subtract(a_3, y, dz_2, L_2);
        vec_vec_mult(dz_2, a_2, dw_2, L_2, L_1);
        transpose_mat(w_2, temp_arr, L_2, L_1);
        mat_vec_mult(temp_arr, dz_2, da_2, L_1, L_2);
        //# Second Layer
        d_ReLU(z_1, temp_arr, L_1);
        hadamard(da_2, temp_arr, dz_1, L_1);
        vec_vec_mult(dz_1, a_1, dw_1, L_1, L_0);
        transpose_mat(w_1, temp_arr, L_1, L_0);
        mat_vec_mult(temp_arr, dz_1, da_1, L_0, L_1);
        //# First Layer
        d_ReLU(z_0, temp_arr, L_0);
        hadamard(da_1, temp_arr, dz_0, L_0);
        vec_vec_mult(dz_0, a_0, dw_0, L_0, A_0);

        // Updates
        //# First Layer
        mat_scalar_mult(dw_0, LEARNING_RATE, temp_arr, L_0, A_0);
        mat_mat_subtract(w_0, temp_arr, w_0, L_0, A_0);
        vec_scalar_mult(db_0, LEARNING_RATE, temp_arr, L_0);
        vec_vec_subtract(b_0, temp_arr, b_0, L_0);
        //# Second  Layer
        mat_scalar_mult(dw_1, LEARNING_RATE, temp_arr, L_1, L_0);
        mat_mat_subtract(w_1, temp_arr, w_1, L_1, L_0);
        vec_scalar_mult(db_1, LEARNING_RATE, temp_arr, L_1);
        vec_vec_subtract(b_1, temp_arr, b_1, L_1);
        //# Third Layer
        mat_scalar_mult(dw_2, LEARNING_RATE, temp_arr, L_2, L_1);
        mat_mat_subtract(w_2, temp_arr, w_2, L_2, L_1);
        vec_scalar_mult(db_2, LEARNING_RATE, temp_arr, L_2);
        vec_vec_subtract(b_2, temp_arr, b_2, L_2);

        
    }

    status = 0;

    // Memory Teardown
cleanup:
    if (trn_f) fclose(trn_f);
    if (lbl_f) fclose(lbl_f);
    free(all_images);
    free(all_labels);
    free(temp_disk_buffer);
    //# Forward Pass
    //## First Layer
    free(a_0);
    free(z_0);
    //## Second Layer
    free(a_1);
    free(z_1);
    //## Third Layer
    free(a_2);
    free(z_2);
    //## Output
    free(a_3);
    //# Backpropagation
    free(y);
    //## Third Layer
    free(dz_2);
    free(dw_2);
    //### db_2 was already freed by dz_2
    free(da_2);
    //## Second Layer
    free(dz_1);
    free(dw_1);
    //### db_1 was already freed by dz_1
    free(da_1);
    //## First Layer
    free(dz_0);
    free(dw_0);
    //### db_0 was already freed by dz_0

    return status;
}

int run_validation(
    float * w_0, float * b_0,
    float * w_1, float * b_1,
    float * w_2, float * b_2)
{    
    int status = 1;
    // Allocate Memory
    size_t i, j = 0;
    float temp_arr[A_0 * L_0];
    //# Data buffers and file handles
    float * all_images = NULL;
    uint8_t * temp_disk_buffer = NULL;
    uint8_t * all_labels = NULL;
    FILE * evl_f = NULL;
    FILE * lbl_f = NULL;
    //# Forward pass
    //## First Layer
    float * a_0 = (float *)malloc(A_0 * sizeof(float));
    float * z_0 = (float *)malloc(L_0 * sizeof(float));
    //## Second Layer
    float * a_1 = (float *)malloc(L_0 * sizeof(float));
    float * z_1 = (float *)malloc(L_1 * sizeof(float));
    //## Third Layer
    float * a_2 = (float *)malloc(L_1 * sizeof(float));
    float * z_2 = (float *)malloc(L_2 * sizeof(float));
    //## Output
    float * a_3 = (float *)malloc(L_2 * sizeof(float));

    //# Verify layer allocations
    if (!a_0 || !z_0 ||
        !a_1 || !z_1 ||
        !a_2 || !z_2 ||
        !a_3) {
        fprintf(stderr, "\nERROR: Out of memory\n");
        goto cleanup;
    }

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

    uint32_t count_correct = 0;
    for (i = 0; i < num_images; ++i) {
        // Defining input values
        memcpy(a_0, &all_images[i * num_pixels], A_0 * sizeof(float));

        // Forward Pass
        //# First Layer
        mat_vec_mult(w_0, a_0, temp_arr, L_0, A_0);
        vec_vec_sum(temp_arr, b_0, z_0, L_0);
        //# Second Layer
        ReLU(z_0, a_1, L_0);
        mat_vec_mult(w_1, a_1, temp_arr, L_1, L_0);
        vec_vec_sum(temp_arr, b_1, z_1, L_1);
        //# Third Layer
        ReLU(z_1, a_2, L_1);
        mat_vec_mult(w_2, a_2, temp_arr, L_2, L_1);
        vec_vec_sum(temp_arr, b_2, z_2, L_2);
        softmax(z_2, a_3, L_2);

        // Validate Result
        size_t max_id = 0;
        for (j = 1; j < L_2; ++j) {
            if (a_3[j] > a_3[max_id]) max_id = j;
        }
        if (all_labels[i] == max_id) {
            ++count_correct;
        }
    }
    printf("\nValidation Results: %u / %u correct\n", count_correct, num_images);

    status = 0;
        // Memory Teardown
cleanup:
    if (evl_f) fclose(evl_f);
    if (lbl_f) fclose(lbl_f);
    free(all_images);
    free(all_labels);
    free(temp_disk_buffer);
    //# Forward Pass
    //## First Layer
    free(a_0);
    free(z_0);
    //## Second Layer
    free(a_1);
    free(z_1);
    //## Third Layer
    free(a_2);
    free(z_2);
    //## Output
    free(a_3);

    return status;
}

int main() {
    int err;

    float * b_0 = (float *)malloc(L_0 * sizeof(float));
    float * w_0 = (float *)malloc(L_0 * A_0 * sizeof(float));
    float * b_1 = (float *)malloc(L_1 * sizeof(float));
    float * w_1 = (float *)malloc(L_1 * L_0 * sizeof(float));
    float * b_2 = (float *)malloc(L_2 * sizeof(float));
    float * w_2 = (float *)malloc(L_2 * L_1 * sizeof(float));

    err = run_training(w_0, b_0, w_1, b_1, w_2, b_2);
    err |= run_validation(w_0, b_0, w_1, b_1, w_2, b_2);

    free(w_0);
    free(b_0);
    free(w_1);
    free(b_1);
    free(w_2);
    free(b_2);

    return err;
}

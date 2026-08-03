// C = A * B, A: MxK, B: KxN, C: MxN
__kernel void matmul(__global const float * A,
                      __global const float * B,
                      __global float * C,
                      const int M, const int K, const int N)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// C = A^T * B, A stored as KxM, B: KxN, C: MxN
__kernel void matmul_at(__global const float * A,
                         __global const float * B,
                         __global float * C,
                         const int M, const int K, const int N)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[k * M + row] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// C = A * B^T, A: MxK, B stored as NxK, C: MxN
__kernel void matmul_bt(__global const float * A,
                         __global const float * B,
                         __global float * C,
                         const int M, const int K, const int N)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[col * K + k];
    }
    C[row * N + col] = sum;
}

// Adds bias[col] to every row of Z
__kernel void add_bias_cols(__global float * Z,
                             __global const float * bias,
                             const int rows, const int cols)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= rows || col >= cols) return;
    Z[row * cols + col] += bias[col];
}

__kernel void relu(__global const float * in,
                    __global float * out,
                    const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    out[i] = (in[i] >= 0.0f) ? in[i] : 0.0f;
}

// out = (z >= 0) ? da : 0  (hadamard product of da with dReLU(z), fused)
__kernel void relu_backward(__global const float * da,
                             __global const float * z,
                             __global float * out,
                             const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    out[i] = (z[i] >= 0.0f) ? da[i] : 0.0f;
}

// One work-item per row; softmax over that row's cols entries
__kernel void softmax_rows(__global const float * Z,
                            __global float * A,
                            const int rows, const int cols)
{
    int row = get_global_id(0);
    if (row >= rows) return;

    __global const float * z_row = Z + row * cols;
    __global float * a_row = A + row * cols;

    float max_v = z_row[0];
    for (int j = 1; j < cols; ++j) {
        if (z_row[j] > max_v) max_v = z_row[j];
    }
    float total = 0.0f;
    for (int j = 0; j < cols; ++j) {
        total += exp(z_row[j] - max_v);
    }
    for (int j = 0; j < cols; ++j) {
        a_row[j] = exp(z_row[j] - max_v) / total;
    }
}

// out[col] = sum over rows of M[row][col], M: rows x cols
__kernel void col_sum(__global const float * M,
                       __global float * out,
                       const int rows, const int cols)
{
    int col = get_global_id(0);
    if (col >= cols) return;

    float sum = 0.0f;
    for (int row = 0; row < rows; ++row) {
        sum += M[row * cols + col];
    }
    out[col] = sum;
}

__kernel void vec_subtract(__global const float * a,
                            __global const float * b,
                            __global float * out,
                            const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    out[i] = a[i] - b[i];
}

// param -= scaled_lr * grad
__kernel void sgd_update(__global float * param,
                          __global const float * grad,
                          const float scaled_lr,
                          const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    param[i] -= scaled_lr * grad[i];
}

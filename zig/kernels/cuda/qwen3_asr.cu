#include <cuda_bf16.h>
#include <math.h>

using B = __nv_bfloat16;

// Eight rows share each weight load; each dot retains sixteen ordered FP32 chains and the same reduction tree.
extern "C" __global__ void __launch_bounds__(64) tf_asr_rows_gemm(const B *x, const B *w, B *out,
                                                                int rows, int n, int k) {
    const int lane = threadIdx.x, column_lane = threadIdx.y;
    const int column = blockIdx.x * 4 + column_lane;
    __shared__ float sums[8][4][16];
    float acc[8] = {};
    for (int base = lane; base < k; base += 128) {
        float weights[8], inputs[8][8];
        // Prefetch eight K positions; each accumulator still visits them in increasing order.
#pragma unroll
        for (int part = 0; part < 8; ++part) {
            const int q = base + part * 16;
            if (q < k) {
                weights[part] = column < n ? __bfloat162float(w[(long long)column * k + q]) : 0.f;
#pragma unroll
                for (int row = 0; row < 8; ++row)
                    if (row < rows) inputs[row][part] = __bfloat162float(x[(long long)row * k + q]);
            }
        }
#pragma unroll
        for (int part = 0; part < 8; ++part)
            if (base + part * 16 < k) {
#pragma unroll
                for (int row = 0; row < 8; ++row)
                    if (row < rows) acc[row] = __fmaf_rn(inputs[row][part], weights[part], acc[row]);
            }
    }
#pragma unroll
    for (int row = 0; row < 8; ++row)
        if (row < rows) sums[row][column_lane][lane] = acc[row];
    __syncthreads();
    for (int step = 8; step; step /= 2) {
        if (lane < step) {
#pragma unroll
            for (int row = 0; row < 8; ++row)
                if (row < rows)
                    sums[row][column_lane][lane] =
                        __fadd_rn(sums[row][column_lane][lane], sums[row][column_lane][lane + step]);
        }
        __syncthreads();
    }
    if (lane == 0 && column < n) {
#pragma unroll
        for (int row = 0; row < 8; ++row)
            if (row < rows) out[(long long)row * n + column] = __float2bfloat16_rn(sums[row][column_lane][0]);
    }
}

extern "C" __global__ void tf_asr_im2col(const void *input, B *out, int height, int width,
                                       int channels, int frames, int start, int mel) {
    const int oh = (height + 1) / 2, ow = (width + 1) / 2;
    const long long count = (long long)oh * ow * channels * 9;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += (long long)blockDim.x * gridDim.x) {
        const int k = i % (channels * 9), row = i / (channels * 9);
        const int c = k / 9, y = row / ow * 2 + k / 3 % 3 - 1, x = row % ow * 2 + k % 3 - 1;
        float v = 0;
        if (y >= 0 && y < height && x >= 0 && x < width) {
            if (mel) {
                if (start + x < frames) v = ((const float *)input)[y * frames + start + x];
            } else v = __bfloat162float(((const B *)input)[((long long)y * width + x) * channels + c]);
        }
        out[i] = __float2bfloat16_rn(v);
    }
}

extern "C" __global__ void tf_asr_conv_pack(const B *x, B *out, int time, int frequency, int channels) {
    const int n = time * frequency * channels;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const int t = i / (frequency * channels), c = i / frequency % channels, f = i % frequency;
        out[i] = x[(f * time + t) * channels + c];
    }
}

extern "C" __global__ void tf_asr_audio_attention(const B *q, const B *k, const B *v, B *out,
                                                 int rows, int window) {
    const int row = blockIdx.x, head = blockIdx.y, tid = threadIdx.x;
    const int first = row / window * window, end = min(first + window, rows);
    const int count = end - first;
    __shared__ float scores[128];
    __shared__ float reduce[128];
    float dot = 0;
    if (tid < count) {
        for (int d = 0; d < 64; ++d)
            dot += __bfloat162float(q[row * 1024 + head * 64 + d]) *
                   __bfloat162float(k[(first + tid) * 1024 + head * 64 + d]);
        dot *= 0.125f;
    } else dot = -INFINITY;
    scores[tid] = dot;
    reduce[tid] = dot;
    __syncthreads();
    for (int step = 64; step; step /= 2) {
        if (tid < step) {
            const float a = reduce[tid], b = reduce[tid + step];
            reduce[tid] = isnan(a) || isnan(b) ? NAN : fmaxf(a, b);
        }
        __syncthreads();
    }
    const float maximum = reduce[0];
    __syncthreads();
    const float weight = tid < count ? expf(scores[tid] - maximum) : 0;
    scores[tid] = weight;
    reduce[tid] = weight;
    __syncthreads();
    for (int step = 64; step; step /= 2) {
        if (tid < step) reduce[tid] += reduce[tid + step];
        __syncthreads();
    }
    const float sum = reduce[0];
    if (tid < 64) {
        float acc = 0;
        for (int j = 0; j < count; ++j)
            acc += (scores[j] / sum) * __bfloat162float(v[(first + j) * 1024 + head * 64 + tid]);
        out[row * 1024 + head * 64 + tid] = __float2bfloat16_rn(acc);
    }
}

/// EmbedKit - Tensor Pooling Kernels (Metal 4 Optimized)
///
/// Batch-optimized pooling operations that process entire batches in single dispatches.
/// These kernels are designed for the tensor shapes commonly used in embedding models:
/// - Input: [batchSize, sequenceLength, dimensions]
/// - Output: [batchSize, dimensions]
///
/// **Metal 4 Optimizations**:
/// - Process entire batch in single dispatch (reduced command buffer overhead)
/// - Coalesced memory access patterns for tensor layouts
/// - SIMD-optimized reductions within each sequence
/// - Designed for unified encoder pipeline chaining
///
/// **Compatibility**: Metal 3.0+ (iOS 16+ / macOS 13+)

#include "../Common/MetalCommon.h"

// ============================================================================
// MARK: - Batch Mean Pooling
// ============================================================================

/// Batch mean pooling - processes all sequences in parallel
///
/// Each threadgroup handles one (batch, dimension) pair.
/// Reduces [batchSize, seqLen, dims] → [batchSize, dims]
///
/// Thread grid: (dimensions, batchSize, 1)
///
/// **Algorithm**:
/// For each batch b, dimension d:
///   output[b, d] = mean(input[b, :, d])
///
/// @param input Token embeddings [batchSize * sequenceLength * dimensions]
/// @param output Pooled embeddings [batchSize * dimensions]
/// @param mask Optional attention masks [batchSize * sequenceLength]
/// @param params Tensor pooling parameters
///
kernel void tensor_mean_pool(
    device const float* input       [[buffer(0)]],
    device float* output            [[buffer(1)]],
    device const int32_t* mask      [[buffer(2)]],
    constant TensorPoolingParams& params [[buffer(3)]],
    uint2 gid                       [[thread_position_in_grid]]
) {
    const int d = gid.x;  // dimension index
    const int b = gid.y;  // batch index

    if (d >= params.dimensions || b >= params.batchSize) return;

    const int seqLen = params.sequenceLength;
    const int dims = params.dimensions;

    // Calculate base offsets
    const int batchInputOffset = b * seqLen * dims;
    const int batchMaskOffset = b * seqLen;
    const int outputIdx = b * dims + d;

    float sum = 0.0f;
    int count = 0;

    // Accumulate with unrolling for better ILP
    int t = 0;
    for (; t <= seqLen - 4; t += 4) {
        const bool m0 = !mask || mask[batchMaskOffset + t] == 1;
        const bool m1 = !mask || mask[batchMaskOffset + t + 1] == 1;
        const bool m2 = !mask || mask[batchMaskOffset + t + 2] == 1;
        const bool m3 = !mask || mask[batchMaskOffset + t + 3] == 1;

        sum = fma(m0 ? input[batchInputOffset + t * dims + d] : 0.0f, 1.0f, sum);
        sum = fma(m1 ? input[batchInputOffset + (t + 1) * dims + d] : 0.0f, 1.0f, sum);
        sum = fma(m2 ? input[batchInputOffset + (t + 2) * dims + d] : 0.0f, 1.0f, sum);
        sum = fma(m3 ? input[batchInputOffset + (t + 3) * dims + d] : 0.0f, 1.0f, sum);

        count += m0 + m1 + m2 + m3;
    }

    // Handle remaining
    for (; t < seqLen; t++) {
        if (!mask || mask[batchMaskOffset + t] == 1) {
            sum = fma(input[batchInputOffset + t * dims + d], 1.0f, sum);
            count++;
        }
    }

    output[outputIdx] = count > 0 ? sum * (1.0f / float(count)) : 0.0f;
}

// ============================================================================
// MARK: - Batch Max Pooling
// ============================================================================

/// Batch max pooling - element-wise maximum across sequences
///
/// Thread grid: (dimensions, batchSize, 1)
///
kernel void tensor_max_pool(
    device const float* input       [[buffer(0)]],
    device float* output            [[buffer(1)]],
    device const int32_t* mask      [[buffer(2)]],
    constant TensorPoolingParams& params [[buffer(3)]],
    uint2 gid                       [[thread_position_in_grid]]
) {
    const int d = gid.x;
    const int b = gid.y;

    if (d >= params.dimensions || b >= params.batchSize) return;

    const int seqLen = params.sequenceLength;
    const int dims = params.dimensions;

    const int batchInputOffset = b * seqLen * dims;
    const int batchMaskOffset = b * seqLen;
    const int outputIdx = b * dims + d;

    float maxVal = -INFINITY;
    bool foundValid = false;

    // Unrolled max finding
    int t = 0;
    for (; t <= seqLen - 4; t += 4) {
        const bool m0 = !mask || mask[batchMaskOffset + t] == 1;
        const bool m1 = !mask || mask[batchMaskOffset + t + 1] == 1;
        const bool m2 = !mask || mask[batchMaskOffset + t + 2] == 1;
        const bool m3 = !mask || mask[batchMaskOffset + t + 3] == 1;

        if (m0) {
            maxVal = metal::max(maxVal, input[batchInputOffset + t * dims + d]);
            foundValid = true;
        }
        if (m1) {
            maxVal = metal::max(maxVal, input[batchInputOffset + (t + 1) * dims + d]);
            foundValid = true;
        }
        if (m2) {
            maxVal = metal::max(maxVal, input[batchInputOffset + (t + 2) * dims + d]);
            foundValid = true;
        }
        if (m3) {
            maxVal = metal::max(maxVal, input[batchInputOffset + (t + 3) * dims + d]);
            foundValid = true;
        }
    }

    for (; t < seqLen; t++) {
        if (!mask || mask[batchMaskOffset + t] == 1) {
            maxVal = metal::max(maxVal, input[batchInputOffset + t * dims + d]);
            foundValid = true;
        }
    }

    output[outputIdx] = foundValid ? maxVal : 0.0f;
}

// ============================================================================
// MARK: - Batch CLS Pooling
// ============================================================================

/// Batch CLS pooling - extract first token embedding for each sequence
///
/// Simply copies the first token (CLS token) embedding for each batch.
/// Highly memory-bound operation, optimal with coalesced access.
///
/// Thread grid: (dimensions, batchSize, 1)
///
kernel void tensor_cls_pool(
    device const float* input       [[buffer(0)]],
    device float* output            [[buffer(1)]],
    constant TensorPoolingParams& params [[buffer(3)]],
    uint2 gid                       [[thread_position_in_grid]]
) {
    const int d = gid.x;
    const int b = gid.y;

    if (d >= params.dimensions || b >= params.batchSize) return;

    const int dims = params.dimensions;
    const int seqLen = params.sequenceLength;

    // CLS token is at position 0 of each sequence
    const int inputIdx = b * seqLen * dims + d;  // First token (t=0)
    const int outputIdx = b * dims + d;

    output[outputIdx] = input[inputIdx];
}

// ============================================================================
// MARK: - Unified Tensor Pooling (Strategy Selection)
// ============================================================================

/// Unified batch pooling kernel with strategy selection
///
/// Single entry point for all pooling strategies, selectable via params.
/// Reduces pipeline state management overhead when using multiple strategies.
///
/// Thread grid: (dimensions, batchSize, 1)
///
/// @param params.poolingStrategy: 0=mean, 1=max, 2=cls
///
kernel void tensor_pool_unified(
    device const float* input       [[buffer(0)]],
    device float* output            [[buffer(1)]],
    device const int32_t* mask      [[buffer(2)]],
    constant TensorPoolingParams& params [[buffer(3)]],
    uint2 gid                       [[thread_position_in_grid]]
) {
    const int d = gid.x;
    const int b = gid.y;

    if (d >= params.dimensions || b >= params.batchSize) return;

    const int seqLen = params.sequenceLength;
    const int dims = params.dimensions;
    const int strategy = params.poolingStrategy;

    const int batchInputOffset = b * seqLen * dims;
    const int batchMaskOffset = b * seqLen;
    const int outputIdx = b * dims + d;

    float result = 0.0f;

    switch (strategy) {
        case 0: {  // Mean pooling
            float sum = 0.0f;
            int count = 0;
            for (int t = 0; t < seqLen; t++) {
                if (!mask || mask[batchMaskOffset + t] == 1) {
                    sum = fma(input[batchInputOffset + t * dims + d], 1.0f, sum);
                    count++;
                }
            }
            result = count > 0 ? sum * (1.0f / float(count)) : 0.0f;
            break;
        }
        case 1: {  // Max pooling
            float maxVal = -INFINITY;
            bool found = false;
            for (int t = 0; t < seqLen; t++) {
                if (!mask || mask[batchMaskOffset + t] == 1) {
                    maxVal = metal::max(maxVal, input[batchInputOffset + t * dims + d]);
                    found = true;
                }
            }
            result = found ? maxVal : 0.0f;
            break;
        }
        case 2: {  // CLS pooling (first token)
            result = input[batchInputOffset + d];
            break;
        }
    }

    output[outputIdx] = result;
}

// ============================================================================
// MARK: - Optimized Threadgroup Pooling (For Large Sequences)
// ============================================================================

/// Threadgroup-cooperative mean pooling for large sequences
///
/// Uses shared memory and SIMD reductions for sequences where seqLen >> dims.
/// Each threadgroup processes one (batch, dimension) with threads splitting
/// the sequence.
///
/// Thread grid: (dims * threadsPerDim, batchSize, 1)
/// Threadgroup: (threadsPerDim, 1, 1)
///
/// Best for: sequenceLength > 512
///
kernel void tensor_mean_pool_cooperative(
    device const float* input       [[buffer(0)]],
    device float* output            [[buffer(1)]],
    device const int32_t* mask      [[buffer(2)]],
    constant TensorPoolingParams& params [[buffer(3)]],
    uint2 tgid                      [[threadgroup_position_in_grid]],
    uint2 tid2                      [[thread_position_in_threadgroup]],
    uint2 tgSize2                   [[threads_per_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_size                  [[threads_per_simdgroup]]
) {
    const int d = tgid.x;  // dimension index
    const int b = tgid.y;  // batch index
    const uint tid = tid2.x;
    const uint tgSize = tgSize2.x;

    if (d >= params.dimensions || b >= params.batchSize) return;

    const int seqLen = params.sequenceLength;
    const int dims = params.dimensions;

    const int batchInputOffset = b * seqLen * dims;
    const int batchMaskOffset = b * seqLen;
    const int outputIdx = b * dims + d;

    // Shared memory for reduction
    threadgroup float sharedSum[256];
    threadgroup int sharedCount[256];

    // Each thread processes a slice of the sequence
    float localSum = 0.0f;
    int localCount = 0;

    for (int t = tid; t < seqLen; t += tgSize) {
        if (!mask || mask[batchMaskOffset + t] == 1) {
            localSum = fma(input[batchInputOffset + t * dims + d], 1.0f, localSum);
            localCount++;
        }
    }

    // SIMD reduction
    localSum = simd_sum(localSum);
    localCount = simd_sum(localCount);

    // Store to shared memory (one per SIMD group)
    if (simd_lane == 0) {
        sharedSum[tid / simd_size] = localSum;
        sharedCount[tid / simd_size] = localCount;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction by first thread
    if (tid == 0) {
        float totalSum = 0.0f;
        int totalCount = 0;
        const int numSimdGroups = (tgSize + simd_size - 1) / simd_size;

        for (int i = 0; i < numSimdGroups; i++) {
            totalSum += sharedSum[i];
            totalCount += sharedCount[i];
        }

        output[outputIdx] = totalCount > 0 ? totalSum * (1.0f / float(totalCount)) : 0.0f;
    }
}

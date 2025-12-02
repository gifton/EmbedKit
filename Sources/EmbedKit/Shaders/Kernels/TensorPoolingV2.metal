/// EmbedKit - Tensor Pooling Kernels (MSL 4.0)
///
/// Sequence pooling operations using Metal 4 tensor primitives.
/// Reduces [B, S, D] token embeddings to [B, D] pooled vectors.
///
/// **Kernels**:
/// - tensor_mean_pool_v2:        Mean pooling with optional mask
/// - tensor_max_pool_v2:         Max pooling with optional mask
/// - tensor_cls_pool_v2:         CLS token extraction
/// - tensor_pool_unified_v2:     Strategy-selectable pooling
/// - tensor_attention_pool_v2:   Attention-weighted pooling
/// - tensor_pool_cooperative_v2: Cooperative mean for long sequences
///
/// **Function Constant Specialization**:
/// Each mask-aware kernel uses HAS_MASK [[function_constant(10)]].
/// Create two PSOs per kernel: one with HAS_MASK=true, one with false.
/// The no-mask PSO eliminates all mask memory accesses at compile time.
///
/// **Compatibility**: Metal 4.0 (iOS 26+ / macOS 26+)
/// **Requires**: __HAVE_TENSOR__ feature flag

#include "../Common/MetalCommon.h"

#if defined(__METAL_VERSION__) && defined(__HAVE_TENSOR__)

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace mpp;
using namespace mpp::tensor_ops;

// ============================================================================
// MARK: - Function Constant for Mask Specialization
// ============================================================================

/// When HAS_MASK is false, the compiler eliminates all mask buffer accesses.
/// This provides ~5-15% speedup for the common no-mask case.
/// Uses index 10 to avoid conflict with normalization constants (0, 1) in MetalCommon.h.
constant bool HAS_MASK [[function_constant(10)]];

// ============================================================================
// MARK: - Kernel 1: tensor_mean_pool_v2
// ============================================================================

/// Batch mean pooling over sequence dimension.
///
/// **Algorithm**:
/// For each (batch b, dimension d):
///   output[b, d] = mean(input[b, t, d] for valid t)
///
/// **Mask Semantics**:
/// - HAS_MASK=false: All tokens valid, mask buffer ignored
/// - HAS_MASK=true:  mask[b*S + t] == 1 means valid token
///
/// **Grid**: (dimensions, batchSize, 1) - one thread per (b, d) pair
///
kernel void tensor_mean_pool_v2(
    tensor<device float, dextents<int32_t, 3>, tensor_handle> input [[buffer(0)]],
    tensor<device float, dextents<int32_t, 2>, tensor_handle> output [[buffer(1)]],
    device const int32_t* mask [[buffer(2)]],
    constant TensorPoolingParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int d = gid.x;
    const int b = gid.y;

    if (d >= params.dimensions || b >= params.batchSize) return;

    const int seqLen = params.sequenceLength;
    const int batchMaskOffset = b * seqLen;

    float sum = 0.0f;
    int count = 0;

    // Unrolled loop by 4 for ILP - matches V1 exactly
    int t = 0;
    for (; t <= seqLen - 4; t += 4) {
        // When HAS_MASK=false, compiler eliminates mask access entirely
        const bool m0 = !HAS_MASK || mask[batchMaskOffset + t] == 1;
        const bool m1 = !HAS_MASK || mask[batchMaskOffset + t + 1] == 1;
        const bool m2 = !HAS_MASK || mask[batchMaskOffset + t + 2] == 1;
        const bool m3 = !HAS_MASK || mask[batchMaskOffset + t + 3] == 1;

        // Branch-free accumulation using ternary (V1 pattern)
        sum = fma(m0 ? input[b, t, d] : 0.0f, 1.0f, sum);
        sum = fma(m1 ? input[b, t + 1, d] : 0.0f, 1.0f, sum);
        sum = fma(m2 ? input[b, t + 2, d] : 0.0f, 1.0f, sum);
        sum = fma(m3 ? input[b, t + 3, d] : 0.0f, 1.0f, sum);

        // Bulk count accumulation (branch-free)
        count += m0 + m1 + m2 + m3;
    }

    // Remainder loop
    for (; t < seqLen; ++t) {
        const bool valid = !HAS_MASK || mask[batchMaskOffset + t] == 1;
        if (valid) {
            sum = fma(input[b, t, d], 1.0f, sum);
            count++;
        }
    }

    output[b, d] = count > 0 ? sum * (1.0f / float(count)) : 0.0f;
}

// ============================================================================
// MARK: - Kernel 2: tensor_max_pool_v2
// ============================================================================

/// Batch max pooling over sequence dimension.
///
/// **Algorithm**:
/// output[b, d] = max(input[b, t, d] for valid t)
/// If all tokens masked, output 0.0 (not -INF).
///
/// **Grid**: (dimensions, batchSize, 1)
///
kernel void tensor_max_pool_v2(
    tensor<device float, dextents<int32_t, 3>, tensor_handle> input [[buffer(0)]],
    tensor<device float, dextents<int32_t, 2>, tensor_handle> output [[buffer(1)]],
    device const int32_t* mask [[buffer(2)]],
    constant TensorPoolingParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int d = gid.x;
    const int b = gid.y;

    if (d >= params.dimensions || b >= params.batchSize) return;

    const int seqLen = params.sequenceLength;
    const int batchMaskOffset = b * seqLen;

    float maxVal = -INFINITY;
    int count = 0;

    for (int t = 0; t < seqLen; ++t) {
        const bool valid = !HAS_MASK || mask[batchMaskOffset + t] == 1;
        if (valid) {
            maxVal = max(maxVal, input[b, t, d]);
            count++;
        }
    }

    // Return 0.0 if all masked (spec requirement)
    output[b, d] = count > 0 ? maxVal : 0.0f;
}

// ============================================================================
// MARK: - Kernel 3: tensor_cls_pool_v2
// ============================================================================

/// CLS token pooling - extracts first token embedding.
///
/// **Algorithm**: output[b, d] = input[b, 0, d]
///
/// No mask needed - CLS token is always position 0.
/// No function constant specialization required.
///
/// **Grid**: (dimensions, batchSize, 1)
///
kernel void tensor_cls_pool_v2(
    tensor<device float, dextents<int32_t, 3>, tensor_handle> input [[buffer(0)]],
    tensor<device float, dextents<int32_t, 2>, tensor_handle> output [[buffer(1)]],
    constant TensorPoolingParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int d = gid.x;
    const int b = gid.y;

    if (d >= params.dimensions || b >= params.batchSize) return;

    output[b, d] = input[b, 0, d];
}

// ============================================================================
// MARK: - Kernel 4: tensor_pool_unified_v2
// ============================================================================

/// Unified pooling with runtime strategy selection.
///
/// **Strategies** (params.poolingStrategy):
/// - 0: Mean pooling
/// - 1: Max pooling
/// - 2: CLS pooling
///
/// Reduces PSO switching overhead when strategy varies per-batch.
///
/// **Grid**: (dimensions, batchSize, 1)
///
kernel void tensor_pool_unified_v2(
    tensor<device float, dextents<int32_t, 3>, tensor_handle> input [[buffer(0)]],
    tensor<device float, dextents<int32_t, 2>, tensor_handle> output [[buffer(1)]],
    device const int32_t* mask [[buffer(2)]],
    constant TensorPoolingParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int d = gid.x;
    const int b = gid.y;

    if (d >= params.dimensions || b >= params.batchSize) return;

    const int seqLen = params.sequenceLength;
    const int batchMaskOffset = b * seqLen;

    // CLS path - early exit, no iteration needed
    if (params.poolingStrategy == 2) {
        output[b, d] = input[b, 0, d];
        return;
    }

    float result;
    int count = 0;

    if (params.poolingStrategy == 1) {
        // Max pooling
        result = -INFINITY;
        for (int t = 0; t < seqLen; ++t) {
            const bool valid = !HAS_MASK || mask[batchMaskOffset + t] == 1;
            if (valid) {
                result = max(result, input[b, t, d]);
                count++;
            }
        }
        output[b, d] = count > 0 ? result : 0.0f;
    } else {
        // Mean pooling (default)
        result = 0.0f;
        for (int t = 0; t < seqLen; ++t) {
            const bool valid = !HAS_MASK || mask[batchMaskOffset + t] == 1;
            if (valid) {
                result = fma(input[b, t, d], 1.0f, result);
                count++;
            }
        }
        output[b, d] = count > 0 ? result * (1.0f / float(count)) : 0.0f;
    }
}

// ============================================================================
// MARK: - Kernel 5: tensor_attention_pool_v2
// ============================================================================

/// Attention-weighted pooling using provided weights.
///
/// **Algorithm**:
/// output[b, d] = sum(input[b, t, d] * weights[b, t]) / sum(weights[b, t])
///
/// Weights are assumed non-negative (post-softmax).
/// No function constant needed - weights are always present.
///
/// **Buffer Layout**:
/// - buffer(0): input [B, S, D]
/// - buffer(1): weights [B, S] (raw pointer for flexibility)
/// - buffer(2): output [B, D]
///
/// **Grid**: (dimensions, batchSize, 1)
///
kernel void tensor_attention_pool_v2(
    tensor<device float, dextents<int32_t, 3>, tensor_handle> input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    tensor<device float, dextents<int32_t, 2>, tensor_handle> output [[buffer(2)]],
    constant TensorPoolingParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int d = gid.x;
    const int b = gid.y;

    if (d >= params.dimensions || b >= params.batchSize) return;

    const int seqLen = params.sequenceLength;
    const int batchWeightOffset = b * seqLen;

    float weightedSum = 0.0f;
    float weightTotal = 0.0f;

    // Unrolled by 4 for ILP
    int t = 0;
    for (; t <= seqLen - 4; t += 4) {
        const float w0 = weights[batchWeightOffset + t];
        const float w1 = weights[batchWeightOffset + t + 1];
        const float w2 = weights[batchWeightOffset + t + 2];
        const float w3 = weights[batchWeightOffset + t + 3];

        weightedSum = fma(input[b, t, d], w0, weightedSum);
        weightedSum = fma(input[b, t + 1, d], w1, weightedSum);
        weightedSum = fma(input[b, t + 2, d], w2, weightedSum);
        weightedSum = fma(input[b, t + 3, d], w3, weightedSum);

        weightTotal += w0 + w1 + w2 + w3;
    }

    // Remainder
    for (; t < seqLen; ++t) {
        const float w = weights[batchWeightOffset + t];
        weightedSum = fma(input[b, t, d], w, weightedSum);
        weightTotal += w;
    }

    const float eps = 1e-12f;
    output[b, d] = weightTotal > eps ? weightedSum / weightTotal : 0.0f;
}

// ============================================================================
// MARK: - Kernel 6: tensor_pool_cooperative_v2
// ============================================================================

/// Cooperative mean pooling for long sequences (S > 512).
///
/// Uses SIMD intrinsics for efficient parallel reduction with minimal
/// synchronization (only 1 threadgroup barrier).
///
/// **Thread Organization**:
/// - Grid: (dimensions, batchSize, 1) threadgroups
/// - Threadgroup: (threadsPerGroup, 1, 1) threads
/// - Each threadgroup cooperatively processes one (b, d) pair
///
/// **Reduction Strategy**:
/// 1. Each thread accumulates strided subset of sequence
/// 2. SIMD reduction within each simdgroup (simd_sum)
/// 3. One barrier, then cross-simdgroup reduction in shared memory
///
kernel void tensor_pool_cooperative_v2(
    tensor<device float, dextents<int32_t, 3>, tensor_handle> input [[buffer(0)]],
    tensor<device float, dextents<int32_t, 2>, tensor_handle> output [[buffer(1)]],
    device const int32_t* mask [[buffer(2)]],
    constant TensorPoolingParams& params [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threadsPerGroupVec [[threads_per_threadgroup]],
    uint simdLaneId [[thread_index_in_simdgroup]],
    uint simdGroupId [[simdgroup_index_in_threadgroup]]
) {
    const int d = tgid.x;
    const int b = tgid.y;
    const uint threadsPerGroup = threadsPerGroupVec.x;

    if (d >= params.dimensions || b >= params.batchSize) return;

    const int seqLen = params.sequenceLength;
    const int batchMaskOffset = b * seqLen;

    // Phase 1: Each thread accumulates strided portion
    float localSum = 0.0f;
    int localCount = 0;

    for (int t = tid; t < seqLen; t += threadsPerGroup) {
        const bool valid = !HAS_MASK || mask[batchMaskOffset + t] == 1;
        if (valid) {
            localSum = fma(input[b, t, d], 1.0f, localSum);
            localCount++;
        }
    }

    // Phase 2a: SIMD-level reduction (no barrier needed)
    float simdSum = simd_sum(localSum);
    int simdCount = simd_sum(localCount);

    // Shared memory for cross-simdgroup communication
    // Max 32 simdgroups (1024 threads / 32 lanes)
    threadgroup float simdSums[32];
    threadgroup int simdCounts[32];

    // First lane of each simdgroup writes result
    if (simdLaneId == 0) {
        simdSums[simdGroupId] = simdSum;
        simdCounts[simdGroupId] = simdCount;
    }

    // Single barrier for shared memory coherence
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2b: Final reduction by first simdgroup
    const uint numSimdGroups = (threadsPerGroup + 31) / 32;

    if (simdGroupId == 0) {
        // Each thread in first simdgroup loads one partial result
        float partialSum = (tid < numSimdGroups) ? simdSums[tid] : 0.0f;
        int partialCount = (tid < numSimdGroups) ? simdCounts[tid] : 0;

        // Final SIMD reduction
        float finalSum = simd_sum(partialSum);
        int finalCount = simd_sum(partialCount);

        // Thread 0 writes output
        if (tid == 0) {
            output[b, d] = finalCount > 0 ? finalSum / float(finalCount) : 0.0f;
        }
    }
}

#endif // __HAVE_TENSOR__

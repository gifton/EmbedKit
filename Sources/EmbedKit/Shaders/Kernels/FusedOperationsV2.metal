/// EmbedKit - Fused Operations (MSL 4.0)
///
/// Combined kernels that eliminate intermediate global memory writes by utilizing threadgroup memory.
/// Reduces memory bandwidth and kernel launch overhead.
///
/// **Kernels**:
/// - fused_pool_normalize_v2:           Unified pool + L2 normalize
/// - fused_mean_pool_normalize_v2:      Optimized mean pool + normalize
/// - fused_max_pool_normalize_v2:       Optimized max pool + normalize
/// - fused_attention_pool_normalize_v2: Attention pool + normalize
///
/// **Function Constant Specialization**:
/// - FUSED_POOL_HAS_MASK [[function_constant(11)]]: Mask-aware pooling
/// - FUSED_MEAN_HAS_MASK [[function_constant(12)]]: Mask-aware mean pooling
/// Create two PSOs per kernel: one with HAS_MASK=true, one with false.
/// The no-mask PSO eliminates all mask memory accesses at compile time.
///
/// **Note on Complex Fused Kernels**:
/// fused_embed_pipeline_v2 and fused_pool_similarity_v2 are omitted.
/// Fusing pool+norm+similarity efficiently is impractical due to conflicting
/// grid organizations and shared memory limitations. The high-performance path
/// is to chain fused_pool_normalize_v2 with tensor_similarity_matrix_v2.
///
/// **Compatibility**: Metal 4.0 (iOS 26+ / macOS 26+)
/// **Requires**: __HAVE_TENSOR__ feature flag

#include "../Common/MetalCommon.h"

#if defined(__METAL_VERSION__) && defined(__HAVE_TENSOR__)

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace mpp;
using namespace mpp::tensor_ops;

// ============================================================================
// MARK: - Constants and Function Constants
// ============================================================================

// **Function Constant Registry** (EmbedKit-wide allocation)
// Indices must be unique across all Metal files to avoid PSO conflicts.
//
// Index | Name                      | Type  | File
// ------|---------------------------|-------|---------------------------
//   0   | USE_STABLE_NORMALIZATION  | bool  | MetalCommon.h
//   1   | EPSILON_NORMAL            | float | MetalCommon.h
//  10   | HAS_MASK                  | bool  | TensorPoolingV2.metal
//  11   | FUSED_POOL_HAS_MASK       | bool  | FusedOperationsV2.metal
//  12   | FUSED_MEAN_HAS_MASK       | bool  | FusedOperationsV2.metal
//  13   | FUSED_MAX_HAS_MASK        | bool  | FusedOperationsV2.metal
//
// When adding new function constants, use the next available index (14+).

/// Epsilon for zero-norm / near-zero-norm protection
/// Matches V1 kernels for numerical parity
constant float FUSED_NORM_EPSILON = 1e-12f;

/// Maximum dimensions supported by shared memory layout
/// Supports D ∈ {384, 768, 1024}
constant int FUSED_MAX_DIMS = 1024;

/// Function constant for mask specialization in unified pool kernel
/// Uses index 11 to avoid conflict with MetalCommon.h (0, 1) and TensorPoolingV2 (10)
constant bool FUSED_POOL_HAS_MASK [[function_constant(11)]];

/// Function constant for mask specialization in mean pool kernel
constant bool FUSED_MEAN_HAS_MASK [[function_constant(12)]];

/// Function constant for mask specialization in max pool kernel
constant bool FUSED_MAX_HAS_MASK [[function_constant(13)]];

// ============================================================================
// MARK: - Cooperative Norm Reduction Utility
// ============================================================================

/// Performs L2 normalization reduction on data already present in shared memory.
/// Implements the optimized two-stage cooperative reduction (SIMD + Shared Memory).
///
/// @param sharedData       Shared memory buffer containing pooled data AND workspace
/// @param dims             Number of dimensions D
/// @param workspaceOffset  Offset in sharedData where reduction workspace begins
/// @param tid              Thread index in threadgroup
/// @param threads_per_group Total threads in threadgroup
/// @param simd_lane_id     Lane within SIMD group
/// @param simd_group_id    SIMD group index
///
/// After this function:
/// - sharedData[workspaceOffset + 32] contains the inverse norm (broadcast slot)
/// - Caller MUST issue threadgroup_barrier before reading the broadcast slot
inline void cooperative_norm_reduction(
    threadgroup float* sharedData,
    int dims,
    int workspaceOffset,
    uint tid,
    uint threads_per_group,
    uint simd_lane_id,
    uint simd_group_id
) {
    // Phase 1: Parallel sum of squares from shared memory (4x unrolled for ILP)
    float localSumSq = 0.0f;
    const int stride = threads_per_group;
    const int stride4 = stride * 4;

    int d = tid;
    for (; d + stride * 3 < dims; d += stride4) {
        const float v0 = sharedData[d];
        const float v1 = sharedData[d + stride];
        const float v2 = sharedData[d + stride * 2];
        const float v3 = sharedData[d + stride * 3];
        localSumSq = fma(v0, v0, localSumSq);
        localSumSq = fma(v1, v1, localSumSq);
        localSumSq = fma(v2, v2, localSumSq);
        localSumSq = fma(v3, v3, localSumSq);
    }
    for (; d < dims; d += stride) {
        const float val = sharedData[d];
        localSumSq = fma(val, val, localSumSq);
    }

    // Phase 2: SIMD-level reduction (no barrier needed)
    float warpSum = simd_sum(localSumSq);

    // Leader of each SIMD group writes to workspace
    if (simd_lane_id == 0) {
        sharedData[workspaceOffset + simd_group_id] = warpSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Cross-SIMD reduction (first SIMD group only)
    const uint numSimdGroups = (threads_per_group + 31) / 32;

    if (simd_group_id == 0) {
        float groupSum = (tid < numSimdGroups) ? sharedData[workspaceOffset + tid] : 0.0f;
        float totalSumSq = simd_sum(groupSum);

        // Phase 4: Compute inverse norm (thread 0 only)
        if (tid == 0) {
            const float norm = sqrt(max(totalSumSq, FUSED_NORM_EPSILON));
            // Zero vector produces zero output (not NaN)
            sharedData[workspaceOffset + 32] = (totalSumSq > FUSED_NORM_EPSILON) ? (1.0f / norm) : 0.0f;
        }
    }
    // Note: Caller must issue barrier before reading broadcast slot
}

/// Cooperative parallel count of valid mask entries.
///
/// All threads participate in counting, using SIMD + shared memory reduction.
/// Much faster than single-thread counting for S > 32.
///
/// @param mask             Mask buffer (1=valid, 0=masked)
/// @param batchMaskOffset  Offset to this batch's mask data
/// @param seqLen           Sequence length
/// @param sharedData       Shared memory for reduction (needs 33 floats at workspaceOffset)
/// @param workspaceOffset  Offset in sharedData for workspace
/// @param tid              Thread index in threadgroup
/// @param threads_per_group Total threads in threadgroup
/// @param simd_lane_id     Lane within SIMD group
/// @param simd_group_id    SIMD group index
/// @return Valid token count (broadcast to all threads after barrier)
///
/// Note: Caller MUST issue threadgroup_barrier after this call before using result.
inline int cooperative_mask_count(
    device const int32_t* __restrict__ mask,
    int batchMaskOffset,
    int seqLen,
    threadgroup float* sharedData,
    int workspaceOffset,
    uint tid,
    uint threads_per_group,
    uint simd_lane_id,
    uint simd_group_id
) {
    // Phase 1: Each thread counts strided portion (4x unrolled)
    int localCount = 0;
    int t = tid;
    const int stride = threads_per_group;
    const int stride4 = stride * 4;

    for (; t + stride * 3 < seqLen; t += stride4) {
        localCount += (mask[batchMaskOffset + t] == 1);
        localCount += (mask[batchMaskOffset + t + stride] == 1);
        localCount += (mask[batchMaskOffset + t + stride * 2] == 1);
        localCount += (mask[batchMaskOffset + t + stride * 3] == 1);
    }
    for (; t < seqLen; t += stride) {
        localCount += (mask[batchMaskOffset + t] == 1);
    }

    // Phase 2: SIMD-level reduction
    int warpCount = simd_sum(localCount);

    // Leader of each SIMD group writes to workspace
    if (simd_lane_id == 0) {
        sharedData[workspaceOffset + simd_group_id] = float(warpCount);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Cross-SIMD reduction (first SIMD group only)
    const uint numSimdGroups = (threads_per_group + 31) / 32;
    int totalCount = 0;

    if (simd_group_id == 0) {
        float groupCount = (tid < numSimdGroups) ? sharedData[workspaceOffset + tid] : 0.0f;
        totalCount = int(simd_sum(groupCount));

        // Broadcast result
        if (tid == 0) {
            sharedData[workspaceOffset + 32] = float(totalCount);
        }
    }

    // Caller issues barrier, then all threads read from broadcast slot
    return totalCount;
}

/// Cooperative parallel sum of attention weights.
///
/// Same pattern as cooperative_mask_count but for float weights.
inline float cooperative_weight_sum(
    device const float* __restrict__ weights,
    int batchWeightOffset,
    int seqLen,
    threadgroup float* sharedData,
    int workspaceOffset,
    uint tid,
    uint threads_per_group,
    uint simd_lane_id,
    uint simd_group_id
) {
    // Phase 1: Each thread sums strided portion (4x unrolled)
    float localSum = 0.0f;
    int t = tid;
    const int stride = threads_per_group;
    const int stride4 = stride * 4;

    for (; t + stride * 3 < seqLen; t += stride4) {
        localSum += weights[batchWeightOffset + t];
        localSum += weights[batchWeightOffset + t + stride];
        localSum += weights[batchWeightOffset + t + stride * 2];
        localSum += weights[batchWeightOffset + t + stride * 3];
    }
    for (; t < seqLen; t += stride) {
        localSum += weights[batchWeightOffset + t];
    }

    // Phase 2: SIMD-level reduction
    float warpSum = simd_sum(localSum);

    // Leader of each SIMD group writes to workspace
    if (simd_lane_id == 0) {
        sharedData[workspaceOffset + simd_group_id] = warpSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Cross-SIMD reduction (first SIMD group only)
    const uint numSimdGroups = (threads_per_group + 31) / 32;

    if (simd_group_id == 0) {
        float groupSum = (tid < numSimdGroups) ? sharedData[workspaceOffset + tid] : 0.0f;
        float totalSum = simd_sum(groupSum);

        // Broadcast result
        if (tid == 0) {
            sharedData[workspaceOffset + 32] = totalSum;
        }
    }

    // Caller issues barrier, then all threads read from broadcast slot
    return 0.0f;  // Result is in shared memory
}

/// Write pooled data from shared memory to output, with optional L2 normalization.
///
/// Consolidates the normalization and output phases shared by all fused kernels.
/// Uses 4x loop unrolling for ILP on the output write.
///
/// @param sharedData        Shared memory containing pooled vector [0..dims-1]
/// @param output            Output tensor [B, D]
/// @param b                 Batch index
/// @param dims              Number of dimensions
/// @param workspaceOffset   Offset in sharedData for reduction workspace
/// @param shouldNormalize   Whether to apply L2 normalization
/// @param tid               Thread index in threadgroup
/// @param threads_per_group Total threads in threadgroup
/// @param simd_lane_id      Lane within SIMD group
/// @param simd_group_id     SIMD group index
inline void write_pooled_output(
    threadgroup float* sharedData,
    tensor<device float, dextents<int32_t, 2>, tensor_handle> output,
    int b,
    int dims,
    int workspaceOffset,
    bool shouldNormalize,
    uint tid,
    uint threads_per_group,
    uint simd_lane_id,
    uint simd_group_id
) {
    float scale = 1.0f;

    if (shouldNormalize) {
        // Compute L2 norm via cooperative reduction
        cooperative_norm_reduction(
            sharedData, dims, workspaceOffset,
            tid, threads_per_group, simd_lane_id, simd_group_id
        );
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scale = sharedData[workspaceOffset + 32];  // invNorm
    }

    // Write output (4x unrolled for ILP)
    const int stride = threads_per_group;
    const int stride4 = stride * 4;

    int d = tid;
    for (; d + stride * 3 < dims; d += stride4) {
        output[b, d] = sharedData[d] * scale;
        output[b, d + stride] = sharedData[d + stride] * scale;
        output[b, d + stride * 2] = sharedData[d + stride * 2] * scale;
        output[b, d + stride * 3] = sharedData[d + stride * 3] * scale;
    }
    for (; d < dims; d += stride) {
        output[b, d] = sharedData[d] * scale;
    }
}

// ============================================================================
// MARK: - Kernel 1: fused_pool_normalize_v2 (Unified)
// ============================================================================

/// Fused pooling + L2 normalization (Unified Strategy).
///
/// **Pipeline**:
/// 1. Pool token embeddings [B, S, D] → [B, D] (in threadgroup shared memory)
/// 2. Compute L2 norm of pooled vector
/// 3. Normalize and write to output
///
/// **Strategies** (params.poolingStrategy):
/// - 0: Mean pooling
/// - 1: Max pooling
/// - 2: CLS pooling
///
/// **Thread Organization**:
/// - Grid: (batchSize, 1, 1) threadgroups
/// - Threadgroup: (256, 1, 1) threads recommended
/// - One threadgroup per batch item
///
kernel void fused_pool_normalize_v2(
    tensor<const device float, dextents<int32_t, 3>, tensor_handle> input [[buffer(0)]],
    tensor<device float, dextents<int32_t, 2>, tensor_handle> output [[buffer(1)]],
    device const int32_t* __restrict__ mask [[buffer(2)]],
    constant FusedPoolNormParams& params [[buffer(3)]],
    uint b [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    if (b >= (uint)params.batchSize) return;

    const int seqLen = params.sequenceLength;
    const int dims = params.dimensions;
    const int strategy = params.poolingStrategy;
    const bool shouldNormalize = (params.normalize != 0);

    // Safety check for dimension limits
    if (dims > FUSED_MAX_DIMS) return;

    const int batchMaskOffset = b * seqLen;

    // Shared memory layout (V1 spec):
    // [0...D-1]: Pooled vector
    // [D...D+32]: Reduction workspace (33 floats)
    threadgroup float sharedData[FUSED_MAX_DIMS + 33];
    const int workspaceOffset = dims;

    // ================================================================
    // Phase 1: Pooling into Shared Memory
    // ================================================================

    if (strategy == 2) {
        // CLS pooling - early path, no iteration needed
        for (int d = tid; d < dims; d += threads_per_group) {
            sharedData[d] = (seqLen > 0) ? input[b, 0, d] : 0.0f;
        }
    }
    else if (strategy == 1) {
        // Max pooling (4x unrolled for ILP)
        for (int d = tid; d < dims; d += threads_per_group) {
            float maxVal = -INFINITY;
            int count = 0;
            int t = 0;

            // 4x unrolled loop
            for (; t <= seqLen - 4; t += 4) {
                const bool m0 = !FUSED_POOL_HAS_MASK || mask[batchMaskOffset + t] == 1;
                const bool m1 = !FUSED_POOL_HAS_MASK || mask[batchMaskOffset + t + 1] == 1;
                const bool m2 = !FUSED_POOL_HAS_MASK || mask[batchMaskOffset + t + 2] == 1;
                const bool m3 = !FUSED_POOL_HAS_MASK || mask[batchMaskOffset + t + 3] == 1;

                // Branch-free max using ternary (preserves -INF for masked)
                maxVal = max(maxVal, m0 ? input[b, t, d] : -INFINITY);
                maxVal = max(maxVal, m1 ? input[b, t + 1, d] : -INFINITY);
                maxVal = max(maxVal, m2 ? input[b, t + 2, d] : -INFINITY);
                maxVal = max(maxVal, m3 ? input[b, t + 3, d] : -INFINITY);

                count += m0 + m1 + m2 + m3;
            }

            // Remainder loop
            for (; t < seqLen; ++t) {
                const bool valid = !FUSED_POOL_HAS_MASK || mask[batchMaskOffset + t] == 1;
                if (valid) {
                    maxVal = max(maxVal, input[b, t, d]);
                    count++;
                }
            }
            sharedData[d] = (count > 0) ? maxVal : 0.0f;
        }
    }
    else {
        // Mean pooling (default, strategy == 0)
        // Step 1a: Calculate valid token count (cooperative parallel)
        float count;
        if (FUSED_POOL_HAS_MASK) {
            cooperative_mask_count(
                mask, batchMaskOffset, seqLen,
                sharedData, workspaceOffset,
                tid, threads_per_group, simd_lane_id, simd_group_id
            );
            threadgroup_barrier(mem_flags::mem_threadgroup);
            count = sharedData[workspaceOffset + 32];
        } else {
            count = float(seqLen);
        }
        const float invCount = (count > 0.0f) ? (1.0f / count) : 0.0f;

        // Step 1b: Parallel summation over D, sequential (unrolled) over S
        for (int d = tid; d < dims; d += threads_per_group) {
            float sum = 0.0f;
            int t = 0;

            // 4x unrolled loop for ILP
            for (; t <= seqLen - 4; t += 4) {
                const bool m0 = !FUSED_POOL_HAS_MASK || mask[batchMaskOffset + t] == 1;
                const bool m1 = !FUSED_POOL_HAS_MASK || mask[batchMaskOffset + t + 1] == 1;
                const bool m2 = !FUSED_POOL_HAS_MASK || mask[batchMaskOffset + t + 2] == 1;
                const bool m3 = !FUSED_POOL_HAS_MASK || mask[batchMaskOffset + t + 3] == 1;

                sum = fma(m0 ? input[b, t, d] : 0.0f, 1.0f, sum);
                sum = fma(m1 ? input[b, t + 1, d] : 0.0f, 1.0f, sum);
                sum = fma(m2 ? input[b, t + 2, d] : 0.0f, 1.0f, sum);
                sum = fma(m3 ? input[b, t + 3, d] : 0.0f, 1.0f, sum);
            }
            // Remainder
            for (; t < seqLen; ++t) {
                const bool valid = !FUSED_POOL_HAS_MASK || mask[batchMaskOffset + t] == 1;
                if (valid) {
                    sum = fma(input[b, t, d], 1.0f, sum);
                }
            }
            sharedData[d] = sum * invCount;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ================================================================
    // Phase 2 & 3: L2 Normalization and Output
    // ================================================================

    write_pooled_output(
        sharedData, output, b, dims, workspaceOffset, shouldNormalize,
        tid, threads_per_group, simd_lane_id, simd_group_id
    );
}

// ============================================================================
// MARK: - Kernel 2: fused_mean_pool_normalize_v2 (Optimized Mean)
// ============================================================================

/// Fused mean pooling + L2 normalization (Optimized Path).
///
/// Specialized version eliminating strategy switch overhead.
/// Includes 4x loop unrolling for ILP.
///
kernel void fused_mean_pool_normalize_v2(
    tensor<const device float, dextents<int32_t, 3>, tensor_handle> input [[buffer(0)]],
    tensor<device float, dextents<int32_t, 2>, tensor_handle> output [[buffer(1)]],
    device const int32_t* __restrict__ mask [[buffer(2)]],
    constant FusedPoolNormParams& params [[buffer(3)]],
    uint b [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    if (b >= (uint)params.batchSize) return;

    const int seqLen = params.sequenceLength;
    const int dims = params.dimensions;
    const bool shouldNormalize = (params.normalize != 0);

    if (dims > FUSED_MAX_DIMS) return;

    const int batchMaskOffset = b * seqLen;

    threadgroup float sharedData[FUSED_MAX_DIMS + 33];
    const int workspaceOffset = dims;

    // ================================================================
    // Phase 1: Mean Pooling into Shared Memory
    // ================================================================

    // Calculate valid token count (cooperative parallel)
    float count;
    if (FUSED_MEAN_HAS_MASK) {
        cooperative_mask_count(
            mask, batchMaskOffset, seqLen,
            sharedData, workspaceOffset,
            tid, threads_per_group, simd_lane_id, simd_group_id
        );
        threadgroup_barrier(mem_flags::mem_threadgroup);
        count = sharedData[workspaceOffset + 32];
    } else {
        count = float(seqLen);
    }
    const float invCount = (count > 0.0f) ? (1.0f / count) : 0.0f;

    // Parallel summation over D, sequential (unrolled) over S
    for (int d = tid; d < dims; d += threads_per_group) {
        float sum = 0.0f;
        int t = 0;

        // 4x unrolled loop for ILP
        for (; t <= seqLen - 4; t += 4) {
            const bool m0 = !FUSED_MEAN_HAS_MASK || mask[batchMaskOffset + t] == 1;
            const bool m1 = !FUSED_MEAN_HAS_MASK || mask[batchMaskOffset + t + 1] == 1;
            const bool m2 = !FUSED_MEAN_HAS_MASK || mask[batchMaskOffset + t + 2] == 1;
            const bool m3 = !FUSED_MEAN_HAS_MASK || mask[batchMaskOffset + t + 3] == 1;

            sum = fma(m0 ? input[b, t, d] : 0.0f, 1.0f, sum);
            sum = fma(m1 ? input[b, t + 1, d] : 0.0f, 1.0f, sum);
            sum = fma(m2 ? input[b, t + 2, d] : 0.0f, 1.0f, sum);
            sum = fma(m3 ? input[b, t + 3, d] : 0.0f, 1.0f, sum);
        }
        // Remainder
        for (; t < seqLen; ++t) {
            const bool valid = !FUSED_MEAN_HAS_MASK || mask[batchMaskOffset + t] == 1;
            if (valid) {
                sum = fma(input[b, t, d], 1.0f, sum);
            }
        }
        sharedData[d] = sum * invCount;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ================================================================
    // Phase 2 & 3: L2 Normalization and Output
    // ================================================================

    write_pooled_output(
        sharedData, output, b, dims, workspaceOffset, shouldNormalize,
        tid, threads_per_group, simd_lane_id, simd_group_id
    );
}

// ============================================================================
// MARK: - Kernel 3: fused_max_pool_normalize_v2 (Optimized Max)
// ============================================================================

/// Fused max pooling + L2 normalization (Optimized Path).
///
/// Specialized version eliminating strategy switch overhead.
/// Matches V1 fused_max_pool_normalize for numerical parity.
///
kernel void fused_max_pool_normalize_v2(
    tensor<const device float, dextents<int32_t, 3>, tensor_handle> input [[buffer(0)]],
    tensor<device float, dextents<int32_t, 2>, tensor_handle> output [[buffer(1)]],
    device const int32_t* __restrict__ mask [[buffer(2)]],
    constant FusedPoolNormParams& params [[buffer(3)]],
    uint b [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    if (b >= (uint)params.batchSize) return;

    const int seqLen = params.sequenceLength;
    const int dims = params.dimensions;
    const bool shouldNormalize = (params.normalize != 0);

    if (dims > FUSED_MAX_DIMS) return;

    const int batchMaskOffset = b * seqLen;

    threadgroup float sharedData[FUSED_MAX_DIMS + 33];
    const int workspaceOffset = dims;

    // ================================================================
    // Phase 1: Max Pooling into Shared Memory (4x unrolled for ILP)
    // ================================================================

    for (int d = tid; d < dims; d += threads_per_group) {
        float maxVal = -INFINITY;
        int count = 0;
        int t = 0;

        // 4x unrolled loop
        for (; t <= seqLen - 4; t += 4) {
            const bool m0 = !FUSED_MAX_HAS_MASK || mask[batchMaskOffset + t] == 1;
            const bool m1 = !FUSED_MAX_HAS_MASK || mask[batchMaskOffset + t + 1] == 1;
            const bool m2 = !FUSED_MAX_HAS_MASK || mask[batchMaskOffset + t + 2] == 1;
            const bool m3 = !FUSED_MAX_HAS_MASK || mask[batchMaskOffset + t + 3] == 1;

            // Branch-free max using ternary (preserves -INF for masked)
            maxVal = max(maxVal, m0 ? input[b, t, d] : -INFINITY);
            maxVal = max(maxVal, m1 ? input[b, t + 1, d] : -INFINITY);
            maxVal = max(maxVal, m2 ? input[b, t + 2, d] : -INFINITY);
            maxVal = max(maxVal, m3 ? input[b, t + 3, d] : -INFINITY);

            count += m0 + m1 + m2 + m3;
        }

        // Remainder loop
        for (; t < seqLen; ++t) {
            const bool valid = !FUSED_MAX_HAS_MASK || mask[batchMaskOffset + t] == 1;
            if (valid) {
                maxVal = max(maxVal, input[b, t, d]);
                count++;
            }
        }

        // If all masked, output 0.0 (spec requirement)
        sharedData[d] = (count > 0) ? maxVal : 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ================================================================
    // Phase 2 & 3: L2 Normalization and Output
    // ================================================================

    write_pooled_output(
        sharedData, output, b, dims, workspaceOffset, shouldNormalize,
        tid, threads_per_group, simd_lane_id, simd_group_id
    );
}

// ============================================================================
// MARK: - Kernel 4: fused_attention_pool_normalize_v2
// ============================================================================

/// Fused attention-weighted pooling + L2 normalization.
///
/// **Pipeline**:
/// 1. Compute weighted sum using attention weights
/// 2. Normalize by weight sum (in shared memory)
/// 3. Compute L2 norm
/// 4. Normalize and write output
///
/// **Buffer Layout**:
/// - buffer(0): input [B, S, D]
/// - buffer(1): weights [B, S] (raw pointer for flexibility)
/// - buffer(2): output [B, D]
/// - buffer(3): params
///
/// No function constant needed - weights are always present.
///
kernel void fused_attention_pool_normalize_v2(
    tensor<const device float, dextents<int32_t, 3>, tensor_handle> input [[buffer(0)]],
    device const float* __restrict__ weights [[buffer(1)]],
    tensor<device float, dextents<int32_t, 2>, tensor_handle> output [[buffer(2)]],
    constant FusedPoolNormParams& params [[buffer(3)]],
    uint b [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    if (b >= (uint)params.batchSize) return;

    const int seqLen = params.sequenceLength;
    const int dims = params.dimensions;
    const bool shouldNormalize = (params.normalize != 0);

    if (dims > FUSED_MAX_DIMS) return;

    const int batchWeightOffset = b * seqLen;

    threadgroup float sharedData[FUSED_MAX_DIMS + 33];
    const int workspaceOffset = dims;

    // ================================================================
    // Phase 1: Attention Pooling into Shared Memory
    // ================================================================

    // Calculate total weight sum (cooperative parallel)
    cooperative_weight_sum(
        weights, batchWeightOffset, seqLen,
        sharedData, workspaceOffset,
        tid, threads_per_group, simd_lane_id, simd_group_id
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float weightTotal = sharedData[workspaceOffset + 32];
    const float invWeightTotal = (weightTotal > FUSED_NORM_EPSILON) ? (1.0f / weightTotal) : 0.0f;

    // Parallel weighted summation over D, sequential (unrolled) over S
    for (int d = tid; d < dims; d += threads_per_group) {
        float weightedSum = 0.0f;
        int t = 0;

        // 4x unrolled loop for ILP
        for (; t <= seqLen - 4; t += 4) {
            const float w0 = weights[batchWeightOffset + t];
            const float w1 = weights[batchWeightOffset + t + 1];
            const float w2 = weights[batchWeightOffset + t + 2];
            const float w3 = weights[batchWeightOffset + t + 3];

            weightedSum = fma(input[b, t, d], w0, weightedSum);
            weightedSum = fma(input[b, t + 1, d], w1, weightedSum);
            weightedSum = fma(input[b, t + 2, d], w2, weightedSum);
            weightedSum = fma(input[b, t + 3, d], w3, weightedSum);
        }
        // Remainder
        for (; t < seqLen; ++t) {
            weightedSum = fma(input[b, t, d], weights[batchWeightOffset + t], weightedSum);
        }

        sharedData[d] = weightedSum * invWeightTotal;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ================================================================
    // Phase 2 & 3: L2 Normalization and Output
    // ================================================================

    write_pooled_output(
        sharedData, output, b, dims, workspaceOffset, shouldNormalize,
        tid, threads_per_group, simd_lane_id, simd_group_id
    );
}

#endif // __HAVE_TENSOR__

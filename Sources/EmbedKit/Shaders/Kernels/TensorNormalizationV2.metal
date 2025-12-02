/// EmbedKit - Tensor Normalization Kernels (MSL 4.0)
///
/// L2 normalization for embedding vectors using Metal 4 tensor primitives.
/// Ensures output vectors have unit length for cosine similarity.
///
/// **Kernels**:
/// - tensor_l2_normalize_v2:          Standard fused normalization
/// - tensor_l2_normalize_stable_v2:   High-precision (Kahan-Neumaier)
/// - tensor_l2_normalize_inplace_v2:  Memory-efficient in-place
/// - tensor_compute_norms_v2:         Norm computation only
/// - tensor_normalize_with_norms_v2:  With pre-computed norms
///
/// **Tensor Shapes**:
/// - input / output: [B, D] (row-major, contiguous)
/// - norms: [B]
///
/// **Thread Organization**:
/// - Cooperative kernels: 1D grid (B threadgroups), flexible threads per group
/// - Element-wise kernel: 2D grid (D, B) for tensor_normalize_with_norms_v2
///
/// **Compatibility**: Metal 4.0 (iOS 26+ / macOS 26+)
/// **Requires**: __HAVE_TENSOR__ feature flag

#include "../Common/MetalCommon.h"

#if defined(__METAL_VERSION__) && defined(__HAVE_TENSOR__)

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace mpp;
using namespace mpp::tensor_ops;

// Epsilon for zero-norm / near-zero-norm protection
// Matches V1 kernels for numerical parity
constant float NORM_EPSILON = 1e-12f;

// ============================================================================
// MARK: - Cooperative Reduction Utility
// ============================================================================

/// SIMD-optimized two-stage parallel reduction.
///
/// Uses simd_sum intrinsics for warp-level reduction, then shared memory
/// for cross-warp communication. Only 1 barrier required.
///
/// @param localSum         Thread-local partial sum
/// @param sharedMem        Threadgroup shared memory (min 32 floats)
/// @param tid              Thread index in threadgroup
/// @param threads_per_group Total threads in threadgroup
/// @param simd_lane_id     Lane within SIMD group
/// @param simd_group_id    SIMD group index
/// @return Final sum - only valid on tid == 0
float cooperative_sum_reduction(
    float localSum,
    threadgroup float* sharedMem,
    uint tid,
    uint threads_per_group,
    uint simd_lane_id,
    uint simd_group_id
) {
    // Stage 1: SIMD-level reduction (no barrier needed)
    float warpSum = simd_sum(localSum);

    // Leader of each SIMD group writes to shared memory
    if (simd_lane_id == 0) {
        sharedMem[simd_group_id] = warpSum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 2: Cross-SIMD reduction (first SIMD group only)
    const uint numSimdGroups = (threads_per_group + 31) / 32;
    float finalSum = 0.0f;

    if (simd_group_id == 0) {
        float groupSum = (tid < numSimdGroups) ? sharedMem[tid] : 0.0f;
        finalSum = simd_sum(groupSum);
    }

    return finalSum;
}

// ============================================================================
// MARK: - Kernel 1: tensor_l2_normalize_v2
// ============================================================================

/// Batch L2 normalization - fused computation and application.
///
/// **Algorithm**:
/// For each batch b:
///   sumSq = Σ input[b, d]²
///   norm = sqrt(max(sumSq, ε))
///   output[b, :] = input[b, :] / norm
///
/// **Zero Vector Handling**:
/// Zero vectors produce zero output (not NaN or Inf).
///
/// **Thread Organization**:
/// - Grid: (batchSize, 1, 1) threadgroups
/// - Threadgroup: Flexible size (recommended 256)
///
kernel void tensor_l2_normalize_v2(
    tensor<const device float, dextents<int32_t, 2>, tensor_handle> input [[buffer(0)]],
    tensor<device float, dextents<int32_t, 2>, tensor_handle> output [[buffer(1)]],
    constant TensorNormParams& params [[buffer(2)]],
    uint b [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    if (b >= (uint)params.batchSize) return;

    const int dims = params.dimensions;

    // Shared memory: 32 for reduction + 1 for invNorm broadcast
    threadgroup float sharedData[33];

    // Phase 1: Parallel sum of squares (grid-stride loop, 4x unrolled for ILP)
    float localSumSq = 0.0f;
    const int stride = threads_per_group;
    const int stride4 = stride * 4;

    // Unrolled loop - process 4 elements per iteration
    int d = tid;
    for (; d + stride * 3 < dims; d += stride4) {
        const float v0 = input[b, d];
        const float v1 = input[b, d + stride];
        const float v2 = input[b, d + stride * 2];
        const float v3 = input[b, d + stride * 3];
        localSumSq = fma(v0, v0, localSumSq);
        localSumSq = fma(v1, v1, localSumSq);
        localSumSq = fma(v2, v2, localSumSq);
        localSumSq = fma(v3, v3, localSumSq);
    }

    // Remainder loop
    for (; d < dims; d += stride) {
        const float val = input[b, d];
        localSumSq = fma(val, val, localSumSq);
    }

    // Phase 2: Cooperative reduction
    float totalSumSq = cooperative_sum_reduction(
        localSumSq, sharedData, tid, threads_per_group, simd_lane_id, simd_group_id
    );

    // Phase 3: Compute inverse norm (thread 0 only)
    if (tid == 0) {
        const float norm = sqrt(max(totalSumSq, NORM_EPSILON));
        // Zero vector produces zero output (not NaN)
        sharedData[32] = (totalSumSq > NORM_EPSILON) ? (1.0f / norm) : 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Apply normalization (grid-stride loop, 4x unrolled)
    const float invNorm = sharedData[32];

    // Unrolled loop
    d = tid;
    for (; d + stride * 3 < dims; d += stride4) {
        output[b, d] = input[b, d] * invNorm;
        output[b, d + stride] = input[b, d + stride] * invNorm;
        output[b, d + stride * 2] = input[b, d + stride * 2] * invNorm;
        output[b, d + stride * 3] = input[b, d + stride * 3] * invNorm;
    }

    // Remainder
    for (; d < dims; d += stride) {
        output[b, d] = input[b, d] * invNorm;
    }
}

// ============================================================================
// MARK: - Kernel 2: tensor_l2_normalize_stable_v2
// ============================================================================

/// Stable L2 normalization using Kahan-Neumaier compensated summation.
///
/// **When to Use**:
/// - High dynamic range vectors (values spanning many orders of magnitude)
/// - Very large dimensions (D > 1024)
/// - When maximum numerical precision is required
///
/// **Key Design**:
/// Independent parallel reductions for sums and compensations preserve
/// precision through the reduction. Folding compensation AFTER reduction
/// is critical - folding before would lose the precision benefits.
///
/// **Algorithm** (per thread):
/// Uses Neumaier's improvement over Kahan: checks which operand is larger
/// before computing compensation, which is more robust for mixed magnitudes.
///
kernel void tensor_l2_normalize_stable_v2(
    tensor<const device float, dextents<int32_t, 2>, tensor_handle> input [[buffer(0)]],
    tensor<device float, dextents<int32_t, 2>, tensor_handle> output [[buffer(1)]],
    constant TensorNormParams& params [[buffer(2)]],
    uint b [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    if (b >= (uint)params.batchSize) return;

    const int dims = params.dimensions;

    // Separate shared memory for sums and compensations
    threadgroup float sharedSums[32];
    threadgroup float sharedComps[32];
    threadgroup float sharedInvNorm[1];

    // Phase 1: Kahan-Neumaier accumulation
    float localSum = 0.0f;
    float localComp = 0.0f;

    for (int d = tid; d < dims; d += threads_per_group) {
        const float val = input[b, d];
        const float sq = val * val;

        // Neumaier's improvement: check which operand is larger
        const float t = localSum + sq;
        if (abs(localSum) >= abs(sq)) {
            // localSum is larger, low bits of sq are lost
            localComp += (localSum - t) + sq;
        } else {
            // sq is larger, low bits of localSum are lost
            localComp += (sq - t) + localSum;
        }
        localSum = t;
    }

    // Phase 2: INDEPENDENT parallel reductions (preserves precision)
    float totalSum = cooperative_sum_reduction(
        localSum, sharedSums, tid, threads_per_group, simd_lane_id, simd_group_id
    );
    float totalComp = cooperative_sum_reduction(
        localComp, sharedComps, tid, threads_per_group, simd_lane_id, simd_group_id
    );

    // Phase 3: Fold compensation AFTER reductions (critical for precision)
    if (tid == 0) {
        const float totalSumSq = totalSum + totalComp;
        const float norm = sqrt(max(totalSumSq, NORM_EPSILON));
        sharedInvNorm[0] = (totalSumSq > NORM_EPSILON) ? (1.0f / norm) : 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Apply normalization
    const float invNorm = sharedInvNorm[0];
    for (int d = tid; d < dims; d += threads_per_group) {
        output[b, d] = input[b, d] * invNorm;
    }
}

// ============================================================================
// MARK: - Kernel 3: tensor_l2_normalize_inplace_v2
// ============================================================================

/// In-place L2 normalization - modifies buffer directly.
///
/// **Advantage**: Eliminates output buffer, saving memory bandwidth.
/// **Caution**: Only safe when caller doesn't need original values.
///
/// **Buffer Layout**:
/// - buffer(0): data [B, D] - read then overwritten
/// - buffer(1): params
///
kernel void tensor_l2_normalize_inplace_v2(
    tensor<device float, dextents<int32_t, 2>, tensor_handle> data [[buffer(0)]],
    constant TensorNormParams& params [[buffer(1)]],
    uint b [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    if (b >= (uint)params.batchSize) return;

    const int dims = params.dimensions;
    const int stride = threads_per_group;
    const int stride4 = stride * 4;

    threadgroup float sharedData[33];

    // Phase 1: Sum of squares (4x unrolled)
    float localSumSq = 0.0f;
    int d = tid;
    for (; d + stride * 3 < dims; d += stride4) {
        const float v0 = data[b, d];
        const float v1 = data[b, d + stride];
        const float v2 = data[b, d + stride * 2];
        const float v3 = data[b, d + stride * 3];
        localSumSq = fma(v0, v0, localSumSq);
        localSumSq = fma(v1, v1, localSumSq);
        localSumSq = fma(v2, v2, localSumSq);
        localSumSq = fma(v3, v3, localSumSq);
    }
    for (; d < dims; d += stride) {
        const float val = data[b, d];
        localSumSq = fma(val, val, localSumSq);
    }

    // Phase 2: Cooperative reduction
    float totalSumSq = cooperative_sum_reduction(
        localSumSq, sharedData, tid, threads_per_group, simd_lane_id, simd_group_id
    );

    // Phase 3: Compute inverse norm
    if (tid == 0) {
        const float norm = sqrt(max(totalSumSq, NORM_EPSILON));
        sharedData[32] = (totalSumSq > NORM_EPSILON) ? (1.0f / norm) : 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Apply normalization in-place (4x unrolled)
    const float invNorm = sharedData[32];
    d = tid;
    for (; d + stride * 3 < dims; d += stride4) {
        data[b, d] *= invNorm;
        data[b, d + stride] *= invNorm;
        data[b, d + stride * 2] *= invNorm;
        data[b, d + stride * 3] *= invNorm;
    }
    for (; d < dims; d += stride) {
        data[b, d] *= invNorm;
    }
}

// ============================================================================
// MARK: - Kernel 4: tensor_compute_norms_v2
// ============================================================================

/// Compute L2 norms for a batch of vectors (without normalizing).
///
/// **Use Case**:
/// - When norms are needed for other purposes (e.g., magnitude analysis)
/// - When you want to normalize in a separate pass
/// - For debugging/validation
///
/// **Output**:
/// - norms[b] = ||input[b, :]||₂
/// - Zero vectors produce norm = 0
///
kernel void tensor_compute_norms_v2(
    tensor<const device float, dextents<int32_t, 2>, tensor_handle> input [[buffer(0)]],
    tensor<device float, dextents<int32_t, 1>, tensor_handle> norms [[buffer(1)]],
    constant TensorNormParams& params [[buffer(2)]],
    uint b [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    if (b >= (uint)params.batchSize) return;

    const int dims = params.dimensions;

    threadgroup float sharedData[32];

    // Phase 1: Sum of squares
    float localSumSq = 0.0f;
    for (int d = tid; d < dims; d += threads_per_group) {
        const float val = input[b, d];
        localSumSq = fma(val, val, localSumSq);
    }

    // Phase 2: Cooperative reduction
    float totalSumSq = cooperative_sum_reduction(
        localSumSq, sharedData, tid, threads_per_group, simd_lane_id, simd_group_id
    );

    // Phase 3: Write norm (thread 0 only)
    if (tid == 0) {
        // Return actual norm (0 for zero vectors)
        norms[b] = (totalSumSq > 0.0f) ? sqrt(totalSumSq) : 0.0f;
    }
}

// ============================================================================
// MARK: - Kernel 5: tensor_normalize_with_norms_v2
// ============================================================================

/// Normalize vectors using pre-computed norms.
///
/// **Use Case**:
/// - Norms computed in separate pass (e.g., tensor_compute_norms_v2)
/// - Norms computed on CPU or different GPU pass
/// - When same norms are used multiple times
///
/// **Thread Organization**:
/// - Grid: (dimensions, batchSize, 1) - one thread per element
/// - No cooperative reduction needed (norms already computed)
///
kernel void tensor_normalize_with_norms_v2(
    tensor<const device float, dextents<int32_t, 2>, tensor_handle> input [[buffer(0)]],
    tensor<device float, dextents<int32_t, 2>, tensor_handle> output [[buffer(1)]],
    tensor<const device float, dextents<int32_t, 1>, tensor_handle> norms [[buffer(2)]],
    constant TensorNormParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int d = gid.x;
    const int b = gid.y;

    if (d >= params.dimensions || b >= params.batchSize) return;

    const float norm = norms[b];

    // Zero norm produces zero output (not NaN or Inf)
    const float invNorm = (norm > NORM_EPSILON) ? (1.0f / norm) : 0.0f;

    output[b, d] = input[b, d] * invNorm;
}

#endif // __HAVE_TENSOR__

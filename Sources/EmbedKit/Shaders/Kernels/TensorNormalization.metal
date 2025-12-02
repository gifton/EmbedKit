/// EmbedKit - Tensor Normalization Kernels (Metal 4 Optimized)
///
/// Batch-optimized L2 normalization for embedding vectors.
/// Processes entire batches of vectors in single dispatches.
///
/// **Tensor Shape**: [batchSize, dimensions] → [batchSize, dimensions]
///
/// **Metal 4 Optimizations**:
/// - Batch processing in single dispatch
/// - Two-pass algorithm for numerical stability when enabled
/// - SIMD cooperative reductions for large dimensions
/// - Efficient threadgroup organization
///
/// **Compatibility**: Metal 4.0 (iOS 26+ / macOS 26+)

#include "../Common/MetalCommon.h"

// ============================================================================
// MARK: - Batch L2 Normalization (Simple)
// ============================================================================

/// Batch L2 normalization - each thread handles one vector element
///
/// Two-phase approach:
/// Phase 1: Compute all norms (one threadgroup per vector)
/// Phase 2: Normalize all vectors (parallel across all elements)
///
/// This kernel is Phase 2 - assumes norms are precomputed.
/// For combined norm computation + normalization, use tensor_l2_normalize_fused.
///
/// Thread grid: (dimensions, batchSize, 1)
///
kernel void tensor_l2_normalize_with_norms(
    device const float* input       [[buffer(0)]],
    device float* output            [[buffer(1)]],
    device const float* norms       [[buffer(2)]],  // Precomputed norms [batchSize]
    constant TensorNormParams& params [[buffer(3)]],
    uint2 gid                       [[thread_position_in_grid]]
) {
    const int d = gid.x;
    const int b = gid.y;

    if (d >= params.dimensions || b >= params.batchSize) return;

    const int idx = b * params.dimensions + d;
    const float norm = norms[b];

    // Safe division with epsilon
    const float invNorm = norm > 1e-12f ? (1.0f / norm) : 0.0f;
    output[idx] = input[idx] * invNorm;
}

// ============================================================================
// MARK: - Batch Norm Computation
// ============================================================================

/// Compute L2 norms for a batch of vectors
///
/// Uses threadgroup reduction to compute one norm per vector.
/// Optimized for vectors where dimensions fits in single threadgroup.
///
/// Thread grid: (1, batchSize, 1) with threadgroup size (min(dims, 256), 1, 1)
///
kernel void tensor_compute_norms(
    device const float* input       [[buffer(0)]],
    device float* norms             [[buffer(1)]],  // Output norms [batchSize]
    constant TensorNormParams& params [[buffer(2)]],
    uint b                          [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_size                  [[threads_per_simdgroup]]
) {
    // b = batch index
    const uint tgSize = 256;

    if (b >= params.batchSize) return;

    const int dims = params.dimensions;
    const int batchOffset = b * dims;

    // Shared memory for reduction
    threadgroup float shared[32];  // Max 32 SIMD groups

    // Each thread accumulates partial sum of squares
    float partial = 0.0f;
    for (int i = tid; i < dims; i += tgSize) {
        const float val = input[batchOffset + i];
        partial = fma(val, val, partial);
    }

    // SIMD reduction (warp-level)
    partial = simd_sum(partial);

    // Store to shared memory
    if (simd_lane == 0) {
        shared[tid / simd_size] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction by first SIMD group
    if (tid < simd_size) {
        const int numSimdGroups = (tgSize + simd_size - 1) / simd_size;
        float val = tid < numSimdGroups ? shared[tid] : 0.0f;
        val = simd_sum(val);
        if (tid == 0) {
            norms[b] = sqrt(max(val, 1e-12f));
        }
    }
}

// ============================================================================
// MARK: - Fused Batch L2 Normalization (Single-Pass per Vector)
// ============================================================================

/// Fused batch L2 normalization - computes norm and normalizes in one pass
///
/// Each threadgroup processes ONE complete vector.
/// Thread grid: (1, batchSize, 1) with threadgroup size (threadgroupSize, 1, 1)
///
/// **Algorithm**:
/// 1. Parallel reduction to compute L2 norm
/// 2. Barrier synchronization
/// 3. Parallel normalization
///
/// Best for: dimensions <= 1024 (fits in single threadgroup)
///
kernel void tensor_l2_normalize_fused(
    device const float* input       [[buffer(0)]],
    device float* output            [[buffer(1)]],
    constant TensorNormParams& params [[buffer(2)]],
    uint b                          [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_size                  [[threads_per_simdgroup]]
) {
    // b = batch index
    const uint tgSize = 256;

    if (b >= params.batchSize) return;

    const int dims = params.dimensions;
    const int batchOffset = b * dims;

    // Shared memory for norm reduction and final inverse norm
    threadgroup float shared[33];  // [0-31] for reduction, [32] for invNorm

    // ========================================================================
    // Phase 1: Compute L2 norm via parallel reduction
    // ========================================================================

    float partial = 0.0f;
    for (int i = tid; i < dims; i += tgSize) {
        const float val = input[batchOffset + i];
        partial = fma(val, val, partial);
    }

    // SIMD reduction
    partial = simd_sum(partial);

    if (simd_lane == 0) {
        shared[tid / simd_size] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction and compute inverse norm
    if (tid < simd_size) {
        const int numSimdGroups = (tgSize + simd_size - 1) / simd_size;
        float val = tid < numSimdGroups ? shared[tid] : 0.0f;
        val = simd_sum(val);
        if (tid == 0) {
            const float norm = sqrt(max(val, 1e-12f));
            shared[32] = norm > 1e-12f ? (1.0f / norm) : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========================================================================
    // Phase 2: Normalize vector
    // ========================================================================

    const float invNorm = shared[32];
    for (int i = tid; i < dims; i += tgSize) {
        output[batchOffset + i] = input[batchOffset + i] * invNorm;
    }
}

// ============================================================================
// MARK: - Two-Pass Stable Normalization
// ============================================================================

/// Two-pass numerically stable L2 normalization
///
/// Uses Kahan-Neumaier summation for improved precision with large vectors
/// or vectors with high dynamic range.
///
/// Pass 1: Compute norm with compensated summation
/// Pass 2: Normalize
///
/// Best for: High-precision requirements, very large dimensions
///
kernel void tensor_l2_normalize_stable(
    device const float* input       [[buffer(0)]],
    device float* output            [[buffer(1)]],
    constant TensorNormParams& params [[buffer(2)]],
    uint b                          [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_size                  [[threads_per_simdgroup]]
) {
    // b = batch index
    const uint tgSize = 256;

    if (b >= params.batchSize) return;

    const int dims = params.dimensions;
    const int batchOffset = b * dims;

    threadgroup float shared[33];

    // ========================================================================
    // Phase 1: Kahan-compensated sum of squares
    // ========================================================================

    float sum = 0.0f;
    float compensation = 0.0f;

    for (int i = tid; i < dims; i += tgSize) {
        const float val = input[batchOffset + i];
        const float sq = val * val;

        // Kahan summation
        const float y = sq - compensation;
        const float t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }

    // SIMD reduction (regular, compensations mostly cancel)
    sum = simd_sum(sum);

    if (simd_lane == 0) {
        shared[tid / simd_size] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < simd_size) {
        const int numSimdGroups = (tgSize + simd_size - 1) / simd_size;
        float val = tid < numSimdGroups ? shared[tid] : 0.0f;
        val = simd_sum(val);
        if (tid == 0) {
            const float norm = sqrt(max(val, 1e-12f));
            shared[32] = norm > 1e-12f ? (1.0f / norm) : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========================================================================
    // Phase 2: Normalize
    // ========================================================================

    const float invNorm = shared[32];
    for (int i = tid; i < dims; i += tgSize) {
        output[batchOffset + i] = input[batchOffset + i] * invNorm;
    }
}

// ============================================================================
// MARK: - In-Place Batch Normalization
// ============================================================================

/// In-place batch L2 normalization
///
/// Normalizes vectors in-place, reducing memory bandwidth.
/// Only safe when input and output buffers are the same.
///
kernel void tensor_l2_normalize_inplace(
    device float* data              [[buffer(0)]],
    constant TensorNormParams& params [[buffer(1)]],
    uint b                          [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_size                  [[threads_per_simdgroup]]
) {
    // b = batch index
    const uint tgSize = 256;

    if (b >= params.batchSize) return;

    const int dims = params.dimensions;
    const int batchOffset = b * dims;

    threadgroup float shared[33];

    // Phase 1: Compute norm
    float partial = 0.0f;
    for (int i = tid; i < dims; i += tgSize) {
        const float val = data[batchOffset + i];
        partial = fma(val, val, partial);
    }

    partial = simd_sum(partial);

    if (simd_lane == 0) {
        shared[tid / simd_size] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < simd_size) {
        const int numSimdGroups = (tgSize + simd_size - 1) / simd_size;
        float val = tid < numSimdGroups ? shared[tid] : 0.0f;
        val = simd_sum(val);
        if (tid == 0) {
            const float norm = sqrt(max(val, 1e-12f));
            shared[32] = norm > 1e-12f ? (1.0f / norm) : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Normalize in-place
    const float invNorm = shared[32];
    for (int i = tid; i < dims; i += tgSize) {
        data[batchOffset + i] *= invNorm;
    }
}

/// EmbedKit - L2 Normalization Kernel
///
/// This file contains GPU-accelerated L2 (Euclidean) normalization operations
/// for embedding vectors. L2 normalization scales vectors to unit length,
/// which is essential for cosine similarity calculations.
///
/// **Algorithm**: output[i] = input[i] / ||input||₂
///   where ||input||₂ = √(Σ input[i]²)
///
/// **Performance Characteristics**:
/// - Complexity: O(D) per vector, where D = dimensions
/// - Memory Bandwidth: 2D reads + D writes per vector
/// - GPU Utilization: ~80% memory bandwidth limited
/// - Optimal for: D ≥ 128 (amortizes norm computation)
///
/// **Numerical Stability**:
/// Uses FMA (fused multiply-add) for better accuracy and performance
/// Epsilon protection prevents division by zero for zero-norm vectors
///
/// **Compatibility**: Metal 3.0+ (iOS 16+ / macOS 13+)

#include "../Common/MetalCommon.h"

// ============================================================================
// MARK: - L2 Normalization Kernel
// ============================================================================

/// L2 normalization kernel with SIMD group operations
///
/// Normalizes each vector in a batch to unit L2 norm. Uses SIMD group
/// cooperative operations for efficient parallel reduction.
///
/// **Thread Organization**:
/// - Grid: (dimensions, batchSize, 1)
/// - Each SIMD group cooperates to compute norm
/// - Each thread writes one output element
///
/// **Memory Access Pattern**:
/// - Step 1: Parallel norm computation (coalesced reads)
/// - Step 2: Normalized write (coalesced writes)
///
/// **Parameters**:
/// @param input Input vectors [batchSize, dimensions]
/// @param output Output normalized vectors [batchSize, dimensions]
/// @param dimensions Number of dimensions per vector
/// @param gid Thread position in grid (x=dimension, y=vector)
/// @param simd_lane_id Thread index within SIMD group
/// @param simd_size Number of threads per SIMD group (typically 32)
///
/// **Example**:
/// Input:  [3.0, 4.0, 0.0]
/// Output: [0.6, 0.8, 0.0]  (||output|| = 1.0)
///
kernel void l2_normalize(
    device const float* input       [[buffer(0)]],
    device float* output            [[buffer(1)]],
    constant int32_t& dimensions    [[buffer(2)]],
    uint3 gid                       [[thread_position_in_grid]],
    uint simd_lane_id               [[thread_index_in_simdgroup]],
    uint simd_size                  [[threads_per_simdgroup]],
    uint simdgroup_id               [[simdgroup_index_in_threadgroup]],
    uint thread_id                  [[thread_index_in_threadgroup]],
    uint3 threads_per_tg            [[threads_per_threadgroup]]
) {
    const uint vectorIndex = gid.y;
    const uint dimIndex = gid.x;
    const uint baseIndex = vectorIndex * uint(dimensions);

    // Check if this thread corresponds to a valid dimension
    const bool isActive = (dimIndex < uint(dimensions));

    // ========================================================================
    // Function constant-backed epsilon with safe fallback
    // ========================================================================
    const float epsilon = (EPSILON_NORMAL > 0.0f) ? EPSILON_NORMAL : 1e-8f;

    // ========================================================================
    // Stable (two-pass) or fast (single-pass) normalization path
    // ========================================================================

    if (USE_STABLE_NORMALIZATION) {
        // -----------------------------
        // Pass 0: Compute scale = max |x_i|
        // -----------------------------
        threadgroup float partial_max[32];
        float local_max = 0.0f;
        if (isActive) {
            float v = input[baseIndex + dimIndex];
            // Sanitize non-finite values (NaN/Inf → 0)
            if (!isfinite(v)) { v = 0.0f; }
            local_max = fabs(v);
        }

        // SIMD group reduction for max
        local_max = simd_max(local_max);

        // Number of SIMD groups by problem size and actually present in this threadgroup
        const uint num_simd_groups = (uint(dimensions) + simd_size - 1) / simd_size;
        const uint simdgroups_in_tg = (threads_per_tg.x + simd_size - 1) / simd_size;
        const uint present_groups = min(num_simd_groups, simdgroups_in_tg);

        // First lane stores its SIMD group's partial max
        if (simd_lane_id == 0) {
            partial_max[simdgroup_id] = local_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Thread 0 combines all partial maxima
        if (thread_id == 0) {
            float total_max = 0.0f;
            for (uint i = 0; i < present_groups; ++i) {
                total_max = max(total_max, partial_max[i]);
            }
            partial_max[0] = total_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float scale = partial_max[0];

        // If scale is exactly zero, vector is all zeros → write zeros and return
        if (scale == 0.0f) {
            if (isActive) {
                output[baseIndex + dimIndex] = 0.0f;
            }
            return;
        }

        // -----------------------------
        // Pass 1: Accumulate Σ((x_i/scale)^2)
        // -----------------------------
        threadgroup float partial_norms[32];
        float norm_sq_local = 0.0f;
        if (isActive) {
            float raw = input[baseIndex + dimIndex];
            if (!isfinite(raw)) { raw = 0.0f; }
            const float val = raw / scale;
            norm_sq_local = fma(val, val, norm_sq_local);
        }

        // SIMD group reduction for sum of squares
        norm_sq_local = simd_sum(norm_sq_local);

        // First lane stores SIMD group's partial sum
        if (simd_lane_id == 0) {
            partial_norms[simdgroup_id] = norm_sq_local;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Thread 0 combines across SIMD groups
        if (thread_id == 0) {
            float total_norm_sq = 0.0f;
            for (uint i = 0; i < present_groups; ++i) {
                total_norm_sq += partial_norms[i];
            }
            partial_norms[0] = total_norm_sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute the actual norm: scale * sqrt(sum)
        // Check against epsilon threshold for near-zero vectors
        const float norm_squared = scale * scale * partial_norms[0];
        const bool isZeroVector = (norm_squared < (epsilon * epsilon));

        // Compute inverse norm using two-factor scaling to avoid overflow:
        // inv_norm = (1/scale) * (1/sqrt(sum))
        float inv_norm = 0.0f;
        if (!isZeroVector && partial_norms[0] > 0.0f) {
            const float inv_scale = 1.0f / scale;
            const float inv_sqrt = metal::rsqrt(partial_norms[0]);
            inv_norm = inv_scale * inv_sqrt;
        }

        // Write normalized values (zero if below epsilon)
        if (isActive) {
            float raw = input[baseIndex + dimIndex];
            if (!isfinite(raw)) { raw = 0.0f; }
            output[baseIndex + dimIndex] = raw * inv_norm;
        }
    } else {
        // -----------------------------
        // Fast path: single-pass accumulation
        // -----------------------------
        threadgroup float partial_norms[32];
        float norm_squared = 0.0f;

        if (isActive) {
            float val = input[baseIndex + dimIndex];
            if (!isfinite(val)) { val = 0.0f; }
            norm_squared = fma(val, val, norm_squared);
        }

        norm_squared = simd_sum(norm_squared);

        const uint num_simd_groups = (uint(dimensions) + simd_size - 1) / simd_size;
        const uint simdgroups_in_tg = (threads_per_tg.x + simd_size - 1) / simd_size;
        const uint present_groups = min(num_simd_groups, simdgroups_in_tg);

        if (simd_lane_id == 0) {
            partial_norms[simdgroup_id] = norm_squared;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (thread_id == 0) {
            float total_norm_squared = 0.0f;
            for (uint i = 0; i < present_groups; ++i) {
                total_norm_squared += partial_norms[i];
            }
            partial_norms[0] = total_norm_squared;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        norm_squared = partial_norms[0];

        const bool isZeroVector = (norm_squared < epsilon);
        const float inv_norm = isZeroVector ? 0.0f : metal::rsqrt(norm_squared);

        if (isActive) {
            float raw = input[baseIndex + dimIndex];
            if (!isfinite(raw)) { raw = 0.0f; }
            output[baseIndex + dimIndex] = raw * inv_norm;
        }
    }
}

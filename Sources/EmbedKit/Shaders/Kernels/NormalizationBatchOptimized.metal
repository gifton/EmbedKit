/// EmbedKit - Batch-Optimized L2 Normalization Kernel
///
/// This file contains GPU-accelerated L2 normalization with improved occupancy
/// for batch processing by utilizing multiple SIMD groups per threadgroup more efficiently.
///
/// **Batch Optimization Strategy**:
/// - Small vectors (dim ≤ 32): Each SIMD group processes its own vector (4× throughput)
/// - Medium vectors (33-64): 2 SIMD groups per vector, 2 vectors per threadgroup (2× throughput)
/// - Large vectors (> 64): Multiple SIMD groups per vector as needed
///
/// **Algorithm**: output[i] = input[i] / ||input||₂
///   where ||input||₂ = √(Σ input[i]²)
///
/// **Performance Improvements**:
/// - 2-4× throughput for batch processing
/// - Better GPU occupancy and resource utilization
/// - Reduced kernel launch overhead
///
/// **Compatibility**: Metal 4.0 (iOS 26+ / macOS 26+)

#include "../Common/MetalCommon.h"

// ============================================================================
// MARK: - Batch-Optimized L2 Normalization Kernel
// ============================================================================

/// Batch-optimized L2 normalization kernel for improved throughput
///
/// **Thread Organization**:
/// - Each SIMD group can process its own vector (for small dimensions)
/// - Or multiple SIMD groups cooperate on larger vectors
/// - Adaptive strategy based on dimension size
///
/// **Parameters**:
/// @param input Input vectors [batchSize, dimensions]
/// @param output Output normalized vectors [batchSize, dimensions]
/// @param dimensions Number of dimensions per vector
/// @param vectors_per_threadgroup Number of vectors each threadgroup processes
/// @param gid Thread position in grid
/// @param tid Thread position in threadgroup
/// @param simd_lane_id Thread index within SIMD group
/// @param simd_size Number of threads per SIMD group (typically 32)
/// @param simdgroup_id SIMD group index within threadgroup
/// @param threads_per_tg Total threads per threadgroup
///
kernel void l2_normalize_batch_optimized(
    device const float* input                [[buffer(0)]],
    device float* output                     [[buffer(1)]],
    constant int32_t& dimensions             [[buffer(2)]],
    constant int32_t& vectors_per_threadgroup [[buffer(3)]],
    constant int32_t& batch_size             [[buffer(4)]],
    uint3 gid                                [[thread_position_in_grid]],
    uint3 tid                                [[thread_position_in_threadgroup]],
    uint simd_lane_id                        [[thread_index_in_simdgroup]],
    uint simd_size                            [[threads_per_simdgroup]],
    uint simdgroup_id                        [[simdgroup_index_in_threadgroup]],
    uint3 threads_per_tg                     [[threads_per_threadgroup]]
) {
    // ========================================================================
    // Step 1: Calculate which vector this SIMD group processes
    // ========================================================================

    // Determine how many SIMD groups work on each vector
    const uint simdgroups_per_vector = (uint(dimensions) + simd_size - 1) / simd_size;

    // Calculate which vector within the threadgroup this SIMD group works on
    const uint vector_in_threadgroup = simdgroup_id / simdgroups_per_vector;

    // Calculate which SIMD group within the vector's processing group this is
    const uint simdgroup_in_vector = simdgroup_id % simdgroups_per_vector;

    // Calculate global vector index
    const uint vectorIndex = gid.y * uint(vectors_per_threadgroup) + vector_in_threadgroup;

    // Calculate the dimension index this thread is responsible for
    const uint dimIndex = simdgroup_in_vector * simd_size + simd_lane_id;

    // Base index for this vector in the input/output buffers
    const uint baseIndex = vectorIndex * uint(dimensions);

    // Check if this thread corresponds to a valid dimension
    const bool isActive = (dimIndex < uint(dimensions));

    // Number of SIMD groups actually present in this (possibly non-uniform) threadgroup (ceil)
    const uint simdgroups_in_tg = (threads_per_tg.x + simd_size - 1) / simd_size;

    // Guard against out-of-range vectors in the last (partial) threadgroup row
    if (vectorIndex >= uint(batch_size)) {
        return;
    }

    // Epsilon selection with safe fallback
    const float epsilon = (EPSILON_NORMAL > 0.0f) ? EPSILON_NORMAL : 1e-8f;

    if (USE_STABLE_NORMALIZATION) {
        // -----------------------------
        // Pass 0: Compute per-vector scale = max |x_i|
        // -----------------------------
        threadgroup float partial_max[32];
        float local_max = 0.0f;
        if (isActive) {
            float v = input[baseIndex + dimIndex];
            if (!isfinite(v)) { v = 0.0f; }
            local_max = fabs(v);
        }
        // SIMD group reduction for max
        local_max = simd_max(local_max);

        // First lane writes this SIMD group's partial max
        if (simd_lane_id == 0) {
            partial_max[simdgroup_id] = local_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Combine maxima across SIMD groups that belong to this vector
        const uint start_simdgroup = vector_in_threadgroup * simdgroups_per_vector;
        const uint end_simdgroup = min(start_simdgroup + simdgroups_per_vector,
                                       simdgroups_in_tg);

        if (simdgroup_id == start_simdgroup && simd_lane_id == 0) {
            float total_max = 0.0f;
            for (uint i = start_simdgroup; i < end_simdgroup; ++i) {
                total_max = max(total_max, partial_max[i]);
            }
            partial_max[start_simdgroup] = total_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float scale = partial_max[start_simdgroup];

        // Zero vector: write zeros and return early
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
        // Reduce within SIMD group
        norm_sq_local = simd_sum(norm_sq_local);

        if (simd_lane_id == 0) {
            partial_norms[simdgroup_id] = norm_sq_local;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Combine across this vector's SIMD groups
        if (simdgroup_id == start_simdgroup && simd_lane_id == 0) {
            float total_norm_sq = 0.0f;
            for (uint i = start_simdgroup; i < end_simdgroup; ++i) {
                total_norm_sq += partial_norms[i];
            }
            partial_norms[start_simdgroup] = total_norm_sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute inverse norm using two-factor scaling to avoid overflow
        float inv_norm = 0.0f;
        if (partial_norms[start_simdgroup] > 0.0f) {
            const float inv_scale = 1.0f / scale;
            const float inv_sqrt = metal::rsqrt(partial_norms[start_simdgroup]);
            inv_norm = inv_scale * inv_sqrt;
        }

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

        if (simdgroups_per_vector > 1) {
            if (simd_lane_id == 0) {
                partial_norms[simdgroup_id] = norm_squared;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (simdgroup_id == vector_in_threadgroup * simdgroups_per_vector && simd_lane_id == 0) {
                float total_norm_squared = 0.0f;
                const uint start_simdgroup2 = vector_in_threadgroup * simdgroups_per_vector;
                const uint end_simdgroup2 = min(start_simdgroup2 + simdgroups_per_vector,
                                               simdgroups_in_tg);
                for (uint i = start_simdgroup2; i < end_simdgroup2; ++i) {
                    total_norm_squared += partial_norms[i];
                }
                partial_norms[start_simdgroup2] = total_norm_squared;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            norm_squared = partial_norms[vector_in_threadgroup * simdgroups_per_vector];
        }

        const bool isZeroVector = (norm_squared < epsilon);
        const float inv_norm = isZeroVector ? 0.0f : metal::rsqrt(norm_squared);

        if (isActive) {
            float raw = input[baseIndex + dimIndex];
            if (!isfinite(raw)) { raw = 0.0f; }
            output[baseIndex + dimIndex] = raw * inv_norm;
        }
    }
}

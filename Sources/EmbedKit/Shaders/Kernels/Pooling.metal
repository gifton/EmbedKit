/// EmbedKit - Pooling Kernels
///
/// This file contains GPU-accelerated pooling operations for token embeddings.
/// Pooling reduces a sequence of token embeddings to a single sentence embedding.
///
/// **Pooling Strategies**:
/// 1. Mean Pooling: Average of all token embeddings (with optional masking)
/// 2. Max Pooling: Element-wise maximum across tokens
/// 3. Attention-Weighted: Weighted average using attention scores
///
/// **Common Use Cases**:
/// - Sentence embeddings from token embeddings (BERT, etc.)
/// - Sequence summarization
/// - Information aggregation
///
/// **Compatibility**: Metal 3.0+ (iOS 16+ / macOS 13+)

#include "../Common/MetalCommon.h"

// ============================================================================
// MARK: - Mean Pooling Kernel
// ============================================================================

/// Mean pooling with optional attention masking
///
/// Computes the mean of token embeddings across the sequence dimension,
/// optionally ignoring masked tokens (e.g., padding tokens).
///
/// **Algorithm**: output[d] = Σ(input[t][d] * mask[t]) / Σ(mask[t])
///   where t ∈ [0, sequenceLength), d ∈ [0, dimensions)
///
/// **Performance**: O(S * D) where S = sequence length, D = dimensions
///
/// **Parameters**:
/// @param input Token embeddings [sequenceLength, dimensions]
/// @param output Pooled embedding [dimensions]
/// @param mask Optional attention mask [sequenceLength] (1=valid, 0=masked)
/// @param params Pooling parameters (sequence length, dimensions)
/// @param gid Thread ID (one thread per dimension)
///
kernel void mean_pool(
    device const float* input       [[buffer(0)]],
    device float* output            [[buffer(1)]],
    device const int32_t* mask      [[buffer(2)]],
    constant PoolingParams& params  [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]],
    uint simd_lane_id               [[thread_index_in_simdgroup]],
    uint simd_size                  [[threads_per_simdgroup]]
) {
    if (gid >= params.dimensions) return;

    float sum = 0.0f;
    int count = 0;

    const int32_t seqLen = params.sequenceLength;
    const int32_t dim = params.dimensions;

    // ========================================================================
    // Accumulate sum with loop unrolling for better performance
    // ========================================================================

    // Process 4 elements at a time when possible
    int i = 0;
    for (; i <= seqLen - 4; i += 4) {
        // Prefetch mask values
        const bool m0 = !mask || mask[i] == 1;
        const bool m1 = !mask || mask[i + 1] == 1;
        const bool m2 = !mask || mask[i + 2] == 1;
        const bool m3 = !mask || mask[i + 3] == 1;

        // Vectorized accumulation with FMA for better performance
        sum = fma(m0 ? input[i * dim + gid] : 0.0f, 1.0f, sum);
        sum = fma(m1 ? input[(i + 1) * dim + gid] : 0.0f, 1.0f, sum);
        sum = fma(m2 ? input[(i + 2) * dim + gid] : 0.0f, 1.0f, sum);
        sum = fma(m3 ? input[(i + 3) * dim + gid] : 0.0f, 1.0f, sum);

        count += m0 + m1 + m2 + m3;
    }

    // Handle remaining elements
    for (; i < seqLen; i++) {
        if (!mask || mask[i] == 1) {
            sum = fma(input[i * dim + gid], 1.0f, sum);
            count++;
        }
    }

    // ========================================================================
    // Compute mean with safe division
    // ========================================================================

    // Use reciprocal multiplication instead of division for better performance
    output[gid] = count > 0 ? sum * (1.0f / float(count)) : 0.0f;
}

// ============================================================================
// MARK: - Max Pooling Kernel
// ============================================================================

/// Max pooling with optional attention masking
///
/// Computes the element-wise maximum across the sequence dimension,
/// optionally ignoring masked tokens.
///
/// **Algorithm**: output[d] = max(input[t][d]) for all valid t
///
/// **Performance**: O(S * D) where S = sequence length, D = dimensions
///
/// **Parameters**:
/// @param input Token embeddings [sequenceLength, dimensions]
/// @param output Pooled embedding [dimensions]
/// @param mask Optional attention mask [sequenceLength]
/// @param params Pooling parameters
/// @param gid Thread ID (one thread per dimension)
///
kernel void max_pool(
    device const float* input       [[buffer(0)]],
    device float* output            [[buffer(1)]],
    device const int32_t* mask      [[buffer(2)]],
    constant PoolingParams& params  [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= params.dimensions) return;

    float maxVal = -INFINITY;
    bool foundValid = false;

    const int32_t seqLen = params.sequenceLength;
    const int32_t dim = params.dimensions;

    // ========================================================================
    // Find maximum with loop unrolling
    // ========================================================================

    // Unroll loop by 4 for better performance
    int i = 0;
    for (; i <= seqLen - 4; i += 4) {
        // Check mask values
        const bool m0 = !mask || mask[i] == 1;
        const bool m1 = !mask || mask[i + 1] == 1;
        const bool m2 = !mask || mask[i + 2] == 1;
        const bool m3 = !mask || mask[i + 3] == 1;

        // Load values conditionally
        if (m0) {
            maxVal = metal::max(maxVal, input[i * dim + gid]);
            foundValid = true;
        }
        if (m1) {
            maxVal = metal::max(maxVal, input[(i + 1) * dim + gid]);
            foundValid = true;
        }
        if (m2) {
            maxVal = metal::max(maxVal, input[(i + 2) * dim + gid]);
            foundValid = true;
        }
        if (m3) {
            maxVal = metal::max(maxVal, input[(i + 3) * dim + gid]);
            foundValid = true;
        }
    }

    // Handle remaining elements
    for (; i < seqLen; i++) {
        if (!mask || mask[i] == 1) {
            maxVal = metal::max(maxVal, input[i * dim + gid]);
            foundValid = true;
        }
    }

    // Output maximum value, or 0.0 if no valid tokens found
    output[gid] = foundValid ? maxVal : 0.0f;
}

// ============================================================================
// MARK: - Attention-Weighted Pooling Kernel
// ============================================================================

/// Attention-weighted pooling with normalized weights
///
/// Computes a weighted average of token embeddings using provided attention weights.
/// Commonly used with attention mechanisms in transformer models.
///
/// **Algorithm**: output[d] = Σ(input[t][d] * weights[t]) / Σ(weights[t])
///
/// **Note**: Assumes weights are pre-normalized (e.g., softmax output)
///   For numerical stability, normalizes by weight sum
///
/// **Performance**: O(S * D) where S = sequence length, D = dimensions
///
/// **Parameters**:
/// @param input Token embeddings [sequenceLength, dimensions]
/// @param weights Attention weights [sequenceLength] (typically from softmax)
/// @param output Pooled embedding [dimensions]
/// @param params Pooling parameters
/// @param gid Thread ID (one thread per dimension)
///
kernel void attention_weighted_pool(
    device const float* input       [[buffer(0)]],
    device const float* weights     [[buffer(1)]],
    device float* output            [[buffer(2)]],
    constant PoolingParams& params  [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= params.dimensions) return;

    float weightedSum = 0.0f;
    float weightSum = 0.0f;

    const int32_t seqLen = params.sequenceLength;
    const int32_t dim = params.dimensions;

    // ========================================================================
    // Compute weighted sum with loop unrolling
    // ========================================================================

    // Unroll by 4 for better performance
    int i = 0;
    for (; i <= seqLen - 4; i += 4) {
        // Load weights
        const float w0 = weights[i];
        const float w1 = weights[i + 1];
        const float w2 = weights[i + 2];
        const float w3 = weights[i + 3];

        // Load input values
        const float v0 = input[i * dim + gid];
        const float v1 = input[(i + 1) * dim + gid];
        const float v2 = input[(i + 2) * dim + gid];
        const float v3 = input[(i + 3) * dim + gid];

        // Accumulate weighted sum using FMA
        weightedSum = fma(v0, w0, weightedSum);
        weightedSum = fma(v1, w1, weightedSum);
        weightedSum = fma(v2, w2, weightedSum);
        weightedSum = fma(v3, w3, weightedSum);
        weightSum += w0 + w1 + w2 + w3;
    }

    // Handle remaining elements with FMA
    for (; i < seqLen; i++) {
        const float weight = weights[i];
        weightedSum = fma(input[i * dim + gid], weight, weightedSum);
        weightSum += weight;
    }

    // ========================================================================
    // Normalize by weight sum with epsilon protection
    // ========================================================================

    // Use reciprocal for division with epsilon protection
    const float invWeightSum = (weightSum > EPSILON_NORMAL) ? (1.0f / weightSum) : 0.0f;
    output[gid] = weightedSum * invWeightSum;
}

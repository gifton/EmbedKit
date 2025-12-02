#ifndef EMBEDKIT_METAL_COMMON_H
#define EMBEDKIT_METAL_COMMON_H

/// EmbedKit Metal Shaders - Common Definitions
///
/// This header contains shared type definitions, constants, and utilities
/// used across all Metal compute kernels in EmbedKit.
///
/// **Alignment Requirements**:
/// All structs are explicitly aligned to 16 bytes to match Swift's memory layout
/// and ensure optimal GPU memory access patterns.
///
/// **Include Order**:
/// This file should be included first in all kernel .metal files
///
/// **Compatibility**:
/// Metal 4.0 (iOS 26+ / macOS 26+)

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// MARK: - Metal Configuration
// ============================================================================

// Metal 4 optimizations
#pragma METAL fast_math enable

// ============================================================================
// MARK: - Numerical Stability Controls (Function Constants)
// ============================================================================

/// Enable more numerically stable algorithms (e.g., two-pass normalization)
/// Specialize via MTLFunctionConstantValues. Defaults to false if not specialized.
constant bool USE_STABLE_NORMALIZATION [[function_constant(0)]];

/// Epsilon for division/zero checks. Specialize via MTLFunctionConstantValues.
/// If not specialized (default 0), kernels should fall back to a safe default (e.g., 1e-8).
constant float EPSILON_NORMAL [[function_constant(1)]];

// ============================================================================
// MARK: - Parameter Structures
// ============================================================================

/// Parameters for pooling operations (mean, max, attention-weighted)
///
/// **Memory Layout**: 16 bytes total (4 x Int32)
/// **Alignment**: Matches Swift's PoolingParams struct
///
/// Fields:
/// - sequenceLength: Number of tokens in the sequence
/// - dimensions: Embedding dimensionality
/// - _padding0, _padding1: Explicit padding to 16 bytes
///
struct PoolingParams {
    int32_t sequenceLength;
    int32_t dimensions;
    int32_t _padding0;  // Explicit padding to 16 bytes
    int32_t _padding1;
};

// Compile-time validation of struct size
static_assert(sizeof(PoolingParams) == 16, "PoolingParams must be exactly 16 bytes");

/// Parameters for cosine similarity matrix calculations
///
/// **Memory Layout**: 16 bytes total (4 x Int32)
/// **Alignment**: Matches Swift's SimilarityParams struct
///
/// Fields:
/// - queryCount: Number of query vectors
/// - keyCount: Number of key vectors to compare against
/// - dimensions: Vector dimensionality
/// - _padding0: Explicit padding to 16 bytes
///
struct SimilarityParams {
    int32_t queryCount;
    int32_t keyCount;
    int32_t dimensions;
    int32_t _padding0;  // Explicit padding to 16 bytes
};

static_assert(sizeof(SimilarityParams) == 16, "SimilarityParams must be exactly 16 bytes");

/// Parameters for batch cosine similarity calculations
///
/// **Memory Layout**: 16 bytes total (4 x Int32)
/// **Alignment**: Matches Swift's BatchSimilarityParams struct
///
/// Fields:
/// - pairCount: Number of vector pairs to process
/// - dimensions: Vector dimensionality
/// - _padding0, _padding1: Explicit padding to 16 bytes
///
struct BatchSimilarityParams {
    int32_t pairCount;
    int32_t dimensions;
    int32_t _padding0;  // Explicit padding to 16 bytes
    int32_t _padding1;
};

static_assert(sizeof(BatchSimilarityParams) == 16, "BatchSimilarityParams must be exactly 16 bytes");

// ============================================================================
// MARK: - Tensor Parameter Structures (Metal 4 Optimized)
// ============================================================================

/// Parameters for batch tensor pooling operations
///
/// Enables processing entire batches of sequences in a single dispatch.
/// Thread grid: (dimensions, batchSize, 1)
///
/// **Memory Layout**: 16 bytes total (4 x Int32)
///
struct TensorPoolingParams {
    int32_t batchSize;        // Number of sequences in batch
    int32_t sequenceLength;   // Tokens per sequence
    int32_t dimensions;       // Embedding dimensions
    int32_t poolingStrategy;  // 0=mean, 1=max, 2=cls
};

static_assert(sizeof(TensorPoolingParams) == 16, "TensorPoolingParams must be exactly 16 bytes");

/// Parameters for batch tensor normalization
///
/// **Memory Layout**: 16 bytes total (4 x Int32)
///
struct TensorNormParams {
    int32_t batchSize;      // Number of vectors to normalize
    int32_t dimensions;     // Vector dimensions
    int32_t _padding0;
    int32_t _padding1;
};

static_assert(sizeof(TensorNormParams) == 16, "TensorNormParams must be exactly 16 bytes");

/// Parameters for fused pooling + normalization pipeline
///
/// Combines pooling and L2 normalization in single dispatch.
///
/// **Memory Layout**: 32 bytes total (8 x Int32)
///
struct FusedPoolNormParams {
    int32_t batchSize;
    int32_t sequenceLength;
    int32_t dimensions;
    int32_t poolingStrategy;  // 0=mean, 1=max, 2=cls
    int32_t normalize;        // 1=apply L2 norm, 0=skip
    int32_t _padding0;
    int32_t _padding1;
    int32_t _padding2;
};

static_assert(sizeof(FusedPoolNormParams) == 32, "FusedPoolNormParams must be exactly 32 bytes");

/// Parameters for batch similarity computation
///
/// Computes similarities between all pairs of query and key embeddings.
/// Output shape: [queryBatchSize, keyBatchSize]
///
/// **Memory Layout**: 16 bytes total (4 x Int32)
///
struct TensorSimilarityParams {
    int32_t queryBatchSize;   // Number of query embeddings
    int32_t keyBatchSize;     // Number of key embeddings
    int32_t dimensions;       // Embedding dimensions
    int32_t metric;           // 0=cosine, 1=dot, 2=euclidean
};

static_assert(sizeof(TensorSimilarityParams) == 16, "TensorSimilarityParams must be exactly 16 bytes");

/// Parameters for complete embedding pipeline
///
/// Fused: token embeddings → pooling → normalization → similarity
/// Reduces memory bandwidth by keeping intermediates on GPU.
///
/// **Memory Layout**: 32 bytes total (8 x Int32)
///
struct EmbeddingPipelineParams {
    int32_t batchSize;         // Number of sequences
    int32_t sequenceLength;    // Tokens per sequence
    int32_t dimensions;        // Embedding dimensions
    int32_t poolingStrategy;   // 0=mean, 1=max, 2=cls
    int32_t normalize;         // 1=apply L2 norm
    int32_t computeSimilarity; // 1=compute similarity matrix
    int32_t _padding0;
    int32_t _padding1;
};

static_assert(sizeof(EmbeddingPipelineParams) == 32, "EmbeddingPipelineParams must be exactly 32 bytes");

// ============================================================================
// MARK: - Utility Functions for Tensor Operations
// ============================================================================

/// Compute L2 norm of a vector
inline float compute_l2_norm(device const float* vec, int dims) {
    float sum = 0.0f;
    for (int i = 0; i < dims; i++) {
        sum = fma(vec[i], vec[i], sum);
    }
    return sqrt(max(sum, 1e-12f));
}

/// Compute L2 norm using threadgroup cooperative reduction
template<int BLOCK_SIZE>
inline float compute_l2_norm_cooperative(
    device const float* vec,
    int dims,
    threadgroup float* shared,
    uint tid,
    uint simd_lane,
    uint simd_size
) {
    // Each thread accumulates partial sum
    float partial = 0.0f;
    for (int i = tid; i < dims; i += BLOCK_SIZE) {
        partial = fma(vec[i], vec[i], partial);
    }

    // SIMD reduction first (warp-level)
    partial = simd_sum(partial);

    // Store to shared memory (one per SIMD group)
    if (simd_lane == 0) {
        shared[tid / simd_size] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction across SIMD groups
    if (tid < simd_size) {
        float val = (tid < (BLOCK_SIZE / simd_size)) ? shared[tid] : 0.0f;
        val = simd_sum(val);
        if (tid == 0) {
            shared[0] = sqrt(max(val, 1e-12f));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return shared[0];
}

/// Compute dot product of two vectors
inline float compute_dot_product(device const float* a, device const float* b, int dims) {
    float sum = 0.0f;
    for (int i = 0; i < dims; i++) {
        sum = fma(a[i], b[i], sum);
    }
    return sum;
}

/// Compute cosine similarity between two vectors
inline float compute_cosine_similarity(
    device const float* a,
    device const float* b,
    int dims
) {
    float dot = 0.0f;
    float normA = 0.0f;
    float normB = 0.0f;

    for (int i = 0; i < dims; i++) {
        dot = fma(a[i], b[i], dot);
        normA = fma(a[i], a[i], normA);
        normB = fma(b[i], b[i], normB);
    }

    float denom = sqrt(max(normA, 1e-12f)) * sqrt(max(normB, 1e-12f));
    return dot / denom;
}

#endif // EMBEDKIT_METAL_COMMON_H

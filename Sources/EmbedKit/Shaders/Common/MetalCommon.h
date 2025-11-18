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
/// Requires Metal 3.0+ (iOS 16+ / macOS 13+)

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// MARK: - Metal Configuration
// ============================================================================

// Metal 3 optimizations
#pragma METAL internals : enable
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

#endif // EMBEDKIT_METAL_COMMON_H

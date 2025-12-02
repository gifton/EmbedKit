# MSL 4.0 Tensor Kernel Specification

**Version**: 1.0
**Target**: Metal 4.0 / MSL 4.0 (iOS 26+ / macOS 26+)
**Xcode**: 16+ with Swift 6.2
**Purpose**: Migrate EmbedKit GPU kernels to use native Metal 4 tensor operations

---

## Table of Contents

1. [Project Context](#1-project-context)
2. [Current Architecture](#2-current-architecture)
3. [MSL 4.0 Migration Goals](#3-msl-40-migration-goals)
4. [Kernel Specifications](#4-kernel-specifications)
5. [Parameter Structures](#5-parameter-structures)
6. [Memory Layouts](#6-memory-layouts)
7. [Performance Targets](#7-performance-targets)
8. [Testing Strategy](#8-testing-strategy)
9. [Code Style Guidelines](#9-code-style-guidelines)
10. [Deliverables](#10-deliverables)

---

## 1. Project Context

### 1.1 What is EmbedKit?

EmbedKit is a Swift package for on-device text embedding generation. It converts text into dense vector representations (embeddings) for semantic search, RAG, and similarity applications.

### 1.2 Embedding Pipeline

```
Text Input → Tokenization → Model Inference → Token Embeddings → Pooling → Normalization → Output
                                                    ↓
                                          [B, S, D] tensor
                                                    ↓
                                          GPU Kernels (this spec)
                                                    ↓
                                          [B, D] normalized embeddings
```

Where:
- **B** = Batch size (1-128 typical)
- **S** = Sequence length (tokens per text, 32-512 typical)
- **D** = Embedding dimensions (384, 768, or 1024 typical)

### 1.3 Why Metal 4 Tensors?

Current kernels use raw `device float*` buffers with manual indexing. Metal 4 introduces native tensor types that:

1. **Express shape semantics** - Compiler understands dimensions
2. **Enable hardware tensor cores** - Apple Silicon has dedicated matrix units
3. **Provide optimized primitives** - `matmul2d`, `reduce_rows`, `reduce_columns`
4. **Reduce boilerplate** - Built-in broadcasting, dimension handling

---

## 2. Current Architecture

### 2.1 File Structure

```
Sources/EmbedKit/Shaders/
├── Common/
│   └── MetalCommon.h          # Shared types, utilities, param structs
└── Kernels/
    ├── TensorPooling.metal    # Mean, max, CLS, attention pooling
    ├── TensorNormalization.metal  # L2 normalization variants
    ├── Similarity.metal       # Cosine similarity matrix
    └── FusedOperations.metal  # Combined pool+norm+similarity
```

### 2.2 Current Kernel Inventory

| Kernel | Purpose | Input Shape | Output Shape |
|--------|---------|-------------|--------------|
| `tensor_mean_pool` | Mean pooling over sequence | [B, S, D] | [B, D] |
| `tensor_max_pool` | Max pooling over sequence | [B, S, D] | [B, D] |
| `tensor_cls_pool` | Extract first token | [B, S, D] | [B, D] |
| `tensor_pool_unified` | Strategy-selectable pooling | [B, S, D] | [B, D] |
| `tensor_attention_pool` | Weighted pooling | [B, S, D] + [B, S] | [B, D] |
| `tensor_l2_normalize_fused` | Batch L2 normalization | [B, D] | [B, D] |
| `cosine_similarity` | Similarity matrix | [Q, D] × [K, D] | [Q, K] |
| `fused_pool_normalize_unified` | Pool + L2 norm | [B, S, D] | [B, D] |
| `tensor_similarity_matrix_normalized` | Similarity (pre-normalized) | [Q, D] × [K, D] | [Q, K] |

### 2.3 Current Implementation Pattern

Current kernels use manual buffer indexing:

```metal
// Current pattern (to be replaced)
kernel void tensor_mean_pool(
    device const float* input       [[buffer(0)]],
    device float* output            [[buffer(1)]],
    device const int32_t* mask      [[buffer(2)]],
    constant TensorPoolingParams& params [[buffer(3)]],
    uint2 gid                       [[thread_position_in_grid]]
) {
    const int d = gid.x;  // dimension index
    const int b = gid.y;  // batch index

    // Manual offset calculation
    const int batchInputOffset = b * params.sequenceLength * params.dimensions;

    // Manual iteration over sequence
    for (int t = 0; t < params.sequenceLength; t++) {
        sum += input[batchInputOffset + t * params.dimensions + d];
    }
}
```

---

## 3. MSL 4.0 Migration Goals

### 3.1 Target Pattern

Replace manual indexing with native tensor operations:

```metal
// Target pattern using MSL 4.0 tensors
#include <metal_tensor>
using namespace metal::tensor_ops;

kernel void tensor_mean_pool_v2(
    tensor<float, shape<dynamic, dynamic, dynamic>> input,  // [B, S, D]
    tensor<float, shape<dynamic, dynamic>> output,          // [B, D]
    constant TensorPoolingParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Use reduce_rows for sequence dimension reduction
    // Native tensor slicing and broadcasting
}
```

### 3.2 Key MSL 4.0 Features to Use

| Feature | Header | Use Case |
|---------|--------|----------|
| `tensor<T, shape<...>>` | `<metal_tensor>` | Type-safe tensor views |
| `matmul2d` | `<metal_tensor_ops>` | Similarity matrix computation |
| `reduce_rows` | `<metal_tensor_ops>` | Mean/sum pooling over sequence |
| `reduce_columns` | `<metal_tensor_ops>` | Batch-wise operations |
| `cooperative_tensor` | `<metal_cooperative_tensor>` | Shared memory tensor tiles |

### 3.3 Expected Benefits

| Metric | Current | Expected with Tensor Ops |
|--------|---------|--------------------------|
| Similarity matrix (128×128, 384D) | ~2.1ms | ~0.8ms (2.6× faster) |
| Fused pool+norm (batch=32) | ~0.4ms | ~0.15ms (2.7× faster) |
| Memory bandwidth utilization | ~65% | ~85% |
| Code complexity | High | Medium |

---

## 4. Kernel Specifications

### 4.1 Tensor Mean Pooling

**Kernel Name**: `tensor_mean_pool_v2`

**Purpose**: Compute mean of token embeddings across sequence dimension, respecting attention mask.

**Tensor Signature**:
```metal
kernel void tensor_mean_pool_v2(
    tensor<float, shape<dynamic, dynamic, dynamic>> input,   // [B, S, D]
    tensor<float, shape<dynamic, dynamic>> output,           // [B, D]
    tensor<int32_t, shape<dynamic, dynamic>> mask,           // [B, S] optional
    constant TensorPoolingParams& params [[buffer(3)]],
    ...
)
```

**Algorithm**:
```
For each batch b:
    For each dimension d:
        sum = 0, count = 0
        For each token t where mask[b,t] == 1:
            sum += input[b, t, d]
            count += 1
        output[b, d] = sum / count
```

**Implementation Guidance**:
- Use `reduce_rows` on input[:, :, d] with custom reduce operation
- Handle mask by zeroing masked positions before reduction
- Compute valid token count separately for division

**Thread Organization**:
- Grid: (dimensions, batchSize, 1)
- Each thread handles one (batch, dimension) pair

---

### 4.2 Tensor Max Pooling

**Kernel Name**: `tensor_max_pool_v2`

**Purpose**: Element-wise maximum across sequence dimension.

**Tensor Signature**:
```metal
kernel void tensor_max_pool_v2(
    tensor<float, shape<dynamic, dynamic, dynamic>> input,   // [B, S, D]
    tensor<float, shape<dynamic, dynamic>> output,           // [B, D]
    tensor<int32_t, shape<dynamic, dynamic>> mask,           // [B, S] optional
    constant TensorPoolingParams& params [[buffer(3)]],
    ...
)
```

**Algorithm**:
```
For each batch b, dimension d:
    maxVal = -INF
    For each token t where mask[b,t] == 1:
        maxVal = max(maxVal, input[b, t, d])
    output[b, d] = maxVal (or 0 if no valid tokens)
```

**Implementation Guidance**:
- Use `reduce_rows` with `max` reduction operation
- Set masked positions to -INF before reduction

---

### 4.3 Tensor L2 Normalization

**Kernel Name**: `tensor_l2_normalize_v2`

**Purpose**: L2 normalize each vector in a batch.

**Tensor Signature**:
```metal
kernel void tensor_l2_normalize_v2(
    tensor<float, shape<dynamic, dynamic>> input,   // [B, D]
    tensor<float, shape<dynamic, dynamic>> output,  // [B, D]
    constant TensorNormParams& params [[buffer(2)]],
    ...
)
```

**Algorithm**:
```
For each batch b:
    norm = sqrt(sum(input[b, :]²))
    output[b, :] = input[b, :] / max(norm, 1e-12)
```

**Implementation Guidance**:
- Use `reduce_rows` with sum-of-squares reduction
- Apply rsqrt and multiply in single pass for output
- Consider cooperative tensors for large dimensions

**Numerical Stability**:
- Use epsilon = 1e-12 for division safety
- Optional: Two-pass algorithm with scaling for high dynamic range

---

### 4.4 Tensor Similarity Matrix

**Kernel Name**: `tensor_similarity_matrix_v2`

**Purpose**: Compute cosine similarity between all query-key pairs.

**Tensor Signature**:
```metal
kernel void tensor_similarity_matrix_v2(
    tensor<float, shape<dynamic, dynamic>> queries,  // [Q, D] (normalized)
    tensor<float, shape<dynamic, dynamic>> keys,     // [K, D] (normalized)
    tensor<float, shape<dynamic, dynamic>> output,   // [Q, K]
    constant TensorSimilarityParams& params [[buffer(3)]],
    ...
)
```

**Algorithm**:
```
output = matmul(queries, transpose(keys))
// For normalized vectors, cosine similarity = dot product
```

**Implementation Guidance**:
- **Primary approach**: Use `matmul2d` with transposed keys
- This is the highest-impact optimization - matrix multiply is perfect for tensor cores
- Pre-normalized inputs allow direct use of dot product result

**Variant for Non-Normalized Inputs**:
```metal
kernel void tensor_similarity_matrix_full_v2(...)
// Compute norms inline and divide
```

---

### 4.5 Fused Pool + Normalize

**Kernel Name**: `fused_pool_normalize_v2`

**Purpose**: Combine pooling and L2 normalization in single kernel to eliminate intermediate buffer.

**Tensor Signature**:
```metal
kernel void fused_pool_normalize_v2(
    tensor<float, shape<dynamic, dynamic, dynamic>> input,  // [B, S, D]
    tensor<float, shape<dynamic, dynamic>> output,          // [B, D]
    tensor<int32_t, shape<dynamic, dynamic>> mask,          // [B, S] optional
    constant FusedPoolNormParams& params [[buffer(3)]],
    ...
)
```

**Algorithm**:
```
For each batch b:
    // Phase 1: Pool
    pooled = pool(input[b, :, :], strategy, mask[b, :])

    // Phase 2: Normalize (in shared memory, no global write)
    norm = sqrt(sum(pooled²))
    output[b, :] = pooled / max(norm, 1e-12)
```

**Implementation Guidance**:
- Use cooperative tensors to keep pooled result in threadgroup memory
- Single global memory write at end
- Strategy selection via params.poolingStrategy (0=mean, 1=max, 2=cls)

---

### 4.6 Tensor Attention-Weighted Pooling

**Kernel Name**: `tensor_attention_pool_v2`

**Purpose**: Weighted sum of token embeddings using attention weights.

**Tensor Signature**:
```metal
kernel void tensor_attention_pool_v2(
    tensor<float, shape<dynamic, dynamic, dynamic>> input,   // [B, S, D]
    tensor<float, shape<dynamic, dynamic>> weights,          // [B, S]
    tensor<float, shape<dynamic, dynamic>> output,           // [B, D]
    constant TensorPoolingParams& params [[buffer(3)]],
    ...
)
```

**Algorithm**:
```
For each batch b:
    weightSum = sum(weights[b, :])
    For each dimension d:
        output[b, d] = sum(input[b, :, d] * weights[b, :]) / weightSum
```

**Implementation Guidance**:
- Broadcast weights across dimension axis
- Element-wise multiply then reduce

---

### 4.7 Cooperative Tensor Pooling (Large Sequences)

**Kernel Name**: `tensor_pool_cooperative_v2`

**Purpose**: Handle sequences > 512 tokens with threadgroup cooperation.

**Design**:
- Use `cooperative_tensor` for shared memory tiles
- Split sequence across threads in threadgroup
- Final reduction in shared memory

**When to Use**: sequenceLength > 512

---

### 4.8 Unified Pooling Entry Point

**Kernel Name**: `tensor_pool_unified_v2`

**Purpose**: Single entry point with strategy selection to reduce PSO management overhead.

**Strategy Enum**:
```metal
// params.poolingStrategy values:
// 0 = mean
// 1 = max
// 2 = cls (first token)
// 3 = attention (requires weights buffer)
```

---

## 5. Parameter Structures

All structures are 16-byte aligned for optimal GPU access. These MUST match the Swift-side definitions exactly.

### 5.1 TensorPoolingParams (16 bytes)

```metal
struct TensorPoolingParams {
    int32_t batchSize;        // Number of sequences
    int32_t sequenceLength;   // Tokens per sequence
    int32_t dimensions;       // Embedding dimensions
    int32_t poolingStrategy;  // 0=mean, 1=max, 2=cls
};
static_assert(sizeof(TensorPoolingParams) == 16);
```

### 5.2 TensorNormParams (16 bytes)

```metal
struct TensorNormParams {
    int32_t batchSize;      // Number of vectors
    int32_t dimensions;     // Vector dimensions
    int32_t _padding0;
    int32_t _padding1;
};
static_assert(sizeof(TensorNormParams) == 16);
```

### 5.3 TensorSimilarityParams (16 bytes)

```metal
struct TensorSimilarityParams {
    int32_t queryBatchSize;   // Number of query embeddings
    int32_t keyBatchSize;     // Number of key embeddings
    int32_t dimensions;       // Embedding dimensions
    int32_t metric;           // 0=cosine, 1=dot, 2=euclidean
};
static_assert(sizeof(TensorSimilarityParams) == 16);
```

### 5.4 FusedPoolNormParams (32 bytes)

```metal
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
static_assert(sizeof(FusedPoolNormParams) == 32);
```

---

## 6. Memory Layouts

### 6.1 Tensor Storage Convention

All tensors use **row-major contiguous storage**:

```
3D Tensor [B, S, D]:
    Physical: [b0_s0_d0, b0_s0_d1, ..., b0_s0_dD, b0_s1_d0, ..., bB_sS_dD]
    Stride:   [S*D, D, 1]

2D Tensor [B, D]:
    Physical: [b0_d0, b0_d1, ..., b0_dD, b1_d0, ..., bB_dD]
    Stride:   [D, 1]
```

### 6.2 Buffer Binding Convention

| Buffer Index | Content | Type |
|--------------|---------|------|
| 0 | Primary input | tensor or device float* |
| 1 | Secondary input or output | tensor or device float* |
| 2 | Mask or weights (optional) | tensor or device int32_t*/float* |
| 3 | Parameters struct | constant T& |

### 6.3 Alignment Requirements

- All buffers: 16-byte aligned
- Parameter structs: 16-byte aligned, sizes are multiples of 16
- Tensor dimensions: No padding required (Metal handles internally)

---

## 7. Performance Targets

### 7.1 Latency Targets (Apple M-series)

| Operation | Batch | Dimensions | Target Latency |
|-----------|-------|------------|----------------|
| Mean pool | 32 | 384 | < 0.1ms |
| L2 normalize | 32 | 384 | < 0.05ms |
| Similarity matrix | 64×64 | 384 | < 0.5ms |
| Similarity matrix | 128×128 | 384 | < 1.5ms |
| Fused pool+norm | 32 | 384 | < 0.15ms |

### 7.2 Throughput Targets

- Pooling: > 50,000 sequences/second (batch=32, S=128, D=384)
- Similarity: > 10M pair comparisons/second

### 7.3 Memory Bandwidth Utilization

- Target: > 80% of theoretical peak
- Current baseline: ~65%

---

## 8. Testing Strategy

### 8.1 Correctness Verification

Each kernel must pass:

1. **Numerical parity** with CPU reference implementation (tolerance: 1e-5 relative error)
2. **Edge cases**:
   - Empty batch (B=0)
   - Single element (B=1, S=1, D=1)
   - All masked tokens
   - Zero vectors (for normalization)
   - Large dimensions (D=1024, D=2048)
3. **Boundary conditions**:
   - Non-power-of-2 dimensions
   - Odd sequence lengths

### 8.2 CPU Reference Functions

Reference implementations exist in `MetalAccelerator.swift` as fallback paths. Search for `cpu()` or `cpuFallback()` closures.

### 8.3 Test Files

Existing tests to maintain parity with:
- `TensorOperationsTests.swift`
- `TensorStorageManagerTests.swift`
- `Metal4EdgeCaseTests.swift`

### 8.4 Benchmark Comparison

Run benchmarks before/after to quantify improvements:
```swift
// Example benchmark pattern
let start = CFAbsoluteTimeGetCurrent()
for _ in 0..<1000 {
    _ = try await accelerator.tensorPoolNormalize(...)
}
let elapsed = CFAbsoluteTimeGetCurrent() - start
```

---

## 9. Code Style Guidelines

### 9.1 File Header

```metal
/// EmbedKit - [Kernel Category] (MSL 4.0 Tensor Implementation)
///
/// [Brief description of kernels in this file]
///
/// **Tensor Shapes**:
/// - Input: [shape description]
/// - Output: [shape description]
///
/// **Compatibility**: Metal 4.0 (iOS 26+ / macOS 26+)
/// **Requires**: <metal_tensor>, <metal_tensor_ops>
```

### 9.2 Include Order

```metal
#include <metal_stdlib>
#include <metal_tensor>
#include <metal_tensor_ops>
#include "../Common/MetalCommon.h"

using namespace metal;
using namespace metal::tensor_ops;
```

### 9.3 Kernel Documentation

```metal
/// Brief description
///
/// **Algorithm**: [Mathematical description]
///
/// **Thread Organization**:
/// - Grid: (x, y, z)
/// - Each thread handles: [description]
///
/// **Performance**: O(complexity)
///
/// @param input [shape] description
/// @param output [shape] description
/// @param params Parameters struct
```

### 9.4 Numerical Constants

```metal
// Use function constants for configurability
constant float EPSILON [[function_constant(1)]];

// Inline fallbacks
const float eps = EPSILON > 0 ? EPSILON : 1e-12f;
```

### 9.5 SIMD Reductions

Prefer SIMD intrinsics over manual reductions:
```metal
// Good
float sum = simd_sum(partial);

// Avoid manual reduction loops when SIMD suffices
```

---

## 10. Deliverables

### 10.1 New Files to Create

| File | Purpose |
|------|---------|
| `TensorPoolingV2.metal` | MSL 4.0 pooling kernels |
| `TensorNormalizationV2.metal` | MSL 4.0 normalization kernels |
| `TensorSimilarityV2.metal` | MSL 4.0 similarity with matmul2d |
| `FusedOperationsV2.metal` | MSL 4.0 fused pipelines |
| `TensorOpsCommon.h` | Shared tensor utilities |

### 10.2 Kernel Naming Convention

New kernels use `_v2` suffix to allow gradual migration:
- `tensor_mean_pool` → `tensor_mean_pool_v2`
- `fused_pool_normalize_unified` → `fused_pool_normalize_v2`

### 10.3 Integration Points

The Swift side (`MetalAccelerator.swift`) will:
1. Load new PSOs for `_v2` kernels
2. Route to new kernels when Metal 4 is available
3. Maintain old kernels as fallback (removed later)

### 10.4 Success Criteria

1. All new kernels compile without warnings
2. Numerical parity with existing kernels (< 1e-5 error)
3. Performance improvement of 2x+ on similarity matrix
4. All existing tests pass
5. No regressions in edge cases

---

## Appendix A: Existing Kernel Source Files

The following files contain the current implementations. Reference these for:
- Algorithm details
- Edge case handling
- Parameter usage patterns

### A.1 MetalCommon.h
Location: `Sources/EmbedKit/Shaders/Common/MetalCommon.h`

Contains:
- All parameter struct definitions
- Utility functions (`compute_l2_norm`, `compute_dot_product`, etc.)
- Function constants for numerical stability control

### A.2 TensorPooling.metal
Location: `Sources/EmbedKit/Shaders/Kernels/TensorPooling.metal`

Kernels:
- `tensor_mean_pool`
- `tensor_max_pool`
- `tensor_cls_pool`
- `tensor_pool_unified`
- `tensor_attention_pool`
- `tensor_mean_pool_cooperative`

### A.3 TensorNormalization.metal
Location: `Sources/EmbedKit/Shaders/Kernels/TensorNormalization.metal`

Kernels:
- `tensor_l2_normalize_with_norms`
- `tensor_compute_norms`
- `tensor_l2_normalize_fused`
- `tensor_l2_normalize_stable`
- `tensor_l2_normalize_inplace`

### A.4 Similarity.metal
Location: `Sources/EmbedKit/Shaders/Kernels/Similarity.metal`

Kernels:
- `cosine_similarity`
- `cosine_similarity_batch`

### A.5 FusedOperations.metal
Location: `Sources/EmbedKit/Shaders/Kernels/FusedOperations.metal`

Kernels:
- `fused_mean_pool_normalize`
- `fused_max_pool_normalize`
- `fused_pool_normalize_unified`
- `fused_attention_pool_normalize`
- `tensor_similarity_matrix_normalized`
- `tensor_similarity_matrix_full`
- `fused_embed_compare_pipeline`

---

## Appendix B: Quick Reference - Thread Grid Sizes

| Kernel | Grid Size | Threadgroup Size |
|--------|-----------|------------------|
| Pooling (per-element) | (dimensions, batchSize, 1) | (256, 1, 1) or auto |
| Pooling (cooperative) | (1, batchSize, 1) | (min(dims, 256), 1, 1) |
| Normalization | (1, batchSize, 1) | (256, 1, 1) |
| Similarity matrix | (keyCount, queryCount, 1) | (16, 16, 1) typical |
| Fused operations | (1, batchSize, 1) | (256, 1, 1) |

---

## Appendix C: Questions for Clarification

Before implementation, please confirm:

1. **Tensor API availability**: Are `metal_tensor` and `metal_tensor_ops` headers available in the Xcode 16 toolchain?

2. **Cooperative tensor support**: Is `metal_cooperative_tensor` available for shared memory tensor tiles?

3. **matmul2d specifics**: What are the exact function signatures and descriptor types for `matmul2d_descriptor`?

4. **Backward compatibility**: Should the V2 kernels completely replace V1, or coexist for gradual rollout?

5. **Testing hardware**: What Apple Silicon variants should be tested (M1, M2, M3, M4)?

---

*Document prepared for external kernel development agent. Last updated: November 2024*

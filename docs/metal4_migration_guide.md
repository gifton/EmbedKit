# EmbedKit Metal 4 Migration Guide

## Overview

This guide details the migration path for EmbedKit to leverage Metal 4's features introduced in iOS 18 / macOS 15. EmbedKit's GPU acceleration currently handles embedding post-processing (normalization, pooling, similarity) and can benefit from Metal 4's unified architecture and tensor operations.

## Table of Contents

1. [Current Architecture Analysis](#current-architecture-analysis)
2. [Metal 4 Opportunities](#metal-4-opportunities)
3. [Unified Encoder Migration](#unified-encoder-migration)
4. [Tensor Operations for Embeddings](#tensor-operations-for-embeddings)
5. [Fused Kernel Optimizations](#fused-kernel-optimizations)
6. [Performance Projections](#performance-projections)
7. [Implementation Plan](#implementation-plan)

---

## Current Architecture Analysis

### Metal Components

```
Sources/EmbedKit/
├── Acceleration/
│   ├── MetalAccelerator.swift      # Main GPU coordinator (~700 lines)
│   ├── AccelerationManager.swift   # VectorAccelerate integration
│   ├── ComputePreference.swift     # GPU/CPU routing
│   └── MetalTypes.swift            # Buffer types
└── Shaders/
    ├── Common/
    │   └── MetalCommon.h           # Shared definitions
    └── Kernels/
        ├── Normalization.metal             # L2 normalization
        ├── NormalizationBatchOptimized.metal  # Batch L2 norm
        ├── Pooling.metal                   # Mean/Max pooling
        └── Similarity.metal                # Cosine similarity matrix
```

### Current GPU Operations

| Operation | Kernel | Use Case |
|-----------|--------|----------|
| L2 Normalize | `l2_normalize` | Unit-length embedding vectors |
| L2 Normalize Batch | `l2_normalize_batch_optimized` | Batch normalization |
| Mean Pooling | `mean_pool` | Token → sentence embedding |
| Max Pooling | `max_pool` | Token → sentence embedding |
| Attention Pooling | `attention_weighted_pool` | Weighted token aggregation |
| Cosine Similarity | `cosine_similarity` | Pairwise similarity matrix |
| Cosine Batch | `cosine_similarity_batch` | Batched similarity |

### Current Architecture Patterns

```swift
// Current: Traditional command buffer pattern
guard let cmd = queue.makeCommandBuffer(),
      let enc = cmd.makeComputeCommandEncoder()
else { return cpuFallback() }

enc.setComputePipelineState(pso)
enc.setBuffer(inputBuf, offset: 0, index: 0)
enc.setBuffer(outputBuf, offset: 0, index: 1)
enc.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
enc.endEncoding()

// Async completion via continuation
let _: Void = await withCheckedContinuation { cont in
    cmd.addCompletedHandler { _ in cont.resume(returning: ()) }
    cmd.commit()
}
```

### Strengths of Current Implementation

- ✅ Already uses actor isolation (`MetalAccelerator` is an actor)
- ✅ Proper async/await patterns
- ✅ CPU fallback for all operations
- ✅ Threshold-based GPU routing (avoids GPU overhead for small workloads)
- ✅ Tiled computation for large similarity matrices

### Areas for Metal 4 Improvement

- ❌ Separate command buffer per operation
- ❌ No tensor type support
- ❌ No fused operations (pooling + normalization separate)
- ❌ Manual async completion handling (can use `.completed`)

---

## Metal 4 Opportunities

### 1. Unified Command Encoder

Metal 4's unified encoder eliminates the need for separate compute/blit encoders:

```swift
// Metal 4: Unified encoder
let commandBuffer = MTL4CommandBuffer(device: device, queue: queue)
let encoder = commandBuffer.makeUnifiedEncoder()!

// All operations in single encoder
encoder.setComputePipelineState(normalizePipeline)
encoder.dispatchThreadgroups(...)
encoder.setComputePipelineState(poolPipeline)  // Switch pipeline, same encoder
encoder.dispatchThreadgroups(...)

encoder.endEncoding()
_ = await commandBuffer.completed
```

**Benefit for EmbedKit**: Embedding pipelines often chain operations (tokenize → inference → pool → normalize). Unified encoder reduces synchronization overhead.

### 2. Native Tensor Types

Metal 4 introduces tensor types that map directly to embedding semantics:

```metal
// Metal 4 tensor types
using tensor2d_float = metal::tensor<float, 2>;

// Perfect for embeddings: [batchSize, dimensions]
kernel void normalize_tensor(
    tensor2d_float embeddings [[buffer(0)]],
    tensor2d_float output [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    // Access embedding vector directly
    auto embedding = embeddings[tid];
    float norm = tensor_l2_norm(embedding);
    output[tid] = embedding / max(norm, 1e-8f);
}
```

**Benefit for EmbedKit**: Cleaner code, potential hardware acceleration for tensor operations.

### 3. Native Async Completion

Metal 4 (iOS 18+/macOS 15+) provides native async completion:

```swift
// Current: Continuation-based
let _: Void = await withCheckedContinuation { cont in
    cmd.addCompletedHandler { _ in cont.resume(returning: ()) }
    cmd.commit()
}

// Metal 4: Native async
cmd.commit()
_ = await cmd.completed
```

**Status**: ✅ Already addressed in macOS 15+ platform target update.

### 4. Fused Operations

Metal 4 enables more efficient fused kernels:

| Current (Separate) | Metal 4 (Fused) | Savings |
|--------------------|-----------------|---------|
| Mean Pool → L2 Norm | `mean_pool_normalized` | 1 pass, 50% memory BW |
| Similarity → Top-K | `similarity_topk_fused` | 1 pass, no intermediate |
| Pool → Norm → Similarity | `embedding_compare_fused` | Single dispatch |

---

## Unified Encoder Migration

### MetalAccelerator Updates

```swift
public actor MetalAccelerator {
    // ...existing properties...

    // Metal 4 unified execution
    private func executeUnified<T>(
        _ operation: (MTLComputeCommandEncoder) async throws -> T
    ) async throws -> T {
        guard let queue = commandQueue,
              let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder()
        else {
            throw MetalError.deviceNotAvailable
        }

        let result = try await operation(encoder)

        encoder.endEncoding()
        cmd.commit()
        _ = await cmd.completed  // Metal 4 native async

        return result
    }

    // Updated pooling with unified execution
    public func meanPoolNormalized(
        embeddings: [Float],
        sequenceLength: Int,
        dimensions: Int,
        mask: [Int]? = nil
    ) async -> [Float] {
        guard isAvailable else { return cpuFallback() }

        do {
            return try await executeUnified { encoder in
                // Fused mean pooling + L2 normalization
                encoder.setComputePipelineState(psoMeanPoolNormalized)
                // ...buffer setup...
                encoder.dispatchThreadgroups(...)

                return readResults()
            }
        } catch {
            return cpuFallback()
        }
    }
}
```

### Pipeline Chaining Example

```swift
/// Complete embedding post-processing pipeline
public func processEmbeddings(
    tokenEmbeddings: [[Float]],  // [sequenceLength, dimensions]
    masks: [[Int]]?
) async -> [Float] {
    try await executeUnified { encoder in
        // Step 1: Mean pooling
        encoder.setComputePipelineState(psoMeanPool)
        encoder.setBuffer(tokenBuffer, offset: 0, index: 0)
        encoder.setBuffer(pooledBuffer, offset: 0, index: 1)
        encoder.dispatchThreadgroups(poolGridSize, threadsPerThreadgroup: poolThreads)

        // Step 2: L2 normalization (same encoder, no sync needed)
        encoder.setComputePipelineState(psoL2Normalize)
        encoder.setBuffer(pooledBuffer, offset: 0, index: 0)
        encoder.setBuffer(normalizedBuffer, offset: 0, index: 1)
        encoder.dispatchThreadgroups(normGridSize, threadsPerThreadgroup: normThreads)

        return readNormalizedResults()
    }
}
```

---

## Tensor Operations for Embeddings

### New Tensor Types

```swift
// Sources/EmbedKit/Acceleration/TensorTypes.swift

/// 2D tensor for batch embeddings [batchSize, dimensions]
public struct EmbeddingTensor: Sendable {
    public let buffer: any MTLBuffer
    public let batchSize: Int
    public let dimensions: Int

    public init(batchSize: Int, dimensions: Int, device: any MTLDevice) {
        self.batchSize = batchSize
        self.dimensions = dimensions
        let size = batchSize * dimensions * MemoryLayout<Float>.stride
        self.buffer = device.makeBuffer(length: size, options: .storageModeShared)!
    }

    /// Create from existing embedding vectors
    public init(embeddings: [[Float]], device: any MTLDevice) {
        self.batchSize = embeddings.count
        self.dimensions = embeddings.first?.count ?? 0
        let flat = embeddings.flatMap { $0 }
        self.buffer = device.makeBuffer(
            bytes: flat,
            length: flat.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
    }
}

/// 3D tensor for token embeddings [batchSize, sequenceLength, dimensions]
public struct TokenEmbeddingTensor: Sendable {
    public let buffer: any MTLBuffer
    public let batchSize: Int
    public let sequenceLength: Int
    public let dimensions: Int

    public var shape: (Int, Int, Int) {
        (batchSize, sequenceLength, dimensions)
    }
}
```

### Tensor-Based Shader Updates

```metal
// Shaders/Kernels/TensorPooling.metal

#include <metal_stdlib>
using namespace metal;

// Metal 4 tensor type
using tensor2d_float = metal::tensor<float, 2>;
using tensor3d_float = metal::tensor<float, 3>;

/// Tensor-based mean pooling with optional mask
kernel void mean_pool_tensor(
    tensor3d_float tokens [[buffer(0)]],      // [batch, seq, dim]
    tensor2d_float output [[buffer(1)]],      // [batch, dim]
    device const int* mask [[buffer(2)]],     // [batch, seq] optional
    constant uint& has_mask [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]     // (batch, dim)
) {
    const uint batch = gid.x;
    const uint dim = gid.y;

    if (batch >= tokens.shape[0] || dim >= tokens.shape[2]) return;

    float sum = 0.0f;
    int count = 0;

    for (uint t = 0; t < tokens.shape[1]; t++) {
        bool valid = !has_mask || mask[batch * tokens.shape[1] + t] == 1;
        if (valid) {
            sum += tokens[batch][t][dim];
            count++;
        }
    }

    output[batch][dim] = (count > 0) ? (sum / float(count)) : 0.0f;
}

/// Fused pooling + normalization using tensor intrinsics
kernel void pool_normalize_tensor(
    tensor3d_float tokens [[buffer(0)]],      // [batch, seq, dim]
    tensor2d_float output [[buffer(1)]],      // [batch, dim]
    uint batch_id [[thread_position_in_grid]]
) {
    if (batch_id >= tokens.shape[0]) return;

    // Mean pool this batch
    auto pooled = tensor_reduce_mean(tokens[batch_id], /*axis=*/0);

    // L2 normalize
    float norm = tensor_l2_norm(pooled);
    output[batch_id] = pooled / max(norm, 1e-8f);
}
```

---

## Fused Kernel Optimizations

### Current: Separate Operations

```
Token Embeddings → [Mean Pool] → Pooled → [L2 Norm] → Normalized → [Similarity] → Matrix
       ↓                 ↓                    ↓                          ↓
   3 dispatches    2 buffer copies     1 buffer copy              1 buffer read
```

### Metal 4: Fused Pipeline

```
Token Embeddings → [Pool+Norm+Similarity Fused] → Matrix
       ↓                       ↓
   1 dispatch             1 buffer read
```

### Fused Embedding Comparison Kernel

```metal
// Shaders/Kernels/FusedEmbeddingCompare.metal

/// Fused operation: Pool tokens, normalize, compute similarity
/// Reduces memory bandwidth by 3x compared to separate operations
kernel void embedding_compare_fused(
    tensor3d_float query_tokens [[buffer(0)]],    // [1, seq_q, dim]
    tensor3d_float doc_tokens [[buffer(1)]],      // [N, seq_d, dim]
    device float* similarities [[buffer(2)]],      // [N]
    uint doc_id [[thread_position_in_grid]]
) {
    if (doc_id >= doc_tokens.shape[0]) return;

    // Pool and normalize query (could be cached)
    auto query_pooled = tensor_reduce_mean(query_tokens[0], 0);
    float query_norm = tensor_l2_norm(query_pooled);
    auto query_normalized = query_pooled / max(query_norm, 1e-8f);

    // Pool and normalize document
    auto doc_pooled = tensor_reduce_mean(doc_tokens[doc_id], 0);
    float doc_norm = tensor_l2_norm(doc_pooled);
    auto doc_normalized = doc_pooled / max(doc_norm, 1e-8f);

    // Cosine similarity (dot product of normalized vectors)
    similarities[doc_id] = tensor_dot_product(query_normalized, doc_normalized);
}
```

### Batched Embedding Processing

```metal
/// Process entire batch of embeddings in one dispatch
kernel void batch_embed_process(
    tensor3d_float all_tokens [[buffer(0)]],      // [batch, seq, dim]
    tensor2d_float embeddings [[buffer(1)]],      // [batch, dim]
    constant PoolingConfig& config [[buffer(2)]],
    uint batch_id [[thread_position_in_grid]]
) {
    if (batch_id >= all_tokens.shape[0]) return;

    auto tokens = all_tokens[batch_id];
    float3 result;

    switch (config.pooling_strategy) {
        case POOL_MEAN:
            result = tensor_reduce_mean(tokens, 0);
            break;
        case POOL_MAX:
            result = tensor_reduce_max(tokens, 0);
            break;
        case POOL_CLS:
            result = tokens[0];  // First token
            break;
    }

    if (config.normalize) {
        float norm = tensor_l2_norm(result);
        result = result / max(norm, 1e-8f);
    }

    embeddings[batch_id] = result;
}
```

---

## Performance Projections

### Estimated Improvements

Based on VectorAccelerate's Metal 4 benchmarks and EmbedKit's operation profile:

| Operation | Current (ms) | Metal 4 (ms) | Improvement |
|-----------|--------------|--------------|-------------|
| L2 Normalize (1K × 384) | 0.8 | 0.5 | 37% |
| L2 Normalize (10K × 768) | 3.2 | 1.9 | 41% |
| Mean Pool (512 seq × 384) | 0.4 | 0.25 | 38% |
| Mean Pool (1K seq × 768) | 1.1 | 0.6 | 45% |
| Similarity Matrix (1K × 1K) | 4.5 | 2.8 | 38% |
| Similarity Matrix (10K × 10K) | 420 | 230 | 45% |
| **Fused Pool+Norm** | 1.2 | 0.5 | 58% |
| **Fused Pool+Norm+Sim** | 5.8 | 2.2 | 62% |

### Memory Bandwidth Savings

| Operation Chain | Current | Metal 4 Fused | Reduction |
|-----------------|---------|---------------|-----------|
| Pool → Norm | 4D reads, 2D writes | 2D reads, D writes | 50% |
| Pool → Norm → Similarity | 4D + 2D + N×D | 2D + N | 60-70% |
| Batch Embed (100 docs) | 100 dispatches | 1 dispatch | 99% dispatch overhead |

---

## Implementation Plan

### Phase 1: Foundation (Current Release) ✅ COMPLETE
- [x] Update platform targets to macOS 15+/iOS 18+
- [x] Remove `@available` decorators
- [x] Use native `MTLCommandBuffer.completed` async
- [x] Add `TensorTypes.swift` with EmbeddingTensor types
  - `EmbeddingTensor` - 2D tensor [batchSize, dimensions]
  - `TokenEmbeddingTensor` - 3D tensor [batchSize, sequenceLength, dimensions]
  - `TensorPoolingParams`, `TensorNormalizationParams`, `TensorSimilarityParams`

### Phase 2: Unified Encoder (v0.2.0) ✅ COMPLETE
- [x] Implement `executeUnified()` helper in MetalAccelerator
- [x] Migrate pooling operations to unified encoder
- [x] Migrate normalization operations
- [x] Migrate similarity operations
- [x] Add pipeline chaining for embedding post-processing
  - `meanPoolNormalized()` - Fused mean pooling + L2 normalization
  - `processEmbeddingsBatch()` - Complete embedding pipeline

### Phase 3: Tensor Operations (v0.3.0) ✅ COMPLETE
- [x] Add Metal 4 tensor shader variants
  - `TensorPooling.metal` - batch pooling kernels
  - `TensorNormalization.metal` - batch L2 normalization
  - `FusedOperations.metal` - fused pipelines
- [x] Implement tensor-based pooling kernels
  - `tensor_mean_pool`, `tensor_max_pool`, `tensor_cls_pool`
  - `tensor_pool_unified` - unified with strategy selection
  - `tensor_mean_pool_cooperative` - for large sequences
- [x] Add tensor-based normalization
  - `tensor_l2_normalize_fused` - single-pass per vector
  - `tensor_l2_normalize_stable` - Kahan-compensated
  - `tensor_l2_normalize_inplace` - memory-efficient
- [x] Implement fused operations (now in Phase 3!)
  - `fused_mean_pool_normalize`, `fused_max_pool_normalize`
  - `fused_pool_normalize_unified` - complete pipeline
  - `tensor_similarity_matrix_normalized/full`
- [x] Update Swift API
  - `tensorPoolNormalize()` - batch pool+norm
  - `tensorSimilarityMatrix()` - batch similarity
  - `tensorPipelinesAvailable` - availability check
  - `FusedPoolNormParams`, `EmbeddingPipelineParams`

### Phase 4: Performance Optimization (v0.4.0) ✅ COMPLETE
- [x] Profile and tune threadgroup sizes for M-series chips
  - `GPUDeviceCapabilities` - M1/M2/M3/M4 family detection
  - `ThreadgroupOptimizer` - optimal dispatch parameters per device
  - Automatic SIMD width detection (32 for Apple Silicon, 64 for AMD)
- [x] Add adaptive kernel selection (fused vs separate based on workload)
  - `AdaptiveKernelSelector` - performance-history-based selection
  - Automatic CPU/GPU/fused/progressive routing
  - Workload size thresholds with adaptive learning
- [x] Implement progressive similarity computation for large batches
  - `ProgressiveSimilarityComputer` - tiled computation
  - Automatic tile sizing based on GPU memory
  - Device-specific tile configurations
- [x] Add residency hints for embedding buffers
  - `BufferResidencyManager` - tracks frequently used buffers
  - LRU eviction with access count weighting
  - Configurable max resident memory
- [x] Integrate optimization infrastructure
  - `GPUOptimizer` - combined entry point
  - MetalAccelerator integration with optimized dispatch
  - Public API for kernel selection and performance recording

---

## Testing Strategy

### Unit Tests

```swift
@Suite("Metal 4 Tensor Operations")
struct Metal4TensorTests {

    @Test("Tensor pooling produces correct results")
    func tensorPooling() async throws {
        let accelerator = await MetalAccelerator()
        let tokens = generateTestTokens(batch: 10, seq: 128, dim: 384)

        let tensor = TokenEmbeddingTensor(tokens: tokens, device: accelerator.device)
        let pooled = try await accelerator.meanPoolTensor(tensor)

        // Verify against CPU reference
        let cpuPooled = cpuMeanPool(tokens)
        #expect(pooled.isApproximatelyEqual(to: cpuPooled, tolerance: 1e-5))
    }

    @Test("Fused pool+norm matches separate operations")
    func fusedPoolNorm() async throws {
        let accelerator = await MetalAccelerator()
        let tokens = generateTestTokens(batch: 100, seq: 256, dim: 768)

        // Separate operations
        let pooled = try await accelerator.meanPool(tokens)
        let normalized = try await accelerator.l2Normalize(pooled)

        // Fused operation
        let fused = try await accelerator.meanPoolNormalized(tokens)

        #expect(fused.isApproximatelyEqual(to: normalized, tolerance: 1e-5))
    }
}
```

### Performance Tests

```swift
@Suite("Metal 4 Performance")
struct Metal4PerformanceTests {

    @Test("Fused operations are faster than separate")
    func fusedPerformance() async throws {
        let accelerator = await MetalAccelerator()
        let tokens = generateLargeTokens(batch: 1000, seq: 512, dim: 768)

        // Measure separate
        let separateTime = try await measure {
            let pooled = try await accelerator.meanPool(tokens)
            _ = try await accelerator.l2Normalize(pooled)
        }

        // Measure fused
        let fusedTime = try await measure {
            _ = try await accelerator.meanPoolNormalized(tokens)
        }

        // Fused should be at least 30% faster
        #expect(fusedTime < separateTime * 0.7)
    }
}
```

---

## Conclusion

EmbedKit's Metal 4 migration offers significant performance improvements:

1. **Unified Encoders**: Eliminate per-operation synchronization overhead
2. **Tensor Types**: Cleaner code, potential hardware acceleration
3. **Fused Kernels**: 50-60% memory bandwidth reduction
4. **Native Async**: Cleaner async/await patterns (already implemented)

The migration can be completed incrementally across releases, with each phase providing measurable benefits while maintaining backward compatibility through CPU fallbacks.

---

*Migration Guide Version: 4.0*
*Created: November 2025*
*Updated: November 2025 - Phase 1, 2, 3 & 4 Complete*
*Target: EmbedKit v0.4.0+ with Metal 4 (iOS 18.0+ / macOS 15.0+)*

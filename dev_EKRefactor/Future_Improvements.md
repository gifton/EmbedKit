# EmbedKit Future Improvements

**Last Updated**: 2025-11-25
**Version**: 1.0.0

This document tracks potential improvements and enhancements for EmbedKit, organized by component. Items are prioritized by impact and complexity.

---

## Priority Legend
- **P0**: Critical - Should be addressed soon
- **P1**: High - Significant value, plan for next release
- **P2**: Medium - Nice to have, schedule when capacity allows
- **P3**: Low - Future consideration, backlog

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Completed

---

## 1. Core Component

### Types & Protocols (`Core/Types.swift`, `Core/Protocols.swift`)

| Priority | Status | Improvement | Description | Complexity |
|----------|--------|-------------|-------------|------------|
| P2 | [ ] | Generic embedding dimensions | Support compile-time dimension checking via phantom types (`Embedding<D384>`) | Medium |
| P2 | [ ] | Streaming embeddings | Add `AsyncSequence`-based embedding generation for large document sets | Medium |
| P3 | [ ] | Embedding versioning | Track embedding model version in metadata for cache invalidation | Low |
| P2 | [ ] | Batch options builder | Fluent API for `BatchOptions` construction | Low |

### Metrics & Profiling (`Core/Metrics.swift`, `Core/Profiler.swift`)

| Priority | Status | Improvement | Description | Complexity |
|----------|--------|-------------|-------------|------------|
| P1 | [x] | Histogram metrics | Add latency histograms (p50, p95, p99) instead of just averages | Medium |
| P2 | [ ] | Metal GPU metrics | Integrate Metal performance counters (GPU time, bandwidth) | Medium |
| P2 | [ ] | Memory pressure tracking | Real-time memory pressure monitoring with automatic backoff | Medium |
| P3 | [ ] | OpenTelemetry export | Export metrics to OpenTelemetry-compatible backends | High |

### Error Handling (`Core/Errors.swift`)

| Priority | Status | Improvement | Description | Complexity |
|----------|--------|-------------|-------------|------------|
| P1 | [x] | Error recovery hints | Add `recoverySuggestion` to all error cases | Low |
| P2 | [ ] | Structured error codes | Numeric error codes for programmatic handling | Low |
| P3 | [ ] | Error aggregation | Batch error collection with partial success support | Medium |

---

## 2. Acceleration Component

### Metal Accelerator (`Acceleration/MetalAccelerator.swift`)

| Priority | Status | Improvement | Description | Complexity |
|----------|--------|-------------|-------------|------------|
| P1 | [ ] | Float16 compute path | Add Float16 versions of all kernels for memory bandwidth savings | High |
| P1 | [x] | Async command encoding | Overlap CPU encoding with GPU execution using triple buffering | Medium |
| P2 | [ ] | Metal Performance Shaders | Evaluate MPS for matrix operations where beneficial | Medium |
| P2 | [ ] | Shared memory optimization | Use threadgroup memory for frequently accessed data in kernels | Medium |
| P2 | [ ] | Metal 4 mesh shaders | Investigate mesh shader benefits for similarity matrix computation | High |
| P3 | [ ] | Metal debugging integration | Add Metal debugger capture triggers for profiling | Low |

### GPU Optimizer (`Acceleration/GPUOptimizer.swift`)

| Priority | Status | Improvement | Description | Complexity |
|----------|--------|-------------|-------------|------------|
| P1 | [x] | Persistent performance DB | Store learned kernel performance across app launches | Medium |
| P2 | [ ] | Workload prediction | Predict optimal kernel based on input characteristics | High |
| P2 | [ ] | M4 Dynamic Caching | Leverage M4's hardware-accelerated cache management | Medium |
| P2 | [ ] | Buffer pooling | Reuse Metal buffers to reduce allocation overhead | Medium |
| P3 | [ ] | Multi-GPU support | Distribute workloads across multiple GPUs (Mac Pro) | High |

### Tensor Operations (`Acceleration/TensorTypes.swift`, Shaders)

| Priority | Status | Improvement | Description | Complexity |
|----------|--------|-------------|-------------|------------|
| P1 | [x] | Attention-weighted pooling GPU | Move attention pooling to GPU (currently CPU-only) | Medium |
| P2 | [ ] | Sparse attention masks | Optimize for sparse attention patterns in long sequences | High |
| P2 | [ ] | Batched similarity | Single-dispatch similarity for entire embedding stores | Medium |
| P3 | [ ] | Custom Metal library caching | Cache compiled metallib per device for faster startup | Low |

---

## 3. Tokenization Component

### WordPiece Tokenizer (`Tokenization/WordPieceTokenizer.swift`)

| Priority | Status | Improvement | Description | Complexity |
|----------|--------|-------------|-------------|------------|
| P1 | [ ] | Aho-Corasick vocabulary | Replace linear search with Aho-Corasick for O(n) tokenization | High |
| P1 | [x] | Batch tokenization | Parallel tokenization for large batches | Medium |
| P2 | [ ] | Pre-tokenization caching | Cache common words/phrases for repeated documents | Low |
| P2 | [ ] | Unicode normalization options | Support NFD, NFC, NFKD, NFKC normalization modes | Medium |
| P3 | [ ] | Custom vocab loading | Support runtime vocabulary hot-reloading | Low |

### BPE & SentencePiece Tokenizers

| Priority | Status | Improvement | Description | Complexity |
|----------|--------|-------------|-------------|------------|
| P2 | [ ] | BPE merge optimization | Use priority queue for faster merge operations | Medium |
| P2 | [ ] | SentencePiece native | Consider calling sentencepiece C++ library via Swift wrapper | High |
| P3 | [ ] | Tokenizer model files | Support HuggingFace tokenizer.json format directly | High |

### Vocabulary (`Tokenization/Vocabulary.swift`)

| Priority | Status | Improvement | Description | Complexity |
|----------|--------|-------------|-------------|------------|
| P2 | [ ] | Memory-mapped vocabulary | Use mmap for large vocabularies to reduce memory | Medium |
| P3 | [ ] | Vocabulary compression | Compress rarely-used tokens in memory | Medium |

---

## 4. Models Component

### AppleEmbeddingModel (`Models/AppleEmbeddingModel.swift`)

| Priority | Status | Improvement | Description | Complexity |
|----------|--------|-------------|-------------|------------|
| P0 | [x] | Batch GPU operations | Use new tensor operations for batch pooling/normalization | Medium |
| P1 | [ ] | Pipeline fusion | Fuse tokenization → inference → pooling when possible | High |
| P2 | [ ] | Model quantization support | Support INT8/INT4 quantized CoreML models | Medium |
| P2 | [ ] | Speculative execution | Pre-tokenize next batch while current inference runs | Medium |
| P2 | [ ] | Dynamic shape support | Better handling of variable-length CoreML inputs | Medium |
| P3 | [ ] | Model ensemble | Support multiple model voting/averaging | Medium |

### LocalCoreMLModel (`Models/LocalCoreMLModel.swift`)

| Priority | Status | Improvement | Description | Complexity |
|----------|--------|-------------|-------------|------------|
| P2 | [ ] | On-device model fine-tuning | Support personalized embedding adaptation | High |
| P3 | [ ] | Model compression | Automatic model pruning/distillation | High |

---

## 5. Backend Component

### CoreML Backend (`Backends/CoreMLBackend.swift`)

| Priority | Status | Improvement | Description | Complexity |
|----------|--------|-------------|-------------|------------|
| P1 | [x] | Stateful predictions | Use MLPredictionOptions with persistent state | Medium |
| P1 | [x] | Flexible shapes | Better dynamic shape detection and validation | Medium |
| P2 | [ ] | Neural Engine targeting | Explicit ANE optimization hints | Medium |
| P2 | [ ] | CoreML async predictions | Leverage newer async CoreML APIs | Low |
| P3 | [ ] | Model inspection | Extract model architecture info for auto-configuration | Medium |

---

## 6. Storage Component

### EmbeddingStore (`Storage/EmbeddingStore.swift`)

| Priority | Status | Improvement | Description | Complexity |
|----------|--------|-------------|-------------|------------|
| P1 | [ ] | GPU-accelerated search | Use Metal for similarity computation during search | High |
| P1 | [ ] | Incremental index updates | Support add/remove without full rebuild | Medium |
| P2 | [ ] | Hybrid search | Combine semantic + keyword search (BM25 + embeddings) | High |
| P2 | [ ] | Filtered search | Support metadata-based pre/post filtering | Medium |
| P2 | [ ] | Disk-backed storage | Persist embeddings to disk with lazy loading | Medium |
| P3 | [ ] | Distributed store | Multi-node embedding storage for large collections | High |

### Index Implementations

| Priority | Status | Improvement | Description | Complexity |
|----------|--------|-------------|-------------|------------|
| P2 | [ ] | GPU HNSW traversal | Accelerate HNSW neighbor search on GPU | High |
| P2 | [ ] | Product quantization | PQ compression for billion-scale indices | High |
| P3 | [ ] | Learned indices | ML-based index structures for specific data distributions | High |

---

## 7. Optimization Component

### Adaptive Batcher (`Optimization/AdaptiveBatcher.swift`)

| Priority | Status | Improvement | Description | Complexity |
|----------|--------|-------------|-------------|------------|
| P1 | [x] | Request prioritization | Support priority levels for urgent requests | Medium |
| P2 | [ ] | Deadline-aware batching | Consider request deadlines when forming batches | Medium |
| P2 | [ ] | Batch size learning | Auto-tune batch sizes based on observed performance | Medium |
| P3 | [ ] | Preemption support | Allow high-priority requests to interrupt batches | High |

### Quantization (`Optimization/Quantization.swift`)

| Priority | Status | Improvement | Description | Complexity |
|----------|--------|-------------|-------------|------------|
| P1 | [ ] | GPU quantization | Perform quantization/dequantization on GPU | Medium |
| P2 | [ ] | INT4 quantization | Add 4-bit quantization for 8x compression | Medium |
| P2 | [ ] | Per-channel quantization | Improved accuracy with per-dimension scale factors | Medium |
| P3 | [ ] | Binary embeddings | 1-bit embeddings for extremely fast similarity | Medium |

---

## 8. Caching Component

### Token Cache (`Caches/TokenCache.swift`)

| Priority | Status | Improvement | Description | Complexity |
|----------|--------|-------------|-------------|------------|
| P2 | [ ] | TTL support | Time-based cache expiration | Low |
| P2 | [ ] | Size-aware eviction | Consider entry size in LRU decisions | Low |
| P3 | [ ] | Distributed cache | Share token cache across processes | High |

### Persistent Cache (`Caches/PersistentCache.swift`)

| Priority | Status | Improvement | Description | Complexity |
|----------|--------|-------------|-------------|------------|
| P1 | [ ] | Embedding cache | Cache computed embeddings by content hash | Medium |
| P2 | [ ] | Cache compression | Compress cached embeddings using quantization | Medium |
| P2 | [ ] | Cache warming | Pre-populate cache on app launch | Low |
| P3 | [ ] | Cross-app cache | Shared embedding cache using app groups | Medium |

---

## 9. API Component

### Convenience API (`API/ConvenienceAPI.swift`)

| Priority | Status | Improvement | Description | Complexity |
|----------|--------|-------------|-------------|------------|
| P2 | [ ] | Result builders | SwiftUI-style DSL for embedding pipelines | Medium |
| P2 | [ ] | Async iterators | `for await embedding in model.embed(documents)` | Low |
| P3 | [ ] | Combine publishers | Reactive API for embedding streams | Low |

### SwiftUI Support (`API/SwiftUISupport.swift`)

| Priority | Status | Improvement | Description | Complexity |
|----------|--------|-------------|-------------|------------|
| P2 | [ ] | @Environment model | SwiftUI environment injection for models | Low |
| P2 | [ ] | Search view modifiers | `.searchable` integration with semantic search | Medium |
| P3 | [ ] | Preview mocking | Mock models for SwiftUI previews | Low |

---

## 10. Cross-Cutting Improvements

### Performance

| Priority | Status | Improvement | Description | Complexity |
|----------|--------|-------------|-------------|------------|
| P1 | [ ] | Zero-copy pipelines | Eliminate array copies between pipeline stages | High |
| P1 | [x] | Accelerate BLAS | Use vDSP/BLAS for CPU fallback paths | Medium |
| P2 | [ ] | SIMD optimizations | Manual SIMD for critical CPU paths | Medium |
| P2 | [ ] | Memory pooling | Reuse allocations across embedding operations | Medium |

### Testing

| Priority | Status | Improvement | Description | Complexity |
|----------|--------|-------------|-------------|------------|
| P1 | [ ] | Benchmark suite | Automated performance regression tests | Medium |
| P2 | [ ] | Fuzz testing | Property-based testing for tokenizers | Medium |
| P2 | [ ] | Integration tests | End-to-end tests with real models | Medium |
| P3 | [ ] | Chaos testing | Test behavior under resource pressure | Medium |

### Documentation

| Priority | Status | Improvement | Description | Complexity |
|----------|--------|-------------|-------------|------------|
| P1 | [ ] | Migration guide | Guide for upgrading between major versions | Low |
| P2 | [ ] | Architecture docs | High-level architecture documentation | Low |
| P2 | [ ] | Performance tuning guide | Best practices for production deployment | Low |
| P3 | [ ] | Video tutorials | Screen recordings of common workflows | Low |

---

## Completed Improvements

### Metal 4 Migration (v2.0)
- [x] Tensor-based batch operations (Phase 3)
- [x] GPU optimizer with adaptive kernel selection (Phase 4)
- [x] Device-specific threadgroup tuning
- [x] Progressive similarity computation for large batches
- [x] Fused pool+normalize kernels

### AppleEmbeddingModel GPU Integration (v2.0)
- [x] Batch GPU pooling/normalization via tensorPoolNormalize
- [x] Single GPU dispatch for entire micro-batches
- [x] Automatic fallback to per-item processing for variable-shape outputs

### Error Handling Improvements (v2.0)
- [x] Added `recoverySuggestion` to all error types (EmbedKitError, CacheError, AccelerationError, EmbeddingStoreError, QuantizationError, ONNXBackendError)
- [x] Added `failureReason` to EmbedKitError for additional context

### Tokenization Improvements (v2.0)
- [x] Added `encodeBatch` to Tokenizer protocol with parallel processing
- [x] Default implementation uses Swift concurrency (TaskGroup) with controlled parallelism
- [x] BatchOptions.tokenizationConcurrency defaults to parallel processing (ProcessInfo.activeProcessorCount)

### Histogram Metrics (v2.0)
- [x] Added `LatencyHistogram` struct with configurable bucket boundaries
- [x] Added `HistogramStatistics` with p50, p75, p90, p95, p99, p999 percentiles
- [x] Updated `ModelMetrics` with `latencyStats: HistogramStatistics?`
- [x] Updated `StageMetrics` with tokenization/inference/pooling histogram stats
- [x] Human-readable `distributionSummary` and `summary` formatters

### GPU Optimizer Persistence (v2.0)
- [x] Added persistent performance database to `AdaptiveKernelSelector`
- [x] JSON-based storage in caches directory with ISO8601 timestamps
- [x] Device family compatibility checking (invalidates data on hardware change)
- [x] 7-day expiration filtering for stale performance records
- [x] `savePerformanceHistory()`, `loadPerformanceHistory()`, `clearPerformanceHistory()` methods
- [x] Comprehensive test suite (11 tests) covering persistence workflow

### Async Command Encoding / Triple Buffering (v2.0)
- [x] `CommandBufferPool` - manages triple-buffered command submissions with DispatchSemaphore
- [x] `MetalBufferPool` - reusable buffer pool with power-of-2 size classes to reduce allocation overhead
- [x] `streamingPoolNormalize()` - processes multiple batches with overlapped CPU/GPU execution
- [x] Public accessors `tripleBufferPool`, `metalBufferPool`, `bufferPoolStatistics`
- [x] DispatchQueue-based thread-safe synchronization (Swift 6 async-compatible)
- [x] Comprehensive test suite (13 tests) covering pool operations and streaming

### GPU Attention-Weighted Pooling (v2.0)
- [x] Added `.attention` case to `PoolingStrategy` enum with `metalIndex = 3`
- [x] `tensor_attention_pool` kernel in TensorPooling.metal for batch attention pooling
- [x] `fused_attention_pool_normalize` kernel in FusedOperations.metal for combined pool+norm
- [x] `tensorAttentionPoolNormalize()` Swift API for GPU-accelerated attention pooling
- [x] CPU fallback for cases without weights (degrades to mean pooling)
- [x] Comprehensive test suite (10 tests) covering correctness and GPU/CPU parity

### CoreML Backend Stateful Predictions & Flexible Shapes (v2.0)
- [x] Stateful predictions with reusable `MLPredictionOptions` and output backing buffers
- [x] `useStatefulPredictions` configuration to enable/disable feature
- [x] Pre-allocated output buffers via `maxSequenceLengthHint` and `hiddenDimensionHint`
- [x] `DimensionConstraint` enum: `.fixed`, `.flexible(min:max:)`, `.enumerated`
- [x] `ShapeConstraints` struct for validating input shapes against model constraints
- [x] Automatic detection and caching of shape constraints on model load
- [x] Public API: `inputShapeConstraints(for:)`, `allInputShapeConstraints`, `hasFlexibleInputs`
- [x] Comprehensive test suite (17 tests) covering constraints, validation, and backend config

### Request Prioritization (v2.0)
- [x] `RequestPriority` enum with `.low`, `.normal`, `.high`, `.urgent` levels
- [x] Priority-aware insertion in `AdaptiveBatcher` queue (higher priority = earlier processing)
- [x] `embed(_:priority:)` and `embedConcurrently(_:priority:)` API methods
- [x] Priority-specific latency multipliers (urgent: 0.25x, high: 0.5x, low: configurable)
- [x] `urgentTriggersFlush` option for immediate processing of urgent requests
- [x] Metrics tracking: `requestsByPriority`, `queueDepthByPriority` in `BatcherMetrics`
- [x] `highestPendingPriority` and `hasUrgentPending` convenience properties
- [x] Configurable: `enablePriorityScheduling`, `lowPriorityLatencyMultiplier`
- [x] Comprehensive test suite (23 tests) covering ordering, metrics, and latency behavior

### Accelerate BLAS for CPU Fallback (v2.0)
- [x] `AccelerateBLAS` module with SIMD-optimized vector operations using vDSP
- [x] `dotProduct`, `sumOfSquares`, `magnitude` using vDSP primitives
- [x] `normalize`, `normalizeInPlace` using vDSP_vsdiv
- [x] `cosineSimilarity`, `cosineDistance` for embedding comparisons
- [x] `euclideanDistance`, `manhattanDistance`, `chebyshevDistance` for all metric types
- [x] `meanPool`, `maxPool`, `attentionPool` using vDSP_vadd/vDSP_vmax
- [x] Batch distance functions for efficient multi-candidate comparisons
- [x] Updated `PoolingHelpers` to delegate to AccelerateBLAS
- [x] Updated `Embedding` type to use AccelerateBLAS for similarity/distance
- [x] Updated `EmbeddingStore` CPU fallback to use AccelerateBLAS
- [x] Updated `AccelerationManager.cpuDistance` to use AccelerateBLAS
- [x] Comprehensive test suite (44 tests) covering all operations

---

## Notes

### Decision Log
- **2025-11-25**: Created initial improvements tracking document after comprehensive codebase review
- **2025-11-25**: Completed Metal 4 migration (Phases 1-4)
- **2025-11-25**: Integrated tensorPoolNormalize into AppleEmbeddingModel.embedBatch() - single GPU dispatch for entire micro-batches (~62% faster than per-item processing)
- **2025-11-25**: Added recoverySuggestion and failureReason to all error types for better developer experience
- **2025-11-25**: Added parallel batch tokenization via encodeBatch protocol method with TaskGroup-based concurrency
- **2025-11-25**: Added LatencyHistogram with bucket-based distribution and percentile calculations (p50-p999)
- **2025-11-26**: Added persistent performance database to GPU optimizer for cross-session kernel selection learning
- **2025-11-26**: Added triple buffering infrastructure (CommandBufferPool, MetalBufferPool) for overlapped CPU/GPU execution
- **2025-11-26**: Added GPU-accelerated attention-weighted pooling with fused kernels (~1.5x faster than CPU for large batches)
- **2025-11-26**: Added CoreML backend stateful predictions and flexible shape detection/validation
- **2025-11-26**: Added request prioritization to AdaptiveBatcher with priority levels, configurable latency multipliers, and comprehensive metrics
- **2025-11-26**: Added AccelerateBLAS module for SIMD-optimized CPU fallback paths using vDSP (3-10x faster for large vectors)

### Contributing
When adding new improvements:
1. Assign appropriate priority based on impact/urgency
2. Estimate complexity (Low/Medium/High)
3. Add to relevant component section
4. Update status as work progresses
5. Move to "Completed Improvements" when done

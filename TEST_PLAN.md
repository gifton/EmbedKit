# EmbedKit Test Plan

**Document Version:** 1.0
**Date:** 2025-10-21
**Framework:** EmbedKit - High-Performance Text Embedding Framework
**Target Platforms:** macOS 14+, iOS 17+, tvOS 17+, watchOS 10+, visionOS 1+

---

## Executive Summary

This test plan identifies all performance-critical hot paths in EmbedKit and defines comprehensive testing strategies for correctness, performance, concurrency, and numerical stability. The framework operates at the intersection of machine learning, GPU computing, and systems programming, requiring rigorous validation of both functional correctness and performance characteristics.

### Hot Path Categories

1. **Tokenization Pipeline** - Text preprocessing and vocabulary lookup
2. **Model Inference** - CoreML backend execution
3. **Pooling Operations** - Token embedding aggregation (CPU & GPU)
4. **Normalization** - L2 normalization for embeddings (CPU & GPU)
5. **Metal GPU Kernels** - SIMD-optimized compute shaders
6. **Embedding Cache** - LRU cache operations
7. **End-to-End Pipeline** - Complete text → embedding flow
8. **Similarity Computation** - Cosine similarity and distance metrics

---

## 1. Hot Path Analysis

### 1.1 Tokenization Pipeline

**Components:**
- `/Users/goftin/dev/gsuite/VSK/EmbedKit/Sources/EmbedKit/Tokenization/BERTTokenizer.swift`
- `/Users/goftin/dev/gsuite/VSK/EmbedKit/Sources/EmbedKit/Tokenization/AdvancedTokenizer.swift`

**Critical Operations:**
- `tokenize(_:)` - Single text tokenization
- `tokenize(batch:)` - Batch tokenization
- `wordpieceTokenize(_:)` - WordPiece subword splitting
- `cleanText(_:)` - Text normalization
- Vocabulary lookup (hash table access)

**Performance Characteristics:**
- **Complexity:** O(n) where n = text length
- **Memory:** O(maxSequenceLength) per input
- **Bottlenecks:**
  - String manipulation (lowercasing, regex)
  - WordPiece greedy longest-match algorithm
  - Vocabulary hash lookups

**Numerical Considerations:**
- Token ID range validation
- Attention mask binary correctness
- Padding consistency

---

### 1.2 CoreML Inference Backend

**Component:**
- `/Users/goftin/dev/gsuite/VSK/EmbedKit/Sources/EmbedKit/Core/CoreMLBackend.swift`

**Critical Operations:**
- `generateEmbeddings(for:)` - Single input inference
- `generateEmbeddings(for:)` - Batch inference
- `createModelInput(from:)` - MLMultiArray conversion
- `extractModelOutput(from:)` - Output tensor extraction

**Performance Characteristics:**
- **Complexity:** O(1) per batch (model-dependent)
- **Memory:** ~1.5MB per 512-token sequence (768D)
- **Bottlenecks:**
  - MLMultiArray allocation and copying
  - Neural Engine scheduling overhead
  - Memory copying between CPU/GPU

**Numerical Considerations:**
- Float32 precision preservation
- Array bounds checking
- Batch dimension alignment

---

### 1.3 Pooling Operations

**Components:**
- `/Users/goftin/dev/gsuite/VSK/EmbedKit/Sources/EmbedKit/Core/EmbeddingPipeline.swift` (CPU)
- `/Users/goftin/dev/gsuite/VSK/EmbedKit/Sources/EmbedKit/Acceleration/MetalPoolingProcessor.swift` (GPU)

**Critical Operations:**
- `poolMean(tokenEmbeddings:attentionMask:)` - Mean pooling
- `poolMax(tokenEmbeddings:attentionMask:)` - Max pooling
- CLS token extraction
- Attention-weighted pooling

**Performance Characteristics:**
- **CPU Complexity:** O(sequence_length × dimensions)
- **GPU Complexity:** O(dimensions) with parallel reduction
- **Memory:** O(dimensions) output
- **Bottlenecks:**
  - CPU: Loop iteration overhead, cache misses
  - GPU: Thread synchronization, memory bandwidth

**Numerical Considerations:**
- Division by zero (empty attention mask)
- Floating-point accumulation order
- Max pooling with -∞ initialization

---

### 1.4 L2 Normalization

**Components:**
- `/Users/goftin/dev/gsuite/VSK/EmbedKit/Sources/EmbedKit/Core/EmbeddingPipeline.swift` (CPU)
- `/Users/goftin/dev/gsuite/VSK/EmbedKit/Sources/EmbedKit/Acceleration/MetalVectorProcessor.swift` (GPU)

**Critical Operations:**
- `normalize(_:)` - Single vector normalization
- `normalizeVectors(_:)` - Batch normalization
- L2 norm computation: ||v||₂ = √(Σvᵢ²)
- Vector scaling: v_norm = v / ||v||₂

**Performance Characteristics:**
- **CPU Complexity:** O(n) where n = dimensions
- **GPU Complexity:** O(n/threads) with SIMD reduction
- **Memory:** O(batch_size × dimensions)
- **Bottlenecks:**
  - Square root computation
  - SIMD reduction overhead
  - Division operations

**Numerical Considerations:**
- **Critical:** Division by zero for zero vectors
- Numerical stability: ||v||₂² may overflow for large values
- Precision loss in sqrt() operation
- Epsilon handling for near-zero norms

---

### 1.5 Metal GPU Kernels

**Component:**
- `/Users/goftin/dev/gsuite/VSK/EmbedKit/Sources/EmbedKit/Acceleration/MetalShaderLibrary.swift`

**Critical Kernels:**

#### l2_normalize
- SIMD group reduction for norm computation
- Fast inverse square root (`rsqrt`)
- Coalesced memory access patterns

#### mean_pool
- Loop unrolling (4-way)
- Vectorized accumulation
- Reciprocal multiplication instead of division

#### max_pool
- 4-way unrolled loop
- SIMD max reduction
- Proper -∞ initialization

#### cosine_similarity
- float4 vectorization for dot products
- Fast inverse square root (`rsqrt`)
- Clamping to [-1, 1] range

#### cosine_similarity_batch
- SIMD lane-level parallelism
- Group reduction operations
- Memory coalescing

**Performance Characteristics:**
- **Threadgroup size:** Pipeline-dependent (typically 32-256)
- **Memory bandwidth:** Critical for large batches
- **Occupancy:** Determined by register pressure
- **Bottlenecks:**
  - Thread divergence in masked operations
  - Memory access patterns
  - Synchronization barriers

**Numerical Considerations:**
- Fast math mode enabled (reduced precision)
- rsqrt() approximation accuracy
- Floating-point associativity in reductions
- Clamping for numerical stability

---

### 1.6 Embedding Cache (LRU)

**Component:**
- `/Users/goftin/dev/gsuite/VSK/EmbedKit/Sources/EmbedKit/Core/EmbeddingPipeline.swift` (EmbeddingCache actor)

**Critical Operations:**
- `get(_:)` - Cache lookup with SHA256 hashing
- `set(_:embedding:)` - Cache insertion with LRU eviction
- `hitRate()` - Statistics computation
- Access order maintenance (array manipulation)

**Performance Characteristics:**
- **Lookup:** O(1) expected (hash table)
- **Insertion:** O(n) worst case (LRU eviction search)
- **Memory:** O(maxEntries × embedding_size)
- **Bottlenecks:**
  - Array removeAll operations (linear scan)
  - SHA256 hashing overhead
  - Actor isolation synchronization

**Numerical Considerations:**
- Hash collision handling
- Cache hit rate accuracy
- Timestamp precision

---

### 1.7 End-to-End Pipeline

**Component:**
- `/Users/goftin/dev/gsuite/VSK/EmbedKit/Sources/EmbedKit/Core/EmbeddingPipeline.swift`

**Critical Flow:**
```
Text Input
    ↓
Tokenization (BERTTokenizer)
    ↓
Model Inference (CoreMLBackend)
    ↓
Pooling (Mean/Max/CLS/AttentionWeighted)
    ↓
Normalization (L2)
    ↓
DynamicEmbedding Output
```

**Performance Characteristics:**
- **Total Latency:** Sum of pipeline stages + overhead
- **Memory Peak:** Concurrent allocation of all intermediate buffers
- **Bottlenecks:**
  - Sequential execution (no pipeline parallelism)
  - Memory allocations for intermediate tensors
  - Cache coherency between stages

**Numerical Considerations:**
- Error propagation through pipeline
- Precision loss accumulation
- Validation at each stage

---

### 1.8 Similarity Computation

**Components:**
- `/Users/goftin/dev/gsuite/VSK/EmbedKit/Sources/EmbedKit/Core/Embedding.swift`
- `/Users/goftin/dev/gsuite/VSK/EmbedKit/Sources/EmbedKit/Acceleration/MetalSimilarityProcessor.swift`

**Critical Operations:**
- `cosineSimilarity(to:)` - Single pair similarity
- `cosineSimilarityBatch(_:)` - Batch computation
- `cosineSimilarityMatrix(queries:keys:)` - N×M matrix
- Euclidean distance computation

**Performance Characteristics:**
- **Complexity:** O(d) per pair, where d = dimensions
- **GPU Speedup:** ~10-100x for large batches
- **Memory:** O(n × m) for similarity matrix
- **Bottlenecks:**
  - Dot product computation
  - Norm calculations
  - Memory bandwidth for large matrices

**Numerical Considerations:**
- Cosine similarity range: [-1, 1]
- Division by zero for zero vectors
- Numerical stability: (a·b) / (||a|| × ||b||)
- Clamping to valid range

---

## 2. Test Coverage Matrix

| Hot Path | Correctness | Performance | Concurrency | Numerical | Edge Cases | Priority |
|----------|------------|-------------|-------------|-----------|------------|----------|
| Tokenization | ✅ | ⚠️ | ✅ | ✅ | ✅ | P0 |
| Model Inference | ⚠️ | ⚠️ | ❌ | ❌ | ❌ | P0 |
| Pooling (CPU) | ✅ | ❌ | ✅ | ⚠️ | ⚠️ | P1 |
| Pooling (GPU) | ✅ | ⚠️ | ✅ | ❌ | ⚠️ | P0 |
| Normalization (CPU) | ⚠️ | ❌ | ✅ | ❌ | ❌ | P1 |
| Normalization (GPU) | ✅ | ⚠️ | ✅ | ❌ | ⚠️ | P0 |
| Metal Kernels | ✅ | ⚠️ | ✅ | ❌ | ⚠️ | P0 |
| LRU Cache | ❌ | ❌ | ❌ | ✅ | ❌ | P1 |
| End-to-End | ❌ | ❌ | ❌ | ❌ | ❌ | P0 |
| Similarity Ops | ✅ | ⚠️ | ✅ | ❌ | ⚠️ | P0 |

**Legend:**
- ✅ Good coverage
- ⚠️ Partial coverage
- ❌ No/insufficient coverage

---

## 3. Detailed Test Specifications

### 3.1 Tokenization Tests

#### T-FUNC-001: Basic Tokenization Correctness
**Status:** ✅ Implemented
**Priority:** P0
**File:** `Tests/EmbedKitTests/TokenizationTests.swift`

**Test Cases:**
- ✅ Special token insertion ([CLS], [SEP])
- ✅ Attention mask generation
- ✅ Padding to max sequence length
- ✅ Token ID range validation

**Additional Required Tests:**
- [ ] **T-FUNC-001a:** Verify token IDs are within vocabulary bounds
- [ ] **T-FUNC-001b:** Validate attention mask sums match original length
- [ ] **T-FUNC-001c:** Check special token positions are invariant

---

#### T-PERF-001: Tokenization Performance Benchmarks
**Status:** ❌ Missing
**Priority:** P1

**Test Cases:**
```swift
func testTokenizationPerformance() async throws {
    let tokenizer = try await BERTTokenizer(maxSequenceLength: 512)

    // Test data: realistic text lengths
    let shortTexts = generateTexts(count: 1000, avgWords: 10)
    let mediumTexts = generateTexts(count: 1000, avgWords: 50)
    let longTexts = generateTexts(count: 1000, avgWords: 200)

    measure {
        _ = try await tokenizer.tokenize(batch: shortTexts)
    }

    // Performance targets:
    // - Short texts (10 words): < 1ms per text
    // - Medium texts (50 words): < 5ms per text
    // - Long texts (200 words): < 20ms per text
}
```

**Performance Baselines:**
- **Throughput:** ≥ 10,000 texts/second for short inputs
- **Latency:** ≤ 1ms p50, ≤ 10ms p99 for typical text
- **Memory:** ≤ 2KB overhead per input

---

#### T-EDGE-001: Tokenization Edge Cases
**Status:** ⚠️ Partial
**Priority:** P0

**Test Cases:**
- ✅ Empty string
- ✅ Very long text (> max sequence length)
- ✅ Unicode and special characters
- [ ] **T-EDGE-001a:** Text with only punctuation
- [ ] **T-EDGE-001b:** Text with only whitespace variations
- [ ] **T-EDGE-001c:** Maximum vocabulary size stress test
- [ ] **T-EDGE-001d:** WordPiece splitting for out-of-vocabulary words
- [ ] **T-EDGE-001e:** Extremely long single word (> maxInputCharsPerWord)

---

#### T-NUM-001: Tokenization Numerical Correctness
**Status:** ✅ Good
**Priority:** P0

**Test Cases:**
- ✅ Token ID determinism (same input → same output)
- [ ] **T-NUM-001a:** Attention mask binary values (only 0 or 1)
- [ ] **T-NUM-001b:** Original length ≤ maxSequenceLength invariant
- [ ] **T-NUM-001c:** Padding consistency (all padding tokens = PAD ID)

---

### 3.2 CoreML Inference Tests

#### T-FUNC-002: CoreML Backend Correctness
**Status:** ❌ Missing
**Priority:** P0

**Test Cases:**
```swift
func testCoreMLInferenceCorrectness() async throws {
    let backend = CoreMLBackend()
    let modelURL = getTestModelURL() // BERT-base or similar
    try await backend.loadModel(from: modelURL)

    // Test single input
    let tokenized = TokenizedInput(
        tokenIds: [101] + Array(repeating: 100, count: 510) + [102],
        attentionMask: Array(repeating: 1, count: 512),
        tokenTypeIds: nil,
        originalLength: 512
    )

    let output = try await backend.generateEmbeddings(for: tokenized)

    // Validate output structure
    XCTAssertEqual(output.tokenEmbeddings.count, 512)
    XCTAssertEqual(output.tokenEmbeddings[0].count, 768) // BERT-base

    // Validate all embeddings are finite
    for embedding in output.tokenEmbeddings {
        XCTAssertTrue(embedding.allSatisfy { $0.isFinite })
    }
}
```

**Required Tests:**
- [ ] **T-FUNC-002a:** Model loading and metadata extraction
- [ ] **T-FUNC-002b:** Input tensor shape validation
- [ ] **T-FUNC-002c:** Output tensor shape validation
- [ ] **T-FUNC-002d:** Batch inference consistency
- [ ] **T-FUNC-002e:** Neural Engine vs CPU computation equivalence

---

#### T-PERF-002: CoreML Inference Performance
**Status:** ❌ Missing
**Priority:** P0

**Test Cases:**
```swift
func testCoreMLInferenceLatency() async throws {
    let backend = CoreMLBackend()
    try await backend.loadModel(from: modelURL)

    let inputs = generateTestInputs(count: 100)

    // Warmup
    _ = try await backend.generateEmbeddings(for: inputs[0])

    // Measure single inference
    let singleStart = CFAbsoluteTimeGetCurrent()
    _ = try await backend.generateEmbeddings(for: inputs[0])
    let singleLatency = CFAbsoluteTimeGetCurrent() - singleStart

    // Measure batch inference
    let batchStart = CFAbsoluteTimeGetCurrent()
    _ = try await backend.generateEmbeddings(for: inputs)
    let batchLatency = CFAbsoluteTimeGetCurrent() - batchStart

    // Performance targets:
    // - Single inference: < 50ms on Neural Engine
    // - Batch throughput: > 20 inferences/second

    print("Single inference: \(singleLatency * 1000)ms")
    print("Batch throughput: \(100.0 / batchLatency) inferences/sec")
}
```

**Performance Baselines:**
- **Single Inference:** ≤ 50ms on ANE, ≤ 200ms on CPU
- **Batch Throughput:** ≥ 20 inferences/sec
- **Memory:** ≤ 500MB peak allocation

---

#### T-CONCUR-002: CoreML Concurrent Access
**Status:** ❌ Missing
**Priority:** P0

**Test Cases:**
```swift
func testCoreMLConcurrentInference() async throws {
    let backend = CoreMLBackend()
    try await backend.loadModel(from: modelURL)

    let inputs = generateTestInputs(count: 100)

    // Test concurrent access from multiple tasks
    try await withThrowingTaskGroup(of: ModelOutput.self) { group in
        for input in inputs {
            group.addTask {
                try await backend.generateEmbeddings(for: input)
            }
        }

        var outputs: [ModelOutput] = []
        for try await output in group {
            outputs.append(output)
        }

        XCTAssertEqual(outputs.count, 100)
    }
}
```

**Required Tests:**
- [ ] **T-CONCUR-002a:** Actor isolation prevents data races
- [ ] **T-CONCUR-002b:** Concurrent reads during inference
- [ ] **T-CONCUR-002c:** Model load/unload race conditions

---

### 3.3 Pooling Operation Tests

#### T-FUNC-003: Pooling Strategy Correctness
**Status:** ✅ Partial
**Priority:** P0

**Test Cases:**
- ✅ Mean pooling with uniform mask
- ✅ CLS token extraction
- ⚠️ Max pooling validation
- [ ] **T-FUNC-003a:** Attention-weighted pooling
- [ ] **T-FUNC-003b:** Mean pooling with sparse mask
- [ ] **T-FUNC-003c:** Max pooling with negative values

**Correctness Criteria:**
```swift
func testMeanPoolingCorrectness() async throws {
    let embeddings = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]
    let mask = [1, 1, 0] // Ignore third token

    let pooled = poolMean(tokenEmbeddings: embeddings, attentionMask: mask)

    // Expected: [(1+4)/2, (2+5)/2, (3+6)/2] = [2.5, 3.5, 4.5]
    XCTAssertEqual(pooled[0], 2.5, accuracy: 1e-6)
    XCTAssertEqual(pooled[1], 3.5, accuracy: 1e-6)
    XCTAssertEqual(pooled[2], 4.5, accuracy: 1e-6)
}
```

---

#### T-PERF-003: Pooling Performance (CPU vs GPU)
**Status:** ❌ Missing
**Priority:** P0

**Test Cases:**
```swift
func testPoolingPerformanceComparison() async throws {
    let dimensions = 768
    let sequenceLength = 512
    let batchSize = 32

    let embeddings = generateRandomEmbeddings(
        count: sequenceLength,
        dimensions: dimensions
    )

    // CPU pooling
    let cpuStart = CFAbsoluteTimeGetCurrent()
    let cpuResult = poolMean(tokenEmbeddings: embeddings, attentionMask: nil)
    let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart

    // GPU pooling
    guard let accelerator = MetalAccelerator.shared else {
        throw XCTSkip("Metal not available")
    }

    let gpuStart = CFAbsoluteTimeGetCurrent()
    let gpuResult = try await accelerator.poolEmbeddings(
        embeddings,
        strategy: .mean,
        attentionMask: nil
    )
    let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart

    // Verify numerical equivalence
    for i in 0..<dimensions {
        XCTAssertEqual(cpuResult[i], gpuResult[i], accuracy: 1e-4)
    }

    print("CPU: \(cpuTime * 1000)ms, GPU: \(gpuTime * 1000)ms")
    print("Speedup: \(cpuTime / gpuTime)x")

    // Performance targets:
    // - GPU should be faster for large sequences (>128 tokens)
    // - GPU overhead should be < 1ms for small sequences
}
```

**Performance Baselines:**
- **CPU Mean Pooling:** ≤ 1ms for 512×768
- **GPU Mean Pooling:** ≤ 0.2ms for 512×768 (excludes transfer)
- **GPU Speedup:** ≥ 3x for sequences > 256 tokens

---

#### T-NUM-003: Pooling Numerical Stability
**Status:** ❌ Missing
**Priority:** P0

**Test Cases:**
```swift
func testPoolingNumericalStability() async throws {
    // Test with extreme values
    let largeValues = Array(repeating: [Float.greatestFiniteMagnitude * 0.5], count: 10)
    let smallValues = Array(repeating: [Float.leastNormalMagnitude * 10], count: 10)
    let mixedValues = largeValues + smallValues

    // Mean pooling should not overflow or underflow
    let pooledLarge = poolMean(tokenEmbeddings: largeValues, attentionMask: nil)
    XCTAssertTrue(pooledLarge.allSatisfy { $0.isFinite })

    let pooledSmall = poolMean(tokenEmbeddings: smallValues, attentionMask: nil)
    XCTAssertTrue(pooledSmall.allSatisfy { $0.isFinite })

    let pooledMixed = poolMean(tokenEmbeddings: mixedValues, attentionMask: nil)
    XCTAssertTrue(pooledMixed.allSatisfy { $0.isFinite })
}
```

**Required Tests:**
- [ ] **T-NUM-003a:** Division by zero (empty mask)
- [ ] **T-NUM-003b:** Overflow prevention
- [ ] **T-NUM-003c:** Accumulation order independence
- [ ] **T-NUM-003d:** Subnormal number handling

---

### 3.4 Normalization Tests

#### T-FUNC-004: L2 Normalization Correctness
**Status:** ⚠️ Partial
**Priority:** P0

**Test Cases:**
- ✅ Unit vector verification (magnitude = 1)
- [ ] **T-FUNC-004a:** Zero vector handling
- [ ] **T-FUNC-004b:** Direction preservation
- [ ] **T-FUNC-004c:** Batch normalization consistency

**Correctness Criteria:**
```swift
func testNormalizationCorrectness() async throws {
    let vectors = [
        [3.0, 4.0],       // magnitude = 5
        [1.0, 1.0, 1.0],  // magnitude = √3
        [0.0, 0.0]        // zero vector (should error)
    ]

    // Test valid vectors
    for vector in vectors[..<2] {
        let normalized = try normalize(vector)
        let magnitude = sqrt(normalized.reduce(0) { $0 + $1 * $1 })
        XCTAssertEqual(magnitude, 1.0, accuracy: 1e-6)
    }

    // Test zero vector
    XCTAssertThrowsError(try normalize(vectors[2])) { error in
        XCTAssertTrue(error is EmbeddingPipelineError)
    }
}
```

---

#### T-PERF-004: Normalization Performance Benchmarks
**Status:** ❌ Missing
**Priority:** P0

**Test Cases:**
```swift
func testNormalizationPerformance() async throws {
    let dimensions = [384, 768, 1536]
    let batchSizes = [1, 10, 100, 1000]

    for dim in dimensions {
        for batchSize in batchSizes {
            let vectors = generateRandomVectors(count: batchSize, dimensions: dim)

            // CPU benchmark
            let cpuStart = CFAbsoluteTimeGetCurrent()
            for vector in vectors {
                _ = try normalize(vector)
            }
            let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart

            // GPU benchmark
            guard let accelerator = MetalAccelerator.shared else { continue }
            let gpuStart = CFAbsoluteTimeGetCurrent()
            _ = try await accelerator.normalizeVectors(vectors)
            let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart

            print("\(dim)D × \(batchSize): CPU=\(cpuTime*1000)ms, GPU=\(gpuTime*1000)ms")
        }
    }
}
```

**Performance Baselines:**
- **CPU:** ≤ 10µs per 768D vector
- **GPU:** ≤ 1µs per vector for batches > 100
- **GPU Overhead:** ≤ 0.5ms for kernel launch

---

#### T-NUM-004: Normalization Numerical Stability
**Status:** ❌ Missing
**Priority:** P0

**Test Cases:**
```swift
func testNormalizationNumericalStability() async throws {
    // Test vectors that cause numerical issues

    // 1. Very large values (risk of overflow in norm calculation)
    let largeVector = Array(repeating: Float.greatestFiniteMagnitude * 0.1, count: 768)
    let normalizedLarge = try normalize(largeVector)
    XCTAssertTrue(normalizedLarge.allSatisfy { $0.isFinite })

    // 2. Very small values (risk of underflow)
    let smallVector = Array(repeating: Float.leastNormalMagnitude * 100, count: 768)
    let normalizedSmall = try normalize(smallVector)
    XCTAssertTrue(normalizedSmall.allSatisfy { $0.isFinite })

    // 3. Mixed magnitude values (catastrophic cancellation risk)
    var mixedVector = Array(repeating: Float(1e-20), count: 768)
    mixedVector[0] = 1.0
    let normalizedMixed = try normalize(mixedVector)
    XCTAssertTrue(normalizedMixed.allSatisfy { $0.isFinite })

    // 4. Near-zero norm (epsilon handling)
    let nearZeroVector = Array(repeating: Float(1e-40), count: 768)
    XCTAssertThrowsError(try normalize(nearZeroVector))
}
```

**Required Tests:**
- [ ] **T-NUM-004a:** Overflow prevention in ||v||₂²
- [ ] **T-NUM-004b:** Division by zero detection
- [ ] **T-NUM-004c:** Epsilon threshold validation
- [ ] **T-NUM-004d:** Precision loss in sqrt()

---

### 3.5 Metal Kernel Tests

#### T-FUNC-005: Metal Kernel Correctness
**Status:** ✅ Good
**Priority:** P0

**Test Cases:**
- ✅ L2 normalization kernel
- ✅ Mean pooling kernel
- ✅ Cosine similarity kernel
- ⚠️ Max pooling kernel
- [ ] **T-FUNC-005a:** Attention-weighted pooling kernel
- [ ] **T-FUNC-005b:** Batch cosine similarity kernel

**Additional Validation:**
```swift
func testMetalKernelVsCPUReference() async throws {
    guard let accelerator = MetalAccelerator.shared else {
        throw XCTSkip("Metal not available")
    }

    let testCases = [
        (dimensions: 384, batchSize: 1),
        (dimensions: 768, batchSize: 10),
        (dimensions: 1536, batchSize: 100)
    ]

    for testCase in testCases {
        let vectors = generateRandomVectors(
            count: testCase.batchSize,
            dimensions: testCase.dimensions
        )

        // CPU reference implementation
        let cpuResults = vectors.map { cpuNormalize($0) }

        // GPU kernel implementation
        let gpuResults = try await accelerator.normalizeVectors(vectors)

        // Compare with tight tolerance
        for (cpu, gpu) in zip(cpuResults, gpuResults) {
            for (c, g) in zip(cpu, gpu) {
                XCTAssertEqual(c, g, accuracy: 1e-5,
                    "CPU/GPU mismatch for \(testCase.dimensions)D")
            }
        }
    }
}
```

---

#### T-PERF-005: Metal Kernel Performance
**Status:** ⚠️ Partial
**Priority:** P0

**Test Cases:**
```swift
func testMetalKernelThroughput() async throws {
    guard let accelerator = MetalAccelerator.shared else {
        throw XCTSkip("Metal not available")
    }

    let dimensions = 768
    let batchSizes = [1, 10, 50, 100, 500, 1000]

    for batchSize in batchSizes {
        let vectors = generateRandomVectors(count: batchSize, dimensions: dimensions)

        // Warmup
        _ = try await accelerator.normalizeVectors(vectors)

        // Measure
        let iterations = 10
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = try await accelerator.normalizeVectors(vectors)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        let avgTime = elapsed / Double(iterations)
        let throughput = Double(batchSize) / avgTime

        print("Batch \(batchSize): \(avgTime*1000)ms, \(throughput) vectors/sec")
    }
}
```

**Performance Baselines:**
- **Normalization:** ≥ 100,000 vectors/sec for 768D
- **Pooling:** ≥ 50,000 poolings/sec for 512×768
- **Similarity:** ≥ 10,000 pairs/sec for 768D

---

#### T-NUM-005: Metal Kernel Numerical Accuracy
**Status:** ❌ Missing
**Priority:** P0

**Test Cases:**
```swift
func testMetalKernelNumericalAccuracy() async throws {
    guard let accelerator = MetalAccelerator.shared else {
        throw XCTSkip("Metal not available")
    }

    // Test with known values to verify accuracy
    let testVectors = [
        // Unit vectors
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],

        // Known magnitudes
        [3.0, 4.0, 0.0],  // magnitude = 5
        [1.0, 1.0, 1.0],  // magnitude = √3

        // Edge cases
        Array(repeating: Float(1e-20), count: 768),  // Very small
        Array(repeating: Float(1e20), count: 768)    // Very large
    ]

    for vector in testVectors {
        let gpuResult = try await accelerator.normalizeVectors([vector])
        let magnitude = sqrt(gpuResult[0].reduce(0) { $0 + $1 * $1 })

        // Allow some tolerance for fast math mode
        XCTAssertEqual(magnitude, 1.0, accuracy: 1e-4,
            "Metal kernel normalization failed for \(vector.prefix(5))...")
    }
}
```

**Required Tests:**
- [ ] **T-NUM-005a:** Fast math accuracy (rsqrt precision)
- [ ] **T-NUM-005b:** SIMD reduction associativity
- [ ] **T-NUM-005c:** Memory coalescing correctness
- [ ] **T-NUM-005d:** Thread divergence handling

---

### 3.6 LRU Cache Tests

#### T-FUNC-006: LRU Cache Correctness
**Status:** ❌ Missing
**Priority:** P1

**Test Cases:**
```swift
func testLRUCacheCorrectness() async throws {
    let cache = EmbeddingCache(maxEntries: 3)

    let embedding1 = try DynamicEmbedding(values: [1.0, 2.0, 3.0])
    let embedding2 = try DynamicEmbedding(values: [4.0, 5.0, 6.0])
    let embedding3 = try DynamicEmbedding(values: [7.0, 8.0, 9.0])
    let embedding4 = try DynamicEmbedding(values: [10.0, 11.0, 12.0])

    // Insert 3 entries
    await cache.set("text1", embedding: embedding1)
    await cache.set("text2", embedding: embedding2)
    await cache.set("text3", embedding: embedding3)

    // Verify all present
    XCTAssertNotNil(await cache.get("text1"))
    XCTAssertNotNil(await cache.get("text2"))
    XCTAssertNotNil(await cache.get("text3"))

    // Insert 4th entry (should evict text1 as LRU)
    await cache.set("text4", embedding: embedding4)

    // Verify LRU eviction
    XCTAssertNil(await cache.get("text1"))
    XCTAssertNotNil(await cache.get("text2"))
    XCTAssertNotNil(await cache.get("text3"))
    XCTAssertNotNil(await cache.get("text4"))

    // Access text2 to update its recency
    _ = await cache.get("text2")

    // Insert 5th entry (should evict text3, not text2)
    await cache.set("text5", embedding: embedding1)

    XCTAssertNil(await cache.get("text3"))
    XCTAssertNotNil(await cache.get("text2"))
}
```

**Required Tests:**
- [ ] **T-FUNC-006a:** LRU eviction order
- [ ] **T-FUNC-006b:** Access order updates
- [ ] **T-FUNC-006c:** Hit rate calculation
- [ ] **T-FUNC-006d:** Hash collision handling

---

#### T-PERF-006: Cache Performance
**Status:** ❌ Missing
**Priority:** P1

**Test Cases:**
```swift
func testCacheThroughput() async throws {
    let cache = EmbeddingCache(maxEntries: 1000)

    // Populate cache
    for i in 0..<1000 {
        let embedding = try DynamicEmbedding(values: Array(repeating: Float(i), count: 768))
        await cache.set("text_\(i)", embedding: embedding)
    }

    // Measure lookup performance
    let lookupStart = CFAbsoluteTimeGetCurrent()
    for i in 0..<10000 {
        _ = await cache.get("text_\(i % 1000)")
    }
    let lookupTime = CFAbsoluteTimeGetCurrent() - lookupStart

    print("Cache lookups: \(10000.0 / lookupTime) ops/sec")

    // Measure insertion performance
    let insertStart = CFAbsoluteTimeGetCurrent()
    for i in 1000..<2000 {
        let embedding = try DynamicEmbedding(values: Array(repeating: Float(i), count: 768))
        await cache.set("text_\(i)", embedding: embedding)
    }
    let insertTime = CFAbsoluteTimeGetCurrent() - insertStart

    print("Cache insertions: \(1000.0 / insertTime) ops/sec")
}
```

**Performance Baselines:**
- **Lookup:** ≥ 100,000 ops/sec
- **Insertion:** ≥ 10,000 ops/sec
- **Memory:** ≤ maxEntries × embedding_size × 1.2 (20% overhead)

---

#### T-CONCUR-006: Cache Concurrent Access
**Status:** ❌ Missing
**Priority:** P1

**Test Cases:**
```swift
func testCacheConcurrentAccess() async throws {
    let cache = EmbeddingCache(maxEntries: 100)

    // Concurrent reads and writes
    try await withThrowingTaskGroup(of: Void.self) { group in
        // Writer tasks
        for i in 0..<50 {
            group.addTask {
                let embedding = try DynamicEmbedding(values: Array(repeating: Float(i), count: 768))
                await cache.set("text_\(i)", embedding: embedding)
            }
        }

        // Reader tasks
        for i in 0..<50 {
            group.addTask {
                _ = await cache.get("text_\(i)")
            }
        }

        try await group.waitForAll()
    }

    // Verify no corruption occurred
    let hitRate = await cache.hitRate()
    XCTAssertGreaterThanOrEqual(hitRate, 0.0)
    XCTAssertLessThanOrEqual(hitRate, 1.0)
}
```

**Required Tests:**
- [ ] **T-CONCUR-006a:** Actor isolation verification
- [ ] **T-CONCUR-006b:** Read-write race conditions
- [ ] **T-CONCUR-006c:** Eviction during concurrent access

---

### 3.7 End-to-End Pipeline Tests

#### T-FUNC-007: Pipeline Integration
**Status:** ❌ Missing
**Priority:** P0

**Test Cases:**
```swift
func testEndToEndPipeline() async throws {
    let tokenizer = try await BERTTokenizer(maxSequenceLength: 512)
    let backend = CoreMLBackend()
    try await backend.loadModel(from: testModelURL)

    let pipeline = EmbeddingPipeline(
        tokenizer: tokenizer,
        backend: backend,
        configuration: EmbeddingPipelineConfiguration(
            poolingStrategy: .mean,
            normalize: true,
            useGPUAcceleration: true
        )
    )

    // Test single input
    let text = "The quick brown fox jumps over the lazy dog"
    let embedding = try await pipeline.embed(text)

    // Validate output
    XCTAssertEqual(embedding.dimensions, 768) // BERT-base
    XCTAssertTrue(embedding.isFinite)

    // Verify normalization
    let magnitude = embedding.magnitude
    XCTAssertEqual(magnitude, 1.0, accuracy: 1e-5)

    // Test batch
    let texts = [
        "First sentence",
        "Second sentence",
        "Third sentence"
    ]
    let embeddings = try await pipeline.embed(batch: texts)

    XCTAssertEqual(embeddings.count, 3)
    for emb in embeddings {
        XCTAssertTrue(emb.isFinite)
        XCTAssertEqual(emb.magnitude, 1.0, accuracy: 1e-5)
    }
}
```

**Required Tests:**
- [ ] **T-FUNC-007a:** All pooling strategies
- [ ] **T-FUNC-007b:** With and without normalization
- [ ] **T-FUNC-007c:** Cache hit/miss scenarios
- [ ] **T-FUNC-007d:** GPU vs CPU execution equivalence

---

#### T-PERF-007: End-to-End Latency
**Status:** ❌ Missing
**Priority:** P0

**Test Cases:**
```swift
func testEndToEndLatency() async throws {
    let pipeline = try await createTestPipeline()

    let texts = generateRealisticTexts(count: 100)

    // Measure per-stage latency
    var tokenizationTotal: TimeInterval = 0
    var inferenceTotal: TimeInterval = 0
    var poolingTotal: TimeInterval = 0
    var normalizationTotal: TimeInterval = 0

    for text in texts {
        let start = CFAbsoluteTimeGetCurrent()

        // Track each stage (requires pipeline instrumentation)
        _ = try await pipeline.embed(text)

        let total = CFAbsoluteTimeGetCurrent() - start

        // Get stage statistics from pipeline
        let stats = await pipeline.getStatistics()
        tokenizationTotal += stats.tokenizationTime
        inferenceTotal += stats.inferenceTime
        poolingTotal += stats.poolingTime
        normalizationTotal += stats.normalizationTime
    }

    print("Average latencies:")
    print("  Tokenization: \(tokenizationTotal / 100 * 1000)ms")
    print("  Inference: \(inferenceTotal / 100 * 1000)ms")
    print("  Pooling: \(poolingTotal / 100 * 1000)ms")
    print("  Normalization: \(normalizationTotal / 100 * 1000)ms")
}
```

**Performance Baselines:**
- **Total Latency (p50):** ≤ 100ms for typical text
- **Total Latency (p99):** ≤ 300ms
- **Breakdown:**
  - Tokenization: ≤ 5ms
  - Inference: ≤ 80ms
  - Pooling: ≤ 5ms
  - Normalization: ≤ 1ms

---

#### T-MEM-007: Memory Usage Profiling
**Status:** ❌ Missing
**Priority:** P1

**Test Cases:**
```swift
func testPipelineMemoryUsage() async throws {
    let pipeline = try await createTestPipeline()

    // Measure peak memory during embedding
    let memoryBefore = getMemoryUsage()

    let largeTexts = generateTexts(count: 1000, avgWords: 100)
    _ = try await pipeline.embed(batch: largeTexts)

    let memoryPeak = getMemoryUsage()
    let memoryAfter = getMemoryUsage()

    let peakDelta = memoryPeak - memoryBefore
    let leakage = memoryAfter - memoryBefore

    print("Peak memory: \(peakDelta / 1_000_000)MB")
    print("Memory leakage: \(leakage / 1_000_000)MB")

    // Targets:
    // - Peak memory: < 100MB for 1000 texts
    // - No memory leaks (< 1MB residual)
}
```

**Required Tests:**
- [ ] **T-MEM-007a:** Peak memory during batch processing
- [ ] **T-MEM-007b:** Memory leak detection
- [ ] **T-MEM-007c:** GPU memory allocation tracking

---

### 3.8 Similarity Computation Tests

#### T-FUNC-008: Similarity Metrics Correctness
**Status:** ✅ Good
**Priority:** P0

**Test Cases:**
- ✅ Cosine similarity: identical vectors → 1.0
- ✅ Cosine similarity: orthogonal vectors → 0.0
- ✅ Cosine similarity: opposite vectors → -1.0
- ✅ Batch cosine similarity
- [ ] **T-FUNC-008a:** Euclidean distance
- [ ] **T-FUNC-008b:** Cosine distance (1 - similarity)

**Additional Validation:**
```swift
func testSimilarityMetricProperties() async throws {
    let embedding1 = try Embedding<Dim768>.random(in: -1...1)
    let embedding2 = try Embedding<Dim768>.random(in: -1...1)

    // Symmetry: sim(a, b) = sim(b, a)
    let sim1 = embedding1.cosineSimilarity(to: embedding2)
    let sim2 = embedding2.cosineSimilarity(to: embedding1)
    XCTAssertEqual(sim1, sim2, accuracy: 1e-6)

    // Range: -1 ≤ sim ≤ 1
    XCTAssertGreaterThanOrEqual(sim1, -1.0)
    XCTAssertLessThanOrEqual(sim1, 1.0)

    // Self-similarity = 1
    let selfSim = embedding1.cosineSimilarity(to: embedding1)
    XCTAssertEqual(selfSim, 1.0, accuracy: 1e-6)
}
```

---

#### T-PERF-008: Similarity Computation Performance
**Status:** ⚠️ Partial
**Priority:** P0

**Test Cases:**
```swift
func testSimilarityThroughput() async throws {
    let dimensions = 768
    let queryCount = 100
    let candidateCount = 10000

    let queries = (0..<queryCount).map { _ in
        Embedding<Dim768>.random(in: -1...1)
    }
    let candidates = (0..<candidateCount).map { _ in
        Embedding<Dim768>.random(in: -1...1)
    }

    // CPU baseline
    let cpuStart = CFAbsoluteTimeGetCurrent()
    for query in queries {
        for candidate in candidates {
            _ = query.cosineSimilarity(to: candidate)
        }
    }
    let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart

    // GPU accelerated
    guard let accelerator = MetalAccelerator.shared else {
        throw XCTSkip("Metal not available")
    }

    let queryVectors = queries.map { $0.toArray() }
    let candidateVectors = candidates.map { $0.toArray() }

    let gpuStart = CFAbsoluteTimeGetCurrent()
    for query in queryVectors {
        _ = try await accelerator.cosineSimilarity(query: query, keys: candidateVectors)
    }
    let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart

    print("Similarity computation:")
    print("  CPU: \(cpuTime * 1000)ms (\(Double(queryCount * candidateCount) / cpuTime) pairs/sec)")
    print("  GPU: \(gpuTime * 1000)ms (\(Double(queryCount * candidateCount) / gpuTime) pairs/sec)")
    print("  Speedup: \(cpuTime / gpuTime)x")
}
```

**Performance Baselines:**
- **CPU:** ≥ 1,000,000 pairs/sec for 768D
- **GPU:** ≥ 10,000,000 pairs/sec for 768D
- **GPU Speedup:** ≥ 5x for large batches

---

#### T-NUM-008: Similarity Numerical Stability
**Status:** ❌ Missing
**Priority:** P0

**Test Cases:**
```swift
func testSimilarityNumericalStability() async throws {
    // Test with vectors that could cause numerical issues

    // 1. Very similar vectors (precision test)
    let base = Array(repeating: Float(1.0), count: 768)
    var similar = base
    similar[0] += Float(1e-7)  // Tiny perturbation

    let embedding1 = try Embedding<Dim768>(base)
    let embedding2 = try Embedding<Dim768>(similar)

    let similarity = embedding1.cosineSimilarity(to: embedding2)

    // Should be very close to 1.0 but not exactly due to perturbation
    XCTAssertGreaterThan(similarity, 0.99999)
    XCTAssertLessThan(similarity, 1.0)

    // 2. Near-zero vectors (division by zero risk)
    let nearZero1 = Array(repeating: Float(1e-30), count: 768)
    let nearZero2 = Array(repeating: Float(1e-30), count: 768)

    let embedding3 = try Embedding<Dim768>(nearZero1)
    let embedding4 = try Embedding<Dim768>(nearZero2)

    let nearZeroSim = embedding3.cosineSimilarity(to: embedding4)

    // Should handle gracefully (return 0 or error)
    XCTAssertTrue(nearZeroSim.isFinite || nearZeroSim.isNaN)

    // 3. Large magnitude vectors (overflow risk)
    let large = Array(repeating: Float.greatestFiniteMagnitude * 0.1, count: 768)
    let embedding5 = try Embedding<Dim768>(large)

    let largeSelfSim = embedding5.cosineSimilarity(to: embedding5)
    XCTAssertEqual(largeSelfSim, 1.0, accuracy: 1e-5)
}
```

**Required Tests:**
- [ ] **T-NUM-008a:** Division by zero (zero vectors)
- [ ] **T-NUM-008b:** Overflow in dot product
- [ ] **T-NUM-008c:** Underflow in norm calculation
- [ ] **T-NUM-008d:** Range clamping verification

---

## 4. Performance Baselines & Targets

### 4.1 Latency Targets (p50 / p99)

| Operation | Input Size | Target p50 | Target p99 | Measured p50 | Measured p99 | Status |
|-----------|------------|------------|------------|--------------|--------------|--------|
| Tokenization | 50 words | 1ms | 5ms | - | - | ❌ |
| Tokenization | 200 words | 5ms | 20ms | - | - | ❌ |
| Model Inference | 512 tokens | 50ms | 100ms | - | - | ❌ |
| Mean Pooling (CPU) | 512×768 | 1ms | 2ms | - | - | ❌ |
| Mean Pooling (GPU) | 512×768 | 0.2ms | 0.5ms | - | - | ❌ |
| Normalization (CPU) | 768D | 10µs | 50µs | - | - | ❌ |
| Normalization (GPU) | 768D | 1µs | 5µs | - | - | ❌ |
| End-to-End | Typical text | 100ms | 300ms | - | - | ❌ |

### 4.2 Throughput Targets

| Operation | Configuration | Target | Measured | Status |
|-----------|---------------|--------|----------|--------|
| Tokenization | Batch=1000 | 10K texts/sec | - | ❌ |
| Normalization (GPU) | 768D, Batch=1000 | 100K vectors/sec | - | ❌ |
| Pooling (GPU) | 512×768 | 50K ops/sec | - | ❌ |
| Similarity (GPU) | 768D pairs | 10M pairs/sec | - | ❌ |
| Cache Lookup | - | 100K ops/sec | - | ❌ |

### 4.3 Memory Baselines

| Component | Configuration | Target | Measured | Status |
|-----------|---------------|--------|----------|--------|
| Pipeline | Batch=100 | < 100MB | - | ❌ |
| Cache | 1000 entries × 768D | < 3MB | - | ❌ |
| GPU Buffers | 1000×768D | < 50MB | - | ❌ |
| Model | BERT-base | < 500MB | - | ❌ |

---

## 5. Test Data Requirements

### 5.1 Synthetic Test Data

**Purpose:** Controlled validation of numerical correctness

```swift
// Deterministic test vectors
struct TestVectors {
    // Unit vectors (for similarity testing)
    static let unitX = [1.0, 0.0, 0.0]
    static let unitY = [0.0, 1.0, 0.0]
    static let unitZ = [0.0, 0.0, 1.0]

    // Known magnitudes
    static let magnitude5 = [3.0, 4.0, 0.0]  // ||v|| = 5
    static let magnitudeSqrt3 = [1.0, 1.0, 1.0]  // ||v|| = √3

    // Edge cases
    static let nearZero = Array(repeating: Float(1e-30), count: 768)
    static let nearMax = Array(repeating: Float.greatestFiniteMagnitude * 0.1, count: 768)

    // Random but reproducible
    static func randomSeeded(dimensions: Int, seed: UInt64) -> [Float] {
        var rng = SeededRNG(seed: seed)
        return (0..<dimensions).map { _ in Float.random(in: -1...1, using: &rng) }
    }
}
```

### 5.2 Realistic Test Data

**Purpose:** Performance benchmarking and integration testing

```swift
struct RealisticTestData {
    // Text samples from different domains
    static let shortTexts = [
        "Hello world",
        "Quick test",
        "Simple sentence"
    ]

    static let mediumTexts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning models process natural language",
        "Swift provides powerful type safety for embeddings"
    ]

    static let longTexts = [
        """
        This is a longer text sample that might approach the maximum
        sequence length. It contains multiple sentences and covers a
        variety of vocabulary to test tokenization thoroughly.
        """,
        // ... more samples
    ]

    // Domain-specific texts
    static let technicalTexts = loadFromFile("technical_corpus.txt")
    static let conversationalTexts = loadFromFile("chat_corpus.txt")
    static let multilingual = loadFromFile("multilingual_corpus.txt")
}
```

### 5.3 Adversarial Test Data

**Purpose:** Edge case and robustness testing

```swift
struct AdversarialTestData {
    // Extreme lengths
    static let emptyString = ""
    static let singleChar = "a"
    static let maxLength = String(repeating: "word ", count: 1000)

    // Special characters
    static let unicodeEmoji = "😀🎉🚀💯"
    static let combinedDiacritics = "e\u{0301}\u{0308}" // é with umlaut
    static let rtlLanguages = "مرحبا العالم"

    // Pathological cases
    static let onlyPunctuation = "!!@@##$$%%"
    static let onlyWhitespace = "     \t\n   "
    static let repeatedChars = String(repeating: "a", count: 10000)

    // Numerical edge cases
    static let zeroVector = Array(repeating: Float(0.0), count: 768)
    static let nanVector = Array(repeating: Float.nan, count: 768)
    static let infVector = Array(repeating: Float.infinity, count: 768)
}
```

---

## 6. Testing Infrastructure

### 6.1 Performance Measurement Utilities

```swift
/// High-precision timing for performance tests
struct PerformanceTimer {
    private var start: CFAbsoluteTime = 0

    mutating func start() {
        self.start = CFAbsoluteTimeGetCurrent()
    }

    func elapsed() -> TimeInterval {
        return CFAbsoluteTimeGetCurrent() - start
    }

    func elapsedMilliseconds() -> Double {
        return elapsed() * 1000
    }

    func elapsedMicroseconds() -> Double {
        return elapsed() * 1_000_000
    }
}

/// Statistical analysis for performance measurements
struct PerformanceStatistics {
    let samples: [TimeInterval]

    var mean: TimeInterval {
        samples.reduce(0, +) / Double(samples.count)
    }

    var median: TimeInterval {
        let sorted = samples.sorted()
        return sorted[samples.count / 2]
    }

    var p95: TimeInterval {
        let sorted = samples.sorted()
        return sorted[Int(Double(samples.count) * 0.95)]
    }

    var p99: TimeInterval {
        let sorted = samples.sorted()
        return sorted[Int(Double(samples.count) * 0.99)]
    }

    var standardDeviation: TimeInterval {
        let avg = mean
        let variance = samples.map { pow($0 - avg, 2) }.reduce(0, +) / Double(samples.count)
        return sqrt(variance)
    }
}

/// Benchmark runner with warmup
func benchmark<T>(
    name: String,
    warmupIterations: Int = 10,
    measurementIterations: Int = 100,
    operation: () async throws -> T
) async throws -> PerformanceStatistics {
    print("Benchmarking: \(name)")

    // Warmup
    for _ in 0..<warmupIterations {
        _ = try await operation()
    }

    // Measure
    var samples: [TimeInterval] = []
    for _ in 0..<measurementIterations {
        let timer = PerformanceTimer()
        timer.start()
        _ = try await operation()
        samples.append(timer.elapsed())
    }

    let stats = PerformanceStatistics(samples: samples)
    print("  Mean: \(stats.mean * 1000)ms")
    print("  Median: \(stats.median * 1000)ms")
    print("  p95: \(stats.p95 * 1000)ms")
    print("  p99: \(stats.p99 * 1000)ms")

    return stats
}
```

### 6.2 Memory Profiling Utilities

```swift
import os.proc

/// Memory usage measurement
struct MemoryProfiler {
    static func getCurrentMemoryUsage() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }

        return kerr == KERN_SUCCESS ? info.resident_size : 0
    }

    static func profileMemory<T>(operation: () async throws -> T) async rethrows -> (result: T, peakMemory: UInt64, leaked: Int64) {
        let memoryBefore = getCurrentMemoryUsage()

        let result = try await operation()

        let memoryPeak = getCurrentMemoryUsage()
        let memoryAfter = getCurrentMemoryUsage()

        let peakDelta = memoryPeak - memoryBefore
        let leaked = Int64(memoryAfter) - Int64(memoryBefore)

        return (result, peakDelta, leaked)
    }
}
```

### 6.3 Numerical Accuracy Utilities

```swift
/// Floating-point comparison with configurable tolerance
struct NumericalComparison {
    static func assertEqual(
        _ a: Float,
        _ b: Float,
        absoluteTolerance: Float = 1e-6,
        relativeTolerance: Float = 1e-6,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        let absError = abs(a - b)
        let relError = abs((a - b) / max(abs(a), abs(b)))

        let withinAbsolute = absError <= absoluteTolerance
        let withinRelative = relError <= relativeTolerance

        XCTAssertTrue(
            withinAbsolute || withinRelative,
            "Values \(a) and \(b) differ beyond tolerance (abs: \(absError), rel: \(relError))",
            file: file,
            line: line
        )
    }

    static func assertVectorEqual(
        _ a: [Float],
        _ b: [Float],
        absoluteTolerance: Float = 1e-6,
        relativeTolerance: Float = 1e-6,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        XCTAssertEqual(a.count, b.count, "Vector dimensions must match", file: file, line: line)

        for (i, (va, vb)) in zip(a, b).enumerated() {
            assertEqual(
                va, vb,
                absoluteTolerance: absoluteTolerance,
                relativeTolerance: relativeTolerance,
                file: file,
                line: line
            )
        }
    }
}
```

---

## 7. Continuous Integration

### 7.1 CI Pipeline Configuration

```yaml
# .github/workflows/tests.yml
name: EmbedKit Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  unit-tests:
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v3
      - name: Run Unit Tests
        run: swift test --parallel

  performance-tests:
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v3
      - name: Run Performance Benchmarks
        run: swift test --filter PerformanceTests
      - name: Upload Benchmark Results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmark-results.json

  metal-tests:
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v3
      - name: Run Metal Kernel Tests
        run: swift test --filter MetalAccelerationTests

  integration-tests:
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v3
      - name: Download Test Models
        run: ./scripts/download-test-models.sh
      - name: Run Integration Tests
        run: swift test --filter IntegrationTests
```

### 7.2 Regression Detection

```swift
/// Automated regression detection for performance tests
struct PerformanceRegression {
    let threshold: Double = 0.1  // 10% regression threshold

    func checkRegression(
        baseline: TimeInterval,
        current: TimeInterval
    ) -> Bool {
        let delta = (current - baseline) / baseline
        return delta > threshold
    }

    func loadBaseline(testName: String) -> TimeInterval? {
        // Load from JSON file or database
        return nil
    }

    func saveBaseline(testName: String, result: TimeInterval) {
        // Save to JSON file or database
    }
}
```

---

## 8. Known Gaps & Recommendations

### 8.1 Critical Gaps (P0)

1. **CoreML Backend Testing**
   - No tests for model loading/unloading
   - No concurrent inference tests
   - No numerical accuracy validation
   - **Recommendation:** Create test model fixtures, implement full backend test suite

2. **End-to-End Pipeline**
   - No integration tests
   - No latency breakdown analysis
   - No memory profiling
   - **Recommendation:** Implement comprehensive E2E test suite with instrumentation

3. **Normalization Numerical Stability**
   - No zero vector handling tests
   - No overflow/underflow tests
   - **Recommendation:** Add comprehensive numerical stability tests

4. **Metal Kernel Accuracy**
   - No fast math precision validation
   - No SIMD reduction correctness tests
   - **Recommendation:** Add CPU reference comparison tests with tight tolerances

### 8.2 Important Gaps (P1)

1. **LRU Cache**
   - No comprehensive cache tests
   - No performance benchmarks
   - No concurrent access tests
   - **Recommendation:** Full cache implementation testing

2. **CPU Pooling Performance**
   - No CPU pooling benchmarks
   - **Recommendation:** Add CPU baseline measurements for GPU comparison

3. **Memory Leak Detection**
   - No automated leak detection
   - **Recommendation:** Integrate Instruments or manual memory profiling

### 8.3 Performance Monitoring Gaps

1. **No Performance Regression Tracking**
   - **Recommendation:** Implement baseline tracking system in CI

2. **No Real-World Workload Tests**
   - **Recommendation:** Create test suite with production-like data

3. **No GPU Memory Monitoring**
   - **Recommendation:** Add Metal resource tracking

---

## 9. Implementation Priority

### Phase 1: Critical Path Coverage (Week 1-2)

**Priority:** P0 hot paths

1. [ ] CoreML backend correctness tests (T-FUNC-002)
2. [ ] CoreML performance benchmarks (T-PERF-002)
3. [ ] End-to-end pipeline tests (T-FUNC-007, T-PERF-007)
4. [ ] Normalization numerical stability (T-NUM-004)
5. [ ] Metal kernel accuracy validation (T-NUM-005)

**Deliverable:** 80% coverage of P0 hot paths

### Phase 2: Performance Baselines (Week 3)

**Priority:** Establish measurement infrastructure

1. [ ] Performance measurement utilities
2. [ ] Benchmark all hot paths
3. [ ] Document baseline performance
4. [ ] Set up regression detection

**Deliverable:** Performance baseline database

### Phase 3: Numerical Robustness (Week 4)

**Priority:** Edge cases and stability

1. [ ] Tokenization edge cases (T-EDGE-001)
2. [ ] Pooling numerical stability (T-NUM-003)
3. [ ] Similarity computation stability (T-NUM-008)
4. [ ] Adversarial test data suite

**Deliverable:** Comprehensive edge case coverage

### Phase 4: Optimization & Polish (Week 5-6)

**Priority:** P1 components and infrastructure

1. [ ] LRU cache tests (T-FUNC-006, T-PERF-006)
2. [ ] Memory profiling (T-MEM-007)
3. [ ] Concurrency tests (T-CONCUR-002, T-CONCUR-006)
4. [ ] CI/CD integration

**Deliverable:** Complete test suite with automation

---

## 10. Test Execution Guide

### 10.1 Running Tests Locally

```bash
# Run all tests
swift test

# Run specific test suite
swift test --filter EmbeddingTests

# Run with parallel execution
swift test --parallel

# Run performance tests (slower)
swift test --filter PerformanceTests

# Run Metal tests (requires GPU)
swift test --filter MetalAccelerationTests
```

### 10.2 Interpreting Results

**Test Output Format:**
```
Test Suite 'MetalAccelerationTests' passed
    - testVectorNormalization: PASSED (0.15s)
    - testMeanPooling: PASSED (0.08s)
    ⚡ Normalized 100x768D vectors in 12.3ms
```

**Performance Regression Alert:**
```
⚠️  PERFORMANCE REGRESSION DETECTED
Test: testNormalizationPerformance
Baseline: 10.2ms (±0.5ms)
Current:  12.8ms (±0.3ms)
Delta:    +25.5% (threshold: 10%)
```

### 10.3 Debugging Test Failures

**Common Failure Patterns:**

1. **Numerical Precision Issues**
   ```
   XCTAssertEqual failed: ("1.0000001") is not equal to ("1.0")
   ```
   - **Cause:** Floating-point accumulation order
   - **Fix:** Use accuracy parameter or relative tolerance

2. **Metal Kernel Failures**
   ```
   MetalError: Pipeline state creation failed
   ```
   - **Cause:** Shader compilation error
   - **Fix:** Check Metal shader syntax, enable compiler warnings

3. **Actor Isolation Violations**
   ```
   Call to actor-isolated property 'model' from nonisolated context
   ```
   - **Cause:** Swift 6 strict concurrency
   - **Fix:** Add async/await or use nonisolated(unsafe)

---

## Appendix A: Test Code Templates

### A.1 Performance Benchmark Template

```swift
func testPerformance_OPERATION_NAME() async throws {
    measure(metrics: [XCTClockMetric(), XCTMemoryMetric()]) {
        // Setup
        let input = generateTestInput()

        // Warmup
        for _ in 0..<10 {
            _ = performOperation(input)
        }

        // Measure
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<100 {
            _ = performOperation(input)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        // Assert performance target
        XCTAssertLessThan(elapsed / 100, 0.001) // < 1ms per operation
    }
}
```

### A.2 Numerical Correctness Template

```swift
func testNumerical_OPERATION_NAME() async throws {
    // Test with known input/output pairs
    let testCases: [(input: [Float], expected: [Float])] = [
        ([3.0, 4.0], [0.6, 0.8]),  // magnitude = 5
        ([1.0, 1.0, 1.0], [0.577, 0.577, 0.577])  // magnitude = √3
    ]

    for (input, expected) in testCases {
        let result = try normalize(input)

        for (r, e) in zip(result, expected) {
            XCTAssertEqual(r, e, accuracy: 1e-6)
        }
    }

    // Test edge cases
    XCTAssertThrowsError(try normalize([0.0, 0.0])) // zero vector
}
```

### A.3 Concurrency Test Template

```swift
func testConcurrency_OPERATION_NAME() async throws {
    let iterations = 1000

    try await withThrowingTaskGroup(of: Void.self) { group in
        for i in 0..<iterations {
            group.addTask {
                let result = try await self.performOperation(input: i)
                // Validate result
            }
        }

        try await group.waitForAll()
    }

    // Verify no data races or corruption occurred
}
```

---

## Appendix B: Glossary

**Hot Path:** Performance-critical code path that executes frequently and impacts overall system latency or throughput.

**SIMD:** Single Instruction, Multiple Data - parallel processing technique used in Metal kernels.

**ANE:** Apple Neural Engine - dedicated hardware for ML acceleration on Apple Silicon.

**LRU:** Least Recently Used - cache eviction policy.

**L2 Norm:** Euclidean norm, ||v||₂ = √(Σvᵢ²)

**Cosine Similarity:** Similarity metric based on angle between vectors: cos(θ) = (a·b) / (||a|| × ||b||)

**WordPiece:** Subword tokenization algorithm used by BERT models.

**Pooling:** Aggregation operation to reduce sequence of embeddings to single vector.

**Fast Math:** Compiler optimization that trades precision for performance (Metal).

**Actor Isolation:** Swift concurrency feature preventing data races through serialized access.

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-21 | Claude | Initial comprehensive test plan |

---

**End of Test Plan**

# Week 2: Real Implementations & Tokenization

## Overview
Week 2 replaces mocks with real implementations. Focus on CoreML integration, proper tokenization strategies, and performance optimization for EmbedBench measurements.

## Dependencies from Week 1
- ✅ Core protocols defined and working
- ✅ ModelManager loading models
- ✅ EmbedBench can benchmark basic operations
- ✅ Metrics collection infrastructure

---

## Day 1-2: Real Apple Model with CoreML

### File: `Sources/EmbedKit/Models/AppleEmbeddingModel.swift`

```swift
// IMPLEMENTATION CHECKLIST:
// [ ] Load actual CoreML model from system
// [ ] Handle model compilation and caching
// [ ] Implement proper input/output tensors
// [ ] Add error recovery for model loading failures
```

**Key Implementation Details:**

```swift
public actor AppleEmbeddingModel: EmbeddingModel {
    private let coreMLModel: MLModel
    private let tokenizer: AppleTokenizer

    init(variant: AppleModelVariant = .base) async throws {
        // 1. Locate system model
        let modelURL = try Self.locateSystemModel(variant: variant)

        // 2. Configure compute units
        let config = MLModelConfiguration()
        config.computeUnits = .all  // CPU, GPU, ANE

        // 3. Load and compile
        self.coreMLModel = try await MLModel.load(
            contentsOf: modelURL,
            configuration: config
        )

        // 4. Initialize tokenizer
        self.tokenizer = try AppleTokenizer(variant: variant)
    }

    func embed(_ text: String) async throws -> Embedding {
        // Real CoreML inference
        let tokens = try await tokenizer.encode(text)
        let inputArray = try MLMultiArray(tokens)

        let prediction = try await coreMLModel.prediction(
            input: ModelInput(tokens: inputArray)
        )

        return processOutput(prediction.embeddings)
    }
}
```

### File: `Sources/EmbedKit/Backends/CoreMLBackend.swift`

```swift
// IMPLEMENTATION CHECKLIST:
// [ ] MLModel lifecycle management
// [ ] Memory pressure handling
// [ ] Batch prediction optimization
// [ ] Device selection logic
```

**Performance Considerations for EmbedBench:**
- Pre-warm model for consistent latency measurements
- Track model compilation time separately
- Monitor memory usage during batch operations

---

## Day 2-3: Tokenization Strategies

### File: `Sources/EmbedKit/Tokenization/WordPieceTokenizer.swift`

```swift
// IMPLEMENTATION CHECKLIST:
// [ ] Load BERT-style vocabulary
// [ ] Implement WordPiece algorithm
// [ ] Handle unknown tokens properly
// [ ] Add subword tokenization
```

**Implementation Structure:**

```swift
public struct WordPieceTokenizer: Tokenizer {
    private let vocabulary: Vocabulary
    private let maxInputCharsPerWord = 200
    private let unkToken = "[UNK]"
    private let maxSubwordLength = 20

    public func encode(_ text: String, config: TokenizerConfig) async throws -> TokenizedText {
        // 1. Basic tokenization (whitespace + punctuation)
        let basicTokens = basicTokenize(text)

        // 2. WordPiece tokenization
        var outputTokens: [String] = []
        var outputIds: [Int] = []

        for token in basicTokens {
            let subwords = wordpieceTokenize(token)
            outputTokens.append(contentsOf: subwords)
            outputIds.append(contentsOf: subwords.map { vocabulary[$0] ?? unkId })
        }

        // 3. Apply truncation/padding
        return applyConfig(tokens: outputTokens, ids: outputIds, config: config)
    }

    private func wordpieceTokenize(_ word: String) -> [String] {
        // Greedy longest-match-first algorithm
        var tokens: [String] = []
        var start = word.startIndex

        while start < word.endIndex {
            var end = word.endIndex
            var foundSubword: String?

            while start < end {
                var substr = String(word[start..<end])
                if start > word.startIndex {
                    substr = "##" + substr
                }

                if vocabulary.contains(substr) {
                    foundSubword = substr
                    break
                }
                end = word.index(before: end)
            }

            if let subword = foundSubword {
                tokens.append(subword)
                start = end
            } else {
                tokens.append(unkToken)
                break
            }
        }

        return tokens
    }
}
```

### File: `Sources/EmbedKit/Tokenization/BPETokenizer.swift`

```swift
// IMPLEMENTATION CHECKLIST:
// [ ] Implement byte-pair encoding
// [ ] Load merge rules
// [ ] Handle GPT-style tokenization
// [ ] Unicode handling
```

### File: `Sources/EmbedKit/Tokenization/SentencePieceTokenizer.swift`

```swift
// IMPLEMENTATION CHECKLIST:
// [ ] SentencePiece model loading
// [ ] Unigram tokenization
// [ ] Handle special tokens (▁)
```

---

## Day 3-4: Batch Processing Optimization

### File: `Sources/EmbedKit/Processing/BatchProcessor.swift`

```swift
// IMPLEMENTATION CHECKLIST:
// [ ] Dynamic batching based on sequence length
// [ ] Efficient padding strategies
// [ ] Parallel tokenization
// [ ] Memory-efficient processing
```

**Key Optimizations for EmbedBench:**

```swift
public struct BatchProcessor {

    public func processBatch(
        _ texts: [String],
        model: any EmbeddingModel,
        options: BatchOptions
    ) async throws -> BatchResult {

        // 1. Sort by length to minimize padding
        let sorted = options.sortByLength ?
            texts.enumerated().sorted { $0.element.count < $1.element.count } :
            texts.enumerated().map { $0 }

        // 2. Create length-based buckets
        let buckets = createBuckets(sorted, maxBatchSize: options.maxBatchSize)

        // 3. Process each bucket in parallel
        let results = try await withThrowingTaskGroup(of: [Embedding].self) { group in
            for bucket in buckets {
                group.addTask {
                    try await self.processBucket(bucket, model: model)
                }
            }

            var allEmbeddings: [Embedding] = []
            for try await bucketResults in group {
                allEmbeddings.append(contentsOf: bucketResults)
            }
            return allEmbeddings
        }

        // 4. Restore original order
        return restoreOrder(results, originalIndices: sorted.map { $0.offset })
    }

    private func processBucket(
        _ texts: [(offset: Int, element: String)],
        model: any EmbeddingModel
    ) async throws -> [Embedding] {
        // Process similar-length texts together
        // Minimizes padding overhead
    }
}
```

### File: `Sources/EmbedKit/Processing/TokenCache.swift`

```swift
// IMPLEMENTATION CHECKLIST:
// [ ] LRU cache for tokenization results
// [ ] Thread-safe cache operations
// [ ] Memory limit management
// [ ] Cache hit rate metrics for EmbedBench
```

---

## Day 4-5: Performance Validation with EmbedBench

### Integration Points

```swift
// EmbedKit exposes for Week 2 benchmarking:

public extension EmbeddingModel {
    /// Detailed performance metrics
    func detailedMetrics() async -> DetailedMetrics {
        DetailedMetrics(
            tokenizationTime: tokenizationMetrics,
            inferenceTime: inferenceMetrics,
            postProcessingTime: postProcessingMetrics,
            cacheHitRate: cacheMetrics.hitRate,
            memoryPressure: memoryMetrics
        )
    }
}

public struct DetailedMetrics {
    public let tokenizationTime: Distribution
    public let inferenceTime: Distribution
    public let postProcessingTime: Distribution
    public let cacheHitRate: Double
    public let memoryPressure: MemoryMetrics
}

public struct Distribution {
    public let p50: TimeInterval
    public let p95: TimeInterval
    public let p99: TimeInterval
    public let mean: TimeInterval
    public let stdDev: TimeInterval
}
```

### Benchmarking Test Cases

```swift
// Week 2 specific benchmarks for EmbedBench

func benchmarkTokenizers() async throws {
    let tokenizers = [
        WordPieceTokenizer(),
        BPETokenizer(),
        SentencePieceTokenizer()
    ]

    for tokenizer in tokenizers {
        let result = try await measureTokenization(
            tokenizer: tokenizer,
            texts: testDocuments
        )
        print("\(tokenizer): \(result.tokensPerSecond) tokens/sec")
    }
}

func benchmarkBatchSizes() async throws {
    let batchSizes = [1, 8, 16, 32, 64, 128]

    for size in batchSizes {
        let throughput = try await measureThroughput(
            batchSize: size,
            documents: testDocuments
        )
        print("Batch \(size): \(throughput) docs/sec")
    }
}

func benchmarkDevices() async throws {
    let devices: [ComputeDevice] = [.cpu, .gpu, .ane]

    for device in devices {
        let latency = try await measureLatency(device: device)
        print("\(device): \(latency)ms")
    }
}
```

---

## Week 2 Success Criteria

### Must Have (P0)
- [ ] Real Apple model loading and working
- [ ] At least one tokenizer fully implemented
- [ ] Batch processing faster than sequential
- [ ] EmbedBench can measure real performance

### Should Have (P1)
- [ ] Multiple tokenization strategies
- [ ] Token caching working
- [ ] Memory pressure handling
- [ ] Device selection optimization

### Nice to Have (P2)
- [ ] All three tokenizers implemented
- [ ] Advanced batching strategies
- [ ] Profiling integration

---

## Performance Targets

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Single embedding latency | < 10ms | EmbedBench latency test |
| Batch throughput | > 100 docs/sec | EmbedBench throughput test |
| Memory per embedding | < 1KB | Instruments memory profiler |
| Token cache hit rate | > 80% | Internal metrics |
| Model load time | < 1s | First embedding latency |

---

## Testing Strategy

### Unit Tests
```swift
func testWordPieceTokenization()
func testBPETokenization()
func testTokenTruncation()
func testBatchProcessing()
func testCacheEffectiveness()
```

### Integration Tests
```swift
func testRealModelLoading()
func testEndToEndWithRealModel()
func testBatchOptimization()
func testMemoryUnderPressure()
```

### Performance Tests (via EmbedBench)
```swift
func benchmarkLatencyP50P95P99()
func benchmarkThroughputVsBatchSize()
func benchmarkMemoryUsage()
func benchmarkCacheImpact()
```

---

## Code Review Focus Areas

- [ ] CoreML integration correctness
- [ ] Tokenization accuracy vs reference implementations
- [ ] Memory management in batch processing
- [ ] Thread safety in caching
- [ ] Error handling for model loading failures

---

## Dependencies to Add

```swift
// Package.swift additions for Week 2
dependencies: [
    .package(url: "https://github.com/apple/swift-async-algorithms", from: "0.1.0"),
    .package(url: "https://github.com/swift-server/swift-prometheus", from: "1.0.0"),  // For metrics
]
```

---

## Known Challenges & Mitigations

1. **Challenge**: Apple's model format might be proprietary
   **Mitigation**: Have fallback to open-source model

2. **Challenge**: Tokenizer vocabulary files might be large
   **Mitigation**: Lazy loading, memory mapping

3. **Challenge**: Batch processing memory spikes
   **Mitigation**: Streaming processing, chunking

4. **Challenge**: Cache invalidation complexity
   **Mitigation**: Simple LRU, time-based expiry
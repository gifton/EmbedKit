# Week 1: Core Foundation Implementation

## Overview
Week 1 establishes the core protocol architecture and basic types that EmbedBench will depend on for benchmarking. Focus is on clean abstractions and a working Apple model prototype.

## Dependencies for EmbedBench Integration
- Public protocol definitions that EmbedBench can test against
- Measurable operations (embed, embedBatch)
- Metrics collection hooks
- Consistent error types

---

## Day 1-2: Core Protocols & Types

### File: `Sources/EmbedKit/Core/Protocols.swift`

```swift
// IMPLEMENTATION CHECKLIST:
// [ ] Define EmbeddingModel protocol with async/await
// [ ] Include metrics collection points for EmbedBench
// [ ] Ensure all operations are measurable
```

**Key Requirements:**
1. **EmbeddingModel Protocol**
   - Must be `Actor` for thread safety
   - All operations must be `async throws`
   - Include timing hooks that EmbedBench can measure
   - Expose `dimensions`, `device`, `metrics`

2. **Tokenizer Protocol**
   - Separate protocol (not embedded in model)
   - Support streaming tokenization for large texts
   - Expose vocabulary size for benchmarking

3. **ModelBackend Protocol**
   - Abstract compute backend (CoreML, Metal, CPU)
   - Memory usage reporting for EmbedBench
   - Device selection logic

### File: `Sources/EmbedKit/Core/Types.swift`

```swift
// IMPLEMENTATION CHECKLIST:
// [ ] Embedding struct with vector operations
// [ ] ModelID with proper equality/hashing
// [ ] Configuration types with all options exposed
// [ ] Error types with detailed information
```

**Key Types:**
- `Embedding`: Include similarity calculation, normalization
- `ModelID`: String representation for EmbedBench logging
- `EmbeddingConfiguration`: All options public for testing
- `EmbedKitError`: Detailed errors for benchmark failure analysis

### File: `Sources/EmbedKit/Core/Metrics.swift`

```swift
public struct ModelMetrics: Codable, Sendable {
    public let totalRequests: Int
    public let totalTokensProcessed: Int
    public let averageLatency: TimeInterval
    public let p50Latency: TimeInterval
    public let p95Latency: TimeInterval
    public let p99Latency: TimeInterval
    public let throughput: Double  // tokens/second
    public let cacheHitRate: Double
    public let memoryUsage: Int64
    public let lastUsed: Date

    // For EmbedBench
    public let latencyHistogram: [TimeInterval]
    public let tokenHistogram: [Int]
}
```

---

## Day 2-3: ModelManager Implementation

### File: `Sources/EmbedKit/Core/ModelManager.swift`

```swift
// IMPLEMENTATION CHECKLIST:
// [ ] Actor-based thread-safe implementation
// [ ] Model loading with proper resource management
// [ ] Metrics aggregation for EmbedBench
// [ ] Clean error propagation
```

**Core Methods to Implement:**

1. **Model Loading**
```swift
public func loadModel(
    _ spec: ModelSpecification,
    configuration: EmbeddingConfiguration = .default
) async throws -> any EmbeddingModel
```

2. **Direct Embedding (for EmbedBench testing)**
```swift
public func embed(
    _ text: String,
    using modelID: ModelID
) async throws -> (embedding: Embedding, metrics: EmbeddingMetrics)
```

3. **Batch Processing (critical for benchmarks)**
```swift
public func embedBatch(
    _ texts: [String],
    using modelID: ModelID,
    options: BatchOptions
) async throws -> BatchResult

public struct BatchResult {
    let embeddings: [Embedding]
    let totalTime: TimeInterval
    let perItemTimes: [TimeInterval]
    let tokenCounts: [Int]
}
```

### File: `Sources/EmbedKit/Core/ResourceManager.swift`

```swift
// IMPLEMENTATION CHECKLIST:
// [ ] Memory tracking for EmbedBench
// [ ] Model caching logic
// [ ] Cleanup on memory pressure
```

---

## Day 3-4: Apple Model Basic Implementation

### File: `Sources/EmbedKit/Models/AppleEmbeddingModel.swift`

```swift
// IMPLEMENTATION CHECKLIST:
// [ ] CoreML model loading
// [ ] Basic tokenization (can be simple for Week 1)
// [ ] Embedding generation
// [ ] Metrics collection
```

**Minimum Viable Implementation:**

```swift
public actor AppleEmbeddingModel: EmbeddingModel {
    // Week 1: Get it working, optimize later

    public func embed(_ text: String) async throws -> Embedding {
        let startTime = CFAbsoluteTimeGetCurrent()

        // 1. Simple tokenization (improve in Week 2)
        let tokens = simpleTokenize(text)

        // 2. Pad/truncate to fixed size
        let input = prepareInput(tokens, maxLength: 512)

        // 3. Run through CoreML (mock for now if needed)
        let output = try await runModel(input)

        // 4. Pool to fixed dimension
        let vector = poolOutput(output)

        // 5. Record metrics
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        recordMetrics(tokens: tokens.count, time: elapsed)

        return Embedding(
            vector: vector,
            metadata: EmbeddingMetadata(
                modelID: id,
                tokenCount: tokens.count,
                processingTime: elapsed
            )
        )
    }
}
```

### File: `Sources/EmbedKit/Models/MockModel.swift`

```swift
// IMPLEMENTATION CHECKLIST:
// [ ] Deterministic mock for testing
// [ ] Configurable latency simulation
// [ ] Memory usage simulation
```

**Purpose:** Allow EmbedBench to test the API before real models are ready

---

## Day 4-5: Basic Tokenizer & Integration

### File: `Sources/EmbedKit/Tokenization/SimpleTokenizer.swift`

```swift
// IMPLEMENTATION CHECKLIST:
// [ ] Word-level tokenization (sufficient for Week 1)
// [ ] Special token handling
// [ ] Truncation/padding
```

**Week 1 Goal:** Get something working, refine in Week 2

```swift
public struct SimpleTokenizer: Tokenizer {
    // Simple space-based tokenization
    // Add [CLS], [SEP] tokens
    // Basic vocabulary mapping
}
```

### File: `Tests/EmbedKitTests/CoreTests.swift`

```swift
// IMPLEMENTATION CHECKLIST:
// [ ] Protocol conformance tests
// [ ] Basic embedding generation
// [ ] Batch processing
// [ ] Error handling
```

---

## Integration Points for EmbedBench

### 1. Measurement Hooks

```swift
// EmbedKit exposes:
public protocol BenchmarkMeasurable {
    func measureEmbedding(_ text: String) async throws -> MeasurementResult
    func measureBatch(_ texts: [String], batchSize: Int) async throws -> BatchMeasurement
}

public struct MeasurementResult {
    let embedding: Embedding
    let tokenizationTime: TimeInterval
    let inferenceTime: TimeInterval
    let postProcessingTime: TimeInterval
    let totalTime: TimeInterval
    let memoryDelta: Int64
}
```

### 2. EmbedBench Can Test

```swift
// In EmbedBench:
import EmbedKit

class AppleModelBenchmark: Benchmark {
    let model: any EmbeddingModel

    func benchmarkSingleDocument() async throws {
        let result = await model.embed("test document")
        // Measure latency, memory, etc.
    }

    func benchmarkThroughput() async throws {
        let documents = loadTestDocuments()
        let results = await model.embedBatch(documents)
        // Calculate docs/sec, tokens/sec
    }
}
```

---

## Week 1 Success Criteria

### Must Have (P0)
- [ ] Core protocols compile and are usable
- [ ] ModelManager can load/unload models
- [ ] Basic Apple model generates embeddings (even if mock)
- [ ] Batch processing works
- [ ] EmbedBench can import and use EmbedKit

### Should Have (P1)
- [ ] Real CoreML model loading
- [ ] Basic tokenization working
- [ ] Metrics collection operational
- [ ] Memory tracking

### Nice to Have (P2)
- [ ] Multiple device support (CPU/GPU/ANE)
- [ ] Caching layer
- [ ] Advanced error recovery

---

## Daily Standup Template

### Day 1 (Monday)
- [ ] Set up project structure
- [ ] Implement core protocols
- [ ] Define all public types
- [ ] Verify EmbedBench can import

### Day 2 (Tuesday)
- [ ] ModelManager implementation
- [ ] Resource management basics
- [ ] Start Apple model

### Day 3 (Wednesday)
- [ ] Complete Apple model basic implementation
- [ ] Add mock model for testing
- [ ] Integration testing

### Day 4 (Thursday)
- [ ] Simple tokenizer
- [ ] Batch processing optimization
- [ ] Metrics collection

### Day 5 (Friday)
- [ ] Integration with EmbedBench
- [ ] Performance baseline
- [ ] Documentation
- [ ] Week 2 planning

---

## Testing Strategy for Week 1

### Unit Tests
```swift
func testProtocolConformance()
func testEmbeddingGeneration()
func testBatchProcessing()
func testMetricsCollection()
func testErrorHandling()
```

### Integration Tests
```swift
func testModelLoading()
func testEndToEndEmbedding()
func testMemoryManagement()
func testConcurrentRequests()
```

### EmbedBench Validation
```swift
// EmbedBench should be able to:
// 1. Import EmbedKit
// 2. Load a model
// 3. Generate embeddings
// 4. Measure performance
// 5. Compare different configurations
```

---

## Known Limitations (To Address in Week 2+)

1. **Tokenization**: Simple word-level only
2. **Models**: Only Apple model (possibly mocked)
3. **Optimization**: No caching, basic batching
4. **Backends**: CoreML only, no Metal acceleration
5. **Features**: No remote models, no custom tokenizers

---

## Git Branch Strategy

```bash
main
├── dev/week1-core-foundation
│   ├── feature/protocols
│   ├── feature/model-manager
│   ├── feature/apple-model
│   └── feature/tokenizer
└── dev/embedbench-integration
```

---

## Dependencies to Install

```swift
// Package.swift additions for Week 1
dependencies: [
    .package(url: "https://github.com/apple/swift-log.git", from: "1.5.0"),
    .package(url: "https://github.com/apple/swift-metrics.git", from: "2.4.0"),
]
```

---

## Code Review Checklist

- [ ] All protocols have documentation
- [ ] Public API is clean and minimal
- [ ] Async/await used correctly
- [ ] Actor isolation is proper
- [ ] Errors are descriptive
- [ ] Tests pass
- [ ] EmbedBench can use the API
- [ ] No unnecessary dependencies
- [ ] Memory management is correct
- [ ] Thread safety is ensured
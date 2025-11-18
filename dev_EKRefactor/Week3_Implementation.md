# Week 3: Optimization & Multi-Model Support

## Overview
Week 3 focuses on performance optimization, adding support for multiple model providers, and advanced features that EmbedBench can use to create comprehensive benchmarks.

## Dependencies from Week 2
- ✅ Real CoreML model working
- ✅ Tokenizers implemented
- ✅ Batch processing functional
- ✅ Basic performance metrics available

---

## Day 1-2: Metal Acceleration Layer

### File: `Sources/EmbedKit/Acceleration/MetalAccelerator.swift`

```swift
// IMPLEMENTATION CHECKLIST:
// [ ] Metal compute pipeline for post-processing
// [ ] GPU memory management
// [ ] Parallel normalization
// [ ] Similarity matrix computation
```

**Key Implementation:**

```swift
public actor MetalAccelerator {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let normalizeFunction: MTLComputePipelineState
    private let similarityFunction: MTLComputePipelineState

    public init() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw EmbedKitError.deviceNotAvailable(.gpu)
        }
        self.device = device
        self.commandQueue = device.makeCommandQueue()!

        // Load compute shaders
        let library = try await loadMetalLibrary()
        self.normalizeFunction = try await createPipeline(
            function: "batch_normalize",
            library: library
        )
        self.similarityFunction = try await createPipeline(
            function: "cosine_similarity_matrix",
            library: library
        )
    }

    /// GPU-accelerated batch normalization
    public func normalizeBatch(_ embeddings: [[Float]]) async throws -> [[Float]] {
        let commandBuffer = commandQueue.makeCommandBuffer()!

        // Create Metal buffers
        let inputBuffer = device.makeBuffer(
            bytes: embeddings.flatMap { $0 },
            length: embeddings.count * embeddings[0].count * MemoryLayout<Float>.size
        )!

        let outputBuffer = device.makeBuffer(
            length: inputBuffer.length
        )!

        // Dispatch compute
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(normalizeFunction)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)

        let threadgroups = MTLSize(
            width: embeddings.count,
            height: 1,
            depth: 1
        )
        encoder.dispatchThreadgroups(
            threadgroups,
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
        encoder.endEncoding()

        // Execute and wait
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return parseOutput(outputBuffer, count: embeddings.count)
    }

    /// GPU-accelerated similarity matrix computation
    public func computeSimilarityMatrix(
        _ embeddings: [[Float]]
    ) async throws -> [[Float]] {
        // Compute all pairwise cosine similarities on GPU
        // Critical for EmbedBench similarity benchmarks
    }
}
```

### File: `Sources/EmbedKit/Acceleration/Shaders.metal`

```metal
// IMPLEMENTATION CHECKLIST:
// [ ] Batch normalization kernel
// [ ] Cosine similarity kernel
// [ ] Mean pooling kernel
// [ ] Attention-weighted pooling kernel
```

---

## Day 2-3: Additional Model Support

### File: `Sources/EmbedKit/Models/LocalCoreMLModel.swift`

```swift
// IMPLEMENTATION CHECKLIST:
// [ ] Load arbitrary CoreML models
// [ ] Auto-detect input/output shapes
// [ ] Dynamic tokenizer selection
// [ ] Dimension inference
```

**Implementation for Custom Models:**

```swift
public actor LocalCoreMLModel: EmbeddingModel {
    private let modelURL: URL
    private let mlModel: MLModel
    private let tokenizer: any Tokenizer
    public let dimensions: Int

    public init(
        modelURL: URL,
        tokenizer: any Tokenizer? = nil
    ) async throws {
        self.modelURL = modelURL

        // Load and inspect model
        self.mlModel = try await MLModel(contentsOf: modelURL)

        // Auto-detect dimensions from model description
        self.dimensions = try inferDimensions(from: mlModel.modelDescription)

        // Use provided tokenizer or auto-detect
        self.tokenizer = tokenizer ?? try detectTokenizer(from: modelURL)
    }

    private func inferDimensions(from description: MLModelDescription) throws -> Int {
        // Inspect output shape
        guard let outputFeature = description.outputDescriptionsByName.values.first,
              case .multiArray(let shape) = outputFeature.type else {
            throw EmbedKitError.modelLoadFailed("Cannot infer embedding dimensions")
        }

        // Handle different shape formats
        // [batch, dimensions] or [dimensions] or [batch, seq, dimensions]
        return Int(shape.shape.last?.intValue ?? 768)
    }
}
```

### File: `Sources/EmbedKit/Models/ONNXModel.swift`

```swift
// IMPLEMENTATION CHECKLIST:
// [ ] ONNX runtime integration
// [ ] Model conversion if needed
// [ ] Cross-platform support
```

**ONNX Support for HuggingFace Models:**

```swift
public actor ONNXModel: EmbeddingModel {
    private let session: ORTSession
    private let tokenizer: any Tokenizer

    public init(modelPath: String, tokenizerPath: String) async throws {
        // Initialize ONNX Runtime
        let env = try ORTEnvironment(loggingLevel: .warning)

        // Load model
        self.session = try ORTSession(
            env: env,
            modelPath: modelPath,
            sessionOptions: nil
        )

        // Load tokenizer (usually HuggingFace format)
        self.tokenizer = try HuggingFaceTokenizer(configPath: tokenizerPath)
    }
}
```

### File: `Sources/EmbedKit/Models/RemoteModel.swift`

```swift
// IMPLEMENTATION CHECKLIST:
// [ ] API client for remote models
// [ ] Request batching
// [ ] Rate limiting
// [ ] Retry logic
// [ ] Response caching
```

---

## Day 3-4: Advanced Optimization Features

### File: `Sources/EmbedKit/Optimization/AdaptiveBatching.swift`

```swift
// IMPLEMENTATION CHECKLIST:
// [ ] Dynamic batch size based on memory
// [ ] Latency-aware batching
// [ ] Throughput optimization
// [ ] Queue management
```

**Smart Batching for EmbedBench:**

```swift
public actor AdaptiveBatcher {
    private var pendingRequests: [PendingRequest] = []
    private let model: any EmbeddingModel
    private var currentMemoryPressure: Float = 0.0

    public func submitRequest(_ text: String) async -> Embedding {
        // Add to queue
        let request = PendingRequest(text: text)
        pendingRequests.append(request)

        // Decide whether to process now or wait for more
        if shouldProcessBatch() {
            await processPendingBatch()
        }

        return await request.result
    }

    private func shouldProcessBatch() -> Bool {
        // Factors:
        // 1. Queue size
        // 2. Oldest request age
        // 3. Memory pressure
        // 4. Current throughput

        let queueSize = pendingRequests.count
        let oldestAge = pendingRequests.first?.age ?? 0
        let memoryOK = currentMemoryPressure < 0.8

        return (queueSize >= optimalBatchSize()) ||
               (oldestAge > maxLatency) ||
               !memoryOK
    }

    private func optimalBatchSize() -> Int {
        // Dynamic based on recent performance
        // EmbedBench can use this to find optimal batch sizes
        switch currentMemoryPressure {
        case 0..<0.3: return 128
        case 0.3..<0.6: return 64
        case 0.6..<0.8: return 32
        default: return 16
        }
    }
}
```

### File: `Sources/EmbedKit/Optimization/StreamingProcessor.swift`

```swift
// IMPLEMENTATION CHECKLIST:
// [ ] Streaming embeddings for large documents
// [ ] Chunking strategies
// [ ] Overlap handling
// [ ] Progressive results
```

**For Long Document Support:**

```swift
public struct StreamingProcessor {
    public func embedStream(
        _ text: String,
        chunkSize: Int = 512,
        overlap: Int = 50
    ) -> AsyncStream<Embedding> {
        AsyncStream { continuation in
            Task {
                let chunks = createChunks(text, size: chunkSize, overlap: overlap)

                for chunk in chunks {
                    let embedding = try await model.embed(chunk.text)
                    continuation.yield(embedding)
                }

                continuation.finish()
            }
        }
    }
}
```

---

## Day 4-5: Advanced Benchmarking Support

### File: `Sources/EmbedKit/Benchmarking/BenchmarkSupport.swift`

```swift
// IMPLEMENTATION CHECKLIST:
// [ ] Profiling hooks for EmbedBench
// [ ] Memory tracking
// [ ] Energy usage estimation
// [ ] Detailed timing breakdowns
```

**EmbedBench Integration Features:**

```swift
public protocol BenchmarkableModel: EmbeddingModel {
    /// Start profiling session
    func startProfiling() async

    /// Stop profiling and get results
    func stopProfiling() async -> ProfilingResults

    /// Reset all metrics
    func resetMetrics() async

    /// Get current resource usage
    var resourceUsage: ResourceUsage { get async }
}

public struct ProfilingResults {
    public let timingBreakdown: TimingBreakdown
    public let memoryProfile: MemoryProfile
    public let energyEstimate: EnergyEstimate

    public struct TimingBreakdown {
        public let tokenization: Distribution
        public let inference: Distribution
        public let postProcessing: Distribution
        public let total: Distribution
    }

    public struct MemoryProfile {
        public let peak: Int64
        public let average: Int64
        public let allocations: [AllocationEvent]
    }

    public struct EnergyEstimate {
        public let cpuEnergy: Double  // Joules
        public let gpuEnergy: Double
        public let aneEnergy: Double
    }
}
```

### File: `Sources/EmbedKit/Benchmarking/Scenarios.swift`

```swift
// Pre-built benchmark scenarios for EmbedBench

public struct BenchmarkScenarios {

    /// Latency-focused scenario
    public static func latencyScenario() -> BenchmarkConfiguration {
        BenchmarkConfiguration(
            batchSize: 1,
            iterations: 1000,
            warmupIterations: 100,
            texts: Self.shortTexts,
            measureMemory: false,
            measureEnergy: false
        )
    }

    /// Throughput-focused scenario
    public static func throughputScenario() -> BenchmarkConfiguration {
        BenchmarkConfiguration(
            batchSize: 128,
            iterations: 10,
            warmupIterations: 2,
            texts: Self.mixedLengthTexts,
            measureMemory: true,
            measureEnergy: false
        )
    }

    /// Memory stress test
    public static func memoryScenario() -> BenchmarkConfiguration {
        BenchmarkConfiguration(
            batchSize: 256,
            iterations: 5,
            warmupIterations: 1,
            texts: Self.longTexts,
            measureMemory: true,
            measureEnergy: true
        )
    }

    /// Model comparison scenario
    public static func comparisonScenario() -> BenchmarkConfiguration {
        BenchmarkConfiguration(
            batchSize: 32,
            iterations: 100,
            warmupIterations: 10,
            texts: Self.standardTexts,
            measureMemory: true,
            measureEnergy: true
        )
    }
}
```

---

## Week 3 Success Criteria

### Must Have (P0)
- [ ] Metal acceleration working
- [ ] At least one additional model type supported
- [ ] Adaptive batching implemented
- [ ] EmbedBench can run comprehensive benchmarks

### Should Have (P1)
- [ ] ONNX model support
- [ ] Streaming processing
- [ ] Energy profiling
- [ ] Multiple optimization strategies

### Nice to Have (P2)
- [ ] Remote model support
- [ ] Custom shader optimization
- [ ] Advanced caching strategies

---

## Performance Improvements Target

| Metric | Week 2 | Week 3 Target | Method |
|--------|--------|---------------|---------|
| Single latency | 10ms | 5ms | Metal acceleration |
| Batch throughput | 100 docs/sec | 500 docs/sec | Adaptive batching |
| Memory efficiency | 1KB/embed | 500B/embed | Optimized storage |
| GPU utilization | N/A | >80% | Metal optimization |

---

## Testing Strategy

### Performance Tests
```swift
func testMetalAcceleration()
func testAdaptiveBatchingEfficiency()
func testStreamingLargeDocuments()
func testMultiModelComparison()
```

### Stress Tests
```swift
func testHighConcurrency()
func testMemoryPressure()
func testLongRunningStability()
```

### EmbedBench Integration
```swift
func benchmarkAllModels()
func compareOptimizationStrategies()
func measureEnergyEfficiency()
```

---

## Code Review Checklist

- [ ] Metal shader correctness
- [ ] Memory management in streaming
- [ ] Thread safety in adaptive batching
- [ ] Model loading error handling
- [ ] Benchmark measurement accuracy

---

## Risk Mitigation

1. **Risk**: Metal might not provide speedup for small batches
   **Mitigation**: Automatic CPU/GPU selection based on batch size

2. **Risk**: ONNX runtime size/complexity
   **Mitigation**: Optional dependency, CoreML conversion path

3. **Risk**: Memory pressure with large batches
   **Mitigation**: Dynamic batch sizing, streaming fallback
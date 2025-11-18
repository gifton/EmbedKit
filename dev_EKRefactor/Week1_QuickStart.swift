// Week 1 Quick Start: Minimal Working Implementation
// This file shows the exact code structure needed by end of Week 1

import Foundation
import CoreML

// ============================================================================
// MARK: - What EmbedBench Needs From EmbedKit (Week 1)
// ============================================================================

/*
 EmbedBench Requirements:
 1. Can import EmbedKit
 2. Can load a model (even if mocked)
 3. Can generate embeddings
 4. Can measure performance metrics
 5. Can run batch operations
*/

// ============================================================================
// MARK: - Minimal Public API for Week 1
// ============================================================================

public protocol Week1PublicAPI {

    // Core functionality EmbedBench will test
    func loadModel() async throws -> any EmbeddingModel
    func embed(_ text: String) async throws -> Embedding
    func embedBatch(_ texts: [String]) async throws -> [Embedding]

    // Metrics EmbedBench needs to collect
    var metrics: ModelMetrics { get async }
}

// ============================================================================
// MARK: - Week 1 File Structure
// ============================================================================

/*
Sources/EmbedKit/
├── Core/
│   ├── Protocols.swift       // Day 1: Core protocols
│   ├── Types.swift           // Day 1: Basic types
│   └── Metrics.swift         // Day 2: Metrics collection
├── Management/
│   └── ModelManager.swift    // Day 2: Model lifecycle
├── Models/
│   ├── AppleEmbeddingModel.swift  // Day 3: Apple model
│   └── MockModel.swift            // Day 3: For testing
└── Tokenization/
    └── SimpleTokenizer.swift       // Day 4: Basic tokenization

Tests/EmbedKitTests/
├── CoreTests.swift           // Day 4: Protocol tests
├── IntegrationTests.swift    // Day 5: End-to-end
└── EmbedBenchTests.swift     // Day 5: Verify EmbedBench works
*/

// ============================================================================
// MARK: - Day 1 Goal: Get These Compiling
// ============================================================================

// File: Core/Protocols.swift
public protocol EmbeddingModel: Actor {
    var id: ModelID { get }
    var dimensions: Int { get }

    func embed(_ text: String) async throws -> Embedding
    func embedBatch(_ texts: [String], options: BatchOptions) async throws -> [Embedding]

    // For EmbedBench metrics
    var metrics: ModelMetrics { get async }
}

// File: Core/Types.swift
public struct Embedding: Sendable {
    public let vector: [Float]
    public let metadata: EmbeddingMetadata

    public init(vector: [Float], metadata: EmbeddingMetadata) {
        self.vector = vector
        self.metadata = metadata
    }
}

public struct ModelID: Hashable, Codable {
    public let provider: String
    public let name: String
    public let version: String
}

// ============================================================================
// MARK: - Day 2 Goal: ModelManager Working
// ============================================================================

// File: Management/ModelManager.swift
public actor ModelManager {
    private var models: [ModelID: any EmbeddingModel] = [:]

    public init() {}

    public func loadAppleModel() async throws -> any EmbeddingModel {
        // Week 1: Can return MockModel if CoreML not ready
        let model = MockEmbeddingModel()
        models[model.id] = model
        return model
    }
}

// ============================================================================
// MARK: - Day 3 Goal: Mock Model Generating Embeddings
// ============================================================================

// File: Models/MockModel.swift
actor MockEmbeddingModel: EmbeddingModel {
    let id = ModelID(provider: "mock", name: "test", version: "1.0")
    let dimensions = 384

    private var metricsData = MetricsData()

    func embed(_ text: String) async throws -> Embedding {
        let start = CFAbsoluteTimeGetCurrent()

        // Deterministic fake embedding
        let vector = (0..<dimensions).map { i in
            Float(sin(Double(text.count + i))) * 0.5 + 0.5
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        metricsData.record(tokenCount: text.count, time: elapsed)

        return Embedding(
            vector: vector,
            metadata: EmbeddingMetadata(
                modelID: id,
                tokenCount: text.count,
                processingTime: elapsed,
                normalized: true,
                poolingStrategy: .mean,
                truncated: false,
                custom: [:]
            )
        )
    }

    func embedBatch(_ texts: [String], options: BatchOptions) async throws -> [Embedding] {
        // Simple: just map over texts
        var embeddings: [Embedding] = []
        for text in texts {
            embeddings.append(try await embed(text))
        }
        return embeddings
    }

    var metrics: ModelMetrics {
        metricsData.toModelMetrics()
    }
}

// ============================================================================
// MARK: - Day 4 Goal: Simple Integration Test
// ============================================================================

// File: Tests/IntegrationTests.swift
func testWeek1Integration() async throws {
    // This MUST work by end of Week 1
    let manager = ModelManager()
    let model = try await manager.loadAppleModel()

    // Single embedding
    let embedding = try await model.embed("Hello world")
    assert(embedding.dimensions == 384)

    // Batch embedding
    let texts = ["Hello", "World", "Test"]
    let embeddings = try await model.embedBatch(texts, options: BatchOptions())
    assert(embeddings.count == 3)

    // Metrics
    let metrics = await model.metrics
    assert(metrics.totalRequests > 0)

    print("✅ Week 1 Integration Test Passed!")
}

// ============================================================================
// MARK: - Day 5 Goal: EmbedBench Can Use This
// ============================================================================

// In EmbedBench project:
/*
import EmbedKit

class EmbedKitBenchmark {
    let model: any EmbeddingModel

    func benchmarkLatency() async throws {
        let start = CFAbsoluteTimeGetCurrent()
        let _ = try await model.embed("Benchmark text")
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        print("Latency: \(elapsed * 1000)ms")
    }

    func benchmarkThroughput() async throws {
        let documents = Array(repeating: "Test doc", count: 100)
        let start = CFAbsoluteTimeGetCurrent()
        let _ = try await model.embedBatch(documents, options: BatchOptions())
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let throughput = Double(documents.count) / elapsed
        print("Throughput: \(throughput) docs/sec")
    }
}
*/

// ============================================================================
// MARK: - Week 1 Validation Checklist
// ============================================================================

/*
 END OF WEEK 1 CHECKLIST:

 [ ] EmbedKit compiles without errors
 [ ] Can create a ModelManager
 [ ] Can load at least one model (even if mocked)
 [ ] Can generate single embeddings
 [ ] Can generate batch embeddings
 [ ] Metrics are being collected
 [ ] Basic tests pass
 [ ] EmbedBench can import EmbedKit
 [ ] EmbedBench can run basic benchmarks
 [ ] No memory leaks in basic operations

 NOT REQUIRED FOR WEEK 1:
 - Real CoreML models (mock is fine)
 - Sophisticated tokenization
 - Multiple model support
 - Caching
 - Advanced optimizations
*/

// ============================================================================
// MARK: - Support Types Needed by Day 2
// ============================================================================

public struct BatchOptions: Sendable {
    public var maxBatchSize: Int = 32
    public init() {}
}

public struct EmbeddingMetadata: Codable, Sendable {
    public let modelID: ModelID
    public let tokenCount: Int
    public let processingTime: TimeInterval
    public let normalized: Bool
    public let poolingStrategy: PoolingStrategy
    public let truncated: Bool
    public let custom: [String: String]
}

public enum PoolingStrategy: String, Codable, Sendable {
    case mean, max, cls
}

public struct ModelMetrics: Sendable {
    public let totalRequests: Int
    public let totalTokensProcessed: Int
    public let averageLatency: TimeInterval

    public init(requests: Int, tokens: Int, avgLatency: TimeInterval) {
        self.totalRequests = requests
        self.totalTokensProcessed = tokens
        self.averageLatency = avgLatency
    }
}

struct MetricsData {
    private var requests = 0
    private var tokens = 0
    private var totalTime: TimeInterval = 0

    mutating func record(tokenCount: Int, time: TimeInterval) {
        requests += 1
        tokens += tokenCount
        totalTime += time
    }

    func toModelMetrics() -> ModelMetrics {
        ModelMetrics(
            requests: requests,
            tokens: tokens,
            avgLatency: requests > 0 ? totalTime / Double(requests) : 0
        )
    }
}
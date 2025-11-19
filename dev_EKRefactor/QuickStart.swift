// Quick Start: Minimal Working Implementation
// This file shows the core code structure and API usage

import Foundation
import CoreML

// ============================================================================
// MARK: - Public API
// ============================================================================

public protocol PublicAPI {

    // Core functionality
    func loadModel() async throws -> any EmbeddingModel
    func embed(_ text: String) async throws -> Embedding
    func embedBatch(_ texts: [String]) async throws -> [Embedding]

    // Metrics
    var metrics: ModelMetrics { get async }
}

// ============================================================================
// MARK: - File Structure
// ============================================================================

/*
Sources/EmbedKit/
├── Core/
│   ├── Protocols.swift       // Core protocols
│   ├── Types.swift           // Basic types
│   └── Metrics.swift         // Metrics collection
├── Management/
│   └── ModelManager.swift    // Model lifecycle
├── Models/
│   ├── DefaultEmbeddingModel.swift  // CoreML model
│   └── MockModel.swift            // For testing
└── Tokenization/
    └── BertTokenizer.swift       // Tokenization

Tests/EmbedKitTests/
├── CoreComponentsTests.swift // Protocol & Component tests
└── IntegrationTests.swift    // End-to-end
*/

// ============================================================================
// MARK: - Core Implementation
// ============================================================================

// File: Core/Protocols.swift
public protocol EmbeddingModel: Actor {
    var id: String { get }
    var dimension: Int { get }

    func embed(_ text: String) async throws -> Embedding
    func embedBatch(_ texts: [String]) async throws -> BatchResult

    // For EmbedBench metrics
    // var metrics: ModelMetrics { get async } // Implementation detail
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

// ============================================================================
// MARK: - Integration Example
// ============================================================================

// File: Tests/IntegrationTests.swift
func testIntegration() async throws {
    let manager = ModelManager()
    let model = try await manager.loadModel(id: "test") { MockEmbeddingModel(id: "test") }

    // Single embedding
    let embedding = try await model.embed("Hello world")
    assert(embedding.vector.count == 384)

    // Batch embedding
    let texts = ["Hello", "World", "Test"]
    let result = try await model.embedBatch(texts)
    assert(result.embeddings.count == 3)

    print("✅ Integration Test Passed!")
}

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
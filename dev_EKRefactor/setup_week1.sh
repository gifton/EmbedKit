#!/bin/bash

# Week 1 Setup Script
# Creates the folder structure and placeholder files for Week 1 implementation

echo "🚀 Setting up EmbedKit Week 1 structure..."

# Create directory structure
mkdir -p Sources/EmbedKit/Core
mkdir -p Sources/EmbedKit/Management
mkdir -p Sources/EmbedKit/Models
mkdir -p Sources/EmbedKit/Tokenization
mkdir -p Tests/EmbedKitTests
mkdir -p Tests/EmbedBenchIntegration

# Create placeholder files with basic structure

# Day 1: Core Protocols
cat > Sources/EmbedKit/Core/Protocols.swift << 'EOF'
// EmbedKit Core Protocols
// Week 1: Foundation

import Foundation

// MARK: - EmbeddingModel Protocol

public protocol EmbeddingModel: Actor {
    var id: ModelID { get }
    var dimensions: Int { get }
    var device: ComputeDevice { get }

    func embed(_ text: String) async throws -> Embedding
    func embedBatch(_ texts: [String], options: BatchOptions) async throws -> [Embedding]

    func warmup() async throws
    func release() async throws

    var metrics: ModelMetrics { get async }
}

// MARK: - Tokenizer Protocol

public protocol Tokenizer: Sendable {
    func encode(_ text: String, config: TokenizerConfig) async throws -> TokenizedText
    func decode(_ ids: [Int]) async throws -> String
    var vocabularySize: Int { get }
}

// MARK: - ModelBackend Protocol

public protocol ModelBackend: Actor {
    associatedtype Input
    associatedtype Output

    func process(_ input: Input) async throws -> Output
    var isLoaded: Bool { get }
    var memoryUsage: Int64 { get }
}
EOF

# Day 1: Core Types
cat > Sources/EmbedKit/Core/Types.swift << 'EOF'
// EmbedKit Core Types
// Week 1: Foundation

import Foundation

// MARK: - Model Identification

public struct ModelID: Hashable, Codable, CustomStringConvertible {
    public let provider: String
    public let name: String
    public let version: String
    public let variant: String?

    public var description: String {
        let v = variant.map { "-\($0)" } ?? ""
        return "\(provider)/\(name)\(v)@\(version)"
    }

    public init(provider: String, name: String, version: String, variant: String? = nil) {
        self.provider = provider
        self.name = name
        self.version = version
        self.variant = variant
    }
}

// MARK: - Embedding Type

public struct Embedding: Sendable {
    public let vector: [Float]
    public let metadata: EmbeddingMetadata

    public var dimensions: Int { vector.count }

    public init(vector: [Float], metadata: EmbeddingMetadata) {
        self.vector = vector
        self.metadata = metadata
    }

    public func similarity(to other: Embedding) -> Float {
        guard dimensions == other.dimensions else { return 0 }
        let dotProduct = zip(vector, other.vector).reduce(0) { $0 + $1.0 * $1.1 }
        return dotProduct
    }
}

// MARK: - Configuration

public struct EmbeddingConfiguration: Sendable {
    public var maxTokens: Int = 512
    public var poolingStrategy: PoolingStrategy = .mean
    public var normalizeOutput: Bool = true
    public var preferredDevice: ComputeDevice = .auto

    public init() {}
}

// MARK: - Supporting Types

public enum ComputeDevice: String, Sendable {
    case cpu, gpu, ane, auto
}

public enum PoolingStrategy: String, Sendable {
    case mean, max, cls, attention
}

public struct BatchOptions: Sendable {
    public var maxBatchSize: Int = 32
    public var sortByLength: Bool = false

    public init() {}
}
EOF

# Day 2: Model Manager
cat > Sources/EmbedKit/Management/ModelManager.swift << 'EOF'
// EmbedKit Model Manager
// Week 1: Basic Implementation

import Foundation

public actor ModelManager {
    private var loadedModels: [ModelID: any EmbeddingModel] = [:]

    public init() {}

    public func loadAppleModel(
        configuration: EmbeddingConfiguration = EmbeddingConfiguration()
    ) async throws -> any EmbeddingModel {
        // Week 1: Return mock for now
        let model = MockEmbeddingModel(configuration: configuration)
        loadedModels[model.id] = model
        return model
    }

    public func unloadModel(_ id: ModelID) async throws {
        loadedModels.removeValue(forKey: id)
    }
}
EOF

# Day 3: Mock Model
cat > Sources/EmbedKit/Models/MockModel.swift << 'EOF'
// Mock Model for Week 1 Testing

import Foundation

actor MockEmbeddingModel: EmbeddingModel {
    let id = ModelID(provider: "mock", name: "test", version: "1.0")
    let dimensions = 384
    let device = ComputeDevice.cpu
    private let configuration: EmbeddingConfiguration

    init(configuration: EmbeddingConfiguration) {
        self.configuration = configuration
    }

    func embed(_ text: String) async throws -> Embedding {
        // Generate deterministic embedding
        let vector = (0..<dimensions).map { Float($0) * 0.001 }

        return Embedding(
            vector: vector,
            metadata: EmbeddingMetadata(
                modelID: id,
                tokenCount: text.count,
                processingTime: 0.001
            )
        )
    }

    func embedBatch(_ texts: [String], options: BatchOptions) async throws -> [Embedding] {
        var embeddings: [Embedding] = []
        for text in texts {
            embeddings.append(try await embed(text))
        }
        return embeddings
    }

    func warmup() async throws {}
    func release() async throws {}

    var metrics: ModelMetrics {
        ModelMetrics(totalRequests: 1)
    }
}
EOF

# Day 4: Integration Test
cat > Tests/EmbedKitTests/IntegrationTests.swift << 'EOF'
// Week 1 Integration Tests

import XCTest
@testable import EmbedKit

final class Week1IntegrationTests: XCTestCase {

    func testBasicEmbedding() async throws {
        let manager = ModelManager()
        let model = try await manager.loadAppleModel()

        let embedding = try await model.embed("Hello world")

        XCTAssertEqual(embedding.dimensions, 384)
        XCTAssertFalse(embedding.vector.isEmpty)
    }

    func testBatchEmbedding() async throws {
        let manager = ModelManager()
        let model = try await manager.loadAppleModel()

        let texts = ["Hello", "World", "Test"]
        let embeddings = try await model.embedBatch(texts, options: BatchOptions())

        XCTAssertEqual(embeddings.count, 3)
    }
}
EOF

# Day 5: EmbedBench Integration Test
cat > Tests/EmbedBenchIntegration/ValidationTests.swift << 'EOF'
// Validate EmbedBench can use EmbedKit

import XCTest
import EmbedKit

final class EmbedBenchValidation: XCTestCase {

    func testEmbedBenchCanImport() async throws {
        // Verify EmbedBench can use our API
        let manager = ModelManager()
        let model = try await manager.loadAppleModel()

        // Simulate what EmbedBench will do
        let startTime = CFAbsoluteTimeGetCurrent()
        let _ = try await model.embed("Benchmark test")
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        print("Latency: \(elapsed * 1000)ms")
        XCTAssertLessThan(elapsed, 1.0) // Should be fast
    }
}
EOF

# Create Package.swift updates
cat > Package_Week1_Updates.swift << 'EOF'
// Add these to Package.swift for Week 1

let package = Package(
    name: "EmbedKit",
    platforms: [
        .macOS(.v13),
        .iOS(.v16)
    ],
    products: [
        .library(
            name: "EmbedKit",
            targets: ["EmbedKit"]
        )
    ],
    dependencies: [
        // Week 1 dependencies
        .package(url: "https://github.com/apple/swift-log.git", from: "1.5.0"),
    ],
    targets: [
        .target(
            name: "EmbedKit",
            dependencies: [
                .product(name: "Logging", package: "swift-log")
            ]
        ),
        .testTarget(
            name: "EmbedKitTests",
            dependencies: ["EmbedKit"]
        )
    ]
)
EOF

echo "✅ Week 1 structure created!"
echo ""
echo "📝 Next steps:"
echo "1. Review the generated files"
echo "2. Start with Core/Protocols.swift and Core/Types.swift"
echo "3. Run: swift build"
echo "4. Run: swift test"
echo ""
echo "📊 For EmbedBench integration:"
echo "1. In EmbedBench, add EmbedKit as a dependency"
echo "2. Import EmbedKit in your benchmarks"
echo "3. Use the ModelManager to load models"
echo ""
echo "🎯 Week 1 Goal: Get MockModel working and EmbedBench connected!"
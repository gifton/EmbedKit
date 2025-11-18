// ============================================================================
// MARK: - Comprehensive Testing Strategy
// ============================================================================

import XCTest
@testable import EmbedKit

// ============================================================================
// MARK: - Mock Implementations for Testing
// ============================================================================

/// Mock model for testing without real model files
public struct MockEmbeddingModel: EmbeddingModelProtocol {
    public typealias Configuration = MockConfiguration
    public typealias Tokenizer = MockTokenizer
    public typealias Backend = MockBackend

    public let modelIdentifier: ModelIdentifier
    public let configuration: Configuration
    public let tokenizer: Tokenizer
    public let backend: Backend
    public let dimensions: Int

    public var capabilities: ModelCapabilities {
        [.textEmbedding, .onDevice, .batchProcessing]
    }

    public func embed(_ text: String) async throws -> Embedding {
        // Return deterministic embeddings for testing
        let hash = text.hashValue
        let vector = (0..<dimensions).map { i in
            Float(sin(Double(hash + i))) * 0.5 + 0.5
        }
        return Embedding(vector: vector, metadata: ["mock": "true"])
    }

    public func embedBatch(_ texts: [String]) async throws -> [Embedding] {
        try await texts.asyncMap { try await embed($0) }
    }
}

// ============================================================================
// MARK: - Protocol Conformance Tests
// ============================================================================

final class ProtocolConformanceTests: XCTestCase {
    func testModelProtocolConformance() throws {
        // Test that all models conform to the protocol correctly
        let models: [any EmbeddingModelProtocol] = [
            MockEmbeddingModel(/* ... */),
            // AppleEmbeddingModel(/* ... */),
            // BERTModel(/* ... */),
        ]

        for model in models {
            XCTAssertNotNil(model.modelIdentifier)
            XCTAssertGreaterThan(model.dimensions, 0)
            XCTAssertFalse(model.capabilities.isEmpty)
        }
    }

    func testTokenizerProtocolConformance() throws {
        let tokenizers: [any TokenizerProtocol] = [
            WordPieceTokenizer(/* ... */),
            BPETokenizer(/* ... */),
            SentencePieceTokenizer(/* ... */),
        ]

        for tokenizer in tokenizers {
            XCTAssertNotNil(tokenizer.strategy)
            XCTAssertNotNil(tokenizer.vocabulary)
            XCTAssertGreaterThan(tokenizer.vocabulary.size, 0)
        }
    }
}

// ============================================================================
// MARK: - Model Registry Tests
// ============================================================================

final class ModelRegistryTests: XCTestCase {
    var registry: ModelRegistry!

    override func setUp() async throws {
        registry = ModelRegistry.shared
    }

    func testModelRegistration() async throws {
        // Register a mock model
        let factory = MockModelFactory()
        await registry.register(factory)

        // Verify it's available
        let models = await registry.availableModels()
        XCTAssertTrue(models.contains { $0.identifier == factory.modelIdentifier })
    }

    func testModelFiltering() async throws {
        // Test filtering by provider
        let appleModels = await registry.availableModels(provider: .apple)
        XCTAssertTrue(appleModels.allSatisfy { $0.identifier.provider == .apple })

        // Test filtering by capabilities
        let onDeviceModels = await registry.availableModels(capabilities: .onDevice)
        XCTAssertTrue(onDeviceModels.allSatisfy { $0.capabilities.contains(.onDevice) })
    }

    func testModelLoading() async throws {
        let identifier = ModelIdentifier(
            provider: .apple,
            name: "text-embedding",
            version: "1.0",
            variant: "base"
        )

        let model = try await registry.loadModel(identifier)
        XCTAssertNotNil(model)
        XCTAssertEqual(model.modelIdentifier, identifier)
    }

    func testModelCaching() async throws {
        let identifier = ModelIdentifier(
            provider: .apple,
            name: "text-embedding",
            version: "1.0",
            variant: "base"
        )

        // Load model twice
        let model1 = try await registry.loadModel(identifier)
        let model2 = try await registry.loadModel(identifier)

        // Should be the same instance (cached)
        XCTAssertTrue(model1 === model2)  // Reference equality
    }
}

// ============================================================================
// MARK: - Tokenizer Strategy Tests
// ============================================================================

final class TokenizerStrategyTests: XCTestCase {
    func testWordPieceTokenization() async throws {
        let tokenizer = WordPieceTokenizer(vocabulary: testVocabulary)
        let result = try await tokenizer.tokenize("hello world")

        // Should produce WordPiece tokens
        XCTAssertTrue(result.tokens.contains { $0.hasPrefix("##") })
    }

    func testBPETokenization() async throws {
        let tokenizer = BPETokenizer(merges: testMerges)
        let result = try await tokenizer.tokenize("hello world")

        // Should produce byte-pair encoded tokens
        XCTAssertTrue(result.tokens.count > 0)
        XCTAssertTrue(result.tokens.allSatisfy { !$0.hasPrefix("##") })
    }

    func testSentencePieceTokenization() async throws {
        let tokenizer = SentencePieceTokenizer(modelPath: "test.model")
        let result = try await tokenizer.tokenize("hello world")

        // Should produce SentencePiece tokens
        XCTAssertTrue(result.tokens.contains { $0.hasPrefix("▁") })
    }
}

// ============================================================================
// MARK: - Performance Tests
// ============================================================================

final class PerformanceTests: XCTestCase {
    func testEmbeddingPerformance() async throws {
        let model = try await loadTestModel()

        measure {
            let expectation = self.expectation(description: "Embedding")
            Task {
                _ = try await model.embed("This is a test sentence for performance measurement.")
                expectation.fulfill()
            }
            wait(for: [expectation], timeout: 5.0)
        }
    }

    func testBatchEmbeddingPerformance() async throws {
        let model = try await loadTestModel()
        let texts = (0..<100).map { "Test sentence number \($0)" }

        measure {
            let expectation = self.expectation(description: "Batch embedding")
            Task {
                _ = try await model.embedBatch(texts)
                expectation.fulfill()
            }
            wait(for: [expectation], timeout: 30.0)
        }
    }

    func testTokenizationPerformance() async throws {
        let tokenizer = WordPieceTokenizer(vocabulary: largeVocabulary)
        let longText = String(repeating: "This is a test. ", count: 100)

        measure {
            let expectation = self.expectation(description: "Tokenization")
            Task {
                _ = try await tokenizer.tokenize(longText)
                expectation.fulfill()
            }
            wait(for: [expectation], timeout: 5.0)
        }
    }
}

// ============================================================================
// MARK: - Integration Tests
// ============================================================================

final class IntegrationTests: XCTestCase {
    func testEndToEndEmbedding() async throws {
        // Test the complete pipeline
        let api = EmbedKitAPI()

        let embedding = try await api.embed(
            text: "Hello, world!",
            using: "apple/text-embedding-base-v1.0"
        )

        XCTAssertEqual(embedding.dimensions, 768)
        XCTAssertTrue(embedding.isNormalized)
    }

    func testMultiModelConsistency() async throws {
        let text = "The quick brown fox jumps over the lazy dog."

        // Test that different models produce reasonable embeddings
        let models = [
            "apple/text-embedding-base-v1.0",
            "openai/text-embedding-ada-002",
            "huggingface/bert-base-uncased"
        ]

        var embeddings: [Embedding] = []
        let api = EmbedKitAPI()

        for modelId in models {
            if let embedding = try? await api.embed(text: text, using: modelId) {
                embeddings.append(embedding)
            }
        }

        // All should be normalized
        XCTAssertTrue(embeddings.allSatisfy { $0.isNormalized })

        // Cosine similarity between same text should be reasonable
        for i in 0..<embeddings.count {
            for j in i+1..<embeddings.count {
                let similarity = embeddings[i].cosineSimilarity(to: embeddings[j])
                XCTAssertGreaterThan(similarity, 0.5, "Similar text should have reasonable similarity")
            }
        }
    }
}

// ============================================================================
// MARK: - Backward Compatibility Tests
// ============================================================================

final class BackwardCompatibilityTests: XCTestCase {
    func testLegacyAPISupport() async throws {
        // Ensure old API still works
        let tokenizer = try await BERTTokenizer(maxSequenceLength: 512)
        let result = try await tokenizer.tokenize("test")

        XCTAssertNotNil(result)
        XCTAssertEqual(result.tokenIds.first, 101)  // CLS token
    }

    func testMigrationPath() async throws {
        // Test that models can be migrated
        let legacyTokenizer = BERTTokenizer(/* ... */)
        let newTokenizer = WordPieceTokenizer.fromLegacy(legacyTokenizer)

        // Should produce compatible output
        let legacyResult = try await legacyTokenizer.tokenize("test")
        let newResult = try await newTokenizer.tokenize("test")

        XCTAssertEqual(legacyResult.tokenIds, newResult.ids)
    }
}

// ============================================================================
// MARK: - Edge Case Tests
// ============================================================================

final class EdgeCaseTests: XCTestCase {
    func testEmptyInput() async throws {
        let model = try await loadTestModel()
        let embedding = try await model.embed("")

        XCTAssertNotNil(embedding)
        XCTAssertTrue(embedding.isFinite)
    }

    func testVeryLongInput() async throws {
        let model = try await loadTestModel()
        let longText = String(repeating: "word ", count: 10000)

        let embedding = try await model.embed(longText)

        XCTAssertNotNil(embedding)
        XCTAssertTrue(embedding.isFinite)
    }

    func testSpecialCharacters() async throws {
        let model = try await loadTestModel()
        let specialText = "Hello 👋 世界 🌍 مرحبا 💫"

        let embedding = try await model.embed(specialText)

        XCTAssertNotNil(embedding)
        XCTAssertTrue(embedding.isFinite)
    }
}

// ============================================================================
// MARK: - Memory & Resource Tests
// ============================================================================

final class ResourceTests: XCTestCase {
    func testMemoryUsage() async throws {
        let model = try await loadTestModel()

        // Measure memory before
        let beforeMemory = getMemoryUsage()

        // Process many embeddings
        for i in 0..<1000 {
            _ = try await model.embed("Test \(i)")
        }

        // Measure memory after
        let afterMemory = getMemoryUsage()

        // Memory increase should be reasonable
        let increase = afterMemory - beforeMemory
        XCTAssertLessThan(increase, 100_000_000, "Memory usage should be < 100MB")
    }

    func testModelUnloading() async throws {
        let identifier = ModelIdentifier(
            provider: .apple,
            name: "text-embedding",
            version: "1.0",
            variant: "base"
        )

        let registry = ModelRegistry.shared

        // Load model
        _ = try await registry.loadModel(identifier)
        let memoryAfterLoad = getMemoryUsage()

        // Unload model
        try await registry.unloadModel(identifier)
        let memoryAfterUnload = getMemoryUsage()

        // Memory should be released
        XCTAssertLessThan(memoryAfterUnload, memoryAfterLoad)
    }
}

// ============================================================================
// MARK: - Test Helpers
// ============================================================================

extension Array {
    func asyncMap<T>(
        _ transform: @escaping (Element) async throws -> T
    ) async throws -> [T] {
        var results: [T] = []
        for element in self {
            results.append(try await transform(element))
        }
        return results
    }
}

func getMemoryUsage() -> Int64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)

    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_,
                     task_flavor_t(MACH_TASK_BASIC_INFO),
                     $0,
                     &count)
        }
    }

    return result == KERN_SUCCESS ? Int64(info.resident_size) : 0
}
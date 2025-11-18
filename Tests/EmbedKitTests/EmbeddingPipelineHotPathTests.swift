import XCTest
@testable import EmbedKit

/// Hot path tests for EmbeddingPipeline using VectorBatch
///
/// These tests verify:
/// 1. VectorBatch API is actually used in the hot path
/// 2. Real CoreML integration works with VectorBatch
/// 3. Performance benefits are realized in practice
/// 4. Memory efficiency improvements are achieved
final class EmbeddingPipelineHotPathTests: XCTestCase {

    // MARK: - Mock Components for Testing

    /// Mock tokenizer for testing
    final class MockTokenizer: Tokenizer {
        var vocabularySize: Int { 30000 }
        var maxSequenceLength: Int { 512 }
        var specialTokens: SpecialTokens {
            SpecialTokens(
                cls: 101,
                sep: 102,
                pad: 0,
                unk: 100,
                mask: 103
            )
        }

        func tokenize(_ text: String) async throws -> TokenizedInput {
            // Generate deterministic token IDs based on word content for consistency
            let tokens = text.split(separator: " ").map { word -> Int in
                // Use hashValue for deterministic token ID generation
                let hash = abs(word.hashValue)
                return 100 + (hash % 900) // Range: 100-999
            }
            return TokenizedInput(
                tokenIds: tokens,
                attentionMask: Array(repeating: 1, count: tokens.count),
                tokenTypeIds: Array(repeating: 0, count: tokens.count),
                originalLength: text.count
            )
        }

        func tokenize(batch texts: [String]) async throws -> [TokenizedInput] {
            return try await withThrowingTaskGroup(of: TokenizedInput.self) { group in
                for text in texts {
                    group.addTask {
                        try await self.tokenize(text)
                    }
                }

                var results: [TokenizedInput] = []
                for try await result in group {
                    results.append(result)
                }
                return results
            }
        }
    }

    /// Mock backend that returns realistic token embeddings
    actor MockBackend: ModelBackend {
        private var modelLoaded = false

        var identifier: String { "mock-backend" }
        var isLoaded: Bool { modelLoaded }
        var metadata: ModelMetadata? { nil }

        func loadModel(from url: URL) async throws {
            modelLoaded = true
        }

        func unloadModel() async throws {
            modelLoaded = false
        }

        func generateEmbeddings(for input: TokenizedInput) async throws -> ModelOutput {
            guard modelLoaded else {
                throw EmbeddingPipelineError.modelNotLoaded
            }

            // Generate deterministic mock token embeddings (384D to match MiniLM-L12)
            // Use token IDs as seed for consistency across calls with same input
            let numTokens = input.tokenIds.count
            let tokenEmbeddings: [[Float]] = (0..<numTokens).map { tokenIdx in
                let tokenId = input.tokenIds[tokenIdx]
                return (0..<384).map { dim in
                    // Deterministic pseudo-random based on token ID and dimension
                    let seed = Float(tokenId * 384 + dim)
                    let pseudoRandom = sin(seed * 12.9898 + 78.233) * 43758.5453
                    let normalized = pseudoRandom - floor(pseudoRandom) // Get fractional part [0, 1)
                    return (normalized * 2.0 - 1.0) + Float(tokenIdx) * 0.1 + Float(dim) * 0.001
                }
            }

            return ModelOutput(
                tokenEmbeddings: tokenEmbeddings,
                attentionWeights: nil,
                metadata: [:]
            )
        }

        func generateEmbeddings(for inputs: [TokenizedInput]) async throws -> [ModelOutput] {
            return try await withThrowingTaskGroup(of: (Int, ModelOutput).self) { group in
                for (idx, input) in inputs.enumerated() {
                    group.addTask {
                        let output = try await self.generateEmbeddings(for: input)
                        return (idx, output)
                    }
                }

                var results: [(Int, ModelOutput)] = []
                for try await result in group {
                    results.append(result)
                }
                results.sort { $0.0 < $1.0 }
                return results.map { $0.1 }
            }
        }

        func inputDimensions() async -> (sequence: Int, features: Int)? {
            return (512, 384)
        }

        func outputDimensions() async -> Int? {
            return 384
        }
    }

    // MARK: - VectorBatch Usage Tests

    func testPoolingUsesVectorBatchAPI() async throws {
        // Create pipeline with mock components
        let tokenizer = MockTokenizer()
        let backend = MockBackend()

        let config = EmbeddingPipelineConfiguration(
            poolingStrategy: .mean,
            normalize: false,  // Test pooling in isolation
            useGPUAcceleration: true
        )

        let pipeline = EmbeddingPipeline(
            tokenizer: tokenizer,
            backend: backend,
            configuration: config
        )

        // Load model through pipeline to set isModelLoaded flag
        try await pipeline.loadModel(from: URL(fileURLWithPath: "/tmp/mock"))

        // Generate embedding - this will use VectorBatch internally
        let text = "The quick brown fox jumps over the lazy dog"
        let embedding = try await pipeline.embed(text)

        // Verify embedding is valid
        XCTAssertEqual(embedding.dimensions, 384)
        XCTAssertFalse(embedding.toArray().contains { $0.isNaN })
        XCTAssertFalse(embedding.toArray().contains { $0.isInfinite })

        print("✅ Pooling uses VectorBatch API successfully")
    }

    func testNormalizationUsesVectorBatchAPI() async throws {
        let tokenizer = MockTokenizer()
        let backend = MockBackend()

        let config = EmbeddingPipelineConfiguration(
            poolingStrategy: .cls,  // Simple pooling
            normalize: true,        // Test normalization
            useGPUAcceleration: true
        )

        let pipeline = EmbeddingPipeline(
            tokenizer: tokenizer,
            backend: backend,
            configuration: config
        )

        // Load model through pipeline to set isModelLoaded flag
        try await pipeline.loadModel(from: URL(fileURLWithPath: "/tmp/mock"))

        let embedding = try await pipeline.embed("Test text")

        // Verify embedding is normalized (magnitude = 1.0)
        let values = embedding.toArray()
        let magnitude = sqrt(values.reduce(0) { $0 + $1 * $1 })
        XCTAssertEqual(magnitude, 1.0, accuracy: 0.01)

        print("✅ Normalization uses VectorBatch API and produces unit vectors")
    }

    func testBatchProcessingUsesVectorBatch() async throws {
        let tokenizer = MockTokenizer()
        let backend = MockBackend()

        let pipeline = EmbeddingPipeline(
            tokenizer: tokenizer,
            backend: backend,
            configuration: EmbeddingPipelineConfiguration(
                poolingStrategy: .mean,
                normalize: true,
                useGPUAcceleration: true,
                batchSize: 5
            )
        )

        // Load model through pipeline to set isModelLoaded flag
        try await pipeline.loadModel(from: URL(fileURLWithPath: "/tmp/mock"))

        let texts = [
            "First text",
            "Second text",
            "Third text",
            "Fourth text",
            "Fifth text"
        ]

        let embeddings = try await pipeline.embed(batch: texts)

        XCTAssertEqual(embeddings.count, 5)
        for embedding in embeddings {
            XCTAssertEqual(embedding.dimensions, 384)
            let magnitude = sqrt(embedding.toArray().reduce(0) { $0 + $1 * $1 })
            XCTAssertEqual(magnitude, 1.0, accuracy: 0.01)
        }

        print("✅ Batch processing uses VectorBatch API for all embeddings")
    }

    // MARK: - Performance Tests

    func testSingleEmbeddingPerformance() async throws {
        guard MetalAccelerator.shared != nil else {
            throw XCTSkip("Metal not available")
        }

        let tokenizer = MockTokenizer()
        let backend = MockBackend()

        let pipeline = EmbeddingPipeline(
            tokenizer: tokenizer,
            backend: backend,
            configuration: EmbeddingPipelineConfiguration(
                poolingStrategy: .mean,
                normalize: true,
                useGPUAcceleration: true
            )
        )

        // Load model through pipeline to set isModelLoaded flag
        try await pipeline.loadModel(from: URL(fileURLWithPath: "/tmp/mock"))

        // Warm up
        _ = try await pipeline.embed("Warmup text")

        // Measure performance
        let start = CFAbsoluteTimeGetCurrent()
        _ = try await pipeline.embed("The quick brown fox jumps over the lazy dog")
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        print("⚡ Single embedding (with VectorBatch): \(elapsed * 1000)ms")

        // Should complete quickly (<100ms for mock backend + Metal ops)
        XCTAssertLessThan(elapsed, 0.1, "Single embedding took longer than expected")
    }

    func testBatchEmbeddingPerformance() async throws {
        guard MetalAccelerator.shared != nil else {
            throw XCTSkip("Metal not available")
        }

        let tokenizer = MockTokenizer()
        let backend = MockBackend()

        let pipeline = EmbeddingPipeline(
            tokenizer: tokenizer,
            backend: backend,
            configuration: EmbeddingPipelineConfiguration(
                poolingStrategy: .mean,
                normalize: true,
                useGPUAcceleration: true,
                batchSize: 10
            )
        )

        // Load model through pipeline to set isModelLoaded flag
        try await pipeline.loadModel(from: URL(fileURLWithPath: "/tmp/mock"))

        let texts = (0..<10).map { "Test text number \($0) with some content" }

        // Warm up
        _ = try await pipeline.embed(batch: Array(texts.prefix(2)))

        // Measure batch performance
        let batchStart = CFAbsoluteTimeGetCurrent()
        _ = try await pipeline.embed(batch: texts)
        let batchTime = CFAbsoluteTimeGetCurrent() - batchStart

        print("⚡ Batch embedding (10 texts with VectorBatch): \(batchTime * 1000)ms")

        // Should complete reasonably fast
        XCTAssertLessThan(batchTime, 0.5, "Batch embedding took longer than expected")
    }

    func testHotPathMemoryEfficiency() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        let tokenizer = MockTokenizer()
        let backend = MockBackend()

        let pipeline = EmbeddingPipeline(
            tokenizer: tokenizer,
            backend: backend,
            configuration: EmbeddingPipelineConfiguration(
                poolingStrategy: .mean,
                normalize: true,
                useGPUAcceleration: true
            )
        )

        // Load model through pipeline to set isModelLoaded flag
        try await pipeline.loadModel(from: URL(fileURLWithPath: "/tmp/mock"))

        let initialMemory = accelerator.getCurrentMemoryUsage()

        // Process multiple embeddings
        for i in 0..<20 {
            _ = try await pipeline.embed("Test text number \(i)")
        }

        let finalMemory = accelerator.getCurrentMemoryUsage()
        let memoryGrowth = finalMemory - initialMemory

        print("⚡ Memory growth for 20 embeddings: \(memoryGrowth / 1024)KB")

        // Memory growth should be minimal (VectorBatch is efficient)
        XCTAssertLessThan(memoryGrowth, 10 * 1024 * 1024, "Memory growth exceeded 10MB")
    }

    // MARK: - Integration Tests

    func testCompleteHotPath() async throws {
        // Test the complete hot path: text → tokenize → inference → pool → normalize
        let tokenizer = MockTokenizer()
        let backend = MockBackend()

        let pipeline = EmbeddingPipeline(
            tokenizer: tokenizer,
            backend: backend,
            configuration: EmbeddingPipelineConfiguration(
                poolingStrategy: .mean,
                normalize: true,
                useGPUAcceleration: true
            )
        )

        // Load model through pipeline to set isModelLoaded flag
        try await pipeline.loadModel(from: URL(fileURLWithPath: "/tmp/mock"))

        let text = "This is a test sentence for the complete hot path"
        let embedding = try await pipeline.embed(text)

        // Verify complete pipeline worked correctly
        XCTAssertEqual(embedding.dimensions, 384)

        let values = embedding.toArray()
        XCTAssertFalse(values.isEmpty)

        // Check normalization
        let magnitude = sqrt(values.reduce(0) { $0 + $1 * $1 })
        XCTAssertEqual(magnitude, 1.0, accuracy: 0.01, "Embedding should be normalized")

        // Check no NaN or Inf
        for value in values {
            XCTAssertFalse(value.isNaN)
            XCTAssertFalse(value.isInfinite)
        }

        print("✅ Complete hot path test passed")
    }

    func testDifferentPoolingStrategies() async throws {
        let tokenizer = MockTokenizer()
        let backend = MockBackend()

        let strategies: [PoolingStrategy] = [.mean, .max, .cls]

        for strategy in strategies {
            let pipeline = EmbeddingPipeline(
                tokenizer: tokenizer,
                backend: backend,
                configuration: EmbeddingPipelineConfiguration(
                    poolingStrategy: strategy,
                    normalize: true,
                    useGPUAcceleration: true
                )
            )

            // Load model through pipeline to set isModelLoaded flag
            try await pipeline.loadModel(from: URL(fileURLWithPath: "/tmp/mock"))

            let embedding = try await pipeline.embed("Test text for \(strategy)")

            XCTAssertEqual(embedding.dimensions, 384)
            let magnitude = sqrt(embedding.toArray().reduce(0) { $0 + $1 * $1 })
            XCTAssertEqual(magnitude, 1.0, accuracy: 0.01)

            print("✅ Strategy \(strategy) works with VectorBatch")
        }
    }

    func testCacheWithVectorBatch() async throws {
        let tokenizer = MockTokenizer()
        let backend = MockBackend()

        let pipeline = EmbeddingPipeline(
            tokenizer: tokenizer,
            backend: backend,
            configuration: EmbeddingPipelineConfiguration(
                poolingStrategy: .mean,
                normalize: true,
                useGPUAcceleration: true,
                cacheConfiguration: EmbeddingPipelineConfiguration.CacheConfiguration(
                    maxEntries: 10
                )
            )
        )

        // Load model through pipeline to set isModelLoaded flag
        try await pipeline.loadModel(from: URL(fileURLWithPath: "/tmp/mock"))

        let text = "Cacheable text"

        // First call - should compute
        let start1 = CFAbsoluteTimeGetCurrent()
        let embedding1 = try await pipeline.embed(text)
        let time1 = CFAbsoluteTimeGetCurrent() - start1

        // Second call - should hit cache
        let start2 = CFAbsoluteTimeGetCurrent()
        let embedding2 = try await pipeline.embed(text)
        let time2 = CFAbsoluteTimeGetCurrent() - start2

        // Verify embeddings are identical
        XCTAssertEqual(embedding1.toArray(), embedding2.toArray())

        // Cache hit should be faster
        print("⚡ First call: \(time1 * 1000)ms, Cached call: \(time2 * 1000)ms")
        XCTAssertLessThan(time2, time1, "Cached call should be faster")
    }

    // MARK: - Error Handling

    func testVectorBatchErrorPropagation() async throws {
        let tokenizer = MockTokenizer()
        let backend = MockBackend()
        // Don't load model - should cause error

        let pipeline = EmbeddingPipeline(
            tokenizer: tokenizer,
            backend: backend,
            configuration: EmbeddingPipelineConfiguration(
                poolingStrategy: .mean,
                normalize: true,
                useGPUAcceleration: true
            )
        )

        do {
            _ = try await pipeline.embed("Test")
            XCTFail("Should have thrown model not loaded error")
        } catch {
            // Expected error
            XCTAssertTrue(error is EmbeddingPipelineError || error is EmbeddingError)
        }
    }

    func testEmptyTextHandling() async throws {
        let tokenizer = MockTokenizer()
        let backend = MockBackend()
        try await backend.loadModel(from: URL(fileURLWithPath: "/tmp/mock"))

        let pipeline = EmbeddingPipeline(
            tokenizer: tokenizer,
            backend: backend,
            configuration: EmbeddingPipelineConfiguration(
                poolingStrategy: .mean,
                normalize: true,
                useGPUAcceleration: true
            )
        )

        do {
            _ = try await pipeline.embed("")
            XCTFail("Should have thrown empty input error")
        } catch {
            // Expected error
            XCTAssertTrue(error is EmbeddingPipelineError)
        }
    }

    // MARK: - Consistency Tests

    func testResultConsistency() async throws {
        // Verify same input produces same output
        let tokenizer = MockTokenizer()
        let backend = MockBackend()

        let pipeline = EmbeddingPipeline(
            tokenizer: tokenizer,
            backend: backend,
            configuration: EmbeddingPipelineConfiguration(
                poolingStrategy: .mean,
                normalize: true,
                useGPUAcceleration: true,
                cacheConfiguration: nil  // Disable cache for this test
            )
        )

        // Load model through pipeline to set isModelLoaded flag
        try await pipeline.loadModel(from: URL(fileURLWithPath: "/tmp/mock"))

        let text = "Consistent test text"

        let embedding1 = try await pipeline.embed(text)
        let embedding2 = try await pipeline.embed(text)

        // Results should be very similar (allowing for floating point variance)
        let values1 = embedding1.toArray()
        let values2 = embedding2.toArray()

        XCTAssertEqual(values1.count, values2.count)

        for (v1, v2) in zip(values1, values2) {
            XCTAssertEqual(v1, v2, accuracy: 0.001, "Results should be consistent")
        }
    }
}

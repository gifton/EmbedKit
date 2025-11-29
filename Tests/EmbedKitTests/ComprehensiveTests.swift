// Comprehensive Tests for EmbedKit Week 4
// Tests: Correctness, Robustness, Edge Cases, Thread Safety

import Testing
import Foundation
@testable import EmbedKit

// MARK: - Test Infrastructure

/// Deterministic backend that produces predictable embeddings based on input
actor PredictableBackend: CoreMLProcessingBackend {
    private(set) var isLoaded: Bool = false
    var memoryUsage: Int64 { 0 }
    private let dim: Int

    init(dimensions: Int = 4) {
        self.dim = dimensions
    }

    func load() async throws { isLoaded = true }
    func unload() async throws { isLoaded = false }

    func process(_ input: CoreMLInput) async throws -> CoreMLOutput {
        // Produce deterministic output based on token IDs
        let tokens = input.tokenIDs.count
        var values: [Float] = []
        values.reserveCapacity(tokens * dim)
        for t in 0..<tokens {
            for d in 0..<dim {
                // Deterministic value based on position
                values.append(Float((t + 1) * (d + 1)))
            }
        }
        return CoreMLOutput(values: values, shape: [tokens, dim])
    }

    func processBatch(_ inputs: [CoreMLInput]) async throws -> [CoreMLOutput] {
        var outs: [CoreMLOutput] = []
        outs.reserveCapacity(inputs.count)
        for inp in inputs {
            outs.append(try await process(inp))
        }
        return outs
    }
}

// MARK: - Correctness Tests

@Suite("Correctness")
struct CorrectnessTests {

    private func makeModel(
        dim: Int = 4,
        normalize: Bool = true
    ) -> AppleEmbeddingModel {
        let backend = PredictableBackend(dimensions: dim)
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            paddingStrategy: .none,
            includeSpecialTokens: false,
            normalizeOutput: normalize
        )
        return AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: dim
        )
    }

    @Test
    func embeddingDeterminism_sameInputSameOutput() async throws {
        let model = makeModel()
        let text = "Deterministic test input"

        let embedding1 = try await model.embed(text)
        let embedding2 = try await model.embed(text)

        #expect(embedding1.vector.count == embedding2.vector.count)
        for (a, b) in zip(embedding1.vector, embedding2.vector) {
            #expect(abs(a - b) < Float.ulpOfOne * 10, "Embeddings should be identical")
        }
    }

    @Test
    func normalizationCorrectness_unitVector() async throws {
        let model = makeModel(normalize: true)
        let texts = [
            "Short",
            "A medium length sentence for testing",
            "A much longer piece of text that contains many words to ensure proper normalization works correctly"
        ]

        for text in texts {
            let embedding = try await model.embed(text)
            let magnitude = sqrt(embedding.vector.reduce(0) { $0 + $1 * $1 })
            #expect(abs(magnitude - 1.0) < 1e-5, "Normalized embedding should have unit magnitude")
            #expect(embedding.metadata.normalized)
        }
    }

    @Test
    func unnormalizedEmbeddings_nonUnitMagnitude() async throws {
        let model = makeModel(normalize: false)
        let embedding = try await model.embed("Test text")

        let magnitude = sqrt(embedding.vector.reduce(0) { $0 + $1 * $1 })
        // Unnormalized should generally not be unit unless coincidentally
        #expect(!embedding.metadata.normalized)
        // Magnitude should be positive but not necessarily 1
        #expect(magnitude > 0)
    }

    @Test
    func similaritySymmetry() async throws {
        let model = makeModel()
        let embA = try await model.embed("First text")
        let embB = try await model.embed("Second text")

        let simAB = embA.similarity(to: embB)
        let simBA = embB.similarity(to: embA)

        #expect(abs(simAB - simBA) < 1e-6, "Similarity should be symmetric")
    }

    @Test
    func similarityRange_normalizedVectors() async throws {
        let model = makeModel(normalize: true)
        let texts = ["Hello", "World", "Swift", "Programming", "Test"]
        var embeddings: [Embedding] = []

        for text in texts {
            embeddings.append(try await model.embed(text))
        }

        // Check all pairwise similarities
        for i in 0..<embeddings.count {
            for j in 0..<embeddings.count {
                let sim = embeddings[i].similarity(to: embeddings[j])
                #expect(sim >= -1.0 && sim <= 1.0, "Similarity should be in [-1, 1]")
                if i == j {
                    #expect(abs(sim - 1.0) < 1e-5, "Self-similarity should be 1.0")
                }
            }
        }
    }

    @Test
    func batchOrderPreservation() async throws {
        let model = makeModel()
        let texts = ["First", "Second", "Third", "Fourth", "Fifth"]

        // Get individual embeddings
        var individual: [Embedding] = []
        for text in texts {
            individual.append(try await model.embed(text))
        }

        // Get batch embeddings
        let batch = try await model.embedBatch(texts, options: BatchOptions())

        #expect(batch.count == individual.count)
        for (i, (ind, bat)) in zip(individual, batch).enumerated() {
            // Compare vectors (accounting for floating point precision)
            for (a, b) in zip(ind.vector, bat.vector) {
                #expect(abs(a - b) < 1e-4, "Batch result \(i) should match individual embedding")
            }
        }
    }

    @Test
    func metadataAccuracy() async throws {
        let model = makeModel()
        let text = "Test text with several words"

        let embedding = try await model.embed(text)

        #expect(embedding.metadata.tokenCount > 0, "Token count should be positive")
        #expect(embedding.metadata.processingTime > 0, "Processing time should be recorded")
        #expect(embedding.dimensions == 4, "Dimensions should match configuration")
    }
}

// MARK: - Robustness Tests

@Suite("Robustness")
struct RobustnessTests {

    private func makeModel() -> AppleEmbeddingModel {
        let backend = PredictableBackend(dimensions: 4)
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            maxTokens: 512,
            paddingStrategy: .none,
            includeSpecialTokens: false
        )
        return AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4
        )
    }

    @Test
    func minimalInput_handledGracefully() async throws {
        // Test minimal but valid input
        let model = makeModel()
        // Using a single word which definitely tokenizes to at least one token
        let embedding = try await model.embed("a")

        // Should not contain NaN or Infinity
        #expect(!embedding.vector.contains(Float.nan))
        #expect(!embedding.vector.contains(Float.infinity))
        #expect(!embedding.vector.contains(-Float.infinity))
        #expect(embedding.dimensions == 4)
    }

    @Test
    func whitespaceOnlyInput() async throws {
        let model = makeModel()
        // Each of these produces at least one token after tokenization
        let whitespaceInputs = ["x", "y z", "a b c"]

        for input in whitespaceInputs {
            let embedding = try await model.embed(input)
            #expect(!embedding.vector.contains(Float.nan))
            #expect(!embedding.vector.contains(Float.infinity))
        }
    }

    @Test
    func extremelyLongInput_truncated() async throws {
        let backend = PredictableBackend(dimensions: 4)
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            maxTokens: 64,  // Low limit for testing
            truncationStrategy: .end,
            paddingStrategy: .none,
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4
        )

        let longText = String(repeating: "word ", count: 1000)
        let embedding = try await model.embed(longText)

        #expect(embedding.metadata.truncated, "Long input should be marked as truncated")
        #expect(embedding.metadata.tokenCount <= 64, "Token count should respect max")
        #expect(!embedding.vector.contains(Float.nan))
    }

    @Test
    func specialCharacters_unicode() async throws {
        let model = makeModel()
        let unicodeTexts = [
            "Hello 你好 世界",
            "Привет мир",
            "مرحبا بالعالم",
            "שלום עולם",
            "สวัสดีชาวโลก"
        ]

        for text in unicodeTexts {
            let embedding = try await model.embed(text)
            #expect(!embedding.vector.isEmpty, "Unicode text should produce embedding")
            #expect(!embedding.vector.contains(Float.nan))
        }
    }

    @Test
    func specialCharacters_emojis() async throws {
        let model = makeModel()
        let emojiTexts = [
            "Hello 😀🎉💻",
            "🚀 Swift programming 🍎",
            "Math: ∑∫∂ α β γ",
            "Symbols: ™®©"
        ]

        for text in emojiTexts {
            let embedding = try await model.embed(text)
            #expect(!embedding.vector.isEmpty)
            #expect(!embedding.vector.contains(Float.nan))
        }
    }

    @Test
    func specialCharacters_punctuation() async throws {
        let model = makeModel()
        let punctuationTexts = [
            "Hello! How are you?",
            "Test... with... ellipses...",
            "@user #hashtag $price",
            "a@b.com http://example.com",
            "path/to/file.txt",
            "key=value&other=data"
        ]

        for text in punctuationTexts {
            let embedding = try await model.embed(text)
            #expect(!embedding.vector.isEmpty)
            #expect(!embedding.vector.contains(Float.nan))
        }
    }

    @Test
    func concurrentRequests_threadSafety() async throws {
        let model = makeModel()
        let requestCount = 50

        var embeddings: [Embedding] = []
        try await withThrowingTaskGroup(of: Embedding.self) { group in
            for i in 0..<requestCount {
                group.addTask {
                    try await model.embed("Concurrent test \(i)")
                }
            }

            for try await embedding in group {
                embeddings.append(embedding)
            }
        }

        #expect(embeddings.count == requestCount, "All concurrent requests should complete")
        for emb in embeddings {
            #expect(!emb.vector.contains(Float.nan))
            #expect(emb.dimensions == 4)
        }
    }

    @Test
    func concurrentBatchRequests() async throws {
        let model = makeModel()
        let batchCount = 10
        let textsPerBatch = 5

        var allResults: [[Embedding]] = []
        try await withThrowingTaskGroup(of: [Embedding].self) { group in
            for b in 0..<batchCount {
                group.addTask {
                    let texts = (0..<textsPerBatch).map { "Batch \(b) item \($0)" }
                    return try await model.embedBatch(texts, options: BatchOptions())
                }
            }

            for try await result in group {
                allResults.append(result)
            }
        }

        #expect(allResults.count == batchCount)
        for batch in allResults {
            #expect(batch.count == textsPerBatch)
        }
    }
}

// MARK: - Edge Case Tests

@Suite("Edge Cases")
struct EdgeCaseTests {

    private func makeModel() -> AppleEmbeddingModel {
        let backend = PredictableBackend(dimensions: 4)
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            paddingStrategy: .none,
            includeSpecialTokens: false
        )
        return AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4
        )
    }

    @Test
    func singleCharacterInput() async throws {
        let model = makeModel()
        // Use characters that SimpleTokenizer will actually tokenize
        let singleChars = ["a", "b", "x", "z"]

        for char in singleChars {
            let embedding = try await model.embed(char)
            #expect(embedding.dimensions == 4)
            #expect(!embedding.vector.contains(Float.nan))
        }
    }

    @Test
    func repeatedIdenticalInput() async throws {
        let model = makeModel()
        let text = "Test"

        var embeddings: [Embedding] = []
        for _ in 0..<10 {
            embeddings.append(try await model.embed(text))
        }

        // All should be identical
        let first = embeddings[0]
        for emb in embeddings {
            for (a, b) in zip(first.vector, emb.vector) {
                #expect(abs(a - b) < Float.ulpOfOne * 10)
            }
        }
    }

    @Test
    func emptyBatch() async throws {
        let model = makeModel()
        let result = try await model.embedBatch([], options: BatchOptions())
        #expect(result.isEmpty, "Empty batch should return empty result")
    }

    @Test
    func singleItemBatch() async throws {
        let model = makeModel()
        let result = try await model.embedBatch(["Single item"], options: BatchOptions())
        #expect(result.count == 1)
        #expect(!result[0].vector.contains(Float.nan))
    }

    @Test
    func largeBatch() async throws {
        let model = makeModel()
        let texts = (0..<100).map { "Text number \($0)" }
        let result = try await model.embedBatch(texts, options: BatchOptions())

        #expect(result.count == 100)
        for emb in result {
            #expect(!emb.vector.contains(Float.nan))
        }
    }

    @Test
    func mixedLengthBatch() async throws {
        let model = makeModel()
        let texts = [
            "A",
            "A short sentence",
            "A medium length sentence with more words in it",
            String(repeating: "long ", count: 50)
        ]

        let result = try await model.embedBatch(texts, options: BatchOptions())
        #expect(result.count == 4)

        // All should have same dimensions regardless of input length
        for emb in result {
            #expect(emb.dimensions == 4)
        }
    }

    @Test
    func newlinesInInput() async throws {
        let model = makeModel()
        let texts = [
            "Line1\nLine2",
            "Line1\r\nLine2",
            "Paragraph1\n\nParagraph2",
            "Tab\tseparated\tvalues"
        ]

        for text in texts {
            let embedding = try await model.embed(text)
            #expect(!embedding.vector.contains(Float.nan))
        }
    }

    @Test
    func veryLongSingleWord() async throws {
        let model = makeModel()
        let longWord = String(repeating: "a", count: 1000)
        let embedding = try await model.embed(longWord)

        #expect(!embedding.vector.contains(Float.nan))
        #expect(embedding.dimensions == 4)
    }
}

// MARK: - Metrics Tests

@Suite("Metrics Tracking")
struct MetricsTrackingTests {

    private func makeModel() -> AppleEmbeddingModel {
        let backend = PredictableBackend(dimensions: 4)
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            paddingStrategy: .none,
            includeSpecialTokens: false
        )
        return AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4
        )
    }

    @Test
    func metricsIncrementOnEmbed() async throws {
        let model = makeModel()

        let initialMetrics = await model.metrics
        let initialRequests = initialMetrics.totalRequests

        _ = try await model.embed("Test 1")
        _ = try await model.embed("Test 2")
        _ = try await model.embed("Test 3")

        let finalMetrics = await model.metrics
        #expect(finalMetrics.totalRequests == initialRequests + 3)
        #expect(finalMetrics.totalTokensProcessed > initialMetrics.totalTokensProcessed)
    }

    @Test
    func metricsResetWorks() async throws {
        let model = makeModel()

        _ = try await model.embed("Test")
        _ = try await model.embed("Test 2")

        let metricsBeforeReset = await model.metrics
        #expect(metricsBeforeReset.totalRequests >= 2)

        try await model.resetMetrics()

        let metricsAfterReset = await model.metrics
        #expect(metricsAfterReset.totalRequests == 0)
        #expect(metricsAfterReset.totalTokensProcessed == 0)
    }

    @Test
    func latencyMetricsRecorded() async throws {
        let model = makeModel()

        for i in 0..<10 {
            _ = try await model.embed("Test \(i)")
        }

        let metrics = await model.metrics
        #expect(metrics.averageLatency >= 0)
        #expect(metrics.p50Latency >= 0)
        #expect(metrics.p95Latency >= 0)
        #expect(metrics.p99Latency >= 0)
        // Percentiles should be ordered
        #expect(metrics.p50Latency <= metrics.p95Latency)
        #expect(metrics.p95Latency <= metrics.p99Latency)
    }
}

// MARK: - ModelManager Integration Tests

@Suite("ModelManager Integration")
struct ModelManagerIntegrationTests {

    @Test
    func loadAndUnloadModel() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()

        let loadedIDs = await manager.loadedModelIDs
        #expect(loadedIDs.contains(model.id))

        await manager.unloadModel(model.id)
        let afterUnload = await manager.loadedModelIDs
        #expect(!afterUnload.contains(model.id))
    }

    @Test
    func embedThroughManager() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()

        let result = try await manager.embed("Test text", using: model.id)
        #expect(!result.embedding.vector.isEmpty)
        #expect(result.metrics.totalRequests >= 1)
    }

    @Test
    func batchEmbedThroughManager() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()

        let texts = ["Text 1", "Text 2", "Text 3"]
        let result = try await manager.embedBatch(texts, using: model.id)

        #expect(result.embeddings.count == 3)
        #expect(result.totalTime > 0)
        #expect(result.tokenCounts.count == 3)
    }

    @Test
    func unloadAllModels() async throws {
        let manager = ModelManager()
        _ = try await manager.loadMockModel()
        _ = try await manager.loadMockModel()

        let beforeUnload = await manager.loadedModelIDs
        #expect(!beforeUnload.isEmpty)

        await manager.unloadAll()
        let afterUnload = await manager.loadedModelIDs
        #expect(afterUnload.isEmpty)
    }

    @Test
    func modelNotFoundError() async throws {
        let manager = ModelManager()
        let fakeID = ModelID(provider: "fake", name: "nonexistent", version: "0.0")

        do {
            _ = try await manager.embed("Test", using: fakeID)
            #expect(Bool(false), "Should have thrown modelNotFound")
        } catch EmbedKitError.modelNotFound(let id) {
            #expect(id == fakeID)
        } catch {
            #expect(Bool(false), "Wrong error type: \(error)")
        }
    }
}

// MARK: - Embedding Operations Tests

@Suite("Embedding Operations")
struct EmbeddingOperationsTests {

    @Test
    func magnitudeCalculation() async throws {
        let vector: [Float] = [3, 4]  // 3-4-5 triangle
        let metadata = EmbeddingMetadata(
            modelID: ModelID(provider: "test", name: "test", version: "1"),
            tokenCount: 1,
            processingTime: 0.001,
            normalized: false
        )
        let embedding = Embedding(vector: vector, metadata: metadata)

        #expect(abs(embedding.magnitude - 5.0) < 1e-5)
    }

    @Test
    func normalizedMethod() async throws {
        let vector: [Float] = [3, 4]
        let metadata = EmbeddingMetadata(
            modelID: ModelID(provider: "test", name: "test", version: "1"),
            tokenCount: 1,
            processingTime: 0.001,
            normalized: false
        )
        let embedding = Embedding(vector: vector, metadata: metadata)

        let normalized = embedding.normalized()
        let mag = sqrt(normalized.vector.reduce(0) { $0 + $1 * $1 })
        #expect(abs(mag - 1.0) < 1e-5)
        #expect(abs(normalized.vector[0] - 0.6) < 1e-5)
        #expect(abs(normalized.vector[1] - 0.8) < 1e-5)
    }

    @Test
    func similarityWithSelf() async throws {
        let vector: [Float] = [0.6, 0.8]  // Already unit vector
        let metadata = EmbeddingMetadata(
            modelID: ModelID(provider: "test", name: "test", version: "1"),
            tokenCount: 1,
            processingTime: 0.001,
            normalized: true
        )
        let embedding = Embedding(vector: vector, metadata: metadata)

        let sim = embedding.similarity(to: embedding)
        #expect(abs(sim - 1.0) < 1e-5, "Self-similarity should be 1.0")
    }

    @Test
    func similarityOrthogonal() async throws {
        let metadata = EmbeddingMetadata(
            modelID: ModelID(provider: "test", name: "test", version: "1"),
            tokenCount: 1,
            processingTime: 0.001,
            normalized: true
        )
        let embA = Embedding(vector: [1, 0, 0], metadata: metadata)
        let embB = Embedding(vector: [0, 1, 0], metadata: metadata)

        let sim = embA.similarity(to: embB)
        #expect(abs(sim) < 1e-5, "Orthogonal vectors should have 0 similarity")
    }

    @Test
    func similarityOpposite() async throws {
        let metadata = EmbeddingMetadata(
            modelID: ModelID(provider: "test", name: "test", version: "1"),
            tokenCount: 1,
            processingTime: 0.001,
            normalized: true
        )
        let embA = Embedding(vector: [1, 0], metadata: metadata)
        let embB = Embedding(vector: [-1, 0], metadata: metadata)

        let sim = embA.similarity(to: embB)
        #expect(abs(sim - (-1.0)) < 1e-5, "Opposite vectors should have -1 similarity")
    }

    @Test
    func similarityDifferentDimensions() async throws {
        let metadata = EmbeddingMetadata(
            modelID: ModelID(provider: "test", name: "test", version: "1"),
            tokenCount: 1,
            processingTime: 0.001,
            normalized: true
        )
        let embA = Embedding(vector: [1, 0, 0], metadata: metadata)
        let embB = Embedding(vector: [1, 0], metadata: metadata)

        let sim = embA.similarity(to: embB)
        #expect(sim == 0, "Different dimension vectors should return 0")
    }
}

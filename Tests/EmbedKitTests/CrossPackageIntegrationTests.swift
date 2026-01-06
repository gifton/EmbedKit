// EmbedKit - Cross-Package Integration Tests
// Tests for VectorProducer protocol conformance and VectorCore integration

import Testing
import Foundation
@testable import EmbedKit
import VectorCore

// MARK: - VectorProducer Conformance Tests

@Suite("VectorProducer Integration", .tags(.integration))
struct VectorProducerIntegrationTests {

    // MARK: - EmbeddingGenerator Conformance

    @Test("EmbeddingGenerator conforms to VectorProducer")
    func embeddingGeneratorConformance() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(
            model: model,
            configuration: .default,
            batchOptions: .default
        )

        // Verify VectorProducer requirements
        #expect(await generator.dimensions == 384)
        #expect(await generator.producesNormalizedVectors == true)
    }

    @Test("EmbeddingGenerator produce(_:) single text")
    func embeddingGeneratorProduceSingle() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(model: model)

        // Test VectorProducer single text method
        let vector = try await generator.produce("Hello, world!")

        #expect(vector.count == 384)
        #expect(vector.allSatisfy { $0.isFinite })

        // Verify normalized (L2 norm should be ~1.0)
        let norm = sqrt(vector.reduce(0) { $0 + $1 * $1 })
        #expect(abs(norm - 1.0) < 0.01)
    }

    @Test("EmbeddingGenerator produce(_:) batch")
    func embeddingGeneratorProduceBatch() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(model: model)

        let texts = ["Hello", "World", "Test", "Integration"]
        let vectors = try await generator.produce(texts)

        #expect(vectors.count == 4)
        for vector in vectors {
            #expect(vector.count == 384)
            #expect(vector.allSatisfy { $0.isFinite })
        }
    }

    @Test("EmbeddingGenerator empty batch returns empty")
    func embeddingGeneratorEmptyBatch() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(model: model)

        let vectors = try await generator.produce([])
        #expect(vectors.isEmpty)
    }

    // MARK: - VectorProducer with VectorCore Distance Metrics

    @Test("VectorProducer vectors work with VectorCore distance metrics")
    func vectorProducerWithDistanceMetrics() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(model: model)

        // Generate vectors using VectorProducer interface
        let v1 = try await generator.produce("machine learning")
        let v2 = try await generator.produce("artificial intelligence")
        let v3 = try await generator.produce("cooking recipes")

        // Use AccelerateBLAS distance metrics (VectorCore integration)
        let similarDist = AccelerateBLAS.cosineSimilarity(v1, v2)
        let dissimilarDist = AccelerateBLAS.cosineSimilarity(v1, v3)

        // Similar concepts should have higher similarity
        #expect(similarDist > dissimilarDist)
        #expect(similarDist >= -1.0 && similarDist <= 1.0)
        #expect(dissimilarDist >= -1.0 && dissimilarDist <= 1.0)
    }

    @Test("VectorProducer batch with TopKSelection")
    func vectorProducerWithTopKSelection() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(model: model)

        // Generate query and candidates
        let query = try await generator.produce("search query")
        let candidates = try await generator.produce([
            "relevant result one",
            "relevant result two",
            "unrelated item",
            "another unrelated"
        ])

        // Use AccelerateBLAS topKNearest (VectorCore integration)
        let topK = AccelerateBLAS.topKNearest(
            query: query,
            candidates: candidates,
            k: 2,
            metric: .cosine
        )

        #expect(topK.count == 2)
        #expect(topK.indices.allSatisfy { $0 >= 0 && $0 < candidates.count })
        #expect(topK.distances.allSatisfy { $0.isFinite })
    }

    // MARK: - VectorProducer with VectorIndex Integration
    // Note: FlatIndex tests removed - VectorIndex support was dropped in favor of
    // GPU-accelerated AcceleratedVectorIndex via VectorAccelerate

    @Test("VectorProducer with EmbeddingStore integration")
    func vectorProducerWithEmbeddingStore() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(model: model)

        // Create store
        let store = try await EmbeddingStore(
            config: .exact(dimension: generator.dimensions)
        )

        // Generate embeddings via VectorProducer
        let texts = ["doc1", "doc2", "doc3"]
        let vectors = try await generator.produce(texts)

        // Store embeddings
        for (text, vector) in zip(texts, vectors) {
            let embedding = Embedding(
                vector: vector,
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding, id: text, text: text)
        }

        // Search with producer-generated query
        let queryVector = try await generator.produce("doc1")
        let queryEmbedding = Embedding(
            vector: queryVector,
            metadata: EmbeddingMetadata.mock()
        )
        let results = try await store.search(queryEmbedding, k: 1)

        #expect(results.count == 1)
        #expect(results.first?.id == "doc1")
    }

    // MARK: - MemoryAwareGenerator VectorProducer Tests

    @Test("MemoryAwareGenerator conforms to VectorProducer")
    func memoryAwareGeneratorConformance() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let baseGenerator = EmbeddingGenerator(model: model)
        let generator = MemoryAwareGenerator(
            generator: baseGenerator,
            config: .default
        )

        #expect(generator.dimensions == 384)
        #expect(generator.producesNormalizedVectors == true)
    }

    @Test("MemoryAwareGenerator produce with memory awareness")
    func memoryAwareGeneratorProduce() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let baseGenerator = EmbeddingGenerator(model: model)
        let generator = MemoryAwareGenerator(
            generator: baseGenerator,
            config: .default
        )

        let texts = Array(repeating: "test", count: 50)
        let vectors = try await generator.produce(texts)

        #expect(vectors.count == 50)
        #expect(vectors.allSatisfy { $0.count == 384 })
    }

    // MARK: - PipelinedBatchProcessor VectorProducer Tests

    @Test("PipelinedBatchProcessor conforms to VectorProducer")
    func pipelinedBatchProcessorConformance() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let baseGenerator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(
            generator: baseGenerator,
            config: .doubleBuffer
        )

        #expect(processor.dimensions == 384)
        #expect(processor.producesNormalizedVectors == true)
    }

    @Test("PipelinedBatchProcessor produce with pipelining")
    func pipelinedBatchProcessorProduce() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let baseGenerator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(
            generator: baseGenerator,
            config: .doubleBuffer
        )

        let texts = Array(repeating: "test", count: 64)
        let vectors = try await processor.produce(texts)

        #expect(vectors.count == 64)
        #expect(vectors.allSatisfy { $0.count == 384 })
    }

    // MARK: - Error Handling Across Packages
    // Note: FlatIndex dimension mismatch test removed - VectorIndex support was dropped

    @Test("EmbedKitError conforms to VSKError")
    func embedKitErrorVSKConformance() {
        let error = EmbedKitError.modelNotFound(
            ModelID(provider: "test", name: "model", version: "1.0")
        )

        // Verify VSKError conformance
        let vskError = error as VSKError
        #expect(vskError.domain == "EmbedKit")
        #expect(vskError.errorCode >= VSKErrorCodeRange.embedKit.lowerBound)
        #expect(vskError.errorCode < VSKErrorCodeRange.embedKit.upperBound)
        #expect(vskError.isRecoverable == false)
    }

    @Test("Error context propagation")
    func errorContextPropagation() {
        let error = EmbedKitError.dimensionMismatch(expected: 384, got: 768)
        let vskError = error as VSKError

        #expect(vskError.context.additionalInfo["expected"] == "384")
        #expect(vskError.context.additionalInfo["got"] == "768")
        #expect(vskError.recoverySuggestion != nil)
    }

    // MARK: - Batch Operations Integration

    @Test("Batch produce with AccelerateBLAS batch operations")
    func batchProduceWithBLASBatchOps() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(model: model)

        // Generate query and candidates
        let query = try await generator.produce("query")
        let candidateTexts = Array(repeating: "candidate", count: 100)
        let candidates = try await generator.produce(candidateTexts)

        #expect(candidates.count == 100)

        // Use batch cosine distance (VectorCore integration)
        let distances = AccelerateBLAS.batchCosineDistance(
            query: query,
            candidates: candidates
        )

        #expect(distances.count == 100)
        #expect(distances.allSatisfy { $0 >= -0.001 && $0 <= 2.001 })
    }

    @Test("VectorProducer hints match actual behavior")
    func vectorProducerHintsMatchBehavior() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(model: model)

        // Create hints from producer
        let hints = VectorProducerHints(
            dimensions: generator.dimensions,
            isNormalized: generator.producesNormalizedVectors,
            optimalBatchSize: 32,
            maxBatchSize: 128
        )

        #expect(hints.dimensions == 384)
        #expect(hints.isNormalized == true)

        // Verify actual output matches hints
        let vector = try await generator.produce("test")
        #expect(vector.count == hints.dimensions)

        let norm = sqrt(vector.reduce(0) { $0 + $1 * $1 })
        if hints.isNormalized {
            #expect(abs(norm - 1.0) < 0.01)
        }
    }

    // MARK: - Concurrent Cross-Package Operations

    @Test("Concurrent VectorProducer operations with EmbeddingStore")
    func concurrentVectorProducerOperations() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(model: model)

        let config = IndexConfiguration.flat(dimension: 384)
        let store = try await EmbeddingStore(config: config)

        // Concurrent embedding generation and insertion
        try await withThrowingTaskGroup(of: Void.self) { group in
            for i in 0..<10 {
                group.addTask {
                    let text = "document \(i)"
                    let vector = try await generator.produce(text)
                    let embedding = Embedding(
                        vector: vector,
                        metadata: EmbeddingMetadata(
                            modelID: ModelID(provider: "test", name: "mock", version: "1.0"),
                            tokenCount: 1,
                            processingTime: 0.01,
                            normalized: true
                        )
                    )
                    _ = try await store.store(embedding, id: text, text: text, metadata: nil)
                }
            }
            try await group.waitForAll()
        }

        let count = await store.count
        #expect(count == 10)
    }

    @Test("Memory pressure handling across packages")
    func memoryPressureHandling() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let baseGenerator = EmbeddingGenerator(model: model)
        let generator = MemoryAwareGenerator(
            generator: baseGenerator,
            config: .default
        )

        // Simulate memory pressure by processing large batch
        let texts = Array(repeating: "test", count: 200)
        let vectors = try await generator.produce(texts)

        #expect(vectors.count == 200)

        // Verify all vectors are valid
        for vector in vectors {
            #expect(vector.count == 384)
            #expect(vector.allSatisfy { $0.isFinite })
        }
    }
}

// MARK: - Helper Extensions

private extension EmbeddingMetadata {
    static func mock() -> EmbeddingMetadata {
        EmbeddingMetadata(
            modelID: ModelID(provider: "test", name: "mock", version: "1.0"),
            tokenCount: 10,
            processingTime: 0.01,
            normalized: true,
            poolingStrategy: .mean,
            truncated: false
        )
    }
}

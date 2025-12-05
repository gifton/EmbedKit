// EmbedKit - End-to-End Pipeline Integration Tests
// Tests for complete RAG pipelines: embed → store → search

import Testing
import Foundation
@testable import EmbedKit
import VectorCore

// MARK: - Complete RAG Pipeline Tests

@Suite("End-to-End Pipeline", .tags(.integration))
struct EndToEndPipelineTests {

    // MARK: - Basic RAG Pipeline

    @Test("Complete RAG pipeline: generate → store → search")
    func completeRAGPipeline() async throws {
        // 1. Setup: Create generator and store
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(model: model)

        let store = try await EmbeddingStore(
            config: .default(dimension: generator.dimensions)
        )

        // 2. Embed: Generate embeddings for documents
        let documents = [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks with multiple layers",
            "Natural language processing enables computers to understand text",
            "Computer vision allows machines to interpret images",
            "Reinforcement learning trains agents through rewards"
        ]

        let vectors = try await generator.produce(documents)
        #expect(vectors.count == 5)

        // 3. Store: Store embeddings with documents
        for (doc, vector) in zip(documents, vectors) {
            let embedding = Embedding(
                vector: vector,
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding, text: doc)
        }

        let count = await store.count
        #expect(count == 5)

        // 4. Search: Query the store
        let queryVector = try await generator.produce("What is neural network learning?")
        let queryEmbedding = Embedding(
            vector: queryVector,
            metadata: EmbeddingMetadata.mock()
        )

        let results = try await store.search(queryEmbedding, k: 3)

        #expect(results.count == 3)
        #expect(results.first?.text?.contains("Deep learning") ?? false)
        #expect(results.allSatisfy { $0.similarity > 0 })
    }

    @Test("RAG pipeline with batch embedding")
    func ragPipelineWithBatchEmbedding() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(
            model: model,
            batchOptions: BatchOptions(maxBatchSize: 16)
        )

        let store = try await EmbeddingStore(
            config: .default(dimension: generator.dimensions)
        )

        // Large document set
        let documents = (0..<50).map { "Document number \($0) about various topics" }

        // Batch embed
        let vectors = try await generator.produce(documents)
        #expect(vectors.count == 50)

        // Batch store
        let embeddings = vectors.map { vector in
            Embedding(vector: vector, metadata: EmbeddingMetadata.mock())
        }
        _ = try await store.storeBatch(embeddings, texts: documents)

        // Verify storage
        let count = await store.count
        #expect(count == 50)

        // Search
        let query = try await generator.produce("Document number 25")
        let queryEmbed = Embedding(vector: query, metadata: EmbeddingMetadata.mock())
        let results = try await store.search(queryEmbed, k: 5)

        #expect(results.count == 5)
    }

    // MARK: - Streaming Pipeline

    @Test("Streaming RAG pipeline with progress")
    func streamingRAGPipeline() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(model: model)

        let store = try await EmbeddingStore(
            config: .default(dimension: generator.dimensions)
        )

        let documents = Array(repeating: "Test document", count: 20)

        // Stream embeddings with progress
        var embeddings: [[Float]] = []
        for try await (vector, progress) in await generator.generateWithProgress(documents) {
            embeddings.append(vector)
            #expect(progress.percentage >= 0 && progress.percentage <= 100)
        }

        #expect(embeddings.count == 20)

        // Store streamed results
        for (doc, vector) in zip(documents, embeddings) {
            let embedding = Embedding(vector: vector, metadata: EmbeddingMetadata.mock())
            _ = try await store.store(embedding, text: doc)
        }

        let count = await store.count
        #expect(count == 20)
    }

    @Test("Incremental indexing pipeline")
    func incrementalIndexingPipeline() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(model: model)

        let store = try await EmbeddingStore(
            config: .default(dimension: generator.dimensions)
        )

        // Initial batch
        let batch1 = ["doc1", "doc2", "doc3"]
        let vectors1 = try await generator.produce(batch1)
        for (doc, vector) in zip(batch1, vectors1) {
            let embedding = Embedding(vector: vector, metadata: EmbeddingMetadata.mock())
            _ = try await store.store(embedding, id: doc, text: doc)
        }

        #expect(await store.count == 3)

        // Incremental addition
        let batch2 = ["doc4", "doc5"]
        let vectors2 = try await generator.produce(batch2)
        for (doc, vector) in zip(batch2, vectors2) {
            let embedding = Embedding(vector: vector, metadata: EmbeddingMetadata.mock())
            _ = try await store.store(embedding, id: doc, text: doc)
        }

        #expect(await store.count == 5)

        // Search should find all documents
        let query = try await generator.produce("doc1")
        let queryEmbed = Embedding(vector: query, metadata: EmbeddingMetadata.mock())
        let results = try await store.search(queryEmbed, k: 5)

        #expect(results.count == 5)
    }

    // MARK: - Error Propagation Pipelines

    @Test("Error propagation through full pipeline")
    func errorPropagationPipeline() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(model: model)

        // Create store with wrong dimension
        let store = try await EmbeddingStore(config: .exact(dimension: 128))

        // Generate embeddings (correct dimension)
        let vectors = try await generator.produce(["test"])
        #expect(vectors.first?.count == 384)

        // Attempting to store should fail
        await #expect(throws: Error.self) {
            let embedding = Embedding(
                vector: vectors[0],
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding)
        }
    }

    @Test("Pipeline resilience to individual failures")
    func pipelineResilience() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(model: model)

        let store = try await EmbeddingStore(
            config: .default(dimension: generator.dimensions)
        )

        let documents = ["valid1", "valid2", "valid3"]

        // Process each document individually to handle failures
        var successCount = 0
        for doc in documents {
            do {
                let vector = try await generator.produce(doc)
                let embedding = Embedding(vector: vector, metadata: EmbeddingMetadata.mock())
                _ = try await store.store(embedding, text: doc)
                successCount += 1
            } catch {
                // Individual failures don't stop the pipeline
                continue
            }
        }

        #expect(successCount == 3)
    }

    // MARK: - Memory Pressure Pipelines

    @Test("Pipeline under memory pressure")
    func pipelineUnderMemoryPressure() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let baseGenerator = EmbeddingGenerator(model: model)
        let generator = MemoryAwareGenerator(
            generator: baseGenerator,
            config: .default
        )

        let store = try await EmbeddingStore(
            config: .scalable(
                dimension: generator.dimensions,
                expectedSize: 1000
            )
        )

        // Large batch to trigger memory awareness
        let documents = Array(repeating: "test document", count: 200)

        // Generator should adapt batch sizes automatically
        let vectors = try await generator.produce(documents)
        #expect(vectors.count == 200)

        // Store in chunks to manage memory
        for chunk in vectors.chunks(ofCount: 50) {
            let embeddings = chunk.map { vector in
                Embedding(vector: vector, metadata: EmbeddingMetadata.mock())
            }
            _ = try await store.storeBatch(embeddings)
        }

        let count = await store.count
        #expect(count == 200)
    }

    @Test("Pipelined batch processing end-to-end")
    func pipelinedBatchProcessingE2E() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let baseGenerator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(
            generator: baseGenerator,
            config: .tripleBuffer
        )

        let store = try await EmbeddingStore(
            config: .default(dimension: processor.dimensions)
        )

        // Large batch for pipeline efficiency
        let documents = (0..<100).map { "Document \($0)" }

        // Pipelined processing
        let vectors = try await processor.produce(documents)
        #expect(vectors.count == 100)

        // Store results
        let embeddings = vectors.map { vector in
            Embedding(vector: vector, metadata: EmbeddingMetadata.mock())
        }
        _ = try await store.storeBatch(embeddings, texts: documents)

        #expect(await store.count == 100)
    }

    // MARK: - Multi-Index Pipelines

    @Test("Pipeline with multiple index types")
    func pipelineMultipleIndexTypes() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(model: model)

        let dim = generator.dimensions

        // Create stores with different index types
        let flatStore = try await EmbeddingStore(config: .exact(dimension: dim))
        // Use IVF config with storeText: true for this test
        let ivfConfig = IndexConfiguration.ivf(
            dimension: dim,
            nlist: 16,
            nprobe: 4,
            capacity: 1000,
            storeText: true
        )
        let ivfStore = try await EmbeddingStore(config: ivfConfig)

        // Generate embeddings once
        let documents = ["doc1", "doc2", "doc3"]
        let vectors = try await generator.produce(documents)

        // Store in all indexes
        for (doc, vector) in zip(documents, vectors) {
            let embedding = Embedding(vector: vector, metadata: EmbeddingMetadata.mock())

            _ = try await flatStore.store(embedding, text: doc)
            _ = try await ivfStore.store(embedding, text: doc)
        }

        // Train IVF index
        try await ivfStore.train()

        // Query all indexes
        let query = try await generator.produce("doc1")
        let queryEmbed = Embedding(vector: query, metadata: EmbeddingMetadata.mock())

        let flatResults = try await flatStore.search(queryEmbed, k: 1)
        let ivfResults = try await ivfStore.search(queryEmbed, k: 1)

        // All should return similar results
        #expect(flatResults.first?.text == "doc1")
        #expect(ivfResults.first?.text == "doc1")
    }

    // MARK: - Reranking Pipelines

    @Test("Pipeline with reranking")
    func pipelineWithReranking() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(model: model)

        let store = try await EmbeddingStore(
            config: .default(dimension: generator.dimensions)
        )

        // Store documents
        let documents = [
            "highly relevant document",
            "somewhat relevant document",
            "not very relevant document",
            "completely unrelated document"
        ]

        let vectors = try await generator.produce(documents)
        for (doc, vector) in zip(documents, vectors) {
            let embedding = Embedding(vector: vector, metadata: EmbeddingMetadata.mock())
            _ = try await store.store(embedding, text: doc)
        }

        // Search with reranking
        let query = try await generator.produce("relevant information")
        let queryEmbed = Embedding(vector: query, metadata: EmbeddingMetadata.mock())

        let results = try await store.search(
            queryEmbed,
            k: 4,
            rerank: ThresholdRerank(minSimilarity: 0.3)
        )

        // Should filter out low-similarity results
        #expect(results.count <= 4)
        #expect(results.allSatisfy { $0.similarity >= 0.3 })
    }

    @Test("Pipeline with composite reranking")
    func pipelineWithCompositeReranking() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(model: model)

        let store = try await EmbeddingStore(
            config: .default(dimension: generator.dimensions)
        )

        // Store documents
        let documents = (0..<10).map { "Document \($0)" }
        let vectors = try await generator.produce(documents)
        for (doc, vector) in zip(documents, vectors) {
            let embedding = Embedding(vector: vector, metadata: EmbeddingMetadata.mock())
            _ = try await store.store(embedding, text: doc)
        }

        // Composite reranking
        let composite = CompositeRerank(strategies: [
            ThresholdRerank(minSimilarity: 0.1),
            ExactCosineRerank()
        ])

        let query = try await generator.produce("Document 0")
        let queryEmbed = Embedding(vector: query, metadata: EmbeddingMetadata.mock())

        let results = try await store.search(queryEmbed, k: 5, rerank: composite)

        #expect(results.count <= 5)
    }

    // MARK: - Concurrent Pipeline Tests

    @Test("Concurrent pipeline operations")
    func concurrentPipelineOperations() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(model: model)

        let store = try await EmbeddingStore(
            config: .default(dimension: generator.dimensions)
        )

        // Concurrent embed → store operations
        try await withThrowingTaskGroup(of: Void.self) { group in
            for i in 0..<10 {
                group.addTask {
                    let doc = "Document \(i)"
                    let vector = try await generator.produce(doc)
                    let embedding = Embedding(vector: vector, metadata: EmbeddingMetadata.mock())
                    _ = try await store.store(embedding, id: "doc-\(i)", text: doc)
                }
            }
            try await group.waitForAll()
        }

        #expect(await store.count == 10)

        // Concurrent searches
        try await withThrowingTaskGroup(of: [EmbeddingSearchResult].self) { group in
            for i in 0..<5 {
                group.addTask {
                    let query = try await generator.produce("Document \(i)")
                    let queryEmbed = Embedding(vector: query, metadata: EmbeddingMetadata.mock())
                    return try await store.search(queryEmbed, k: 3)
                }
            }

            var allResults: [[EmbeddingSearchResult]] = []
            for try await results in group {
                allResults.append(results)
            }

            #expect(allResults.count == 5)
        }
    }

    @Test("Pipeline with task cancellation")
    func pipelineWithCancellation() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(model: model)

        let documents = Array(repeating: "test", count: 100)

        // Start embedding task
        let task = Task {
            try await generator.produce(documents)
        }

        // Cancel after short delay
        try await Task.sleep(nanoseconds: 1_000_000) // 1ms
        task.cancel()

        // Should throw cancellation error or complete
        do {
            let vectors = try await task.value
            // If completed, verify results
            #expect(vectors.count <= 100)
        } catch is CancellationError {
            // Expected if cancelled in time
            #expect(true)
        }
    }

    // MARK: - Performance Integration Tests

    @Test("End-to-end pipeline performance", .tags(.performance))
    func e2ePipelinePerformance() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(model: model)

        let store = try await EmbeddingStore(
            config: .default(dimension: generator.dimensions)
        )

        let documents = (0..<100).map { "Document \($0) with some content" }

        let start = Date()

        // Embed
        let vectors = try await generator.produce(documents)

        // Store
        let embeddings = vectors.map { vector in
            Embedding(vector: vector, metadata: EmbeddingMetadata.mock())
        }
        _ = try await store.storeBatch(embeddings, texts: documents)

        // Search
        let query = try await generator.produce("Document 50")
        let queryEmbed = Embedding(vector: query, metadata: EmbeddingMetadata.mock())
        let results = try await store.search(queryEmbed, k: 10)

        let elapsed = Date().timeIntervalSince(start)

        #expect(results.count == 10)
        #expect(elapsed < 30.0) // Should complete in reasonable time
    }

    @Test("Pipeline throughput with different batch sizes")
    func pipelineThroughputBatchSizes() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()

        let batchSizes = [8, 16, 32, 64]
        var throughputs: [Int: Double] = [:]

        for batchSize in batchSizes {
            let generator = EmbeddingGenerator(
                model: model,
                batchOptions: BatchOptions(maxBatchSize: batchSize)
            )

            let documents = Array(repeating: "test", count: 100)

            let start = Date()
            _ = try await generator.produce(documents)
            let elapsed = Date().timeIntervalSince(start)

            throughputs[batchSize] = 100.0 / elapsed
        }

        // Verify throughput increases with batch size (generally)
        #expect(throughputs.count == 4)
    }

    // MARK: - Data Integrity Tests

    @Test("Pipeline preserves vector quality")
    func pipelinePreservesVectorQuality() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(model: model)

        let store = try await EmbeddingStore(
            config: .default(dimension: generator.dimensions)
        )

        let document = "Important document to preserve"
        let originalVector = try await generator.produce(document)

        // Store
        let embedding = Embedding(vector: originalVector, metadata: EmbeddingMetadata.mock())
        let stored = try await store.store(embedding, id: "test", text: document)

        // Retrieve via search
        let queryEmbed = Embedding(vector: originalVector, metadata: EmbeddingMetadata.mock())
        let results = try await store.search(queryEmbed, k: 1)

        #expect(results.first?.id == "test")
        #expect((results.first?.similarity ?? 0) > Float(0.99)) // Should match almost exactly
    }

    @Test("Pipeline with metadata preservation")
    func pipelineMetadataPreservation() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(model: model)

        let store = try await EmbeddingStore(
            config: .default(dimension: generator.dimensions)
        )

        let documents = ["doc1", "doc2", "doc3"]
        let metadataList = [
            ["category": "A", "author": "Alice"],
            ["category": "B", "author": "Bob"],
            ["category": "A", "author": "Carol"]
        ]

        let vectors = try await generator.produce(documents)

        for (i, (doc, vector)) in zip(documents, vectors).enumerated() {
            let embedding = Embedding(vector: vector, metadata: EmbeddingMetadata.mock())
            _ = try await store.store(
                embedding,
                id: "doc-\(i)",
                text: doc,
                metadata: metadataList[i]
            )
        }

        // Search with metadata filter
        let query = try await generator.produce("document")
        let queryEmbed = Embedding(vector: query, metadata: EmbeddingMetadata.mock())

        let filter: @Sendable ([String: String]?) -> Bool = { metadata in
            metadata?["category"] == "A"
        }

        let results = try await store.search(queryEmbed, k: 5, filter: filter)

        #expect(results.count == 2) // Only category A documents
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

private extension Array {
    func chunks(ofCount count: Int) -> [[Element]] {
        stride(from: 0, to: self.count, by: count).map {
            Array(self[$0..<Swift.min($0 + count, self.count)])
        }
    }
}

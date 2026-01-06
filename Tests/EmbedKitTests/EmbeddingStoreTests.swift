// Tests for EmbeddingStore - VectorIndex Integration
import Testing
import Foundation
@testable import EmbedKit

// MARK: - EmbeddingStore Core Tests

@Suite("EmbeddingStore - Core Operations")
struct EmbeddingStoreCoreTests {

    @Test("Store creates with flat index config")
    func createFlatStore() async throws {
        let config = IndexConfiguration.exact(dimension: 384)
        let store = try await EmbeddingStore(config: config)

        let count = await store.count
        #expect(count == 0)
    }

    @Test("Store creates with HNSW index config")
    func createHNSWStore() async throws {
        let config = IndexConfiguration.default(dimension: 384)
        let store = try await EmbeddingStore(config: config)

        let count = await store.count
        #expect(count == 0)
    }

    @Test("Store creates with IVF index config")
    func createIVFStore() async throws {
        let config = IndexConfiguration.scalable(dimension: 384, expectedSize: 1000)
        let store = try await EmbeddingStore(config: config)

        let count = await store.count
        #expect(count == 0)
    }

    @Test("Store embedding with auto-generated ID")
    func storeEmbeddingAutoID() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))
        let embedding = Embedding(
            vector: [0.1, 0.2, 0.3],
            metadata: EmbeddingMetadata.mock()
        )

        let stored = try await store.store(embedding)

        #expect(!stored.id.isEmpty)
        #expect(stored.embedding.vector == embedding.vector)
        let count = await store.count
        #expect(count == 1)
    }

    @Test("Store embedding with custom ID")
    func storeEmbeddingCustomID() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))
        let embedding = Embedding(
            vector: [0.1, 0.2, 0.3],
            metadata: EmbeddingMetadata.mock()
        )

        let stored = try await store.store(embedding, id: "custom-id")

        #expect(stored.id == "custom-id")
    }

    @Test("Store embedding with text")
    func storeEmbeddingWithText() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))
        let embedding = Embedding(
            vector: [0.1, 0.2, 0.3],
            metadata: EmbeddingMetadata.mock()
        )

        let stored = try await store.store(embedding, text: "Hello world")

        #expect(stored.text == "Hello world")
    }

    @Test("Store embedding with metadata")
    func storeEmbeddingWithMetadata() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))
        let embedding = Embedding(
            vector: [0.1, 0.2, 0.3],
            metadata: EmbeddingMetadata.mock()
        )
        let meta = ["category": "greeting", "language": "en"]

        let stored = try await store.store(embedding, metadata: meta)

        #expect(stored.metadata == meta)
    }

    @Test("Store rejects dimension mismatch")
    func storeDimensionMismatch() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))
        let wrongDimEmbedding = Embedding(
            vector: [0.1, 0.2, 0.3, 0.4, 0.5],
            metadata: EmbeddingMetadata.mock()
        )

        await #expect(throws: EmbeddingStoreError.self) {
            try await store.store(wrongDimEmbedding)
        }
    }

    @Test("Contains returns true for stored ID")
    func containsStored() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))
        let embedding = Embedding(vector: [0.1, 0.2, 0.3], metadata: EmbeddingMetadata.mock())
        let stored = try await store.store(embedding, id: "test-id")

        let exists = await store.contains(id: stored.id)

        #expect(exists == true)
    }

    @Test("Contains returns false for unknown ID")
    func containsUnknown() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        let exists = await store.contains(id: "nonexistent")

        #expect(exists == false)
    }

    @Test("Remove deletes embedding")
    func removeEmbedding() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))
        let embedding = Embedding(vector: [0.1, 0.2, 0.3], metadata: EmbeddingMetadata.mock())
        let stored = try await store.store(embedding, id: "to-remove")

        try await store.remove(id: stored.id)

        let exists = await store.contains(id: stored.id)
        #expect(exists == false)
    }

    @Test("Clear removes all embeddings")
    func clearStore() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        // Add multiple embeddings
        for i in 0..<5 {
            let embedding = Embedding(
                vector: [Float(i), Float(i + 1), Float(i + 2)],
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding)
        }

        let countBefore = await store.count
        #expect(countBefore == 5)

        try await store.clear()

        let countAfter = await store.count
        #expect(countAfter == 0)
    }
}

// MARK: - Search Tests

@Suite("EmbeddingStore - Search")
struct EmbeddingStoreSearchTests {

    @Test("Search returns similar embeddings")
    func searchSimilar() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        // Store some embeddings
        let e1 = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let e2 = Embedding(vector: [0.9, 0.1, 0.0], metadata: EmbeddingMetadata.mock())
        let e3 = Embedding(vector: [0.0, 1.0, 0.0], metadata: EmbeddingMetadata.mock())

        _ = try await store.store(e1, id: "e1", text: "x-axis")
        _ = try await store.store(e2, id: "e2", text: "almost-x")
        _ = try await store.store(e3, id: "e3", text: "y-axis")

        // Search for x-axis direction
        let query = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let results = try await store.search(query, k: 3)

        #expect(results.count == 3)
        #expect(results[0].id == "e1") // Exact match
        #expect(results[1].id == "e2") // Second closest
    }

    @Test("Search respects k limit")
    func searchKLimit() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        // Store 10 embeddings
        for i in 0..<10 {
            let embedding = Embedding(
                vector: [Float(i) / 10, Float(10 - i) / 10, 0.5],
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding)
        }

        let query = Embedding(vector: [0.5, 0.5, 0.5], metadata: EmbeddingMetadata.mock())
        let results = try await store.search(query, k: 3)

        #expect(results.count == 3)
    }

    @Test("Search returns empty for empty store")
    func searchEmptyStore() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))
        let query = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())

        let results = try await store.search(query, k: 5)

        #expect(results.isEmpty)
    }

    @Test("Search returns stored text")
    func searchReturnsText() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))
        let embedding = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        _ = try await store.store(embedding, id: "e1", text: "Original text")

        let query = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let results = try await store.search(query, k: 1)

        #expect(results.first?.text == "Original text")
    }

    @Test("Search similarity scores are bounded")
    func searchSimilarityBounded() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        // Store normalized vectors
        let e1 = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let e2 = Embedding(vector: [-1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        _ = try await store.store(e1, id: "e1")
        _ = try await store.store(e2, id: "e2")

        let query = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let results = try await store.search(query, k: 2)

        for result in results {
            #expect(result.similarity >= 0)
            #expect(result.similarity <= 1)
        }
    }
}

// MARK: - Batch Operations Tests

@Suite("EmbeddingStore - Batch Operations")
struct EmbeddingStoreBatchTests {

    @Test("Store batch of embeddings")
    func storeBatch() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        let embeddings = [
            Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock()),
            Embedding(vector: [0.0, 1.0, 0.0], metadata: EmbeddingMetadata.mock()),
            Embedding(vector: [0.0, 0.0, 1.0], metadata: EmbeddingMetadata.mock())
        ]

        let stored = try await store.storeBatch(
            embeddings,
            texts: ["x", "y", "z"]
        )

        #expect(stored.count == 3)
        let count = await store.count
        #expect(count == 3)
    }

    @Test("Remove batch of embeddings")
    func removeBatch() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        // Store embeddings
        var ids: [String] = []
        for i in 0..<5 {
            let embedding = Embedding(
                vector: [Float(i), 0.0, 0.0],
                metadata: EmbeddingMetadata.mock()
            )
            let stored = try await store.store(embedding)
            ids.append(stored.id)
        }

        // Remove first 3
        try await store.removeBatch(Array(ids.prefix(3)))

        let count = await store.count
        #expect(count == 2)
    }
}

// MARK: - Reranking Tests

@Suite("EmbeddingStore - Reranking")
struct EmbeddingStoreRerankTests {

    @Test("ExactCosineRerank reorders by exact similarity")
    func exactCosineRerank() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        // Store embeddings
        let e1 = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let e2 = Embedding(vector: [0.9, 0.1, 0.0], metadata: EmbeddingMetadata.mock())
        _ = try await store.store(e1, id: "exact")
        _ = try await store.store(e2, id: "close")

        let query = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let results = try await store.search(
            query,
            k: 2,
            rerank: ExactCosineRerank()
        )

        #expect(results.first?.id == "exact")
    }

    @Test("ThresholdRerank filters by similarity")
    func thresholdRerank() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        // Store one similar and one dissimilar embedding
        let similar = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let dissimilar = Embedding(vector: [-1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        _ = try await store.store(similar, id: "similar")
        _ = try await store.store(dissimilar, id: "dissimilar")

        let query = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let results = try await store.search(
            query,
            k: 10,
            rerank: ThresholdRerank(minSimilarity: 0.5)
        )

        // Only similar should pass threshold
        #expect(results.count == 1)
        #expect(results.first?.id == "similar")
    }

    @Test("NoRerank passes through results")
    func noRerank() async throws {
        let query = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let candidates = [
            EmbeddingSearchResult(id: "a", distance: 0.1, similarity: 0.9),
            EmbeddingSearchResult(id: "b", distance: 0.2, similarity: 0.8)
        ]

        let reranked = try await NoRerank().rerank(
            query: query,
            candidates: candidates,
            k: 2
        )

        #expect(reranked.count == 2)
        #expect(reranked[0].id == "a")
    }

    @Test("CompositeRerank chains strategies")
    func compositeRerank() async throws {
        let query = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let candidates = [
            EmbeddingSearchResult(id: "a", distance: 0.1, similarity: 0.9),
            EmbeddingSearchResult(id: "b", distance: 0.3, similarity: 0.7),
            EmbeddingSearchResult(id: "c", distance: 0.6, similarity: 0.4)
        ]

        let composite = CompositeRerank(strategies: [
            ThresholdRerank(minSimilarity: 0.5),
            NoRerank()
        ])

        let reranked = try await composite.rerank(
            query: query,
            candidates: candidates,
            k: 10
        )

        // Only a and b should pass threshold
        #expect(reranked.count == 2)
    }
}

// MARK: - Index Configuration Tests

@Suite("EmbeddingStore - Configuration")
struct EmbeddingStoreConfigTests {

    @Test("Default config uses flat GPU index")
    func defaultConfigFlat() {
        let config = IndexConfiguration.default(dimension: 384)

        #expect(config.indexType == .flat)
        #expect(config.dimension == 384)
        #expect(config.metric == .euclidean)  // GPU-accelerated default
        #expect(config.storeText == true)
    }

    @Test("Exact config uses flat index")
    func exactConfigFlat() {
        let config = IndexConfiguration.exact(dimension: 768)

        #expect(config.indexType == .flat)
        #expect(config.dimension == 768)
    }

    @Test("Flat config with custom capacity")
    func flatConfigCapacity() {
        let config = IndexConfiguration.flat(dimension: 384, capacity: 50_000)

        #expect(config.indexType == .flat)
        #expect(config.capacity == 50_000)
    }

    @Test("Scalable config uses IVF")
    func scalableConfigIVF() {
        let config = IndexConfiguration.scalable(dimension: 384, expectedSize: 100_000)

        #expect(config.indexType == .ivf)
        #expect(config.nlist != nil)
        #expect(config.nprobe != nil)
    }

    @Test("IVF config with custom parameters")
    func ivfCustomParams() {
        let config = IndexConfiguration.ivf(
            dimension: 768,
            nlist: 512,
            nprobe: 32,
            capacity: 200_000
        )

        #expect(config.indexType == .ivf)
        #expect(config.nlist == 512)
        #expect(config.nprobe == 32)
        #expect(config.capacity == 200_000)
    }
}

// MARK: - Search Result Tests

@Suite("EmbeddingSearchResult")
struct EmbeddingSearchResultTests {

    @Test("Similarity is bounded 0-1")
    func similarityBounded() {
        let result = EmbeddingSearchResult(id: "test", distance: 0.5)

        #expect(result.similarity >= 0)
        #expect(result.similarity <= 1)
    }

    @Test("Results are comparable by similarity")
    func resultsComparable() {
        let r1 = EmbeddingSearchResult(id: "a", distance: 0.1, similarity: 0.9)
        let r2 = EmbeddingSearchResult(id: "b", distance: 0.3, similarity: 0.7)

        // Higher similarity should come first
        #expect(r1 > r2)
    }

    @Test("Best returns highest similarity")
    func bestResult() {
        let results = [
            EmbeddingSearchResult(id: "low", distance: 0.5, similarity: 0.5),
            EmbeddingSearchResult(id: "high", distance: 0.1, similarity: 0.9),
            EmbeddingSearchResult(id: "mid", distance: 0.3, similarity: 0.7)
        ]

        #expect(results.best?.id == "high")
    }

    @Test("Filtered by min similarity")
    func filteredResults() {
        let results = [
            EmbeddingSearchResult(id: "a", distance: 0.1, similarity: 0.9),
            EmbeddingSearchResult(id: "b", distance: 0.3, similarity: 0.7),
            EmbeddingSearchResult(id: "c", distance: 0.6, similarity: 0.4)
        ]

        let filtered = results.filtered(minSimilarity: 0.6)

        #expect(filtered.count == 2)
    }

    @Test("Texts extraction")
    func textsExtraction() {
        let results = [
            EmbeddingSearchResult(id: "a", distance: 0.1, text: "Hello"),
            EmbeddingSearchResult(id: "b", distance: 0.2, text: nil),
            EmbeddingSearchResult(id: "c", distance: 0.3, text: "World")
        ]

        #expect(results.texts == ["Hello", "World"])
    }
}

// MARK: - Mock Metadata Extension

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

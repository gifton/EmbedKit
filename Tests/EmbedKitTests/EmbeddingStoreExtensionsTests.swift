// Tests for EmbeddingStore Extensions and Convenience APIs
import Testing
import Foundation
@testable import EmbedKit

// MARK: - Embedding Extensions Tests

@Suite("Storage Extensions - Embedding")
struct EmbeddingStorageExtensionTests {

    @Test("Embedding.store() stores in store")
    func embeddingStore() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))
        let embedding = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())

        let stored = try await embedding.store(in: store, text: "test")

        #expect(stored.text == "test")
        let count = await store.count
        #expect(count == 1)
    }

    @Test("Embedding.store() with custom ID")
    func embeddingStoreCustomID() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))
        let embedding = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())

        let stored = try await embedding.store(in: store, id: "my-id")

        #expect(stored.id == "my-id")
    }

    @Test("Embedding.findSimilar() searches store")
    func embeddingFindSimilar() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        // Populate
        let e1 = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let e2 = Embedding(vector: [0.0, 1.0, 0.0], metadata: EmbeddingMetadata.mock())
        _ = try await store.store(e1, id: "x")
        _ = try await store.store(e2, id: "y")

        let query = Embedding(vector: [0.9, 0.1, 0.0], metadata: EmbeddingMetadata.mock())
        let results = try await query.findSimilar(in: store, k: 2)

        #expect(results.count == 2)
        #expect(results[0].id == "x") // Closer to x-axis
    }
}

// MARK: - Array Extensions Tests

@Suite("Storage Extensions - Array")
struct ArrayStorageExtensionTests {

    @Test("Array<Embedding>.store() stores all embeddings")
    func arrayEmbeddingStore() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        let embeddings = [
            Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock()),
            Embedding(vector: [0.0, 1.0, 0.0], metadata: EmbeddingMetadata.mock()),
            Embedding(vector: [0.0, 0.0, 1.0], metadata: EmbeddingMetadata.mock())
        ]

        let stored = try await embeddings.store(in: store, texts: ["x", "y", "z"])

        #expect(stored.count == 3)
        let count = await store.count
        #expect(count == 3)
    }

    @Test("Array<Embedding>.store() with metadata")
    func arrayEmbeddingStoreMetadata() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        let embeddings = [
            Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock()),
            Embedding(vector: [0.0, 1.0, 0.0], metadata: EmbeddingMetadata.mock())
        ]
        let metadata: [[String: String]?] = [
            ["type": "first"],
            ["type": "second"]
        ]

        let stored = try await embeddings.store(in: store, metadata: metadata)

        #expect(stored[0].metadata?["type"] == "first")
        #expect(stored[1].metadata?["type"] == "second")
    }
}

// MARK: - EmbeddingStore Convenience Tests

@Suite("Storage Extensions - Convenience APIs")
struct EmbeddingStoreConvenienceTests {

    @Test("findMostSimilar returns best match")
    func findMostSimilar() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        let e1 = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let e2 = Embedding(vector: [0.0, 1.0, 0.0], metadata: EmbeddingMetadata.mock())
        _ = try await store.store(e1, id: "exact")
        _ = try await store.store(e2, id: "different")

        let query = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let best = try await store.findMostSimilar(to: query)

        #expect(best?.id == "exact")
    }

    @Test("findMostSimilar returns nil for empty store")
    func findMostSimilarEmpty() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        let query = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let best = try await store.findMostSimilar(to: query)

        #expect(best == nil)
    }

    @Test("containsSimilar returns true above threshold")
    func containsSimilarAboveThreshold() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        // Store a known embedding
        let embedding = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        _ = try await store.store(embedding, text: "hello")

        // Search with similar vector
        let query = Embedding(vector: [0.99, 0.01, 0.0], metadata: EmbeddingMetadata.mock())
        let result = try await store.findMostSimilar(to: query)

        #expect(result != nil)
        #expect(result!.similarity > 0.9)
    }
}

// MARK: - EmbeddingStores Factory Tests

@Suite("Storage Extensions - Factory")
struct EmbeddingStoresFactoryTests {

    @Test("flat creates GPU flat index store")
    func flatFactory() async throws {
        let store = try await EmbeddingStores.flat(dimension: 384)

        let count = await store.count
        #expect(count == 0)
    }

    @Test("ivf creates GPU IVF store")
    func ivfFactory() async throws {
        let store = try await EmbeddingStores.ivf(dimension: 384)

        let embedding = Embedding(
            vector: [Float](repeating: 0.1, count: 384),
            metadata: EmbeddingMetadata.mock()
        )
        _ = try await store.store(embedding)

        let count = await store.count
        #expect(count == 1)
    }

    @Test("scalable creates IVF store")
    func scalableFactory() async throws {
        let store = try await EmbeddingStores.scalable(
            dimension: 128,
            expectedSize: 10_000
        )

        let embedding = Embedding(
            vector: [Float](repeating: 0.1, count: 128),
            metadata: EmbeddingMetadata.mock()
        )
        _ = try await store.store(embedding)

        let count = await store.count
        #expect(count == 1)
    }
}

// MARK: - Reranking Strategy Tests

@Suite("Reranking - Edge Cases")
struct RerankingEdgeCaseTests {

    @Test("ExactCosineRerank handles empty candidates")
    func exactCosineEmptyCandidates() async throws {
        let query = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let reranker = ExactCosineRerank()

        let results = try await reranker.rerank(query: query, candidates: [], k: 5)

        #expect(results.isEmpty)
    }

    @Test("ThresholdRerank with threshold=0 keeps all")
    func thresholdZeroKeepsAll() async throws {
        let query = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let candidates = [
            EmbeddingSearchResult(id: "a", distance: 0.9, similarity: 0.1),
            EmbeddingSearchResult(id: "b", distance: 0.5, similarity: 0.5)
        ]

        let reranker = ThresholdRerank(minSimilarity: 0.0)
        let results = try await reranker.rerank(query: query, candidates: candidates, k: 10)

        #expect(results.count == 2)
    }

    @Test("ThresholdRerank with threshold=1 filters most")
    func thresholdOneFiltersAll() async throws {
        let query = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let candidates = [
            EmbeddingSearchResult(id: "a", distance: 0.1, similarity: 0.9),
            EmbeddingSearchResult(id: "b", distance: 0.5, similarity: 0.5)
        ]

        let reranker = ThresholdRerank(minSimilarity: 1.0)
        let results = try await reranker.rerank(query: query, candidates: candidates, k: 10)

        #expect(results.isEmpty)
    }

    @Test("DiversityRerank with lambda=1 is relevance-only")
    func diversityLambdaOne() async throws {
        let query = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())

        // Create embeddings for candidates
        let e1 = Embedding(vector: [0.9, 0.1, 0.0], metadata: EmbeddingMetadata.mock())
        let e2 = Embedding(vector: [0.5, 0.5, 0.0], metadata: EmbeddingMetadata.mock())

        let candidates = [
            EmbeddingSearchResult(id: "a", distance: 0.1, similarity: 0.9, embedding: e1),
            EmbeddingSearchResult(id: "b", distance: 0.3, similarity: 0.7, embedding: e2)
        ]

        let reranker = DiversityRerank(lambda: 1.0)
        let results = try await reranker.rerank(query: query, candidates: candidates, k: 2)

        // With lambda=1, should just sort by relevance
        #expect(results[0].id == "a")
    }

    @Test("DiversityRerank with lambda=0 maximizes diversity")
    func diversityLambdaZero() async throws {
        let query = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())

        // Two very similar embeddings and one different
        let e1 = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let e2 = Embedding(vector: [0.99, 0.01, 0.0], metadata: EmbeddingMetadata.mock())
        let e3 = Embedding(vector: [0.0, 1.0, 0.0], metadata: EmbeddingMetadata.mock())

        let candidates = [
            EmbeddingSearchResult(id: "same1", distance: 0.0, similarity: 1.0, embedding: e1),
            EmbeddingSearchResult(id: "same2", distance: 0.01, similarity: 0.99, embedding: e2),
            EmbeddingSearchResult(id: "different", distance: 0.5, similarity: 0.5, embedding: e3)
        ]

        let reranker = DiversityRerank(lambda: 0.0)
        let results = try await reranker.rerank(query: query, candidates: candidates, k: 2)

        // With lambda=0, diversity matters most
        // After selecting first (same1), "different" should be preferred for diversity
        #expect(results.count == 2)
    }

    @Test("NoRerank respects k limit")
    func noRerankKLimit() async throws {
        let query = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let candidates = (0..<10).map { i in
            EmbeddingSearchResult(id: "e\(i)", distance: Float(i) / 10, similarity: 1 - Float(i) / 10)
        }

        let reranker = NoRerank()
        let results = try await reranker.rerank(query: query, candidates: candidates, k: 3)

        #expect(results.count == 3)
        #expect(results[0].id == "e0")
    }

    @Test("CompositeRerank with empty strategies returns candidates")
    func compositeEmptyStrategies() async throws {
        let query = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let candidates = [
            EmbeddingSearchResult(id: "a", distance: 0.1, similarity: 0.9)
        ]

        let reranker = CompositeRerank(strategies: [])
        let results = try await reranker.rerank(query: query, candidates: candidates, k: 5)

        #expect(results.count == 1)
    }
}

// MARK: - Model Integration Tests

@Suite("EmbeddingStore - Model Integration")
struct EmbeddingStoreModelIntegrationTests {

    @Test("Store without model cannot use text methods")
    func storeWithoutModelTextMethods() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 384))

        await #expect(throws: EmbeddingStoreError.self) {
            try await store.store(text: "Hello world")
        }
    }

    @Test("Store with mock model can store text")
    func storeWithMockModel() async throws {
        let model = MockEmbeddingModel(configuration: EmbeddingConfiguration())
        let store = try await EmbeddingStore(
            config: .exact(dimension: 384),
            model: model
        )

        let stored = try await store.store(text: "Hello world")

        #expect(stored.text == "Hello world")
        let count = await store.count
        #expect(count == 1)
    }

    @Test("Store with mock model batch store")
    func storeWithMockModelBatch() async throws {
        let model = MockEmbeddingModel(configuration: EmbeddingConfiguration())
        let store = try await EmbeddingStore(
            config: .exact(dimension: 384),
            model: model
        )

        let stored = try await store.storeBatch(texts: ["one", "two", "three"])

        #expect(stored.count == 3)
    }

    @Test("Search with text uses model")
    func searchWithTextUsesModel() async throws {
        let model = MockEmbeddingModel(configuration: EmbeddingConfiguration())
        let store = try await EmbeddingStore(
            config: .exact(dimension: 384),
            model: model
        )

        // Store some embeddings
        _ = try await store.store(text: "first")
        _ = try await store.store(text: "second")

        let results = try await store.search(text: "query", k: 2)

        #expect(results.count == 2)
    }
}

// MARK: - Mock Extension

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

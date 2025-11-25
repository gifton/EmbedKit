// Tests for EmbeddingStore Index Type Specific Behaviors
import Testing
import Foundation
@testable import EmbedKit

// MARK: - HNSW Index Tests

@Suite("EmbeddingStore - HNSW Index")
struct EmbeddingStoreHNSWTests {

    @Test("HNSW index creates with default config")
    func hnswDefaultConfig() async throws {
        let config = IndexConfiguration.default(dimension: 128)
        let store = try await EmbeddingStore(config: config)

        #expect(config.indexType == .hnsw)
        #expect(config.hnswConfig != nil)

        let count = await store.count
        #expect(count == 0)
    }

    @Test("HNSW index creates with fast config")
    func hnswFastConfig() async throws {
        let config = IndexConfiguration(
            indexType: .hnsw,
            dimension: 128,
            metric: .cosine,
            hnswConfig: .fast
        )
        let store = try await EmbeddingStore(config: config)

        // Store and search should work
        let embedding = Embedding(
            vector: [Float](repeating: 0.1, count: 128),
            metadata: EmbeddingMetadata.mock()
        )
        _ = try await store.store(embedding)

        let count = await store.count
        #expect(count == 1)
    }

    @Test("HNSW index creates with accurate config")
    func hnswAccurateConfig() async throws {
        let config = IndexConfiguration(
            indexType: .hnsw,
            dimension: 64,
            metric: .cosine,
            hnswConfig: .accurate
        )
        let store = try await EmbeddingStore(config: config)

        // Store multiple embeddings
        for i in 0..<20 {
            let embedding = Embedding(
                vector: (0..<64).map { _ in Float.random(in: 0...1) },
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding, id: "e\(i)")
        }

        let count = await store.count
        #expect(count == 20)
    }

    @Test("HNSW search returns approximate neighbors")
    func hnswApproximateSearch() async throws {
        let config = IndexConfiguration.default(dimension: 32)
        let store = try await EmbeddingStore(config: config)

        // Insert many vectors
        for i in 0..<100 {
            let embedding = Embedding(
                vector: (0..<32).map { j in Float(i + j) / 100 },
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding, id: "e\(i)")
        }

        let query = Embedding(
            vector: (0..<32).map { Float($0) / 100 },
            metadata: EmbeddingMetadata.mock()
        )
        let results = try await store.search(query, k: 10)

        #expect(results.count == 10)
        // Results should include e0 or nearby vectors
    }

    @Test("HNSW custom M and efConstruction")
    func hnswCustomParams() async throws {
        let hnswConfig = HNSWConfiguration(m: 8, efConstruction: 50, efSearch: 20)
        let config = IndexConfiguration(
            indexType: .hnsw,
            dimension: 16,
            metric: .cosine,
            hnswConfig: hnswConfig
        )
        let store = try await EmbeddingStore(config: config)

        for i in 0..<50 {
            let embedding = Embedding(
                vector: [Float](repeating: Float(i) / 50, count: 16),
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding)
        }

        let query = Embedding(
            vector: [Float](repeating: 0.5, count: 16),
            metadata: EmbeddingMetadata.mock()
        )
        let results = try await store.search(query, k: 5)

        #expect(results.count == 5)
    }
}

// MARK: - IVF Index Tests

@Suite("EmbeddingStore - IVF Index")
struct EmbeddingStoreIVFTests {

    @Test("IVF index creates with scalable config")
    func ivfScalableConfig() async throws {
        let config = IndexConfiguration.scalable(dimension: 64, expectedSize: 1000)
        let store = try await EmbeddingStore(config: config)

        #expect(config.indexType == .ivf)
        #expect(config.ivfConfig != nil)

        let count = await store.count
        #expect(count == 0)
    }

    @Test("IVF index stores and retrieves")
    func ivfStoreAndRetrieve() async throws {
        let config = IndexConfiguration.scalable(dimension: 32, expectedSize: 100)
        let store = try await EmbeddingStore(config: config)

        let embedding = Embedding(
            vector: [Float](repeating: 0.5, count: 32),
            metadata: EmbeddingMetadata.mock()
        )
        _ = try await store.store(embedding, id: "test")

        let exists = await store.contains(id: "test")
        #expect(exists == true)
    }

    @Test("IVF custom nlist and nprobe")
    func ivfCustomParams() async throws {
        let ivfConfig = IVFConfiguration(nlist: 16, nprobe: 4)
        let config = IndexConfiguration(
            indexType: .ivf,
            dimension: 32,
            metric: .euclidean,
            ivfConfig: ivfConfig
        )
        let store = try await EmbeddingStore(config: config)

        for i in 0..<30 {
            let embedding = Embedding(
                vector: (0..<32).map { _ in Float.random(in: 0...1) },
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding)
        }

        let count = await store.count
        #expect(count == 30)
    }

    @Test("IVF search with many vectors")
    func ivfSearchManyVectors() async throws {
        let config = IndexConfiguration.scalable(dimension: 16, expectedSize: 200)
        let store = try await EmbeddingStore(config: config)

        // Insert vectors
        for i in 0..<100 {
            let embedding = Embedding(
                vector: (0..<16).map { j in Float(i * 16 + j) / 1600 },
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding, id: "v\(i)")
        }

        let query = Embedding(
            vector: (0..<16).map { Float($0) / 16 },
            metadata: EmbeddingMetadata.mock()
        )
        let results = try await store.search(query, k: 10)

        #expect(results.count == 10)
    }
}

// MARK: - Flat Index Tests

@Suite("EmbeddingStore - Flat Index")
struct EmbeddingStoreFlatTests {

    @Test("Flat index provides exact results")
    func flatExactResults() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        // Store known vectors
        let e1 = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let e2 = Embedding(vector: [0.0, 1.0, 0.0], metadata: EmbeddingMetadata.mock())
        let e3 = Embedding(vector: [0.0, 0.0, 1.0], metadata: EmbeddingMetadata.mock())

        _ = try await store.store(e1, id: "x")
        _ = try await store.store(e2, id: "y")
        _ = try await store.store(e3, id: "z")

        let query = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let results = try await store.search(query, k: 3)

        // Flat index should always return exact nearest neighbor first
        #expect(results[0].id == "x")
        #expect(results[0].similarity > 0.99) // Should be ~1.0
    }

    @Test("Flat index search is deterministic")
    func flatDeterministic() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        for i in 0..<10 {
            let embedding = Embedding(
                vector: [Float(i) / 10, Float(10 - i) / 10, 0.5],
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding, id: "e\(i)")
        }

        let query = Embedding(vector: [0.5, 0.5, 0.5], metadata: EmbeddingMetadata.mock())

        // Run same search multiple times
        var firstResults: [String]?
        for _ in 0..<5 {
            let results = try await store.search(query, k: 5)
            let ids = results.map { $0.id }

            if let first = firstResults {
                #expect(ids == first) // Should be identical
            } else {
                firstResults = ids
            }
        }
    }

    @Test("Flat index handles large dimensions")
    func flatLargeDimensions() async throws {
        let dimension = 1024
        let store = try await EmbeddingStore(config: .exact(dimension: dimension))

        let embedding = Embedding(
            vector: [Float](repeating: 0.01, count: dimension),
            metadata: EmbeddingMetadata.mock()
        )
        _ = try await store.store(embedding)

        let query = Embedding(
            vector: [Float](repeating: 0.01, count: dimension),
            metadata: EmbeddingMetadata.mock()
        )
        let results = try await store.search(query, k: 1)

        #expect(results.count == 1)
        #expect(results[0].similarity > 0.99)
    }
}

// MARK: - Optimize Tests

@Suite("EmbeddingStore - Optimize")
struct EmbeddingStoreOptimizeTests {

    @Test("Optimize flat index is no-op")
    func optimizeFlatNoOp() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))
        let embedding = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        _ = try await store.store(embedding)

        // Should not throw
        try await store.optimize()

        let count = await store.count
        #expect(count == 1)
    }

    @Test("Optimize HNSW index")
    func optimizeHNSW() async throws {
        let store = try await EmbeddingStore(config: .default(dimension: 16))

        for i in 0..<20 {
            let embedding = Embedding(
                vector: [Float](repeating: Float(i) / 20, count: 16),
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding)
        }

        // Should not throw
        try await store.optimize()

        let count = await store.count
        #expect(count == 20)
    }
}

// MARK: - Statistics Tests

@Suite("EmbeddingStore - Statistics")
struct EmbeddingStoreStatisticsTests {

    @Test("Statistics reflect correct count")
    func statisticsCount() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        for i in 0..<7 {
            let embedding = Embedding(
                vector: [Float(i), 0.0, 0.0],
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding)
        }

        let stats = await store.statistics()

        #expect(stats.vectorCount == 7)
    }

    @Test("Statistics reflect correct dimension")
    func statisticsDimension() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 256))

        let stats = await store.statistics()

        #expect(stats.dimension == 256)
    }

    @Test("Statistics reflect index type")
    func statisticsIndexType() async throws {
        let flatStore = try await EmbeddingStore(config: .exact(dimension: 3))
        let hnswStore = try await EmbeddingStore(config: .default(dimension: 3))

        let flatStats = await flatStore.statistics()
        let hnswStats = await hnswStore.statistics()

        #expect(flatStats.indexType == "Flat")
        #expect(hnswStats.indexType == "HNSW")
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

// Tests for EmbeddingStore Index Type Specific Behaviors
import Testing
import Foundation
@testable import EmbedKit

// MARK: - IVF Index Tests

@Suite("EmbeddingStore - IVF Index")
struct EmbeddingStoreIVFTests {

    @Test("IVF index creates with scalable config")
    func ivfScalableConfig() async throws {
        let config = IndexConfiguration.scalable(dimension: 64, expectedSize: 1000)
        let store = try await EmbeddingStore(config: config)

        #expect(config.indexType == .ivf)
        #expect(config.nlist != nil)
        #expect(config.nprobe != nil)

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
        let config = IndexConfiguration.ivf(
            dimension: 32,
            nlist: 16,
            nprobe: 4,
            metric: .euclidean,
            capacity: 1000
        )
        let store = try await EmbeddingStore(config: config)

        for i in 0..<30 {
            let embedding = Embedding(
                vector: (0..<32).map { _ in Float.random(in: 0...1) },
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding)
        }

        // Train the IVF index
        try await store.train()

        let count = await store.count
        #expect(count == 30)
    }

    @Test("IVF search with many vectors")
    func ivfSearchManyVectors() async throws {
        let config = IndexConfiguration.ivf(
            dimension: 16,
            nlist: 8,
            nprobe: 4,
            capacity: 200
        )
        let store = try await EmbeddingStore(config: config)

        // Insert vectors
        for i in 0..<100 {
            let embedding = Embedding(
                vector: (0..<16).map { j in Float(i * 16 + j) / 1600 },
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding, id: "v\(i)")
        }

        // Train the IVF index
        try await store.train()

        let query = Embedding(
            vector: (0..<16).map { Float($0) / 16 },
            metadata: EmbeddingMetadata.mock()
        )
        let results = try await store.search(query, k: 10)

        #expect(results.count == 10)
    }

    @Test("IVF training is required for search")
    func ivfTrainingRequired() async throws {
        let config = IndexConfiguration.ivf(
            dimension: 16,
            nlist: 4,
            nprobe: 2,
            capacity: 50
        )
        let store = try await EmbeddingStore(config: config)

        // Insert vectors
        for i in 0..<20 {
            let embedding = Embedding(
                vector: (0..<16).map { _ in Float.random(in: 0...1) },
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding)
        }

        // Before training, isTrained should be false
        let trainedBefore = await store.isTrained
        #expect(trainedBefore == false)

        // Train
        try await store.train()

        // After training, isTrained should be true
        let trainedAfter = await store.isTrained
        #expect(trainedAfter == true)
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

    @Test("Flat index with default config")
    func flatDefaultConfig() async throws {
        let config = IndexConfiguration.default(dimension: 128)
        let store = try await EmbeddingStore(config: config)

        #expect(config.indexType == .flat)

        let count = await store.count
        #expect(count == 0)
    }

    @Test("Flat index search with many vectors")
    func flatSearchManyVectors() async throws {
        let store = try await EmbeddingStore(config: .flat(dimension: 32, capacity: 200))

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
}

// MARK: - Compaction Tests

@Suite("EmbeddingStore - Compaction")
struct EmbeddingStoreCompactionTests {

    @Test("Compact reclaims deleted space")
    func compactReclaimsSpace() async throws {
        let store = try await EmbeddingStore(config: .flat(dimension: 16, capacity: 50))

        // Store embeddings
        for i in 0..<20 {
            let embedding = Embedding(
                vector: [Float](repeating: Float(i) / 20, count: 16),
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding, id: "e\(i)")
        }

        // Remove some
        for i in 0..<10 {
            try await store.remove(id: "e\(i)")
        }

        let countBefore = await store.count
        #expect(countBefore == 10)

        // Compact
        let remapped = try await store.compact()

        // Some handles may have been remapped
        #expect(remapped >= 0)

        // Count should still be correct
        let countAfter = await store.count
        #expect(countAfter == 10)
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

    @Test("Statistics show memory usage")
    func statisticsMemory() async throws {
        let store = try await EmbeddingStore(config: .flat(dimension: 256, capacity: 100))

        for i in 0..<10 {
            let embedding = Embedding(
                vector: [Float](repeating: Float(i) / 10, count: 256),
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding)
        }

        let memUsage = await store.memoryUsage
        #expect(memUsage > 0)
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

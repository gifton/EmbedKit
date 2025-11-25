// Tests for EmbeddingStore Concurrency
import Testing
import Foundation
@testable import EmbedKit

// MARK: - Concurrent Access Tests

@Suite("EmbeddingStore - Concurrency")
struct EmbeddingStoreConcurrencyTests {

    @Test("Concurrent stores complete without error")
    func concurrentStores() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        await withTaskGroup(of: Void.self) { group in
            for i in 0..<50 {
                group.addTask {
                    let embedding = Embedding(
                        vector: [Float(i), Float(i + 1), Float(i + 2)],
                        metadata: EmbeddingMetadata.mock()
                    )
                    _ = try? await store.store(embedding, id: "concurrent-\(i)")
                }
            }
        }

        let count = await store.count
        #expect(count == 50)
    }

    @Test("Concurrent searches complete without error")
    func concurrentSearches() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        // Populate store
        for i in 0..<20 {
            let embedding = Embedding(
                vector: [Float(i) / 20, Float(20 - i) / 20, 0.5],
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding)
        }

        // Concurrent searches
        await withTaskGroup(of: [EmbeddingSearchResult].self) { group in
            for i in 0..<30 {
                group.addTask {
                    let query = Embedding(
                        vector: [Float(i % 10) / 10, 0.5, 0.5],
                        metadata: EmbeddingMetadata.mock()
                    )
                    return (try? await store.search(query, k: 5)) ?? []
                }
            }

            var allResults: [[EmbeddingSearchResult]] = []
            for await results in group {
                allResults.append(results)
            }

            #expect(allResults.count == 30)
            for results in allResults {
                #expect(results.count == 5)
            }
        }
    }

    @Test("Concurrent store and search")
    func concurrentStoreAndSearch() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        // Pre-populate
        for i in 0..<10 {
            let embedding = Embedding(
                vector: [Float(i), 0.0, 0.0],
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding)
        }

        await withTaskGroup(of: Void.self) { group in
            // Writers
            for i in 10..<30 {
                group.addTask {
                    let embedding = Embedding(
                        vector: [Float(i), 0.0, 0.0],
                        metadata: EmbeddingMetadata.mock()
                    )
                    _ = try? await store.store(embedding)
                }
            }

            // Readers
            for _ in 0..<20 {
                group.addTask {
                    let query = Embedding(
                        vector: [5.0, 0.0, 0.0],
                        metadata: EmbeddingMetadata.mock()
                    )
                    _ = try? await store.search(query, k: 5)
                }
            }
        }

        let count = await store.count
        #expect(count >= 10) // At least the initial ones
    }

    @Test("Concurrent removes are safe")
    func concurrentRemoves() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        // Populate
        var ids: [String] = []
        for i in 0..<30 {
            let embedding = Embedding(
                vector: [Float(i), 0.0, 0.0],
                metadata: EmbeddingMetadata.mock()
            )
            let stored = try await store.store(embedding)
            ids.append(stored.id)
        }

        // Concurrent removes
        await withTaskGroup(of: Void.self) { group in
            for id in ids.prefix(20) {
                group.addTask {
                    try? await store.remove(id: id)
                }
            }
        }

        let count = await store.count
        #expect(count == 10)
    }

    @Test("High volume concurrent operations")
    func highVolumeConcurrent() async throws {
        let store = try await EmbeddingStore(config: .default(dimension: 3)) // HNSW

        await withTaskGroup(of: Void.self) { group in
            // 100 concurrent stores
            for i in 0..<100 {
                group.addTask {
                    let embedding = Embedding(
                        vector: [Float.random(in: 0...1), Float.random(in: 0...1), Float.random(in: 0...1)],
                        metadata: EmbeddingMetadata.mock()
                    )
                    _ = try? await store.store(embedding, id: "high-\(i)")
                }
            }
        }

        let count = await store.count
        #expect(count == 100)
    }
}

// MARK: - Edge Cases Tests

@Suite("EmbeddingStore - Edge Cases")
struct EmbeddingStoreEdgeCaseTests {

    @Test("Search with k=0 returns empty")
    func searchKZero() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))
        let embedding = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        _ = try await store.store(embedding)

        let query = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let results = try await store.search(query, k: 0)

        #expect(results.isEmpty)
    }

    @Test("Search with k larger than store size")
    func searchKLargerThanStore() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        for i in 0..<5 {
            let embedding = Embedding(
                vector: [Float(i), 0.0, 0.0],
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding)
        }

        let query = Embedding(vector: [2.5, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let results = try await store.search(query, k: 100)

        #expect(results.count == 5) // Returns all available
    }

    @Test("Store with duplicate ID overwrites")
    func duplicateIDOverwrites() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        let e1 = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let e2 = Embedding(vector: [0.0, 1.0, 0.0], metadata: EmbeddingMetadata.mock())

        _ = try await store.store(e1, id: "same-id", text: "first")
        _ = try await store.store(e2, id: "same-id", text: "second")

        let count = await store.count
        #expect(count == 1) // Only one entry with this ID
    }

    @Test("Remove nonexistent ID is safe")
    func removeNonexistent() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        // Should not throw
        try await store.remove(id: "nonexistent-id")

        let count = await store.count
        #expect(count == 0)
    }

    @Test("Store and search with zero vector")
    func zeroVector() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        let zeroEmb = Embedding(vector: [0.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        _ = try await store.store(zeroEmb, id: "zero")

        let query = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let results = try await store.search(query, k: 1)

        #expect(results.count == 1)
    }

    @Test("Store with very long text")
    func veryLongText() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))
        let longText = String(repeating: "Hello world. ", count: 1000)

        let embedding = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let stored = try await store.store(embedding, text: longText)

        #expect(stored.text == longText)
    }

    @Test("Store with Unicode text")
    func unicodeText() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))
        let unicodeText = "Hello 世界 🌍 مرحبا العالم"

        let embedding = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let stored = try await store.store(embedding, text: unicodeText)

        #expect(stored.text == unicodeText)
    }

    @Test("Store with empty text")
    func emptyText() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        let embedding = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let stored = try await store.store(embedding, text: "")

        #expect(stored.text == "")
    }

    @Test("Store with special metadata keys")
    func specialMetadataKeys() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))
        let meta = [
            "key with spaces": "value",
            "key.with.dots": "value",
            "key:with:colons": "value",
            "": "empty key"
        ]

        let embedding = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let stored = try await store.store(embedding, metadata: meta)

        #expect(stored.metadata == meta)
    }

    @Test("Clear empty store is safe")
    func clearEmptyStore() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        // Should not throw
        try await store.clear()

        let count = await store.count
        #expect(count == 0)
    }

    @Test("Statistics returns valid data")
    func statisticsValid() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        for i in 0..<5 {
            let embedding = Embedding(
                vector: [Float(i), 0.0, 0.0],
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding)
        }

        let stats = await store.statistics()

        #expect(stats.vectorCount == 5)
        #expect(stats.dimension == 3)
    }
}

// MARK: - Distance Metric Tests

@Suite("EmbeddingStore - Distance Metrics")
struct EmbeddingStoreDistanceMetricTests {

    @Test("Cosine distance search orders correctly")
    func cosineSearch() async throws {
        let config = IndexConfiguration(
            indexType: .flat,
            dimension: 3,
            metric: .cosine
        )
        let store = try await EmbeddingStore(config: config)

        // Parallel vector (most similar for cosine)
        let parallel = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        // Opposite vector (least similar for cosine)
        let opposite = Embedding(vector: [-1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())

        _ = try await store.store(parallel, id: "parallel")
        _ = try await store.store(opposite, id: "opposite")

        let query = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let results = try await store.search(query, k: 2)

        #expect(results[0].id == "parallel")
    }

    @Test("Euclidean distance search orders correctly")
    func euclideanSearch() async throws {
        let config = IndexConfiguration(
            indexType: .flat,
            dimension: 3,
            metric: .euclidean
        )
        let store = try await EmbeddingStore(config: config)

        let near = Embedding(vector: [1.1, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let far = Embedding(vector: [5.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())

        _ = try await store.store(near, id: "near")
        _ = try await store.store(far, id: "far")

        let query = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let results = try await store.search(query, k: 2)

        #expect(results[0].id == "near")
    }

    @Test("Dot product search orders correctly")
    func dotProductSearch() async throws {
        let config = IndexConfiguration(
            indexType: .flat,
            dimension: 3,
            metric: .dotProduct
        )
        let store = try await EmbeddingStore(config: config)

        let highDot = Embedding(vector: [1.0, 1.0, 1.0], metadata: EmbeddingMetadata.mock())
        let lowDot = Embedding(vector: [0.1, 0.1, 0.1], metadata: EmbeddingMetadata.mock())

        _ = try await store.store(highDot, id: "high")
        _ = try await store.store(lowDot, id: "low")

        let query = Embedding(vector: [1.0, 1.0, 1.0], metadata: EmbeddingMetadata.mock())
        let results = try await store.search(query, k: 2)

        // Higher dot product should come first
        #expect(results[0].id == "high")
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

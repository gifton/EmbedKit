// Tests for EmbeddingStore GPU Acceleration
import Testing
import Foundation
@testable import EmbedKit

// MARK: - GPU Acceleration Tests

@Suite("EmbeddingStore - GPU Acceleration")
struct EmbeddingStoreGPUAccelerationTests {

    @Test("Store reports acceleration available")
    func accelerationAvailable() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        // GPU acceleration is always available (required for EmbeddingStore)
        // isAccelerationAvailable is a nonisolated computed property
        #expect(store.isAccelerationAvailable == true)
    }

    @Test("Compute distances using GPU")
    func computeDistancesGPU() async throws {
        let store = try await EmbeddingStore(config: .exact(dimension: 3))

        let query = Embedding(
            vector: [1.0, 0.0, 0.0],
            metadata: EmbeddingMetadata.mock()
        )
        let candidates = [
            Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock()),
            Embedding(vector: [0.0, 1.0, 0.0], metadata: EmbeddingMetadata.mock()),
            Embedding(vector: [0.5, 0.5, 0.0], metadata: EmbeddingMetadata.mock())
        ]

        let distances = try await store.computeDistances(from: query, to: candidates)

        #expect(distances.count == 3)
        // Cosine distance: 0 = identical, larger = more different
        #expect(distances[0] < 0.1) // Same vector
    }

    @Test("GPU statistics accessible")
    func gpuStatistics() async throws {
        let store = try await EmbeddingStore(config: .flat(dimension: 16, capacity: 100))

        // Store some vectors
        for i in 0..<10 {
            let embedding = Embedding(
                vector: [Float](repeating: Float(i) / 10, count: 16),
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding)
        }

        let stats = await store.statistics()

        #expect(stats.vectorCount == 10)
        #expect(stats.gpuMemoryBytes > 0)
    }

    @Test("Memory usage reflects stored vectors")
    func memoryUsage() async throws {
        let store = try await EmbeddingStore(config: .flat(dimension: 256, capacity: 50))

        let initialMemory = await store.memoryUsage

        // Store vectors
        for i in 0..<20 {
            let embedding = Embedding(
                vector: [Float](repeating: Float(i) / 20, count: 256),
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding)
        }

        let finalMemory = await store.memoryUsage

        // Memory should have increased
        #expect(finalMemory >= initialMemory)
    }
}

// MARK: - Large Batch GPU Tests

@Suite("EmbeddingStore - Large Batch GPU Operations")
struct EmbeddingStoreLargeBatchGPUTests {

    @Test("Large batch distance computation")
    func largeBatchDistances() async throws {
        let store = try await EmbeddingStore(config: .flat(dimension: 128, capacity: 200))

        let query = Embedding(
            vector: [Float](repeating: 0.1, count: 128),
            metadata: EmbeddingMetadata.mock()
        )

        // Create many candidates
        var candidates: [Embedding] = []
        for _ in 0..<100 {
            let vec = (0..<128).map { _ in Float.random(in: 0...1) }
            candidates.append(Embedding(vector: vec, metadata: EmbeddingMetadata.mock()))
        }

        let distances = try await store.computeDistances(from: query, to: candidates)

        #expect(distances.count == 100)
        for distance in distances {
            #expect(distance >= 0)
        }
    }

    @Test("Batch insert uses GPU")
    func batchInsertGPU() async throws {
        let store = try await EmbeddingStore(config: .flat(dimension: 64, capacity: 500))

        // Store many vectors
        for i in 0..<100 {
            let embedding = Embedding(
                vector: (0..<64).map { _ in Float.random(in: 0...1) },
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding, id: "e\(i)")
        }

        let count = await store.count
        #expect(count == 100)

        // Search should work
        let query = Embedding(
            vector: [Float](repeating: 0.5, count: 64),
            metadata: EmbeddingMetadata.mock()
        )
        let results = try await store.search(query, k: 10)
        #expect(results.count == 10)
    }
}

// MARK: - Persistence Tests

@Suite("EmbeddingStore - GPU Persistence")
struct EmbeddingStoreGPUPersistenceTests {

    @Test("Save and load recreates GPU index")
    func saveLoadRecreatesGPUIndex() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        // Create store and add vectors
        let store = try await EmbeddingStore(config: .flat(dimension: 8, capacity: 50))
        for i in 0..<5 {
            let embedding = Embedding(
                vector: [Float](repeating: Float(i) / 5, count: 8),
                metadata: EmbeddingMetadata.mock()
            )
            _ = try await store.store(embedding, id: "e\(i)", text: "doc\(i)")
        }

        // Save
        try await store.save(to: tempDir)

        // Load - this recreates a new GPU index
        let loaded = try await EmbeddingStore.load(from: tempDir)

        // Verify data preserved
        let count = await loaded.count
        #expect(count == 5)

        // Search should work
        let query = Embedding(
            vector: [Float](repeating: 0.0, count: 8),
            metadata: EmbeddingMetadata.mock()
        )
        let results = try await loaded.search(query, k: 3)
        #expect(results.count == 3)
        #expect(results[0].text == "doc0")
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

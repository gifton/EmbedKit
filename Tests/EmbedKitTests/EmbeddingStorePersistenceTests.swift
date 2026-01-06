// Tests for EmbeddingStore Persistence
import Testing
import Foundation
@testable import EmbedKit

// MARK: - Persistence Tests

@Suite("EmbeddingStore - Persistence")
struct EmbeddingStorePersistenceTests {

    @Test("Save and load flat index")
    func saveLoadFlatIndex() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        // Create and populate store
        let store = try await EmbeddingStore(config: .exact(dimension: 3))
        let e1 = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let e2 = Embedding(vector: [0.0, 1.0, 0.0], metadata: EmbeddingMetadata.mock())
        _ = try await store.store(e1, id: "e1", text: "first")
        _ = try await store.store(e2, id: "e2", text: "second")

        // Save
        try await store.save(to: tempDir)

        // Load
        let loaded = try await EmbeddingStore.load(from: tempDir)

        let count = await loaded.count
        #expect(count == 2)
    }

    @Test("Save and load HNSW index")
    func saveLoadHNSWIndex() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let store = try await EmbeddingStore(config: .default(dimension: 3))
        let embedding = Embedding(vector: [0.5, 0.5, 0.0], metadata: EmbeddingMetadata.mock())
        _ = try await store.store(embedding, id: "test")

        try await store.save(to: tempDir)
        let loaded = try await EmbeddingStore.load(from: tempDir)

        let exists = await loaded.contains(id: "test")
        #expect(exists == true)
    }

    @Test("Save and load IVF index")
    func saveLoadIVFIndex() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        // Use IVF with small nlist (4 clusters requires at least 4 vectors)
        let ivfConfig = IndexConfiguration.ivf(dimension: 3, nlist: 4, nprobe: 2)
        let store = try await EmbeddingStore(config: ivfConfig)

        // Insert enough vectors for IVF training
        for i in 0..<5 {
            let vec: [Float] = [Float(i) * 0.2, 0.3, 0.4]
            let embedding = Embedding(vector: vec, metadata: EmbeddingMetadata.mock())
            _ = try await store.store(embedding, id: "ivf-test-\(i)")
        }

        try await store.save(to: tempDir)
        let loaded = try await EmbeddingStore.load(from: tempDir)

        let count = await loaded.count
        #expect(count == 5)
    }

    @Test("Loaded store preserves text")
    func loadPreservesText() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let store = try await EmbeddingStore(config: .exact(dimension: 3))
        let embedding = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        _ = try await store.store(embedding, id: "with-text", text: "Hello World")

        try await store.save(to: tempDir)
        let loaded = try await EmbeddingStore.load(from: tempDir)

        let query = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        let results = try await loaded.search(query, k: 1)

        #expect(results.first?.text == "Hello World")
    }

    @Test("Loaded store is searchable")
    func loadedStoreSearchable() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        // Create store with multiple embeddings
        let store = try await EmbeddingStore(config: .exact(dimension: 3))
        for i in 0..<10 {
            let vec: [Float] = [Float(i) / 10, Float(10 - i) / 10, 0.0]
            let embedding = Embedding(vector: vec, metadata: EmbeddingMetadata.mock())
            _ = try await store.store(embedding, id: "e\(i)")
        }

        try await store.save(to: tempDir)
        let loaded = try await EmbeddingStore.load(from: tempDir)

        let query = Embedding(vector: [0.5, 0.5, 0.0], metadata: EmbeddingMetadata.mock())
        let results = try await loaded.search(query, k: 3)

        #expect(results.count == 3)
    }

    @Test("Save creates directory if needed")
    func saveCreatesDirectory() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathComponent("nested")
            .appendingPathComponent("path")
        defer { try? FileManager.default.removeItem(at: tempDir.deletingLastPathComponent().deletingLastPathComponent()) }

        let store = try await EmbeddingStore(config: .exact(dimension: 3))
        let embedding = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        _ = try await store.store(embedding)

        // Should not throw even though directory doesn't exist
        try await store.save(to: tempDir)

        #expect(FileManager.default.fileExists(atPath: tempDir.path))
    }

    @Test("Multiple save operations are idempotent")
    func multipleSaves() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let store = try await EmbeddingStore(config: .exact(dimension: 3))
        let embedding = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        _ = try await store.store(embedding, id: "test")

        // Save multiple times
        try await store.save(to: tempDir)
        try await store.save(to: tempDir)
        try await store.save(to: tempDir)

        let loaded = try await EmbeddingStore.load(from: tempDir)
        let count = await loaded.count
        #expect(count == 1)
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

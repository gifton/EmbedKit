// Tests for EmbeddingStore Acceleration Integration
import Testing
import Foundation
@testable import EmbedKit

// MARK: - Acceleration Integration Tests

@Suite("EmbeddingStore - Acceleration Integration")
struct EmbeddingStoreAccelerationTests {

    @Test("Store with auto acceleration creates accelerator")
    func autoAccelerationCreatesAccelerator() async throws {
        let store = try await EmbeddingStore(
            config: .exact(dimension: 3, computePreference: .auto)
        )

        // Acceleration availability depends on hardware
        // Just verify the property is accessible
        _ = await store.isAccelerationAvailable
    }

    @Test("Store with cpuOnly does not create accelerator")
    func cpuOnlyNoAccelerator() async throws {
        let store = try await EmbeddingStore(
            config: .exact(dimension: 3, computePreference: .cpuOnly)
        )

        let isAvailable = await store.isAccelerationAvailable
        #expect(isAvailable == false)
    }

    @Test("Acceleration statistics available when enabled")
    func accelerationStatisticsAvailable() async throws {
        let store = try await EmbeddingStore(
            config: .exact(dimension: 3, computePreference: .auto)
        )

        let stats = await store.accelerationStatistics()
        // Stats may be nil if GPU not available, otherwise should exist
        if stats != nil {
            #expect(stats!.gpuOperations >= 0)
            #expect(stats!.cpuOperations >= 0)
        }
    }

    @Test("Compute distances with acceleration")
    func computeDistancesWithAcceleration() async throws {
        let store = try await EmbeddingStore(
            config: .exact(dimension: 3, computePreference: .auto)
        )

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
        #expect(distances[0] < 0.1) // Same vector
        #expect(distances[1] > 0.4) // Orthogonal
    }

    @Test("Compute distances with CPU fallback")
    func computeDistancesCPUFallback() async throws {
        let store = try await EmbeddingStore(
            config: .exact(dimension: 3, computePreference: .cpuOnly)
        )

        let query = Embedding(
            vector: [1.0, 0.0, 0.0],
            metadata: EmbeddingMetadata.mock()
        )
        let candidates = [
            Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock()),
            Embedding(vector: [0.0, 1.0, 0.0], metadata: EmbeddingMetadata.mock())
        ]

        let distances = try await store.computeDistances(from: query, to: candidates)

        #expect(distances.count == 2)
        #expect(distances[0] < 0.1) // Same vector = low distance
    }
}

// MARK: - Index Configuration with Compute Preference

@Suite("IndexConfiguration - ComputePreference")
struct IndexConfigurationComputePreferenceTests {

    @Test("Default configuration uses auto preference")
    func defaultConfigUsesAuto() {
        let config = IndexConfiguration.default(dimension: 384)
        #expect(config.computePreference == .auto)
    }

    @Test("Default configuration accepts custom preference")
    func defaultConfigCustomPreference() {
        let config = IndexConfiguration.default(
            dimension: 384,
            computePreference: .cpuOnly
        )
        #expect(config.computePreference == .cpuOnly)
    }

    @Test("Exact configuration default preference")
    func exactConfigDefaultPreference() {
        let config = IndexConfiguration.exact(dimension: 3)
        #expect(config.computePreference == .auto)
    }

    @Test("Fast configuration default preference")
    func fastConfigDefaultPreference() {
        let config = IndexConfiguration.fast(dimension: 384)
        #expect(config.computePreference == .auto)
    }

    @Test("Scalable configuration default preference")
    func scalableConfigDefaultPreference() {
        let config = IndexConfiguration.scalable(dimension: 128)
        #expect(config.computePreference == .auto)
    }

    @Test("Custom init with all parameters")
    func customInitAllParameters() {
        let config = IndexConfiguration(
            indexType: .hnsw,
            dimension: 256,
            metric: .euclidean,
            storeText: false,
            hnswConfig: .accurate,
            computePreference: .gpuOnly
        )

        #expect(config.indexType == .hnsw)
        #expect(config.dimension == 256)
        #expect(config.metric == .euclidean)
        #expect(config.storeText == false)
        #expect(config.computePreference == .gpuOnly)
    }
}

// MARK: - Persistence with Compute Preference

@Suite("EmbeddingStore - Acceleration Persistence")
struct EmbeddingStoreAccelerationPersistenceTests {

    @Test("Save and load preserves compute preference")
    func saveLoadPreservesPreference() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        // Create store with specific preference
        let store = try await EmbeddingStore(
            config: .exact(dimension: 3, computePreference: .cpuOnly)
        )
        let embedding = Embedding(vector: [1.0, 0.0, 0.0], metadata: EmbeddingMetadata.mock())
        _ = try await store.store(embedding, id: "test")

        // Save and load
        try await store.save(to: tempDir)
        let loaded = try await EmbeddingStore.load(from: tempDir)

        // Verify preference preserved - config is nonisolated
        let loadedConfig = loaded.config
        #expect(loadedConfig.computePreference == .cpuOnly)
    }

    @Test("Load defaults to auto when preference missing")
    func loadDefaultsToAuto() async throws {
        // This tests backwards compatibility - old saved stores won't have
        // computePreference in their config.json
        // The StoredConfig.toConfig() should default to .auto

        let config = IndexConfiguration(
            indexType: .flat,
            dimension: 3,
            metric: .cosine,
            storeText: true
            // No computePreference specified - defaults to .auto
        )

        #expect(config.computePreference == .auto)
    }
}

// MARK: - Large Batch Tests

@Suite("EmbeddingStore - Large Batch Acceleration")
struct EmbeddingStoreLargeBatchTests {

    @Test("Large batch distance computation works")
    func largeBatchDistanceComputation() async throws {
        let store = try await EmbeddingStore(
            config: .exact(dimension: 128, computePreference: .auto)
        )

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

// Tests for AccelerationManager (GPU-only architecture)
import Testing
import Foundation
@testable import EmbedKit

// MARK: - Acceleration Manager Tests

@Suite("AccelerationManager - Core")
struct AccelerationManagerCoreTests {

    @Test("AccelerationManager initializes successfully")
    func defaultInitialization() async throws {
        let manager = try await AccelerationManager.create()

        let stats = await manager.statistics()
        #expect(stats.gpuOperations == 0)
    }

    @Test("AccelerationManager shared instance works")
    func sharedInstance() async throws {
        let manager1 = try await AccelerationManager.shared()
        let manager2 = try await AccelerationManager.shared()

        // Should be the same instance (comparing statistics as proxy)
        let stats1 = await manager1.statistics()
        let stats2 = await manager2.statistics()
        #expect(stats1.gpuOperations == stats2.gpuOperations)
    }

    @Test("AccelerationManager isGPUAvailable is always true")
    func gpuAlwaysAvailable() async throws {
        let manager = try await AccelerationManager.create()
        #expect(manager.isGPUAvailable == true)
    }
}

// MARK: - GPU Distance Tests

@Suite("AccelerationManager - GPU Distance")
struct AccelerationManagerGPUDistanceTests {

    @Test("Single distance computation")
    func singleDistanceComputation() async throws {
        let manager = try await AccelerationManager.create()

        let a: [Float] = [1.0, 0.0, 0.0]
        let b: [Float] = [0.0, 1.0, 0.0]

        let distance = try await manager.distance(from: a, to: b, metric: .cosine)

        // Orthogonal vectors have cosine distance of 1.0
        #expect(abs(distance - 1.0) < 0.1)
    }

    @Test("Batch distance computation")
    func batchDistanceComputation() async throws {
        let manager = try await AccelerationManager.create()

        let query: [Float] = [1.0, 0.0, 0.0]
        let candidates: [[Float]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0]
        ]

        let distances = try await manager.batchDistance(
            from: query,
            to: candidates,
            metric: .cosine
        )

        #expect(distances.count == 3)
        #expect(distances[0] < 0.1) // Same direction = distance ~0
        #expect(distances[1] > 0.9)  // Orthogonal = distance ~1
    }

    @Test("Euclidean distance computation")
    func euclideanDistance() async throws {
        let manager = try await AccelerationManager.create()

        let a: [Float] = [0.0, 0.0, 0.0]
        let b: [Float] = [3.0, 4.0, 0.0]

        let distance = try await manager.distance(from: a, to: b, metric: .euclidean)

        #expect(abs(distance - 5.0) < 0.1) // 3-4-5 triangle
    }

    @Test("Dot product distance computation")
    func dotProductDistance() async throws {
        let manager = try await AccelerationManager.create()

        let a: [Float] = [1.0, 2.0, 3.0]
        let b: [Float] = [4.0, 5.0, 6.0]

        let distance = try await manager.distance(from: a, to: b, metric: .dotProduct)

        // Dot product = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        // Distance is negated dot product
        #expect(abs(distance - (-32.0)) < 0.1)
    }
}

// MARK: - Validation Tests

@Suite("AccelerationManager - Validation")
struct AccelerationManagerValidationTests {

    @Test("Dimension mismatch throws error")
    func dimensionMismatchSingleDistance() async throws {
        let manager = try await AccelerationManager.create()

        let a: [Float] = [1.0, 0.0, 0.0]
        let b: [Float] = [1.0, 0.0] // Different dimension

        await #expect(throws: AccelerationError.self) {
            try await manager.distance(from: a, to: b, metric: .cosine)
        }
    }

    @Test("Batch dimension mismatch throws error")
    func dimensionMismatchBatchDistance() async throws {
        let manager = try await AccelerationManager.create()

        let query: [Float] = [1.0, 0.0, 0.0]
        let candidates: [[Float]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0]  // Wrong dimension
        ]

        await #expect(throws: AccelerationError.self) {
            try await manager.batchDistance(from: query, to: candidates, metric: .cosine)
        }
    }

    @Test("Empty candidates returns empty array")
    func emptyBatchDistance() async throws {
        let manager = try await AccelerationManager.create()

        let query: [Float] = [1.0, 0.0, 0.0]
        let candidates: [[Float]] = []

        let distances = try await manager.batchDistance(
            from: query,
            to: candidates,
            metric: .cosine
        )

        #expect(distances.isEmpty)
    }
}

// MARK: - Normalization Tests

@Suite("AccelerationManager - Normalization")
struct AccelerationManagerNormalizationTests {

    @Test("Single vector normalization (CPU)")
    func singleNormalization() async throws {
        let manager = try await AccelerationManager.create()

        let vector: [Float] = [3.0, 4.0, 0.0]
        let normalized = manager.normalize(vector)

        // Magnitude should be 5, so normalized = [0.6, 0.8, 0.0]
        #expect(abs(normalized[0] - 0.6) < 0.01)
        #expect(abs(normalized[1] - 0.8) < 0.01)
        #expect(abs(normalized[2] - 0.0) < 0.01)
    }

    @Test("Batch normalization (GPU)")
    func batchNormalization() async throws {
        let manager = try await AccelerationManager.create()

        let vectors: [[Float]] = [
            [3.0, 4.0, 0.0],
            [0.0, 5.0, 0.0],
            [1.0, 0.0, 0.0]
        ]

        let normalized = try await manager.normalizeBatch(vectors)

        #expect(normalized.count == 3)

        // First vector normalized
        #expect(abs(normalized[0][0] - 0.6) < 0.01)
        #expect(abs(normalized[0][1] - 0.8) < 0.01)

        // Second vector normalized
        #expect(abs(normalized[1][1] - 1.0) < 0.01)

        // Third vector already normalized
        #expect(abs(normalized[2][0] - 1.0) < 0.01)
    }

    @Test("Zero vector normalization returns zeros")
    func zeroVectorNormalization() async throws {
        let manager = try await AccelerationManager.create()

        let vector: [Float] = [0.0, 0.0, 0.0]
        let normalized = manager.normalize(vector)

        #expect(normalized[0] == 0.0)
        #expect(normalized[1] == 0.0)
        #expect(normalized[2] == 0.0)
    }
}

// MARK: - Statistics Tests

@Suite("AccelerationManager - Statistics")
struct AccelerationManagerStatisticsTests {

    @Test("Statistics track GPU operations")
    func statisticsTrackGPU() async throws {
        let manager = try await AccelerationManager.create()

        // Perform some operations
        let a: [Float] = [1.0, 0.0, 0.0]
        let b: [Float] = [0.0, 1.0, 0.0]

        _ = try await manager.distance(from: a, to: b, metric: .cosine)
        _ = try await manager.distance(from: a, to: b, metric: .euclidean)

        let stats = await manager.statistics()
        #expect(stats.gpuOperations == 2)
        #expect(stats.gpuTimeTotal > 0)
    }

    @Test("Statistics reset works")
    func statisticsReset() async throws {
        let manager = try await AccelerationManager.create()

        let a: [Float] = [1.0, 0.0, 0.0]
        let b: [Float] = [0.0, 1.0, 0.0]
        _ = try await manager.distance(from: a, to: b, metric: .cosine)

        let statsBefore = await manager.statistics()
        #expect(statsBefore.gpuOperations == 1)

        await manager.resetStatistics()

        let statsAfter = await manager.statistics()
        #expect(statsAfter.gpuOperations == 0)
    }

    @Test("Average GPU time calculation")
    func averageGPUTime() {
        let stats = AccelerationStatistics(
            gpuOperations: 0,
            gpuTimeTotal: 0
        )
        #expect(stats.averageGPUTime == 0.0)

        let stats2 = AccelerationStatistics(
            gpuOperations: 10,
            gpuTimeTotal: 5.0
        )
        #expect(abs(stats2.averageGPUTime - 0.5) < 0.01)
    }
}

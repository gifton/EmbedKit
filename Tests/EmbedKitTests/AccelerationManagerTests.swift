// Tests for AccelerationManager
import Testing
import Foundation
@testable import EmbedKit

// MARK: - Acceleration Manager Tests

@Suite("AccelerationManager - Core")
struct AccelerationManagerCoreTests {

    @Test("AccelerationManager initializes with default preference")
    func defaultInitialization() async {
        let manager = await AccelerationManager()

        let stats = await manager.statistics()
        #expect(stats.gpuOperations == 0)
        #expect(stats.cpuOperations == 0)
    }

    @Test("AccelerationManager initializes with CPU-only preference")
    func cpuOnlyInitialization() async {
        let manager = await AccelerationManager(preference: .cpuOnly)

        // GPU should not be initialized in CPU-only mode
        // but isGPUAvailable reflects hardware, not preference
        let stats = await manager.statistics()
        #expect(stats.gpuOperations == 0)
    }

    @Test("AccelerationManager shared instance works")
    func sharedInstance() async {
        let manager1 = await AccelerationManager.shared()
        let manager2 = await AccelerationManager.shared()

        // Should be the same instance (comparing statistics as proxy)
        let stats1 = await manager1.statistics()
        let stats2 = await manager2.statistics()
        #expect(stats1.gpuOperations == stats2.gpuOperations)
    }

    @Test("AccelerationManager preference can be changed")
    func preferenceChange() async {
        let manager = await AccelerationManager(preference: .cpuOnly)
        await manager.setPreference(.auto)

        // Should not throw
        let stats = await manager.statistics()
        #expect(stats.gpuOperations >= 0)
    }

    @Test("AccelerationManager thresholds can be updated")
    func thresholdsUpdate() async {
        let manager = await AccelerationManager()

        let newThresholds = AccelerationThresholds(
            minCandidatesForGPU: 500,
            minDimensionForGPU: 32
        )
        await manager.setThresholds(newThresholds)

        // Should not throw
        let stats = await manager.statistics()
        #expect(stats.gpuOperations >= 0)
    }
}

// MARK: - CPU Distance Tests

@Suite("AccelerationManager - CPU Distance")
struct AccelerationManagerCPUDistanceTests {

    @Test("Single distance computation uses CPU")
    func singleDistanceUsesCPU() async throws {
        let manager = await AccelerationManager(preference: .cpuOnly)

        let a: [Float] = [1.0, 0.0, 0.0]
        let b: [Float] = [0.0, 1.0, 0.0]

        let distance = try await manager.distance(from: a, to: b, metric: .cosine)

        // Orthogonal vectors have cosine distance of 1.0
        #expect(abs(distance - 1.0) < 0.01)

        let stats = await manager.statistics()
        #expect(stats.cpuOperations == 1)
        #expect(stats.gpuOperations == 0)
    }

    @Test("Batch distance with CPU-only mode")
    func batchDistanceCPUOnly() async throws {
        let manager = await AccelerationManager(preference: .cpuOnly)

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
        #expect(distances[0] < 0.01) // Same direction = distance ~0
        #expect(distances[1] > 0.9)  // Orthogonal = distance ~1

        let stats = await manager.statistics()
        #expect(stats.cpuOperations == 1)
        #expect(stats.gpuOperations == 0)
    }

    @Test("Euclidean distance computation")
    func euclideanDistance() async throws {
        let manager = await AccelerationManager(preference: .cpuOnly)

        let a: [Float] = [0.0, 0.0, 0.0]
        let b: [Float] = [3.0, 4.0, 0.0]

        let distance = try await manager.distance(from: a, to: b, metric: .euclidean)

        #expect(abs(distance - 5.0) < 0.01) // 3-4-5 triangle
    }

    @Test("Dot product distance computation")
    func dotProductDistance() async throws {
        let manager = await AccelerationManager(preference: .cpuOnly)

        let a: [Float] = [1.0, 2.0, 3.0]
        let b: [Float] = [4.0, 5.0, 6.0]

        let distance = try await manager.distance(from: a, to: b, metric: .dotProduct)

        // Dot product = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        // Distance is negated dot product
        #expect(abs(distance - (-32.0)) < 0.01)
    }

    @Test("Manhattan distance computation")
    func manhattanDistance() async throws {
        let manager = await AccelerationManager(preference: .cpuOnly)

        let a: [Float] = [1.0, 2.0, 3.0]
        let b: [Float] = [4.0, 6.0, 8.0]

        let distance = try await manager.distance(from: a, to: b, metric: .manhattan)

        // Manhattan = |1-4| + |2-6| + |3-8| = 3 + 4 + 5 = 12
        #expect(abs(distance - 12.0) < 0.01)
    }

    @Test("Chebyshev distance computation")
    func chebyshevDistance() async throws {
        let manager = await AccelerationManager(preference: .cpuOnly)

        let a: [Float] = [1.0, 2.0, 3.0]
        let b: [Float] = [4.0, 10.0, 5.0]

        let distance = try await manager.distance(from: a, to: b, metric: .chebyshev)

        // Chebyshev = max(|1-4|, |2-10|, |3-5|) = max(3, 8, 2) = 8
        #expect(abs(distance - 8.0) < 0.01)
    }
}

// MARK: - Validation Tests

@Suite("AccelerationManager - Validation")
struct AccelerationManagerValidationTests {

    @Test("Dimension mismatch throws error")
    func dimensionMismatchSingleDistance() async throws {
        let manager = await AccelerationManager(preference: .cpuOnly)

        let a: [Float] = [1.0, 0.0, 0.0]
        let b: [Float] = [1.0, 0.0] // Different dimension

        await #expect(throws: AccelerationError.self) {
            try await manager.distance(from: a, to: b, metric: .cosine)
        }
    }

    @Test("Batch dimension mismatch throws error")
    func dimensionMismatchBatchDistance() async throws {
        let manager = await AccelerationManager(preference: .cpuOnly)

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
        let manager = await AccelerationManager(preference: .cpuOnly)

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

    @Test("Single vector normalization")
    func singleNormalization() async {
        let manager = await AccelerationManager(preference: .cpuOnly)

        let vector: [Float] = [3.0, 4.0, 0.0]
        let normalized = await manager.normalize(vector)

        // Magnitude should be 5, so normalized = [0.6, 0.8, 0.0]
        #expect(abs(normalized[0] - 0.6) < 0.01)
        #expect(abs(normalized[1] - 0.8) < 0.01)
        #expect(abs(normalized[2] - 0.0) < 0.01)
    }

    @Test("Batch normalization")
    func batchNormalization() async throws {
        let manager = await AccelerationManager(preference: .cpuOnly)

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

    @Test("Zero vector normalization returns original")
    func zeroVectorNormalization() async {
        let manager = await AccelerationManager(preference: .cpuOnly)

        let vector: [Float] = [0.0, 0.0, 0.0]
        let normalized = await manager.normalize(vector)

        #expect(normalized[0] == 0.0)
        #expect(normalized[1] == 0.0)
        #expect(normalized[2] == 0.0)
    }
}

// MARK: - Statistics Tests

@Suite("AccelerationManager - Statistics")
struct AccelerationManagerStatisticsTests {

    @Test("Statistics track CPU operations")
    func statisticsTrackCPU() async throws {
        let manager = await AccelerationManager(preference: .cpuOnly)

        // Perform some operations
        let a: [Float] = [1.0, 0.0, 0.0]
        let b: [Float] = [0.0, 1.0, 0.0]

        _ = try await manager.distance(from: a, to: b, metric: .cosine)
        _ = try await manager.distance(from: a, to: b, metric: .euclidean)

        let stats = await manager.statistics()
        #expect(stats.cpuOperations == 2)
        #expect(stats.cpuTimeTotal > 0)
    }

    @Test("Statistics reset works")
    func statisticsReset() async throws {
        let manager = await AccelerationManager(preference: .cpuOnly)

        let a: [Float] = [1.0, 0.0, 0.0]
        let b: [Float] = [0.0, 1.0, 0.0]
        _ = try await manager.distance(from: a, to: b, metric: .cosine)

        let statsBefore = await manager.statistics()
        #expect(statsBefore.cpuOperations == 1)

        await manager.resetStatistics()

        let statsAfter = await manager.statistics()
        #expect(statsAfter.cpuOperations == 0)
        #expect(statsAfter.gpuOperations == 0)
    }

    @Test("GPU utilization calculation")
    func gpuUtilizationCalculation() async {
        // With no operations, utilization should be 0
        let stats = AccelerationStatistics(
            gpuOperations: 0,
            cpuOperations: 0
        )
        #expect(stats.gpuUtilization == 0.0)

        // With some GPU operations
        let stats2 = AccelerationStatistics(
            gpuOperations: 3,
            cpuOperations: 7
        )
        #expect(abs(stats2.gpuUtilization - 0.3) < 0.01)
    }
}

// MARK: - Thresholds Tests

@Suite("AccelerationManager - Thresholds")
struct AccelerationManagerThresholdsTests {

    @Test("Default thresholds are reasonable")
    func defaultThresholds() {
        let thresholds = AccelerationThresholds.default

        #expect(thresholds.minCandidatesForGPU == 1000)
        #expect(thresholds.minDimensionForGPU == 64)
        #expect(thresholds.minBatchForNormalization == 100)
    }

    @Test("Aggressive thresholds lower GPU barrier")
    func aggressiveThresholds() {
        let thresholds = AccelerationThresholds.aggressive

        #expect(thresholds.minCandidatesForGPU < AccelerationThresholds.default.minCandidatesForGPU)
        #expect(thresholds.minDimensionForGPU < AccelerationThresholds.default.minDimensionForGPU)
    }

    @Test("Conservative thresholds raise GPU barrier")
    func conservativeThresholds() {
        let thresholds = AccelerationThresholds.conservative

        #expect(thresholds.minCandidatesForGPU > AccelerationThresholds.default.minCandidatesForGPU)
        #expect(thresholds.minDimensionForGPU > AccelerationThresholds.default.minDimensionForGPU)
    }
}

// MARK: - Compute Preference Tests

@Suite("ComputePreference")
struct ComputePreferenceTests {

    @Test("ComputePreference rawValue encoding")
    func rawValueEncoding() {
        #expect(ComputePreference.auto.rawValue == "auto")
        #expect(ComputePreference.cpuOnly.rawValue == "cpuOnly")
        #expect(ComputePreference.gpuOnly.rawValue == "gpuOnly")
    }

    @Test("ComputePreference rawValue decoding")
    func rawValueDecoding() {
        #expect(ComputePreference(rawValue: "auto") == .auto)
        #expect(ComputePreference(rawValue: "cpuOnly") == .cpuOnly)
        #expect(ComputePreference(rawValue: "gpuOnly") == .gpuOnly)
        #expect(ComputePreference(rawValue: "invalid") == nil)
    }

    @Test("ComputePreference is CaseIterable")
    func caseIterable() {
        let allCases = ComputePreference.allCases
        #expect(allCases.count == 3)
        #expect(allCases.contains(.auto))
        #expect(allCases.contains(.cpuOnly))
        #expect(allCases.contains(.gpuOnly))
    }
}

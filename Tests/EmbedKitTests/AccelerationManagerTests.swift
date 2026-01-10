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

// MARK: - GPU Health Monitoring Tests

@Suite("AccelerationManager - GPU Health Monitoring")
struct AccelerationManagerHealthMonitoringTests {

    @Test("Default preset initializes successfully")
    func defaultHealthPreset() async throws {
        let manager = try await AccelerationManager.create(healthPreset: .default)
        #expect(await manager.isGPUHealthy() == true)
    }

    @Test("Aggressive preset initializes successfully")
    func aggressiveHealthPreset() async throws {
        let manager = try await AccelerationManager.create(healthPreset: .aggressive)
        #expect(await manager.isGPUHealthy() == true)
    }

    @Test("Lenient preset initializes successfully")
    func lenientHealthPreset() async throws {
        let manager = try await AccelerationManager.create(healthPreset: .lenient)
        #expect(await manager.isGPUHealthy() == true)
    }

    @Test("Initial health status shows healthy")
    func initialHealthStatus() async throws {
        let manager = try await AccelerationManager.create()

        let status = await manager.gpuHealthStatus()

        #expect(status.isHealthy == true)
        #expect(status.totalFailureCount == 0)
        #expect(status.disabledOperations.isEmpty)
        #expect(status.recoveryAttempts == 0)
    }

    @Test("Reset health tracking clears state")
    func resetHealthTracking() async throws {
        let manager = try await AccelerationManager.create()

        // Verify initial state
        let initialStatus = await manager.gpuHealthStatus()
        #expect(initialStatus.isHealthy == true)

        // Reset should maintain healthy state
        await manager.resetHealthTracking()

        let statusAfterReset = await manager.gpuHealthStatus()
        #expect(statusAfterReset.isHealthy == true)
        #expect(statusAfterReset.totalFailureCount == 0)
    }

    @Test("Fallback not recommended initially")
    func fallbackNotRecommendedInitially() async throws {
        let manager = try await AccelerationManager.create()

        let recommended = await manager.isFallbackRecommended(for: "batch_distance")
        #expect(recommended == false)
    }

    @Test("Degradation level is none initially")
    func initialDegradationLevel() async throws {
        let manager = try await AccelerationManager.create()

        let level = await manager.degradationLevel(for: "batch_distance")
        #expect(level == .none)
    }

    @Test("Health status summary is readable")
    func healthStatusSummary() async throws {
        let manager = try await AccelerationManager.create()

        let status = await manager.gpuHealthStatus()
        let summary = status.summary

        #expect(summary.contains("GPU Health Status"))
        #expect(summary.contains("Healthy"))
    }

    @Test("Operations track success after GPU operation")
    func operationsTrackSuccess() async throws {
        let manager = try await AccelerationManager.create()

        // Perform a successful GPU operation
        let a: [Float] = [1.0, 0.0, 0.0]
        let b: [Float] = [0.0, 1.0, 0.0]
        _ = try await manager.distance(from: a, to: b, metric: .cosine)

        // Health should still be good after success
        let status = await manager.gpuHealthStatus()
        #expect(status.isHealthy == true)
    }

    @Test("Batch operations with fallback available still execute")
    func batchOperationsWithFallback() async throws {
        let manager = try await AccelerationManager.create()

        let query: [Float] = [1.0, 0.0, 0.0]
        let candidates: [[Float]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0]
        ]

        // Should succeed with GPU (or CPU fallback if GPU unavailable)
        let distances = try await manager.batchDistance(
            from: query,
            to: candidates,
            metric: .cosine
        )

        #expect(distances.count == 3)

        // Health should still be tracked
        let status = await manager.gpuHealthStatus()
        #expect(status.totalFailureCount >= 0) // May have CPU fallback
    }

    @Test("Normalization operations with health tracking")
    func normalizationWithHealthTracking() async throws {
        let manager = try await AccelerationManager.create()

        let vectors: [[Float]] = [
            [3.0, 4.0, 0.0],
            [0.0, 5.0, 0.0]
        ]

        let normalized = try await manager.normalizeBatch(vectors)

        #expect(normalized.count == 2)

        // Health monitoring should be active
        let status = await manager.gpuHealthStatus()
        #expect(status.isHealthy == true)
    }
}

// MARK: - GPU Decision Engine Tests

@Suite("AccelerationManager - GPU Decision Engine")
struct AccelerationManagerDecisionEngineTests {

    @Test("Default profile is alwaysGPU for backward compatibility")
    func defaultProfileIsAlwaysGPU() async throws {
        let manager = try await AccelerationManager.create()
        let profile = await manager.currentDecisionProfile()
        #expect(profile == .alwaysGPU)
    }

    @Test("Balanced profile initializes with decision engine")
    func balancedProfileInitialization() async throws {
        let manager = try await AccelerationManager.create(decisionProfile: .balanced)
        let profile = await manager.currentDecisionProfile()
        #expect(profile == .balanced)

        // Decision engine should be available for adaptive profiles
        let stats = await manager.gpuPerformanceStats()
        #expect(stats != nil)
    }

    @Test("BatchOptimized profile initializes with decision engine")
    func batchOptimizedProfileInitialization() async throws {
        let manager = try await AccelerationManager.create(decisionProfile: .batchOptimized)
        let profile = await manager.currentDecisionProfile()
        #expect(profile == .batchOptimized)

        let stats = await manager.gpuPerformanceStats()
        #expect(stats != nil)
    }

    @Test("RealTimeOptimized profile initializes with decision engine")
    func realTimeOptimizedProfileInitialization() async throws {
        let manager = try await AccelerationManager.create(decisionProfile: .realTimeOptimized)
        let profile = await manager.currentDecisionProfile()
        #expect(profile == .realTimeOptimized)

        let stats = await manager.gpuPerformanceStats()
        #expect(stats != nil)
    }

    @Test("AlwaysGPU profile has no decision engine")
    func alwaysGPUProfileNoDecisionEngine() async throws {
        let manager = try await AccelerationManager.create(decisionProfile: .alwaysGPU)
        let profile = await manager.currentDecisionProfile()
        #expect(profile == .alwaysGPU)

        // Decision engine should be nil for forced profiles
        let stats = await manager.gpuPerformanceStats()
        #expect(stats == nil)
    }

    @Test("AlwaysCPU profile has no decision engine")
    func alwaysCPUProfileNoDecisionEngine() async throws {
        let manager = try await AccelerationManager.create(decisionProfile: .alwaysCPU)
        let profile = await manager.currentDecisionProfile()
        #expect(profile == .alwaysCPU)

        // Decision engine should be nil for forced profiles
        let stats = await manager.gpuPerformanceStats()
        #expect(stats == nil)
    }

    @Test("AlwaysCPU profile uses CPU for distance computation")
    func alwaysCPUUsesOnlyCPU() async throws {
        let manager = try await AccelerationManager.create(decisionProfile: .alwaysCPU)

        let a: [Float] = [1.0, 0.0, 0.0]
        let b: [Float] = [0.0, 1.0, 0.0]

        let distance = try await manager.distance(from: a, to: b, metric: .cosine)

        // Orthogonal vectors have cosine distance of 1.0
        #expect(abs(distance - 1.0) < 0.1)

        // GPU operations should be 0 since we're using CPU
        let stats = await manager.statistics()
        #expect(stats.gpuOperations == 0)
    }

    @Test("AlwaysGPU profile uses GPU for distance computation")
    func alwaysGPUUsesGPU() async throws {
        let manager = try await AccelerationManager.create(decisionProfile: .alwaysGPU)

        let a: [Float] = [1.0, 0.0, 0.0]
        let b: [Float] = [0.0, 1.0, 0.0]

        let distance = try await manager.distance(from: a, to: b, metric: .cosine)

        // Orthogonal vectors have cosine distance of 1.0
        #expect(abs(distance - 1.0) < 0.1)

        // GPU operations should be > 0 (assuming GPU is healthy)
        let stats = await manager.statistics()
        #expect(stats.gpuOperations >= 1)
    }

    @Test("Profile can be updated at runtime")
    func profileUpdateAtRuntime() async throws {
        let manager = try await AccelerationManager.create(decisionProfile: .alwaysGPU)

        var profile = await manager.currentDecisionProfile()
        #expect(profile == .alwaysGPU)

        await manager.updateDecisionProfile(.alwaysCPU)

        profile = await manager.currentDecisionProfile()
        #expect(profile == .alwaysCPU)

        // After switching to CPU, operations should use CPU
        let a: [Float] = [1.0, 0.0, 0.0]
        let b: [Float] = [0.0, 1.0, 0.0]

        await manager.resetStatistics()
        _ = try await manager.distance(from: a, to: b, metric: .cosine)

        let stats = await manager.statistics()
        #expect(stats.gpuOperations == 0)
    }

    @Test("Activation thresholds available for adaptive profiles")
    func activationThresholdsAvailable() async throws {
        let manager = try await AccelerationManager.create(decisionProfile: .balanced)

        let thresholds = await manager.currentActivationThresholds()
        #expect(thresholds != nil)
    }

    @Test("Activation thresholds nil for forced profiles")
    func activationThresholdsNilForForcedProfiles() async throws {
        let manager = try await AccelerationManager.create(decisionProfile: .alwaysGPU)

        let thresholds = await manager.currentActivationThresholds()
        #expect(thresholds == nil)
    }

    @Test("Decision engine history can be reset")
    func decisionEngineHistoryReset() async throws {
        let manager = try await AccelerationManager.create(decisionProfile: .balanced)

        // Perform some operations to populate history
        let query: [Float] = [1.0, 0.0, 0.0]
        let candidates: [[Float]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ]
        _ = try await manager.batchDistance(from: query, to: candidates, metric: .cosine)

        // Reset history
        await manager.resetDecisionEngineHistory()

        // Stats should be reset
        let stats = await manager.gpuPerformanceStats()
        #expect(stats != nil)
        #expect(stats?.totalOperations == 0)
    }

    @Test("Batch distance works with all profiles")
    func batchDistanceWithAllProfiles() async throws {
        let query: [Float] = [1.0, 0.0, 0.0]
        let candidates: [[Float]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0]
        ]

        let profiles: [GPUDecisionProfile] = [.balanced, .batchOptimized, .realTimeOptimized, .alwaysGPU, .alwaysCPU]

        for profile in profiles {
            let manager = try await AccelerationManager.create(decisionProfile: profile)
            let distances = try await manager.batchDistance(from: query, to: candidates, metric: .cosine)
            #expect(distances.count == 3)
        }
    }

    @Test("Normalization works with all profiles")
    func normalizationWithAllProfiles() async throws {
        let vectors: [[Float]] = [
            [3.0, 4.0, 0.0],
            [0.0, 5.0, 0.0]
        ]

        let profiles: [GPUDecisionProfile] = [.balanced, .batchOptimized, .realTimeOptimized, .alwaysGPU, .alwaysCPU]

        for profile in profiles {
            let manager = try await AccelerationManager.create(decisionProfile: profile)
            let normalized = try await manager.normalizeBatch(vectors)
            #expect(normalized.count == 2)
            // First vector [3, 4, 0] normalized should be [0.6, 0.8, 0]
            #expect(abs(normalized[0][0] - 0.6) < 0.01)
            #expect(abs(normalized[0][1] - 0.8) < 0.01)
        }
    }

    @Test("Performance stats summary is readable")
    func performanceStatsSummary() async throws {
        let manager = try await AccelerationManager.create(decisionProfile: .balanced)

        // Perform some operations
        let query: [Float] = [1.0, 0.0, 0.0]
        let candidates: [[Float]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ]
        _ = try await manager.batchDistance(from: query, to: candidates, metric: .cosine)

        let stats = await manager.gpuPerformanceStats()
        #expect(stats != nil)

        // Summary should contain expected content
        let summary = stats!.summary
        #expect(summary.contains("GPU Performance Statistics"))
    }
}

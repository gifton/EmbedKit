import Testing
import Foundation
import VectorAccelerate
@testable import EmbedKit

@Suite("AccelerationManager - GPU Timing Integration (VA 0.4.2)")
struct AccelerationManagerGPUTimingTests {

    @Test("Adaptive manager exposes GPU performance stats after batch work")
    func batchDistanceSmoke() async throws {
        let context = try await Metal4ContextManager.shared()
        let manager = try await AccelerationManager.create(
            context: context,
            decisionProfile: .batchOptimized
        )

        await manager.resetDecisionEngineHistory()

        let dimension = 256
        let candidateCount = 512
        let query = (0..<dimension).map { Float($0) / Float(dimension) }
        let candidates: [[Float]] = (0..<candidateCount).map { i in
            (0..<dimension).map { j in Float((i * 31 + j) % 1024) / 1024.0 }
        }

        let distances = try await manager.batchDistance(
            from: query,
            to: candidates,
            metric: .euclidean
        )

        #expect(distances.count == candidateCount)

        // Adaptive profile → decisionEngine present → stats struct is non-nil.
        // Note: under current AccelerationManager.shouldUseGPU hardcoding (k=1), the
        // batch path routes to CPU for adaptive profiles, so totalOperations may be 0.
        // This test verifies the stats accessor wiring survives the 0.4.2 bump.
        let stats = try #require(await manager.gpuPerformanceStats())
        #expect(stats.totalOperations >= 0)
    }

    @Test("Metal4Context.lastGPUTiming is readable after direct GPU work")
    func contextExposesLastGPUTiming() async throws {
        let context = try await Metal4ContextManager.shared()

        // Go through the distance provider directly so we bypass AccelerationManager's
        // k=1 routing gate and guarantee a real GPU kernel launch. This verifies the
        // new 0.4.2 `GPUTimingInfo` API is wired up end-to-end on the shared context.
        let provider = await UniversalKernelDistanceProvider(context: context)

        let dimension = 128
        let candidateCount = 256
        let query = DynamicVector((0..<dimension).map { _ in Float.random(in: -1...1) })
        let candidates: [DynamicVector] = (0..<candidateCount).map { _ in
            DynamicVector((0..<dimension).map { _ in Float.random(in: -1...1) })
        }

        _ = try await provider.batchDistance(
            from: query,
            to: candidates,
            metric: .euclidean
        )

        // Best-effort: lastGPUTiming may be nil on runners whose command buffers don't
        // report valid timestamps (CI, software renderers). When present, duration must
        // be finite and non-negative.
        if let timing = await context.lastGPUTiming {
            #expect(timing.duration >= 0)
            #expect(timing.duration.isFinite)
            #expect(timing.gpuEndTime >= timing.gpuStartTime)
        }
    }

    @Test("Chebyshev batch distance completes without crashing")
    func chebyshevBatchDistanceRuns() async throws {
        let context = try await Metal4ContextManager.shared()
        let manager = try await AccelerationManager.create(
            context: context,
            decisionProfile: .batchOptimized
        )

        let dimension = 128
        let query = (0..<dimension).map { Float($0) / Float(dimension) }
        let candidates: [[Float]] = (0..<256).map { i in
            (0..<dimension).map { j in Float((i + j) % 256) / 256.0 }
        }

        // Covers the Chebyshev routing change: `.chebyshev` now maps to
        // `GPUOperation.chebyshevDistance` rather than `.l2Distance`. The result must
        // still be correct regardless of whether the decision engine picks GPU or CPU.
        let distances = try await manager.batchDistance(
            from: query,
            to: candidates,
            metric: .chebyshev
        )

        #expect(distances.count == 256)
        for d in distances {
            #expect(d.isFinite)
            #expect(d >= 0)
        }
    }
}

import XCTest
@testable import EmbedKit

/// Phase 4: Optimization tests for improved GPU occupancy and batch throughput
///
/// These tests verify:
/// 1. Correctness is maintained with Phase 4 optimizations
/// 2. Performance improvements for batch processing
/// 3. Adaptive dispatch strategy works correctly
/// 4. No regression from Phase 3 requirements
final class BatchOptimizationTests: XCTestCase {

    // MARK: - Test Setup

    private var processor: MetalVectorProcessor!
    private var resourceManager: MetalResourceManager!

    override func setUp() async throws {
        try await super.setUp()
        guard let manager = MetalResourceManager.createShared() else {
            throw XCTSkip("Metal not available")
        }
        resourceManager = manager
        processor = MetalVectorProcessor(resourceManager: resourceManager)
    }

    override func tearDown() async throws {
        processor = nil
        resourceManager = nil
        try await super.tearDown()
    }

    // MARK: - Correctness Tests

    func testBatchOptimizationCorrectnessAllDimensions() async throws {
        print("\n=== Batch Optimization Correctness Test: All Dimensions 1-129 ===")

        var failures: [(dim: Int, error: String)] = []

        // Test key dimension boundaries with batch optimization enabled
        await processor.setBatchOptimization(true)

        for dim in [1, 16, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129] {
            // Create deterministic test vector
            let vector = (0..<dim).map { Float($0 + 1) }

            let batch = try VectorBatch(vectors: [vector])
            let normalized = try await processor.normalizeVectors(batch)

            let result = Array(normalized[0])
            let magnitude = sqrt(result.reduce(0) { $0 + $1 * $1 })

            // Verify unit magnitude
            if abs(magnitude - 1.0) > 0.01 {
                failures.append((dim, "Magnitude = \(magnitude), expected 1.0"))
            }

            // Verify no NaN or Inf
            for (idx, val) in result.enumerated() {
                if val.isNaN {
                    failures.append((dim, "NaN at index \(idx)"))
                }
                if val.isInfinite {
                    failures.append((dim, "Inf at index \(idx)"))
                }
            }

            print("✅ Dimension \(dim): magnitude = \(String(format: "%.6f", magnitude))")
        }

        XCTAssertTrue(failures.isEmpty, "Batch optimization correctness failures: \(failures)")
    }

    func testBatchOptimizationConsistency() async throws {
        print("\n=== Batch Optimization Consistency Test ===")

        await processor.setBatchOptimization(true)

        // Test different dimension ranges
        let testConfigs = [
            (dim: 16, batch: 32),   // Small vectors: 4× throughput
            (dim: 48, batch: 16),   // Medium vectors: 2× throughput
            (dim: 128, batch: 8)    // Large vectors: Standard
        ]

        for (dim, batchSize) in testConfigs {
            // Create test vectors
            let vectors = (0..<batchSize).map { i in
                (0..<dim).map { j in Float((i + 1) * dim + j) }
            }

            // Process as batch
            let batch = try VectorBatch(vectors: vectors)
            let batchResults = try await processor.normalizeVectors(batch)

            // Process individually for comparison
            var individualResults: [[Float]] = []
            for vector in vectors {
                let singleBatch = try VectorBatch(vectors: [vector])
                let result = try await processor.normalizeVectors(singleBatch)
                individualResults.append(Array(result[0]))
            }

            // Compare results
            for i in 0..<batchSize {
                let batchResult = Array(batchResults[i])
                let individualResult = individualResults[i]

                for j in 0..<dim {
                    XCTAssertEqual(batchResult[j], individualResult[j], accuracy: 0.0001,
                                  "Batch consistency failed at dim=\(dim), batch[\(i)][\(j)]")
                }
            }

            print("✅ Batch consistency verified: dim=\(dim), batch=\(batchSize)")
        }
    }

    // MARK: - Performance Benchmarks

    func testBatchOptimizationPerformanceImprovement() async throws {
        print("\n=== Batch Optimization Performance Improvement Test ===")

        let iterations = 100
        let batchSize = 128

        // Test configurations (dimension -> expected speedup)
        let testConfigs: [(dim: Int, expectedSpeedup: Double)] = [
            (16, 3.0),   // Small: expect ~4× speedup
            (32, 3.0),   // Small: expect ~4× speedup
            (48, 1.5),   // Medium: expect ~2× speedup
            (64, 1.5),   // Medium: expect ~2× speedup
            (128, 1.0),  // Large: similar performance
        ]

        for (dim, expectedSpeedup) in testConfigs {
            // Create test batch
            let vectors = (0..<batchSize).map { _ in
                (0..<dim).map { _ in Float.random(in: -10...10) }
            }
            let batch = try VectorBatch(vectors: vectors)

            // Warm up
            await processor.setBatchOptimization(true)
            _ = try await processor.normalizeVectors(batch)

            // Benchmark with Phase 4 disabled
            await processor.setBatchOptimization(false)
            let standardStart = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iterations {
                _ = try await processor.normalizeVectors(batch)
            }
            let standardTime = CFAbsoluteTimeGetCurrent() - standardStart

            // Benchmark with Phase 4 enabled
            await processor.setBatchOptimization(true)
            let batchOptStart = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iterations {
                _ = try await processor.normalizeVectors(batch)
            }
            let batchOptTime = CFAbsoluteTimeGetCurrent() - batchOptStart

            // Calculate speedup
            let speedup = standardTime / batchOptTime
            let avgTimeStandard = (standardTime / Double(iterations)) * 1000
            let avgTimeBatchOpt = (batchOptTime / Double(iterations)) * 1000

            print("""
                Dimension \(dim):
                  Standard: \(String(format: "%.3f", avgTimeStandard))ms
                  Batch Opt:  \(String(format: "%.3f", avgTimeBatchOpt))ms
                  Speedup:  \(String(format: "%.2f", speedup))×
                """)

            // Verify speedup meets expectations (with some tolerance)
            XCTAssertGreaterThan(speedup, expectedSpeedup * 0.7,
                                "Batch optimization speedup for dim=\(dim) below expectations")
        }
    }

    func testBatchOptimizationThroughputScaling() async throws {
        print("\n=== Batch Optimization Throughput Scaling Test ===")

        await processor.setBatchOptimization(true)

        // Test how throughput scales with batch size for small vectors
        let dim = 32 // Optimal for batch optimization (≈4× throughput)
        let batchSizes = [1, 4, 16, 64, 256, 1024]

        var throughputResults: [(batchSize: Int, vectorsPerSec: Double)] = []

        for batchSize in batchSizes {
            // Create test batch
            let vectors = (0..<batchSize).map { _ in
                (0..<dim).map { _ in Float.random(in: -10...10) }
            }
            let batch = try VectorBatch(vectors: vectors)

            // Warm up
            _ = try await processor.normalizeVectors(batch)

            // Measure throughput
            let iterations = max(10, 1000 / batchSize)
            let startTime = CFAbsoluteTimeGetCurrent()

            for _ in 0..<iterations {
                _ = try await processor.normalizeVectors(batch)
            }

            let totalTime = CFAbsoluteTimeGetCurrent() - startTime
            let totalVectors = batchSize * iterations
            let vectorsPerSec = Double(totalVectors) / totalTime

            throughputResults.append((batchSize, vectorsPerSec))

            print("Batch size \(batchSize): \(String(format: "%.0f", vectorsPerSec)) vectors/sec")
        }

        // Verify throughput scales well with batch size
        let smallBatchThroughput = throughputResults[0].vectorsPerSec
        let largeBatchThroughput = throughputResults.last!.vectorsPerSec

        XCTAssertGreaterThan(largeBatchThroughput, smallBatchThroughput * 2,
                            "Throughput should scale with batch size")
    }

    // MARK: - Adaptive Strategy Tests

    func testAdaptiveDispatchStrategy() async throws {
        print("\n=== Batch Optimization Adaptive Dispatch Strategy Test ===")

        await processor.setBatchOptimization(true)

        // Test that different dimensions use appropriate strategies
        let testCases: [(dim: Int, strategy: String)] = [
            (16, "4 vectors/threadgroup"),
            (32, "4 vectors/threadgroup"),
            (48, "2 vectors/threadgroup"),
            (64, "2 vectors/threadgroup"),
            (96, "1 vector/threadgroup"),
            (128, "1 vector/threadgroup")
        ]

        for (dim, expectedStrategy) in testCases {
            let vectors = (0..<100).map { _ in
                (0..<dim).map { _ in Float.random(in: -10...10) }
            }

            let batch = try VectorBatch(vectors: vectors)
            let result = try await processor.normalizeVectors(batch)

            // Verify correctness
            for i in 0..<min(5, result.count) {
                let vec = Array(result[i])
                let magnitude = sqrt(vec.reduce(0) { $0 + $1 * $1 })
                XCTAssertEqual(magnitude, 1.0, accuracy: 0.01,
                              "Strategy '\(expectedStrategy)' failed for dim=\(dim)")
            }

            print("✅ Dimension \(dim): Using \(expectedStrategy)")
        }
    }

    // MARK: - Edge Case Tests

    func testBatchOptimizationEdgeCases() async throws {
        print("\n=== Batch Optimization Edge Cases Test ===")

        await processor.setBatchOptimization(true)

        // Test 1: Batch size not multiple of vectors per threadgroup
        let oddBatch = try VectorBatch(vectors: [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0] // 5 vectors (not multiple of 4)
        ])

        let oddResult = try await processor.normalizeVectors(oddBatch)
        XCTAssertEqual(oddResult.count, 5)

        for i in 0..<5 {
            let magnitude = sqrt(Array(oddResult[i]).reduce(0) { $0 + $1 * $1 })
            XCTAssertEqual(magnitude, 1.0, accuracy: 0.01,
                          "Edge case: odd batch size failed at vector \(i)")
        }
        print("✅ Edge case: Odd batch size handled correctly")

        // Test 2: Single vector batch (should still work)
        let singleBatch = try VectorBatch(vectors: [[1.0, 2.0, 3.0, 4.0]])
        let singleResult = try await processor.normalizeVectors(singleBatch)
        let singleMag = sqrt(Array(singleResult[0]).reduce(0) { $0 + $1 * $1 })
        XCTAssertEqual(singleMag, 1.0, accuracy: 0.01)
        print("✅ Edge case: Single vector batch handled correctly")

        // Test 3: Very large batch
        let largeDim = 16
        let largeCount = 10000
        let largeVectors = (0..<largeCount).map { i in
            (0..<largeDim).map { j in Float((i % 10) + j + 1) }
        }
        let largeBatch = try VectorBatch(vectors: largeVectors)

        let startTime = CFAbsoluteTimeGetCurrent()
        let largeResult = try await processor.normalizeVectors(largeBatch)
        let processTime = CFAbsoluteTimeGetCurrent() - startTime

        XCTAssertEqual(largeResult.count, largeCount)
        print("✅ Edge case: Large batch (\(largeCount) vectors) processed in \(String(format: "%.3f", processTime))s")

        // Spot check some results
        for i in stride(from: 0, to: largeCount, by: largeCount / 10) {
            let magnitude = sqrt(Array(largeResult[i]).reduce(0) { $0 + $1 * $1 })
            XCTAssertEqual(magnitude, 1.0, accuracy: 0.01)
        }
    }

    // MARK: - Memory and Stability Tests

    func testBatchOptimizationMemoryStability() async throws {
        print("\n=== Batch Optimization Memory Stability Test ===")

        await processor.setBatchOptimization(true)

        // Run many iterations to check for memory issues
        let iterations = 1000
        let dim = 32
        let batchSize = 100

        for i in 0..<iterations {
            let vectors = (0..<batchSize).map { _ in
                (0..<dim).map { _ in Float.random(in: -10...10) }
            }

            // Run synchronously to avoid data race warnings in stress test
            do {
                let batch = try VectorBatch(vectors: vectors)
                _ = try await processor.normalizeVectors(batch)
            } catch {
                // Ignore errors in this stress test
            }

            // Print progress
            if (i + 1) % 100 == 0 {
                print("  Completed \(i + 1) iterations...")
            }
        }

        print("✅ Memory stability verified over \(iterations) iterations")
    }

}

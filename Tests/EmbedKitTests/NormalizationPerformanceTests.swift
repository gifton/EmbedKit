import XCTest
@testable import EmbedKit

/// Performance and numerical stability tests for L2 normalization
/// Tests performance benchmarks and numerical edge cases
final class NormalizationPerformanceTests: XCTestCase {

    // MARK: - Performance Benchmark

    func testPerformanceBenchmark() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        print("\n=== Performance Benchmarks ===")

        let benchmarkDims = [32, 64, 128, 256, 384, 512, 768, 1024]
        let iterations = 100

        for dim in benchmarkDims {
            // Create test vectors
            let vectors = (0..<iterations).map { _ in
                (0..<dim).map { _ in Float.random(in: -10...10) }
            }

            // Warm up
            let warmupBatch = try VectorBatch(vectors: [vectors[0]])
            _ = try await accelerator.normalizeVectors(warmupBatch)

            // Benchmark
            let startTime = CFAbsoluteTimeGetCurrent()

            for vector in vectors {
                let batch = try VectorBatch(vectors: [vector])
                _ = try await accelerator.normalizeVectors(batch)
            }

            let totalTime = CFAbsoluteTimeGetCurrent() - startTime
            let avgTime = totalTime / Double(iterations) * 1000 // Convert to ms

            print("Dimension \(dim): \(String(format: "%.3f", avgTime))ms per normalization")

            // Performance should be reasonable (< 1ms for most dimensions)
            if dim <= 768 {
                XCTAssertLessThan(avgTime, 1.0,
                                 "Normalization should be < 1ms for dim \(dim)")
            }
        }
    }

    // MARK: - Numerical Stability Test

    func testNumericalStability() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        print("\n=== Numerical Stability Tests ===")

        let testDims = [32, 64, 128]

        for dim in testDims {
            // Test 1: Very large values
            let largeVector = Array(repeating: Float(1e15), count: dim)
            let batch1 = try VectorBatch(vectors: [largeVector])
            let result1 = try await accelerator.normalizeVectors(batch1)
            let mag1 = sqrt(Array(result1[0]).reduce(0) { $0 + $1 * $1 })
            XCTAssertEqual(mag1, 1.0, accuracy: 0.01,
                          "Large values should normalize stably for dim \(dim)")

            // Test 2: Very small values
            let smallVector = Array(repeating: Float(1e-15), count: dim)
            let batch2 = try VectorBatch(vectors: [smallVector])
            let result2 = try await accelerator.normalizeVectors(batch2)
            let mag2 = sqrt(Array(result2[0]).reduce(0) { $0 + $1 * $1 })
            XCTAssertTrue(mag2 < 0.001 || abs(mag2 - 1.0) < 0.01,
                         "Small values should handle stably for dim \(dim)")

            // Test 3: Mixed scales
            var mixedVector = Array(repeating: Float(0), count: dim)
            for i in 0..<dim {
                mixedVector[i] = (i % 2 == 0) ? Float(1e10) : Float(1e-10)
            }
            let batch3 = try VectorBatch(vectors: [mixedVector])
            let result3 = try await accelerator.normalizeVectors(batch3)

            // Check no NaN/Inf
            for val in Array(result3[0]) {
                XCTAssertFalse(val.isNaN, "Mixed scales should not produce NaN")
                XCTAssertFalse(val.isInfinite, "Mixed scales should not produce Inf")
            }

            print("✅ Numerical stability passed for dimension \(dim)")
        }
    }

    // MARK: - Batch Throughput Test

    func testBatchThroughput() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        print("\n=== Batch Throughput Tests ===")

        let dim = 128
        let batchSizes = [1, 10, 50, 100, 500, 1000]

        for batchSize in batchSizes {
            // Create test batch
            let vectors = (0..<batchSize).map { _ in
                (0..<dim).map { _ in Float.random(in: -10...10) }
            }

            let batch = try VectorBatch(vectors: vectors)

            // Warm up
            _ = try await accelerator.normalizeVectors(batch)

            // Benchmark
            let iterations = 10
            let startTime = CFAbsoluteTimeGetCurrent()

            for _ in 0..<iterations {
                _ = try await accelerator.normalizeVectors(batch)
            }

            let totalTime = CFAbsoluteTimeGetCurrent() - startTime
            let avgTime = totalTime / Double(iterations) * 1000 // ms
            let throughput = Double(batchSize) / (avgTime / 1000.0) // vectors/sec

            print("Batch size \(batchSize): \(String(format: "%.2f", avgTime))ms, " +
                  "\(String(format: "%.0f", throughput)) vectors/sec")
        }
    }

    // MARK: - Memory Stability Test

    func testMemoryStability() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        print("\n=== Memory Stability Test ===")

        // Test with large batches to ensure memory is properly managed
        let dim = 768 // Common embedding dimension
        let largeBatchSize = 10000

        // Create large batch
        let vectors = (0..<largeBatchSize).map { _ in
            (0..<dim).map { _ in Float.random(in: -1...1) }
        }

        // Process in chunks to simulate real usage
        let chunkSize = 1000
        var processedCount = 0

        for i in stride(from: 0, to: largeBatchSize, by: chunkSize) {
            let end = min(i + chunkSize, largeBatchSize)
            let chunk = Array(vectors[i..<end])

            let batch = try VectorBatch(vectors: chunk)
            let normalized = try await accelerator.normalizeVectors(batch)

            // Verify results are valid
            for vector in normalized.toArrays() {
                let mag = sqrt(vector.reduce(0) { $0 + $1 * $1 })
                XCTAssertEqual(mag, 1.0, accuracy: 0.01,
                              "Normalized vector should have unit magnitude")
            }

            processedCount += chunk.count
        }

        XCTAssertEqual(processedCount, largeBatchSize,
                      "All vectors should be processed")
        print("✅ Processed \(processedCount) vectors without memory issues")
    }
}
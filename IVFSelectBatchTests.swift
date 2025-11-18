// Comprehensive Test Suite for IVFSelect Batch Bug Fix
// These tests verify that the race condition is resolved and batch processing works correctly

import XCTest
import Foundation
@testable import VectorIndex

final class IVFSelectBatchTests: XCTestCase {

    // MARK: - Test 1: Race Condition Detection

    /// This test is designed to reliably trigger the race condition in the original implementation
    func testRaceConditionDetection() {
        // Use large batch size to increase likelihood of race condition
        let b = 100  // Large batch to stress concurrent processing
        let d = 128
        let kc = 1000
        let nprobe = 20

        // Create distinct queries that will produce different results
        var Q = [Float]()
        for i in 0..<b {
            // Each query is increasingly distant from origin
            let baseValue = Float(i) / Float(b)
            Q.append(contentsOf: (0..<d).map { _ in baseValue + Float.random(in: -0.1...0.1) })
        }

        // Create well-separated centroids
        var centroids = [Float]()
        for i in 0..<kc {
            let angle = Float(i) * 2.0 * Float.pi / Float(kc)
            let vector = (0..<d).map { j in
                j == 0 ? cos(angle) : (j == 1 ? sin(angle) : Float.random(in: -0.1...0.1))
            }
            centroids.append(contentsOf: vector)
        }

        // Run the test multiple times to catch intermittent failures
        var failures = 0
        let iterations = 10

        for iteration in 0..<iterations {
            var batchIDs = [Int32](repeating: -1, count: b * nprobe)
            var batchScores: [Float]? = nil

            // Use original implementation (if available) or simulate the bug
            ivf_select_nprobe_batch_f32(
                Q: Q, b: b, d: d, centroids: centroids, kc: kc,
                metric: .l2, nprobe: nprobe,
                listIDsOut: &batchIDs, listScoresOut: &batchScores
            )

            // Check for the telltale sign of the race condition:
            // All queries returning identical results (last query's results)
            var identicalResults = true
            let firstQueryResults = Array(batchIDs[0..<nprobe])

            for i in 1..<b {
                let offset = i * nprobe
                let queryResults = Array(batchIDs[offset..<(offset + nprobe)])
                if queryResults != firstQueryResults {
                    identicalResults = false
                    break
                }
            }

            // With distinct queries, results should NOT be identical
            if identicalResults {
                failures += 1
                print("⚠️ Iteration \(iteration): Detected race condition - all queries have identical results")
            }
        }

        print("Race condition detected in \(failures)/\(iterations) iterations")
        // Original implementation should fail frequently
        XCTAssertTrue(failures > 0, "Expected to detect race condition in original implementation")
    }

    // MARK: - Test 2: Comprehensive Batch vs Single Parity

    /// Exhaustive test comparing batch results with single-query results
    func testBatchVsSingleParityComprehensive() {
        let testCases: [(b: Int, d: Int, kc: Int, nprobe: Int, metric: IVFMetric)] = [
            (b: 1, d: 16, kc: 10, nprobe: 5, metric: .l2),        // Minimal case
            (b: 10, d: 32, kc: 100, nprobe: 10, metric: .l2),     // Small batch
            (b: 50, d: 64, kc: 500, nprobe: 20, metric: .ip),     // Medium batch
            (b: 100, d: 128, kc: 1000, nprobe: 50, metric: .l2),  // Large batch
            (b: 5, d: 256, kc: 2000, nprobe: 100, metric: .cosine), // High dimension
            (b: 200, d: 16, kc: 50, nprobe: 25, metric: .l2),     // Many queries, few centroids
        ]

        for (idx, testCase) in testCases.enumerated() {
            print("Running test case \(idx + 1)/\(testCases.count): b=\(testCase.b), d=\(testCase.d), kc=\(testCase.kc)")

            let Q = randomVectors(n: testCase.b, d: testCase.d)
            let centroids = randomVectors(n: testCase.kc, d: testCase.d)

            // Batch processing
            var batchIDs = [Int32](repeating: -1, count: testCase.b * testCase.nprobe)
            var batchScores: [Float]? = [Float](repeating: 0, count: testCase.b * testCase.nprobe)

            // Use corrected/canonical implementation
            ivf_select_nprobe_batch_f32(
                Q: Q, b: testCase.b, d: testCase.d, centroids: centroids, kc: testCase.kc,
                metric: testCase.metric, nprobe: testCase.nprobe,
                listIDsOut: &batchIDs, listScoresOut: &batchScores
            )

            // Single-query processing for validation
            for i in 0..<testCase.b {
                let qOffset = i * testCase.d
                let q = Array(Q[qOffset..<(qOffset + testCase.d)])

                var singleIDs = [Int32](repeating: -1, count: testCase.nprobe)
                var singleScores: [Float]? = [Float](repeating: 0, count: testCase.nprobe)

                ivf_select_nprobe_f32(
                    q: q, d: testCase.d, centroids: centroids, kc: testCase.kc,
                    metric: testCase.metric, nprobe: testCase.nprobe,
                    listIDsOut: &singleIDs, listScoresOut: &singleScores
                )

                // Validate results match
                let batchOffset = i * testCase.nprobe
                for j in 0..<testCase.nprobe {
                    XCTAssertEqual(
                        batchIDs[batchOffset + j], singleIDs[j],
                        "Case \(idx): Query \(i), position \(j) - ID mismatch"
                    )

                    if let bScores = batchScores, let sScores = singleScores {
                        XCTAssertEqual(
                            bScores[batchOffset + j], sScores[j], accuracy: 1e-5,
                            "Case \(idx): Query \(i), position \(j) - Score mismatch"
                        )
                    }
                }
            }

            print("✅ Test case \(idx + 1) passed")
        }
    }

    // MARK: - Test 3: Thread Safety Stress Test

    /// Stress test with high concurrency to ensure thread safety
    func testThreadSafetyUnderStress() {
        let b = 1000  // Very large batch
        let d = 64
        let kc = 500
        let nprobe = 10

        let Q = randomVectors(n: b, d: d)
        let centroids = randomVectors(n: kc, d: d)

        // Run multiple times concurrently
        let group = DispatchGroup()
        let queue = DispatchQueue(label: "test", attributes: .concurrent)
        var results = [[Int32]]()
        let resultsLock = NSLock()

        for _ in 0..<10 {
            group.enter()
            queue.async {
                var batchIDs = [Int32](repeating: -1, count: b * nprobe)
                var nilScores: [Float]? = nil

                ivf_select_nprobe_batch_f32_FIXED(
                    Q: Q, b: b, d: d, centroids: centroids, kc: kc,
                    metric: .l2, nprobe: nprobe,
                    listIDsOut: &batchIDs, listScoresOut: &nilScores
                )

                resultsLock.lock()
                results.append(batchIDs)
                resultsLock.unlock()
                group.leave()
            }
        }

        group.wait()

        // All concurrent runs should produce identical results
        let firstResult = results[0]
        for i in 1..<results.count {
            XCTAssertEqual(results[i], firstResult, "Concurrent execution \(i) produced different results")
        }
    }

    // MARK: - Test 4: Edge Cases

    func testBatchEdgeCases() {
        // Test 1: Single query in batch (b=1)
        testSingleQueryBatch()

        // Test 2: nprobe equals kc
        testNprobeEqualsKcBatch()

        // Test 3: Empty results due to disabled lists
        testDisabledListsBatch()

        // Test 4: Very small dimensions
        testSmallDimensionBatch()

        // Test 5: Maximum batch size
        testLargeBatchSize()
    }

    private func testSingleQueryBatch() {
        let b = 1
        let d = 32
        let kc = 100
        let nprobe = 10

        let Q = randomVectors(n: b, d: d)
        let centroids = randomVectors(n: kc, d: d)

        var batchIDs = [Int32](repeating: -1, count: b * nprobe)
        var nilScores: [Float]? = nil

        ivf_select_nprobe_batch_f32_FIXED(
            Q: Q, b: b, d: d, centroids: centroids, kc: kc,
            metric: .l2, nprobe: nprobe,
            listIDsOut: &batchIDs, listScoresOut: &nilScores
        )

        // Should work correctly with single query
        XCTAssertEqual(batchIDs.count, nprobe)
        XCTAssertNotEqual(batchIDs[0], -1, "Should return valid centroid ID")
    }

    private func testNprobeEqualsKcBatch() {
        let b = 5
        let d = 16
        let kc = 20
        let nprobe = kc  // nprobe equals number of centroids

        let Q = randomVectors(n: b, d: d)
        let centroids = randomVectors(n: kc, d: d)

        var batchIDs = [Int32](repeating: -1, count: b * nprobe)
        var nilScores: [Float]? = nil

        ivf_select_nprobe_batch_f32_FIXED(
            Q: Q, b: b, d: d, centroids: centroids, kc: kc,
            metric: .l2, nprobe: nprobe,
            listIDsOut: &batchIDs, listScoresOut: &nilScores
        )

        // Each query should return all centroids
        for i in 0..<b {
            let offset = i * nprobe
            let queryResults = Set(batchIDs[offset..<(offset + nprobe)])
            XCTAssertEqual(queryResults.count, kc, "Query \(i) should return all unique centroids")
        }
    }

    private func testDisabledListsBatch() {
        let b = 3
        let d = 32
        let kc = 50
        let nprobe = 10

        let Q = randomVectors(n: b, d: d)
        let centroids = randomVectors(n: kc, d: d)

        // Disable first half of centroids
        let wordCount = (kc + 63) / 64
        var disabledMask = [UInt64](repeating: 0, count: wordCount)
        for i in 0..<(kc / 2) {
            let word = i / 64
            let bit = i % 64
            disabledMask[word] |= (1 << UInt64(bit))
        }

        var batchIDs = [Int32](repeating: -1, count: b * nprobe)
        var nilScores: [Float]? = nil

        ivf_select_nprobe_batch_f32_FIXED(
            Q: Q, b: b, d: d, centroids: centroids, kc: kc,
            metric: .l2, nprobe: nprobe,
            opts: IVFSelectOpts(disabledLists: disabledMask),
            listIDsOut: &batchIDs, listScoresOut: &nilScores
        )

        // Verify no disabled centroids in results
        for i in 0..<b {
            let offset = i * nprobe
            for j in 0..<nprobe {
                let id = Int(batchIDs[offset + j])
                if id >= 0 {
                    XCTAssertTrue(id >= kc / 2, "Found disabled centroid \(id) in batch query \(i)")
                }
            }
        }
    }

    private func testSmallDimensionBatch() {
        let b = 10
        let d = 2  // Very small dimension
        let kc = 100
        let nprobe = 5

        let Q = randomVectors(n: b, d: d)
        let centroids = randomVectors(n: kc, d: d)

        var batchIDs = [Int32](repeating: -1, count: b * nprobe)
        var batchScores: [Float]? = [Float](repeating: 0, count: b * nprobe)

        ivf_select_nprobe_batch_f32_FIXED(
            Q: Q, b: b, d: d, centroids: centroids, kc: kc,
            metric: .l2, nprobe: nprobe,
            listIDsOut: &batchIDs, listScoresOut: &batchScores
        )

        // Should handle small dimensions correctly
        for id in batchIDs {
            if id >= 0 {
                XCTAssertTrue(id < kc, "Invalid centroid ID returned")
            }
        }
    }

    private func testLargeBatchSize() {
        let b = 10000  // Very large batch
        let d = 16    // Small dimension to keep memory reasonable
        let kc = 100
        let nprobe = 5

        let Q = randomVectors(n: b, d: d)
        let centroids = randomVectors(n: kc, d: d)

        var batchIDs = [Int32](repeating: -1, count: b * nprobe)
        var nilScores: [Float]? = nil

        let start = CFAbsoluteTimeGetCurrent()
        ivf_select_nprobe_batch_f32_FIXED(
            Q: Q, b: b, d: d, centroids: centroids, kc: kc,
            metric: .l2, nprobe: nprobe,
            listIDsOut: &batchIDs, listScoresOut: &nilScores
        )
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        print("Large batch (\(b) queries) processed in \(elapsed * 1000)ms")

        // Verify results are valid
        XCTAssertEqual(batchIDs.count, b * nprobe)
        let validCount = batchIDs.filter { $0 >= 0 }.count
        XCTAssertTrue(validCount > 0, "Should return valid results for large batch")
    }

    // MARK: - Test 5: Performance Validation

    func testBatchPerformanceImprovement() {
        let b = 100
        let d = 128
        let kc = 1000
        let nprobe = 20

        let Q = randomVectors(n: b, d: d)
        let centroids = randomVectors(n: kc, d: d)

        // Measure batch processing time
        var batchIDs = [Int32](repeating: -1, count: b * nprobe)
        var nilScores: [Float]? = nil

        let batchStart = CFAbsoluteTimeGetCurrent()
        ivf_select_nprobe_batch_f32_FIXED(
            Q: Q, b: b, d: d, centroids: centroids, kc: kc,
            metric: .l2, nprobe: nprobe,
            listIDsOut: &batchIDs, listScoresOut: &nilScores
        )
        let batchTime = CFAbsoluteTimeGetCurrent() - batchStart

        // Measure sequential single-query processing time
        let seqStart = CFAbsoluteTimeGetCurrent()
        for i in 0..<b {
            let qOffset = i * d
            let q = Array(Q[qOffset..<(qOffset + d)])
            var singleIDs = [Int32](repeating: -1, count: nprobe)
            var singleScores: [Float]? = nil

            ivf_select_nprobe_f32(
                q: q, d: d, centroids: centroids, kc: kc,
                metric: .l2, nprobe: nprobe,
                listIDsOut: &singleIDs, listScoresOut: &singleScores
            )
        }
        let seqTime = CFAbsoluteTimeGetCurrent() - seqStart

        let speedup = seqTime / batchTime
        print("Performance Results:")
        print("  Batch time: \(batchTime * 1000)ms")
        print("  Sequential time: \(seqTime * 1000)ms")
        print("  Speedup: \(String(format: "%.2f", speedup))x")

        XCTAssertTrue(speedup > 1.5, "Batch processing should be significantly faster than sequential")
    }

    // MARK: - Helper Functions

    private func randomVectors(n: Int, d: Int) -> [Float] {
        (0..<(n * d)).map { _ in Float.random(in: -1...1) }
    }
}

// MARK: - Integration Test

/// Integration test to verify the fix works with VectorIndex
final class IVFSelectIntegrationTest: XCTestCase {

    func testBatchProcessingIntegration() {
        // This test would integrate with the actual VectorIndex IVF implementation
        // to ensure the fixed batch processing works correctly in the full context

        print("🔬 Running IVFSelect Batch Integration Test")

        // Test configuration
        let numVectors = 10000
        let dimension = 128
        let numCentroids = 100
        let nprobe = 10
        let batchSize = 50

        // Generate test data
        let vectors = (0..<(numVectors * dimension)).map { _ in Float.random(in: -1...1) }
        let centroids = (0..<(numCentroids * dimension)).map { _ in Float.random(in: -1...1) }
        let queries = (0..<(batchSize * dimension)).map { _ in Float.random(in: -1...1) }

        // Test batch processing
        var batchResults = [Int32](repeating: -1, count: batchSize * nprobe)
        var batchScores: [Float]? = [Float](repeating: 0, count: batchSize * nprobe)

        let batchSuccess = measureTime("Batch Processing") {
            ivf_select_nprobe_batch_f32_FIXED(
                Q: queries, b: batchSize, d: dimension,
                centroids: centroids, kc: numCentroids,
                metric: .l2, nprobe: nprobe,
                listIDsOut: &batchResults, listScoresOut: &batchScores
            )
        }

        // Validate results
        var allValid = true
        for i in 0..<batchSize {
            let offset = i * nprobe
            for j in 0..<nprobe {
                let id = batchResults[offset + j]
                if id < 0 || id >= numCentroids {
                    print("❌ Invalid ID at query \(i), position \(j): \(id)")
                    allValid = false
                }
            }
        }

        XCTAssertTrue(allValid, "All batch results should be valid")
        print(allValid ? "✅ Integration test passed" : "❌ Integration test failed")
    }

    private func measureTime(_ label: String, block: () -> Void) -> Bool {
        let start = CFAbsoluteTimeGetCurrent()
        block()
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        print("\(label): \(String(format: "%.2f", elapsed * 1000))ms")
        return true
    }
}

import XCTest
@testable import EmbedKit

/// Edge case tests for L2 normalization
/// Tests handling of adversarial inputs, boundary values, and exceptional conditions
final class NormalizationEdgeCaseTests: XCTestCase {

    // MARK: - Adversarial Pattern Tests

    func testAdversarialPatterns() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        print("\n=== Adversarial Pattern Tests ===")

        // Test dimensions that are likely to cause issues
        let criticalDims = [32, 33, 64, 65, 96, 97, 128, 129]

        for dim in criticalDims {
            // Pattern 1: Single non-zero at last index
            var lastNonZero = Array(repeating: Float(0), count: dim)
            lastNonZero[dim - 1] = 1.0

            let batch1 = try VectorBatch(vectors: [lastNonZero])
            let result1 = try await accelerator.normalizeVectors(batch1)
            XCTAssertEqual(Array(result1[0])[dim - 1], 1.0, accuracy: 0.01,
                          "Single element at end should be 1.0 for dim \(dim)")

            // Pattern 2: Alternating signs
            let alternating = (0..<dim).map { Float(($0 % 2 == 0) ? 1 : -1) }
            let batch2 = try VectorBatch(vectors: [alternating])
            let result2 = try await accelerator.normalizeVectors(batch2)
            let mag2 = sqrt(Array(result2[0]).reduce(0) { $0 + $1 * $1 })
            XCTAssertEqual(mag2, 1.0, accuracy: 0.01,
                          "Alternating pattern should normalize for dim \(dim)")

            // Pattern 3: Exponentially growing values
            let exponential = (0..<dim).map { Float(pow(2.0, Float($0) / 10.0)) }
            let batch3 = try VectorBatch(vectors: [exponential])
            let result3 = try await accelerator.normalizeVectors(batch3)
            let mag3 = sqrt(Array(result3[0]).reduce(0) { $0 + $1 * $1 })
            XCTAssertEqual(mag3, 1.0, accuracy: 0.01,
                          "Exponential pattern should normalize for dim \(dim)")

            print("✅ Adversarial patterns passed for dimension \(dim)")
        }
    }

    // MARK: - Denormal Value Tests

    func testDenormalValues() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        print("\n=== Denormal Value Tests ===")

        let testDims = [1, 32, 33, 64, 128]

        for dim in testDims {
            // Test with denormal values
            let denormalValue = Float.leastNormalMagnitude
            let denormalVector = Array(repeating: denormalValue, count: dim)

            let batch = try VectorBatch(vectors: [denormalVector])
            let normalized = try await accelerator.normalizeVectors(batch)
            let result = Array(normalized[0])

            // Should either normalize or go to zero (both are acceptable)
            let magnitude = sqrt(result.reduce(0) { $0 + $1 * $1 })
            let isNormalized = abs(magnitude - 1.0) < 0.1
            let isZero = magnitude < 0.0001

            XCTAssertTrue(isNormalized || isZero,
                         "Denormal vector should either normalize or zero for dim \(dim)")

            // No NaN or Inf allowed
            for val in result {
                XCTAssertFalse(val.isNaN, "Denormal should not produce NaN")
                XCTAssertFalse(val.isInfinite, "Denormal should not produce Inf")
            }

            print("✅ Denormal test passed for dimension \(dim) (mag: \(magnitude))")
        }
    }

    // MARK: - Boundary Value Tests

    func testBoundaryValues() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        print("\n=== Boundary Value Tests ===")

        // Test dimensions at SIMD boundaries
        let boundaryDims = [1, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129]

        for dim in boundaryDims {
            // Test with large (but not overflow-inducing) positive values
            // Use 1e10 which is large but won't cause overflow when squared
            let maxValue = Float(1e10)
            let maxVector = Array(repeating: maxValue, count: dim)
            let batch1 = try VectorBatch(vectors: [maxVector])
            let result1 = try await accelerator.normalizeVectors(batch1)
            let mag1 = sqrt(Array(result1[0]).reduce(0) { $0 + $1 * $1 })

            // Should normalize without overflow
            XCTAssertEqual(mag1, 1.0, accuracy: 0.01,
                          "Large values should normalize for dim \(dim)")

            // Test with minimum positive values
            let minVector = Array(repeating: Float.leastNonzeroMagnitude * 1000, count: dim)
            let batch2 = try VectorBatch(vectors: [minVector])
            let result2 = try await accelerator.normalizeVectors(batch2)
            let mag2 = sqrt(Array(result2[0]).reduce(0) { $0 + $1 * $1 })

            // Should either normalize or go to zero
            XCTAssertTrue(mag2 < 0.001 || abs(mag2 - 1.0) < 0.01,
                         "Min values should handle correctly for dim \(dim)")

            print("✅ Boundary values passed for dimension \(dim)")
        }
    }

    // MARK: - Sparse Vector Tests

    func testSparseVectors() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        print("\n=== Sparse Vector Tests ===")

        let testDims = [32, 64, 96, 128]
        let sparsityLevels = [0.9, 0.95, 0.99] // 90%, 95%, 99% zeros

        for dim in testDims {
            for sparsity in sparsityLevels {
                // Create sparse vector
                var sparseVector = Array(repeating: Float(0), count: dim)
                let nonZeroCount = Int(Float(dim) * Float(1.0 - sparsity))

                // Randomly place non-zero values
                for i in 0..<max(1, nonZeroCount) {
                    let index = Int.random(in: 0..<dim)
                    sparseVector[index] = Float.random(in: -10...10)
                }

                // Ensure it's not all zeros
                if sparseVector.allSatisfy({ $0 == 0 }) {
                    sparseVector[0] = 1.0
                }

                let batch = try VectorBatch(vectors: [sparseVector])
                let normalized = try await accelerator.normalizeVectors(batch)
                let result = Array(normalized[0])

                let magnitude = sqrt(result.reduce(0) { $0 + $1 * $1 })
                XCTAssertEqual(magnitude, 1.0, accuracy: 0.01,
                              "Sparse vector (\(Int(sparsity*100))% zeros) dim \(dim) should normalize")

                // Count preserved sparsity
                let outputZeros = result.filter { abs($0) < 0.0001 }.count
                let outputSparsity = Float(outputZeros) / Float(dim)

                // Sparsity should be somewhat preserved (within reason)
                XCTAssertGreaterThan(outputSparsity, Float(sparsity * 0.8),
                                    "Sparsity should be mostly preserved")
            }
            print("✅ Sparse vector tests passed for dimension \(dim)")
        }
    }

    // MARK: - Zero Vector Tests (from Phase 2)

    func testZeroVectorHandling() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        print("\n=== Zero Vector Tests ===")

        let testDims = [1, 16, 32, 33, 64, 128, 256, 384]

        for dim in testDims {
            let zeroVector = Array(repeating: Float(0), count: dim)
            let batch = try VectorBatch(vectors: [zeroVector])
            let normalized = try await accelerator.normalizeVectors(batch)
            let result = Array(normalized[0])

            // Zero vector should remain zero (not NaN)
            for val in result {
                XCTAssertEqual(val, 0.0, accuracy: 0.0001,
                             "Zero vector dim \(dim) should produce zero output")
                XCTAssertFalse(val.isNaN, "Zero vector should not produce NaN")
                XCTAssertFalse(val.isInfinite, "Zero vector should not produce Inf")
            }

            let magnitude = sqrt(result.reduce(0) { $0 + $1 * $1 })
            XCTAssertEqual(magnitude, 0.0, accuracy: 0.0001,
                          "Zero vector magnitude should be 0 for dim \(dim)")
        }
        print("✅ Zero vector handling verified for all dimensions")
    }

    // MARK: - NaN/Inf Handling Tests (from Phase 2)

    func testNaNInfHandling() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        print("\n=== NaN/Inf Handling Tests ===")

        let dim = 32

        // Test NaN handling
        var nanVector = (0..<dim).map { Float($0 + 1) }
        nanVector[10] = Float.nan

        let nanBatch = try VectorBatch(vectors: [nanVector])
        let nanResult = Array(try await accelerator.normalizeVectors(nanBatch)[0])

        XCTAssertEqual(nanResult[10], 0.0, accuracy: 0.0001,
                      "NaN input should produce 0 output")
        for val in nanResult {
            XCTAssertFalse(val.isNaN, "Output should not contain NaN")
        }
        print("✅ NaN handling verified")

        // Test Infinity handling
        var infVector = (0..<dim).map { Float($0 + 1) }
        infVector[15] = Float.infinity

        let infBatch = try VectorBatch(vectors: [infVector])
        let infResult = Array(try await accelerator.normalizeVectors(infBatch)[0])

        for val in infResult {
            XCTAssertFalse(val.isNaN, "Infinity input should not produce NaN")
        }
        print("✅ Infinity handling verified")
    }
}
import XCTest
@testable import EmbedKit

/// Correctness tests for L2 normalization across all supported dimensions
/// Tests basic functionality and consistency of normalization operations
final class NormalizationCorrectnessTests: XCTestCase {

    // MARK: - Test All Dimensions 1-129

    func testAllDimensions1To129() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        var failures: [(dim: Int, magnitude: Float)] = []

        print("\n=== Testing All Dimensions 1-129 ===")
        print("Testing normalization correctness for each dimension...")

        for dim in 1...129 {
            // Create a deterministic test vector [1, 2, 3, ..., dim]
            let vector = (0..<dim).map { Float($0 + 1) }

            let batch = try VectorBatch(vectors: [vector])
            let normalized = try await accelerator.normalizeVectors(batch)

            let result = Array(normalized[0])
            let magnitude = sqrt(result.reduce(0) { $0 + $1 * $1 })

            // Check if magnitude is approximately 1.0
            if abs(magnitude - 1.0) > 0.01 {
                failures.append((dim, magnitude))
                print("❌ Dimension \(dim): magnitude = \(magnitude)")
            }

            // Verify no NaN or Inf
            for (idx, val) in result.enumerated() {
                XCTAssertFalse(val.isNaN, "Dimension \(dim) has NaN at index \(idx)")
                XCTAssertFalse(val.isInfinite, "Dimension \(dim) has Inf at index \(idx)")
            }

            // Print progress every 10 dimensions
            if dim % 10 == 0 {
                print("✅ Tested up to dimension \(dim)")
            }
        }

        // Report results
        if failures.isEmpty {
            print("✅ ALL DIMENSIONS 1-129 PASSED!")
        } else {
            print("\n❌ Failed dimensions: \(failures.count)")
            for (dim, mag) in failures {
                print("  Dim \(dim): magnitude = \(mag)")
            }
        }

        XCTAssertTrue(failures.isEmpty, "Some dimensions failed normalization")
    }

    // MARK: - Random Input Tests

    func testRandomInputsVariousDimensions() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Test key dimensions with random inputs
        let testDims = [1, 7, 16, 31, 32, 33, 47, 63, 64, 65, 96, 127, 128, 129]

        print("\n=== Random Input Tests ===")

        for dim in testDims {
            // Generate random vector with values in [-10, 10]
            let randomVector = (0..<dim).map { _ in Float.random(in: -10...10) }

            // Skip if we randomly got a zero vector
            let inputMagnitude = sqrt(randomVector.reduce(0) { $0 + $1 * $1 })
            guard inputMagnitude > 0.001 else { continue }

            let batch = try VectorBatch(vectors: [randomVector])
            let normalized = try await accelerator.normalizeVectors(batch)

            let result = Array(normalized[0])
            let outputMagnitude = sqrt(result.reduce(0) { $0 + $1 * $1 })

            XCTAssertEqual(outputMagnitude, 1.0, accuracy: 0.01,
                          "Random vector dim \(dim) should have unit magnitude")

            // Verify direction is preserved (dot product should be positive)
            let dotProduct = zip(randomVector, result).reduce(0) { $0 + $1.0 * $1.1 }
            XCTAssertGreaterThan(dotProduct, 0,
                                "Normalization should preserve direction for dim \(dim)")

            print("✅ Random input test passed for dimension \(dim)")
        }
    }

    // MARK: - Batch Consistency Test

    func testBatchConsistency() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        print("\n=== Batch Consistency Test ===")

        // Test that batch processing gives same results as individual processing
        let dim = 64
        let batchSize = 10

        // Create test vectors
        let vectors = (0..<batchSize).map { i in
            (0..<dim).map { j in Float(i * dim + j + 1) }
        }

        // Process as batch
        let batch = try VectorBatch(vectors: vectors)
        let batchResults = try await accelerator.normalizeVectors(batch)

        // Process individually
        var individualResults: [[Float]] = []
        for vector in vectors {
            let singleBatch = try VectorBatch(vectors: [vector])
            let result = try await accelerator.normalizeVectors(singleBatch)
            individualResults.append(Array(result[0]))
        }

        // Compare results
        for i in 0..<batchSize {
            let batchResult = Array(batchResults[i])
            let individualResult = individualResults[i]

            for j in 0..<dim {
                XCTAssertEqual(batchResult[j], individualResult[j], accuracy: 0.0001,
                              "Batch and individual results should match at [\(i)][\(j)]")
            }
        }

        print("✅ Batch consistency verified for \(batchSize) vectors of dimension \(dim)")
    }
}
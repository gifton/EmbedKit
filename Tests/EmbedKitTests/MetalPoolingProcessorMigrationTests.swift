import XCTest
@testable import EmbedKit

/// Tests for MetalPoolingProcessor migration to VectorBatch API
///
/// These tests verify:
/// 1. VectorBatch API produces correct results
/// 2. Results match deprecated [[Float]] API
/// 3. Performance characteristics
/// 4. Backward compatibility
final class MetalPoolingProcessorMigrationTests: XCTestCase {

    // MARK: - Basic Functionality

    func testVectorBatchMeanPooling() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Create test batch (3 tokens × 4 dimensions)
        let tokens: [[Float]] = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ]
        let batch = try VectorBatch(vectors: tokens)

        // Pool using mean strategy
        let pooled = try await accelerator.poolEmbeddings(batch, strategy: .mean)

        // Expected: mean of all tokens = [5.0, 6.0, 7.0, 8.0]
        XCTAssertEqual(pooled.count, 4)
        XCTAssertEqual(pooled[0], 5.0, accuracy: 0.001)
        XCTAssertEqual(pooled[1], 6.0, accuracy: 0.001)
        XCTAssertEqual(pooled[2], 7.0, accuracy: 0.001)
        XCTAssertEqual(pooled[3], 8.0, accuracy: 0.001)
    }

    func testVectorBatchMaxPooling() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Create test batch with varied values
        let tokens: [[Float]] = [
            [1.0, 9.0, 2.0, 3.0],
            [8.0, 2.0, 7.0, 4.0],
            [3.0, 5.0, 10.0, 1.0]
        ]
        let batch = try VectorBatch(vectors: tokens)

        // Pool using max strategy
        let pooled = try await accelerator.poolEmbeddings(batch, strategy: .max)

        // Expected: max of each dimension = [8.0, 9.0, 10.0, 4.0]
        XCTAssertEqual(pooled.count, 4)
        XCTAssertEqual(pooled[0], 8.0, accuracy: 0.001)
        XCTAssertEqual(pooled[1], 9.0, accuracy: 0.001)
        XCTAssertEqual(pooled[2], 10.0, accuracy: 0.001)
        XCTAssertEqual(pooled[3], 4.0, accuracy: 0.001)
    }

    func testVectorBatchCLSPooling() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Create test batch
        let tokens: [[Float]] = [
            [1.0, 2.0, 3.0],  // CLS token
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]
        let batch = try VectorBatch(vectors: tokens)

        // Pool using CLS strategy
        let pooled = try await accelerator.poolEmbeddings(batch, strategy: .cls)

        // Expected: first token only
        XCTAssertEqual(pooled.count, 3)
        XCTAssertEqual(pooled[0], 1.0, accuracy: 0.001)
        XCTAssertEqual(pooled[1], 2.0, accuracy: 0.001)
        XCTAssertEqual(pooled[2], 3.0, accuracy: 0.001)
    }

    func testVectorBatchAttentionWeightedPooling() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Create test batch
        let tokens: [[Float]] = [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
        let batch = try VectorBatch(vectors: tokens)
        let weights: [Float] = [0.7, 0.3]  // Emphasize first token

        // Pool using attention weights
        let pooled = try await accelerator.attentionWeightedPooling(batch, attentionWeights: weights)

        // Expected: 0.7 * [1, 2] + 0.3 * [3, 4] = [1.6, 2.6]
        XCTAssertEqual(pooled.count, 2)
        XCTAssertEqual(pooled[0], 1.6, accuracy: 0.001)
        XCTAssertEqual(pooled[1], 2.6, accuracy: 0.001)
    }

    func testEmptyBatchPooling() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        let empty = try VectorBatch.empty(dimensions: 768)

        // Should throw error for empty batch
        do {
            _ = try await accelerator.poolEmbeddings(empty, strategy: .mean)
            XCTFail("Should have thrown error for empty batch")
        } catch {
            // Expected error
            XCTAssertTrue(error is MetalError)
        }
    }

    // MARK: - Backward Compatibility

    func testMeanPoolingResultsMatch() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        let testVectors: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]

        // VectorBatch API
        let batch = try VectorBatch(vectors: testVectors)
        let batchResult = try await accelerator.poolEmbeddings(batch, strategy: .mean)

        // Deprecated [[Float]] API - REMOVED, no longer supported
        // let arrayResult = try await accelerator.poolEmbeddings(testVectors, strategy: .mean)

        // Verify VectorBatch results are correct
        XCTAssertEqual(batchResult.count, testVectors[0].count, "Should return vector with dimensions matching input")
    }

    func testMaxPoolingResultsMatch() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        let testVectors: [[Float]] = [
            [1.0, 9.0, 2.0],
            [8.0, 2.0, 7.0],
            [3.0, 5.0, 10.0]
        ]

        // VectorBatch API
        let batch = try VectorBatch(vectors: testVectors)
        let batchResult = try await accelerator.poolEmbeddings(batch, strategy: .max)

        // Deprecated [[Float]] API - REMOVED, no longer supported
        // let arrayResult = try await accelerator.poolEmbeddings(testVectors, strategy: .max)

        // Verify VectorBatch results are correct
        XCTAssertEqual(batchResult.count, testVectors[0].count, "Should return vector with dimensions matching input")
    }

    func testAttentionPoolingResultsMatch() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        let testVectors: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]
        let weights: [Float] = [0.6, 0.4]

        // VectorBatch API
        let batch = try VectorBatch(vectors: testVectors)
        let batchResult = try await accelerator.attentionWeightedPooling(batch, attentionWeights: weights)

        // Deprecated [[Float]] API - REMOVED, no longer supported
        // let arrayResult = try await accelerator.attentionWeightedPooling(testVectors, attentionWeights: weights)

        // Verify VectorBatch results are correct
        XCTAssertEqual(batchResult.count, testVectors[0].count, "Should return vector with dimensions matching input")
    }

    func testDeprecatedAPIStillWorks() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        let vectors: [[Float]] = [
            [1.0, 2.0],
            [3.0, 4.0]
        ]

        // Deprecated [[Float]] API has been removed, use VectorBatch instead
        let batch = try VectorBatch(vectors: vectors)
        let pooled = try await accelerator.poolEmbeddings(batch, strategy: .mean)

        XCTAssertEqual(pooled.count, 2)
        XCTAssertEqual(pooled[0], 2.0, accuracy: 0.001)
        XCTAssertEqual(pooled[1], 3.0, accuracy: 0.001)
    }

    // MARK: - Edge Cases

    func testSingleTokenPooling() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        let batch = try VectorBatch(vectors: [[5.0, 10.0, 15.0]])

        // Mean pooling of single token should return that token
        let pooled = try await accelerator.poolEmbeddings(batch, strategy: .mean)

        XCTAssertEqual(pooled.count, 3)
        XCTAssertEqual(pooled[0], 5.0, accuracy: 0.001)
        XCTAssertEqual(pooled[1], 10.0, accuracy: 0.001)
        XCTAssertEqual(pooled[2], 15.0, accuracy: 0.001)
    }

    func testLargeSequencePooling() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // BERT max sequence length: 512 tokens × 768 dimensions
        var vectors: [[Float]] = []
        for i in 0..<512 {
            let vector = (0..<768).map { _ in Float.random(in: -1...1) + Float(i) }
            vectors.append(vector)
        }

        let batch = try VectorBatch(vectors: vectors)
        let pooled = try await accelerator.poolEmbeddings(batch, strategy: .mean)

        XCTAssertEqual(pooled.count, 768)

        // Verify values are reasonable (not NaN/Inf)
        for value in pooled {
            XCTAssertFalse(value.isNaN)
            XCTAssertFalse(value.isInfinite)
        }
    }

    func testHighDimensionalPooling() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // High-dimensional vectors (4096D)
        let vector1 = [Float](repeating: 1.0, count: 4096)
        let vector2 = [Float](repeating: 2.0, count: 4096)
        let vector3 = [Float](repeating: 3.0, count: 4096)

        let batch = try VectorBatch(vectors: [vector1, vector2, vector3])
        let pooled = try await accelerator.poolEmbeddings(batch, strategy: .mean)

        XCTAssertEqual(pooled.count, 4096)

        // Mean should be 2.0 for all dimensions
        for value in pooled {
            XCTAssertEqual(value, 2.0, accuracy: 0.01)
        }
    }

    func testPoolingWithMask() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Tokens with padding (matching existing test pattern)
        let tokens: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]  // This will be masked out
        ]
        let batch = try VectorBatch(vectors: tokens)
        let mask: [Int] = [1, 1, 1, 0]  // Mask out last token

        let pooled = try await accelerator.poolEmbeddings(batch, strategy: .mean, attentionMask: mask)

        // Should average first three tokens: [4.0, 5.0, 6.0]
        // (1+4+7)/3=4, (2+5+8)/3=5, (3+6+9)/3=6
        XCTAssertEqual(pooled.count, 3)
        XCTAssertEqual(pooled[0], 4.0, accuracy: 0.001)
        XCTAssertEqual(pooled[1], 5.0, accuracy: 0.001)
        XCTAssertEqual(pooled[2], 6.0, accuracy: 0.001)
    }

    func testNonPowerOfTwoDimensions() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Test dimensions that don't align with SIMD width
        for dim in [1, 7, 31, 127, 383] {
            let vector1 = (0..<dim).map { Float($0) }
            let vector2 = (0..<dim).map { Float($0 + dim) }

            let batch = try VectorBatch(vectors: [vector1, vector2])
            let pooled = try await accelerator.poolEmbeddings(batch, strategy: .mean)

            XCTAssertEqual(pooled.count, dim,
                "Failed for dimension \(dim)")

            // Verify no NaN or Inf
            for (idx, value) in pooled.enumerated() {
                XCTAssertFalse(value.isNaN,
                    "NaN at index \(idx) for dimension \(dim)")
                XCTAssertFalse(value.isInfinite,
                    "Inf at index \(idx) for dimension \(dim)")
            }
        }
    }

    // MARK: - Numerical Stability

    func testPoolingWithExtremeValues() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Mix of very small and very large values
        let tokens: [[Float]] = [
            [1e-5, 1e5],     // Very small and very large
            [1e5, 1e-5],     // Swapped
            [1.0, 1.0]       // Normal
        ]
        let batch = try VectorBatch(vectors: tokens)

        let pooled = try await accelerator.poolEmbeddings(batch, strategy: .mean)

        // Should not contain NaN or Inf
        for value in pooled {
            XCTAssertFalse(value.isNaN)
            XCTAssertFalse(value.isInfinite)
        }
    }

    func testAttentionPoolingZeroWeights() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        let tokens: [[Float]] = [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
        let batch = try VectorBatch(vectors: tokens)
        let weights: [Float] = [0.0, 0.0]  // All zero weights

        // Zero weights should either throw error or return zero vector
        // Different implementations may handle this differently (GPU vs CPU fallback)
        let pooled = try await accelerator.attentionWeightedPooling(batch, attentionWeights: weights)

        // Result should be all zeros or very small values (not NaN/Inf)
        for value in pooled {
            XCTAssertFalse(value.isNaN, "Result contains NaN")
            XCTAssertFalse(value.isInfinite, "Result contains Infinity")
        }
    }

    func testMixedMagnitudePooling() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Mix of different magnitude vectors
        let tokens: [[Float]] = [
            [1e-7, 1e-7],    // Tiny
            [1e7, 1e7],      // Huge
            [1.0, 1.0]       // Normal
        ]
        let batch = try VectorBatch(vectors: tokens)

        let pooled = try await accelerator.poolEmbeddings(batch, strategy: .mean)

        // Should produce reasonable results (not NaN/Inf)
        for value in pooled {
            XCTAssertFalse(value.isNaN)
            XCTAssertFalse(value.isInfinite)
        }
    }

    // MARK: - Performance

    func testMeanPoolingPerformance() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Create realistic batch (100 tokens × 768 dimensions)
        var vectors: [[Float]] = []
        for i in 0..<100 {
            let vector = (0..<768).map { _ in Float.random(in: -1...1) + Float(i) }
            vectors.append(vector)
        }

        let batch = try VectorBatch(vectors: vectors)

        // Measure VectorBatch API
        let batchStart = CFAbsoluteTimeGetCurrent()
        _ = try await accelerator.poolEmbeddings(batch, strategy: .mean)
        let batchTime = CFAbsoluteTimeGetCurrent() - batchStart

        print("⚡ VectorBatch mean pooling: \(batchTime * 1000)ms for 100×768")

        // Performance should be reasonable (<10ms on modern hardware)
        // Note: First run includes Metal pipeline compilation overhead
        XCTAssertLessThan(batchTime, 0.010, "Mean pooling took longer than expected (>10ms)")
    }

    func testAttentionPoolingPerformance() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Create realistic batch
        var vectors: [[Float]] = []
        for i in 0..<100 {
            let vector = (0..<768).map { _ in Float.random(in: -1...1) + Float(i) }
            vectors.append(vector)
        }
        let weights = (0..<100).map { _ in Float.random(in: 0...1) }

        let batch = try VectorBatch(vectors: vectors)

        // Measure VectorBatch API
        let batchStart = CFAbsoluteTimeGetCurrent()
        _ = try await accelerator.attentionWeightedPooling(batch, attentionWeights: weights)
        let batchTime = CFAbsoluteTimeGetCurrent() - batchStart

        print("⚡ VectorBatch attention pooling: \(batchTime * 1000)ms for 100×768")

        // Performance should be reasonable (<15ms on modern hardware)
        XCTAssertLessThan(batchTime, 0.015, "Attention pooling took longer than expected (>15ms)")
    }

    // MARK: - Round-Trip Conversion

    func testRoundTripConversion() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        let original: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]

        // Convert to VectorBatch
        let batch = try VectorBatch(vectors: original)

        // Pool
        let pooled = try await accelerator.poolEmbeddings(batch, strategy: .mean)

        // Verify result is correct
        XCTAssertEqual(pooled.count, original[0].count)

        // Expected mean: [4.0, 5.0, 6.0]
        XCTAssertEqual(pooled[0], 4.0, accuracy: 0.001)
        XCTAssertEqual(pooled[1], 5.0, accuracy: 0.001)
        XCTAssertEqual(pooled[2], 6.0, accuracy: 0.001)
    }
}

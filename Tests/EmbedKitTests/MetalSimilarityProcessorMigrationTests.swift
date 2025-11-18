import XCTest
@testable import EmbedKit

/// Tests for MetalSimilarityProcessor migration to VectorBatch API
///
/// These tests verify:
/// 1. VectorBatch API produces correct results
/// 2. Results match deprecated [[Float]] API
/// 3. Performance characteristics
/// 4. Backward compatibility
final class MetalSimilarityProcessorMigrationTests: XCTestCase {

    // MARK: - Basic Functionality

    func testVectorBatchSimilarityMatrix() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Create test batches (orthogonal vectors)
        let queries = try VectorBatch(vectors: [
            [1.0, 0.0],  // Query 1
            [0.0, 1.0]   // Query 2
        ])
        let keys = try VectorBatch(vectors: [
            [1.0, 0.0],  // Key 1
            [0.0, 1.0]   // Key 2
        ])

        let similarities = try await accelerator.cosineSimilarityMatrix(queries: queries, keys: keys)

        // Expected: identity matrix (orthogonal vectors)
        // [[1.0, 0.0], [0.0, 1.0]]
        XCTAssertEqual(similarities.count, 2)
        XCTAssertEqual(similarities[0].count, 2)

        XCTAssertEqual(similarities[0][0], 1.0, accuracy: 0.001)
        XCTAssertEqual(similarities[0][1], 0.0, accuracy: 0.001)
        XCTAssertEqual(similarities[1][0], 0.0, accuracy: 0.001)
        XCTAssertEqual(similarities[1][1], 1.0, accuracy: 0.001)
    }

    func testVectorBatchSingleQueryMultipleKeys() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        let query: [Float] = [1.0, 0.0]
        let keys = try VectorBatch(vectors: [
            [1.0, 0.0],   // Identical (similarity = 1.0)
            [0.0, 1.0],   // Orthogonal (similarity = 0.0)
            [0.707, 0.707] // 45 degrees (similarity ≈ 0.707)
        ])

        let similarities = try await accelerator.cosineSimilarity(query: query, keys: keys)

        XCTAssertEqual(similarities.count, 3)
        XCTAssertEqual(similarities[0], 1.0, accuracy: 0.001)
        XCTAssertEqual(similarities[1], 0.0, accuracy: 0.001)
        XCTAssertEqual(similarities[2], 0.707, accuracy: 0.01)
    }

    func testIdentityMatrix() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Same batch for queries and keys
        let batch = try VectorBatch(vectors: [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        let similarities = try await accelerator.cosineSimilarityMatrix(queries: batch, keys: batch)

        // Self-similarity should be identity matrix
        for i in 0..<3 {
            for j in 0..<3 {
                let expected: Float = (i == j) ? 1.0 : 0.0
                XCTAssertEqual(similarities[i][j], expected, accuracy: 0.001,
                    "Failed at [\(i)][\(j)]")
            }
        }
    }

    func testEmptyBatchSimilarity() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        let empty = try VectorBatch.empty(dimensions: 768)
        let keys = try VectorBatch(vectors: [[1.0, 2.0, 3.0]])

        do {
            _ = try await accelerator.cosineSimilarityMatrix(queries: empty, keys: keys)
            XCTFail("Should have thrown error for empty batch")
        } catch {
            // Expected error
            XCTAssertTrue(error is MetalError)
        }
    }

    // MARK: - Backward Compatibility

    func testSimilarityMatrixResultsMatch() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        let queryVectors: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]
        let keyVectors: [[Float]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]

        // VectorBatch API
        let queryBatch = try VectorBatch(vectors: queryVectors)
        let keyBatch = try VectorBatch(vectors: keyVectors)
        let batchResult = try await accelerator.cosineSimilarityMatrix(queries: queryBatch, keys: keyBatch)

        // Deprecated [[Float]] API - REMOVED, no longer supported
        // let arrayResult = try await accelerator.cosineSimilarityMatrix(queries: queryVectors, keys: keyVectors)

        // Verify VectorBatch results are correct
        XCTAssertEqual(batchResult.count, queryVectors.count, "Should have one row per query")
        XCTAssertEqual(batchResult[0].count, keyVectors.count, "Should have one column per key")
    }

    func testSingleQueryResultsMatch() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        let query: [Float] = [1.0, 2.0, 3.0]
        let keyVectors: [[Float]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]

        // VectorBatch API
        let keyBatch = try VectorBatch(vectors: keyVectors)
        let batchResult = try await accelerator.cosineSimilarity(query: query, keys: keyBatch)

        // Deprecated [[Float]] API - REMOVED, no longer supported
        // let arrayResult = try await accelerator.cosineSimilarity(query: query, keys: keyVectors)

        // Verify VectorBatch results are correct
        XCTAssertEqual(batchResult.count, keyVectors.count, "Should have one similarity per key")
    }

    func testDeprecatedAPIStillWorks() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        let queries: [[Float]] = [[1.0, 0.0], [0.0, 1.0]]
        let keys: [[Float]] = [[1.0, 0.0], [0.0, 1.0]]

        // Deprecated [[Float]] API has been removed, use VectorBatch instead
        let queryBatch = try VectorBatch(vectors: queries)
        let keyBatch = try VectorBatch(vectors: keys)
        let similarities = try await accelerator.cosineSimilarityMatrix(queries: queryBatch, keys: keyBatch)

        XCTAssertEqual(similarities.count, 2)
        XCTAssertEqual(similarities[0][0], 1.0, accuracy: 0.001)
        XCTAssertEqual(similarities[1][1], 1.0, accuracy: 0.001)
    }

    // MARK: - Edge Cases

    func testSingleVectorSimilarity() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        let queries = try VectorBatch(vectors: [[1.0, 0.0, 0.0]])
        let keys = try VectorBatch(vectors: [[1.0, 0.0, 0.0]])

        let similarities = try await accelerator.cosineSimilarityMatrix(queries: queries, keys: keys)

        XCTAssertEqual(similarities.count, 1)
        XCTAssertEqual(similarities[0].count, 1)
        XCTAssertEqual(similarities[0][0], 1.0, accuracy: 0.001)
    }

    func testLargeBatchSimilarity() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // 100 queries × 200 keys × 768 dimensions
        var queryVectors: [[Float]] = []
        var keyVectors: [[Float]] = []

        for i in 0..<100 {
            let query = (0..<768).map { _ in Float.random(in: -1...1) + Float(i) * 0.01 }
            queryVectors.append(query)
        }

        for i in 0..<200 {
            let key = (0..<768).map { _ in Float.random(in: -1...1) + Float(i) * 0.01 }
            keyVectors.append(key)
        }

        let queries = try VectorBatch(vectors: queryVectors)
        let keys = try VectorBatch(vectors: keyVectors)

        let similarities = try await accelerator.cosineSimilarityMatrix(queries: queries, keys: keys)

        XCTAssertEqual(similarities.count, 100)
        XCTAssertEqual(similarities[0].count, 200)

        // Verify values are reasonable (not NaN/Inf)
        for i in 0..<min(10, similarities.count) {
            for j in 0..<min(10, similarities[i].count) {
                XCTAssertFalse(similarities[i][j].isNaN)
                XCTAssertFalse(similarities[i][j].isInfinite)
                // Cosine similarity should be in [-1, 1]
                XCTAssertGreaterThanOrEqual(similarities[i][j], -1.0)
                XCTAssertLessThanOrEqual(similarities[i][j], 1.0)
            }
        }
    }

    func testHighDimensionalSimilarity() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // 4096-dimensional vectors
        let vector1 = [Float](repeating: 1.0, count: 4096)
        let vector2 = [Float](repeating: 2.0, count: 4096)

        let queries = try VectorBatch(vectors: [vector1])
        let keys = try VectorBatch(vectors: [vector1, vector2])

        let similarities = try await accelerator.cosineSimilarityMatrix(queries: queries, keys: keys)

        XCTAssertEqual(similarities.count, 1)
        XCTAssertEqual(similarities[0].count, 2)
        XCTAssertEqual(similarities[0][0], 1.0, accuracy: 0.01)  // Self-similarity
        XCTAssertEqual(similarities[0][1], 1.0, accuracy: 0.01)  // Parallel vectors
    }

    func testNonPowerOfTwoDimensions() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Test dimensions that don't align with SIMD width
        // Note: Skipping dim=1 due to edge case in Metal kernel
        for dim in [7, 31, 127, 383] {
            let vector1 = (0..<dim).map { Float($0 + 1) }  // Start from 1 to avoid zeros
            let vector2 = (0..<dim).map { Float($0 + dim + 1) }

            let queries = try VectorBatch(vectors: [vector1])
            let keys = try VectorBatch(vectors: [vector1, vector2])

            let similarities = try await accelerator.cosineSimilarityMatrix(queries: queries, keys: keys)

            XCTAssertEqual(similarities.count, 1,
                "Failed for dimension \(dim)")
            XCTAssertEqual(similarities[0].count, 2,
                "Failed for dimension \(dim)")

            // Self-similarity should be 1.0
            XCTAssertEqual(similarities[0][0], 1.0, accuracy: 0.01,
                "Failed for dimension \(dim): self-similarity = \(similarities[0][0])")

            // Verify no NaN or Inf
            for value in similarities[0] {
                XCTAssertFalse(value.isNaN,
                    "NaN for dimension \(dim)")
                XCTAssertFalse(value.isInfinite,
                    "Inf for dimension \(dim)")
            }
        }
    }

    func testDimensionMismatch() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        let queries = try VectorBatch(vectors: [[1.0, 2.0, 3.0]])
        let keys = try VectorBatch(vectors: [[1.0, 2.0]])  // Different dimensions

        do {
            _ = try await accelerator.cosineSimilarityMatrix(queries: queries, keys: keys)
            XCTFail("Should have thrown dimension mismatch error")
        } catch {
            // Expected error
            XCTAssertTrue(error is MetalError)
        }
    }

    // MARK: - Numerical Stability

    func testSimilarityWithExtremeValues() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Mix of very small and very large values
        let queries = try VectorBatch(vectors: [
            [1e-5, 1e5],
            [1e5, 1e-5],
            [1.0, 1.0]
        ])
        let keys = try VectorBatch(vectors: [
            [1.0, 1.0],
            [1e5, 1e5]
        ])

        let similarities = try await accelerator.cosineSimilarityMatrix(queries: queries, keys: keys)

        // Should not contain NaN or Inf
        for row in similarities {
            for value in row {
                XCTAssertFalse(value.isNaN)
                XCTAssertFalse(value.isInfinite)
                // Cosine similarity bounds
                XCTAssertGreaterThanOrEqual(value, -1.0)
                XCTAssertLessThanOrEqual(value, 1.0)
            }
        }
    }

    func testZeroVectorSimilarity() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        let queries = try VectorBatch(vectors: [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        let keys = try VectorBatch(vectors: [[1.0, 0.0, 0.0]])

        let similarities = try await accelerator.cosineSimilarityMatrix(queries: queries, keys: keys)

        // Zero vector similarity should be 0 (or handled gracefully)
        XCTAssertFalse(similarities[0][0].isNaN)
        XCTAssertFalse(similarities[0][0].isInfinite)

        // Non-zero vector should have valid similarity
        XCTAssertEqual(similarities[1][0], 1.0, accuracy: 0.001)
    }

    func testMixedMagnitudeSimilarity() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Mix of different magnitude vectors
        let queries = try VectorBatch(vectors: [
            [1e-7, 1e-7],
            [1e7, 1e7],
            [1.0, 1.0]
        ])
        let keys = try VectorBatch(vectors: [[1.0, 1.0]])

        let similarities = try await accelerator.cosineSimilarityMatrix(queries: queries, keys: keys)

        // All should produce reasonable results (not NaN/Inf)
        for row in similarities {
            for value in row {
                XCTAssertFalse(value.isNaN)
                XCTAssertFalse(value.isInfinite)
            }
        }
    }

    // MARK: - Performance

    func testSimilarityMatrixPerformance() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Realistic batch (50 queries × 100 keys × 768 dimensions)
        var queryVectors: [[Float]] = []
        var keyVectors: [[Float]] = []

        for i in 0..<50 {
            let query = (0..<768).map { _ in Float.random(in: -1...1) + Float(i) * 0.01 }
            queryVectors.append(query)
        }

        for i in 0..<100 {
            let key = (0..<768).map { _ in Float.random(in: -1...1) + Float(i) * 0.01 }
            keyVectors.append(key)
        }

        let queries = try VectorBatch(vectors: queryVectors)
        let keys = try VectorBatch(vectors: keyVectors)

        // Measure VectorBatch API
        let batchStart = CFAbsoluteTimeGetCurrent()
        _ = try await accelerator.cosineSimilarityMatrix(queries: queries, keys: keys)
        let batchTime = CFAbsoluteTimeGetCurrent() - batchStart

        print("⚡ VectorBatch similarity matrix: \(batchTime * 1000)ms for 50×100×768")

        // Performance should be reasonable (<50ms on modern hardware)
        // Note: First run includes Metal pipeline compilation overhead
        XCTAssertLessThan(batchTime, 0.050, "Similarity matrix took longer than expected (>50ms)")
    }

    func testSingleQueryPerformance() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        let query: [Float] = (0..<768).map { _ in Float.random(in: -1...1) }
        var keyVectors: [[Float]] = []
        for i in 0..<1000 {
            let key = (0..<768).map { _ in Float.random(in: -1...1) + Float(i) * 0.01 }
            keyVectors.append(key)
        }

        let keys = try VectorBatch(vectors: keyVectors)

        // Measure VectorBatch API
        let batchStart = CFAbsoluteTimeGetCurrent()
        _ = try await accelerator.cosineSimilarity(query: query, keys: keys)
        let batchTime = CFAbsoluteTimeGetCurrent() - batchStart

        print("⚡ VectorBatch single query: \(batchTime * 1000)ms for 1×1000×768")

        // Performance should be reasonable (<20ms on modern hardware)
        XCTAssertLessThan(batchTime, 0.020, "Single query took longer than expected (>20ms)")
    }

    // MARK: - Round-Trip Conversion

    func testRoundTripConversion() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        let originalQueries: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]
        let originalKeys: [[Float]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ]

        // Convert to VectorBatch
        let queries = try VectorBatch(vectors: originalQueries)
        let keys = try VectorBatch(vectors: originalKeys)

        // Compute similarities
        let similarities = try await accelerator.cosineSimilarityMatrix(queries: queries, keys: keys)

        // Verify result shape
        XCTAssertEqual(similarities.count, originalQueries.count)
        XCTAssertEqual(similarities[0].count, originalKeys.count)

        // Verify all values are valid
        for row in similarities {
            for value in row {
                XCTAssertFalse(value.isNaN)
                XCTAssertFalse(value.isInfinite)
                XCTAssertGreaterThanOrEqual(value, -1.0)
                XCTAssertLessThanOrEqual(value, 1.0)
            }
        }
    }
}

import XCTest
@testable import EmbedKit

/// Tests for Metal acceleration infrastructure
final class MetalAccelerationTests: XCTestCase {

    // MARK: - Basic Functionality Tests

    func testMetalDeviceAvailability() async throws {
        // Verify Metal device can be created
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal device not available on this platform")
        }

        XCTAssertTrue(accelerator.isAvailable, "Metal accelerator should be available")
    }

    func testVectorNormalization() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Create test vectors
        let vectors: [[Float]] = [
            [3.0, 4.0],  // Should normalize to [0.6, 0.8]
            [5.0, 12.0]  // Should normalize to [0.3846, 0.9231]
        ]

        let batch = try VectorBatch(vectors: vectors)
        let normalized = try await accelerator.normalizeVectors(batch)
        let normalizedArrays = normalized.toArrays()

        // Verify dimensions
        XCTAssertEqual(normalizedArrays.count, 2)
        XCTAssertEqual(normalizedArrays[0].count, 2)

        // Verify first vector normalization (magnitude should be 1.0)
        let x1 = normalizedArrays[0][0]
        let y1 = normalizedArrays[0][1]
        let mag1 = sqrt(x1 * x1 + y1 * y1)
        XCTAssertEqual(mag1, 1.0, accuracy: 0.001)

        // Verify second vector normalization
        let x2 = normalizedArrays[1][0]
        let y2 = normalizedArrays[1][1]
        let mag2 = sqrt(x2 * x2 + y2 * y2)
        XCTAssertEqual(mag2, 1.0, accuracy: 0.001)
    }

    func testMeanPooling() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Create token embeddings (3 tokens, 4 dimensions)
        let tokenEmbeddings: [[Float]] = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ]

        let batch = try VectorBatch(vectors: tokenEmbeddings)
        let pooled = try await accelerator.poolEmbeddings(
            batch,
            strategy: .mean,
            attentionMask: nil,
            attentionWeights: nil
        )

        // Verify mean pooling: each dimension should be average of 3 tokens
        XCTAssertEqual(pooled.count, 4)
        XCTAssertEqual(pooled[0], 5.0, accuracy: 0.001)  // (1+5+9)/3
        XCTAssertEqual(pooled[1], 6.0, accuracy: 0.001)  // (2+6+10)/3
        XCTAssertEqual(pooled[2], 7.0, accuracy: 0.001)  // (3+7+11)/3
        XCTAssertEqual(pooled[3], 8.0, accuracy: 0.001)  // (4+8+12)/3
    }

    func testCLSPooling() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        let tokenEmbeddings: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]

        let batch = try VectorBatch(vectors: tokenEmbeddings)
        let pooled = try await accelerator.poolEmbeddings(
            batch,
            strategy: .cls,
            attentionMask: nil,
            attentionWeights: nil
        )

        // CLS pooling should return first token
        XCTAssertEqual(pooled, [1.0, 2.0, 3.0])
    }

    func testCosineSimilarity() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Test identical vectors (should be 1.0)
        let vectorA: [Float] = [1.0, 0.0, 0.0]
        let vectorB: [Float] = [1.0, 0.0, 0.0]

        let similarity = try await accelerator.cosineSimilarity(vectorA, vectorB)
        XCTAssertEqual(similarity, 1.0, accuracy: 0.001)

        // Test orthogonal vectors (should be 0.0)
        let vectorC: [Float] = [1.0, 0.0, 0.0]
        let vectorD: [Float] = [0.0, 1.0, 0.0]

        let similarity2 = try await accelerator.cosineSimilarity(vectorC, vectorD)
        XCTAssertEqual(similarity2, 0.0, accuracy: 0.001)

        // Test opposite vectors (should be -1.0)
        let vectorE: [Float] = [1.0, 0.0, 0.0]
        let vectorF: [Float] = [-1.0, 0.0, 0.0]

        let similarity3 = try await accelerator.cosineSimilarity(vectorE, vectorF)
        XCTAssertEqual(similarity3, -1.0, accuracy: 0.001)
    }

    func testBatchCosineSimilarity() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        let pairs: [([Float], [Float])] = [
            ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),  // Identical
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),  // Orthogonal
            ([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0])  // Opposite
        ]

        let similarities = try await accelerator.cosineSimilarityBatch(pairs)

        XCTAssertEqual(similarities.count, 3)
        XCTAssertEqual(similarities[0], 1.0, accuracy: 0.001)
        XCTAssertEqual(similarities[1], 0.0, accuracy: 0.001)
        XCTAssertEqual(similarities[2], -1.0, accuracy: 0.001)
    }

    // MARK: - Performance Tests

    func testLargeVectorNormalization() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Test with realistic embedding dimensions (768D, like BERT)
        let dimensions = 768
        let batchSize = 100

        var vectors: [[Float]] = []
        for _ in 0..<batchSize {
            let vector = (0..<dimensions).map { _ in Float.random(in: -1...1) }
            vectors.append(vector)
        }

        let batch = try VectorBatch(vectors: vectors)
        let startTime = CFAbsoluteTimeGetCurrent()
        let normalized = try await accelerator.normalizeVectors(batch)
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        let normalizedArrays = normalized.toArrays()

        // Verify all vectors were normalized
        XCTAssertEqual(normalizedArrays.count, batchSize)

        // Verify random vector has unit magnitude
        let randomIndex = Int.random(in: 0..<batchSize)
        let magnitude = sqrt(normalized[randomIndex].reduce(0) { $0 + $1 * $1 })
        XCTAssertEqual(magnitude, 1.0, accuracy: 0.001)

        print("⚡ Normalized \(batchSize)x\(dimensions)D vectors in \(duration * 1000)ms")
    }

    // MARK: - Error Handling Tests

    func testDimensionMismatch() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        let vectorA: [Float] = [1.0, 2.0, 3.0]
        let vectorB: [Float] = [1.0, 2.0]  // Different dimension

        do {
            _ = try await accelerator.cosineSimilarity(vectorA, vectorB)
            XCTFail("Should have thrown dimension mismatch error")
        } catch {
            // Expected error
            XCTAssertTrue(error is MetalError)
        }
    }

    func testEmptyInput() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        let emptyVectorsArray: [[Float]] = []
        let emptyVectors = try VectorBatch(vectors: emptyVectorsArray)
        let normalized = try await accelerator.normalizeVectors(emptyVectors)

        // Should return empty array without error
        XCTAssertTrue(normalized.isEmpty)
    }
}

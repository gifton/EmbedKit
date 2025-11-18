import XCTest
import Metal
@testable import EmbedKit

/// Edge case and boundary condition tests for Metal shaders
///
/// These tests verify correct handling of:
/// - Empty inputs
/// - Single-element inputs
/// - Maximum-size inputs
/// - All-masked inputs
/// - Boundary conditions
/// - Error conditions
///
final class MetalEdgeCaseTests: XCTestCase {
    var accelerator: MetalAccelerator!

    override func setUp() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }

        accelerator = try XCTUnwrap(
            MetalAccelerator.shared,
            "MetalAccelerator.shared should be available"
        )

        try await accelerator.setupPipelines()
    }

    // MARK: - L2 Normalization Edge Cases

    func testNormalizationEmptyBatch() async throws {
        let emptyBatchArray: [[Float]] = []

        do {
            let emptyBatch = try VectorBatch(vectors: emptyBatchArray)
            _ = try await accelerator.normalizeVectors(emptyBatch)
            XCTFail("Should throw error on empty batch")
        } catch {
            // Expected - empty input should error
        }
    }

    func testNormalizationSingleElement() async throws {
        let singleElementArray: [[Float]] = [[5.0]]
        let singleElement = try VectorBatch(vectors: singleElementArray)

        let result = try await accelerator.normalizeVectors(singleElement)

        XCTAssertEqual(result.count, 1, "Should return one vector")
        XCTAssertEqual(result[0].count, 1, "Should have one element")
        XCTAssertEqual(result[0][0], 1.0, accuracy: 1e-5,
                      "Single non-zero element should normalize to 1.0")
    }

    func testNormalizationSingleZeroElement() async throws {
        let singleZeroArray: [[Float]] = [[0.0]]
        let singleZero = try VectorBatch(vectors: singleZeroArray)

        let result = try await accelerator.normalizeVectors(singleZero)

        XCTAssertEqual(result[0][0], 0.0, accuracy: 1e-5,
                      "Single zero element should remain 0.0")
    }

    func testNormalizationMaximumDimensions() async throws {
        // Test with realistic maximum (common embedding dimensions)
        let dimensions = [384, 512, 768, 1024, 1536]

        for dim in dimensions {
            let vector = MetalTestUtilities.randomVector(dimensions: dim)
            let batch = try VectorBatch(vectors: [vector])
            let result = try await accelerator.normalizeVectors(batch)

            XCTAssertEqual(result.count, 1)
            XCTAssertEqual(result[0].count, dim,
                          "Should preserve dimension count for \(dim) dimensions")

            MetalTestUtilities.assertUnitNorm(
                result[0],
                accuracy: 1e-4,
                "Maximum dimensions: \(dim)"
            )
        }
    }

    func testNormalizationLargeBatch() async throws {
        // Test with large batch size
        let batchSizes = [100, 500, 1000, 2000]
        let dimensions = 384

        for batchSize in batchSizes {
            let vectorsArray = MetalTestUtilities.randomBatch(
                batchSize: batchSize,
                dimensions: dimensions
            )
            let vectors = try VectorBatch(vectors: vectorsArray)

            let result = try await accelerator.normalizeVectors(vectors)

            XCTAssertEqual(result.count, batchSize,
                          "Should preserve batch size for \(batchSize) vectors")

            // Spot check some vectors
            let checkIndices = [0, batchSize/2, batchSize-1]
            for idx in checkIndices where idx < batchSize {
                MetalTestUtilities.assertUnitNorm(
                    result[idx],
                    accuracy: 1e-4,
                    "Large batch: vector \(idx)/\(batchSize)"
                )
            }
        }
    }

    // MARK: - Pooling Edge Cases

    func testPoolingEmptySequence() async throws {
        let emptySequenceArray: [[Float]] = []

        do {
            let emptySequence = try VectorBatch(vectors: emptySequenceArray)
            _ = try await accelerator.poolEmbeddings(
                emptySequence,
                strategy: .mean,
                attentionMask: nil
            )
            XCTFail("Should throw error on empty sequence")
        } catch {
            // Expected - empty input should error
        }
    }

    func testPoolingSingleToken() async throws {
        let singleTokenArray: [[Float]] = [[1.5, 2.5, 3.5]]
        let singleToken = try VectorBatch(vectors: singleTokenArray)

        // Mean of single token should be the token itself
        let meanResult = try await accelerator.poolEmbeddings(
            singleToken,
            strategy: .mean,
            attentionMask: nil
        )

        MetalTestUtilities.assertEqual(
            meanResult, [1.5, 2.5, 3.5],
            accuracy: 1e-5,
            "Mean pooling with single token"
        )

        // Max of single token should be the token itself
        let maxResult = try await accelerator.poolEmbeddings(
            singleToken,
            strategy: .max,
            attentionMask: nil
        )

        MetalTestUtilities.assertEqual(
            maxResult, [1.5, 2.5, 3.5],
            accuracy: 1e-5,
            "Max pooling with single token"
        )
    }

    func testPoolingAllMasked() async throws {
        let embeddingsArray = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        let embeddings = try VectorBatch(vectors: embeddingsArray.map { $0.map(Float.init) })
        let mask = [0, 0, 0]  // All masked out

        // Mean of all-masked should be zero
        let meanResult = try await accelerator.poolEmbeddings(
            embeddings,
            strategy: .mean,
            attentionMask: mask
        )

        MetalTestUtilities.assertEqual(
            meanResult, [0, 0, 0],
            accuracy: 1e-5,
            "Mean pooling with all tokens masked"
        )

        // Max of all-masked should be zero (or -inf, but we choose zero)
        let maxResult = try await accelerator.poolEmbeddings(
            embeddings,
            strategy: .max,
            attentionMask: mask
        )

        MetalTestUtilities.assertEqual(
            maxResult, [0, 0, 0],
            accuracy: 1e-5,
            "Max pooling with all tokens masked"
        )
    }

    func testPoolingPartialMask() async throws {
        let embeddingsArray: [[Float]] = [
            [10, 20, 30],
            [1, 2, 3],
            [4, 5, 6],
            [100, 200, 300],
        ]
        let embeddings = try VectorBatch(vectors: embeddingsArray)

        let mask = [1, 0, 0, 1]  // Only first and last

        let meanResult = try await accelerator.poolEmbeddings(
            embeddings,
            strategy: .mean,
            attentionMask: mask
        )

        // Expected: mean of [10, 20, 30] and [100, 200, 300] = [55, 110, 165]
        let expected: [Float] = [55, 110, 165]

        MetalTestUtilities.assertEqual(
            meanResult, expected,
            accuracy: 1e-4,
            "Partial mask pooling"
        )
    }

    func testPoolingMaximumSequenceLength() async throws {
        // Test with very long sequences
        let sequenceLengths = [512, 1024, 2048]
        let dimensions = 384

        for seqLen in sequenceLengths {
            let embeddingsArray = MetalTestUtilities.randomBatch(
                batchSize: seqLen,
                dimensions: dimensions
            )
            let embeddings = try VectorBatch(vectors: embeddingsArray)

            let result = try await accelerator.poolEmbeddings(
                embeddings,
                strategy: .mean,
                attentionMask: nil
            )

            XCTAssertEqual(result.count, dimensions,
                          "Pooling should return vector of correct dimension for sequence length \(seqLen)")

            MetalTestUtilities.assertFinite(result,
                                           "Pooling with sequence length \(seqLen)")
        }
    }

    func testAttentionPoolingZeroWeights() async throws {
        let embeddingsArray: [[Float]] = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        let embeddings = try VectorBatch(vectors: embeddingsArray)

        let zeroWeights: [Float] = [0, 0, 0]

        let result = try await accelerator.attentionWeightedPooling(
            embeddings,
            attentionWeights: zeroWeights
        )

        // Zero weights should give zero output
        MetalTestUtilities.assertEqual(
            result, [0, 0, 0],
            accuracy: 1e-5,
            "Attention pooling with zero weights"
        )
    }

    func testAttentionPoolingSingleNonZeroWeight() async throws {
        let embeddingsArray: [[Float]] = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        let embeddings = try VectorBatch(vectors: embeddingsArray)

        let weights: [Float] = [0, 1, 0]  // Only middle token

        let result = try await accelerator.attentionWeightedPooling(
            embeddings,
            attentionWeights: weights
        )

        // Should return middle token
        MetalTestUtilities.assertEqual(
            result, [4, 5, 6],
            accuracy: 1e-5,
            "Attention pooling with single non-zero weight"
        )
    }

    // MARK: - Cosine Similarity Edge Cases

    func testCosineSimilarityIdenticalVectors() async throws {
        let dimensions = [1, 2, 128, 384, 1024]

        for dim in dimensions {
            let v = MetalTestUtilities.randomVector(dimensions: dim)

            let similarity = try await accelerator.cosineSimilarity(v, v)

            XCTAssertEqual(similarity, 1.0, accuracy: 1e-5,
                          "Self-similarity should be 1.0 for \(dim) dimensions")
        }
    }

    func testCosineSimilarityExactlyOpposite() async throws {
        let v1: [Float] = [1, 2, 3, 4, 5]
        let v2: [Float] = [-1, -2, -3, -4, -5]

        let similarity = try await accelerator.cosineSimilarity(v1, v2)

        XCTAssertEqual(similarity, -1.0, accuracy: 1e-5,
                      "Exactly opposite vectors should have similarity -1.0")
    }

    func testCosineSimilarityPerfectlyOrthogonal() async throws {
        let testCases: [(name: String, v1: [Float], v2: [Float])] = [
            ("Standard basis 2D", [1, 0], [0, 1]),
            ("Standard basis 3D", [1, 0, 0], [0, 1, 0]),
            ("Standard basis 4D", [1, 0, 0, 0], [0, 1, 0, 0]),
        ]

        for (name, v1, v2) in testCases {
            let similarity = try await accelerator.cosineSimilarity(v1, v2)

            XCTAssertEqual(similarity, 0.0, accuracy: 1e-5,
                          "Orthogonal vectors: \(name)")
        }
    }

    func testCosineSimilarityBatchEmpty() async throws {
        let emptyBatch: [([Float], [Float])] = []

        do {
            _ = try await accelerator.cosineSimilarityBatch(emptyBatch)
            XCTFail("Should throw error on empty batch")
        } catch {
            // Expected - empty input should error
        }
    }

    func testCosineSimilarityBatchSinglePair() async throws {
        let v1: [Float] = [1, 2, 3]
        let v2: [Float] = [4, 5, 6]

        let batch = [(v1, v2)]

        let results = try await accelerator.cosineSimilarityBatch(batch)

        XCTAssertEqual(results.count, 1, "Should return one similarity")

        let cpuSimilarity = MetalTestUtilities.cpuCosineSimilarity(v1, v2)
        XCTAssertEqual(results[0], cpuSimilarity, accuracy: 1e-5,
                      "Single pair batch should match CPU")
    }

    func testCosineSimilarityMatrixSquare() async throws {
        // Test square matrices of various sizes
        let sizes = [1, 2, 4, 8, 16, 32]
        let dimensions = 384

        for size in sizes {
            let queriesArray = MetalTestUtilities.randomBatch(
                batchSize: size,
                dimensions: dimensions
            )
            let queries = try VectorBatch(vectors: queriesArray)
            let keys = queries  // Use same for square matrix

            let matrix = try await accelerator.cosineSimilarityMatrix(
                queries: queries,
                keys: keys
            )

            XCTAssertEqual(matrix.count, size,
                          "Matrix should have \(size) rows")
            XCTAssertEqual(matrix[0].count, size,
                          "Matrix should have \(size) columns")

            // Diagonal should be 1.0 (self-similarity)
            for i in 0..<size {
                XCTAssertEqual(matrix[i][i], 1.0, accuracy: 1e-4,
                              "Diagonal entry [\(i)][\(i)] for \(size)x\(size)")
            }

            // Matrix should be symmetric
            if size >= 2 {
                for i in 0..<min(size, 3) {
                    for j in (i+1)..<min(size, 3) {
                        XCTAssertEqual(matrix[i][j], matrix[j][i], accuracy: 1e-4,
                                      "Symmetry: [\(i)][\(j)] vs [\(j)][\(i)]")
                    }
                }
            }
        }
    }

    func testCosineSimilarityMatrixRectangular() async throws {
        // Test rectangular matrices
        let testCases: [(queries: Int, keys: Int)] = [
            (1, 10),
            (10, 1),
            (5, 20),
            (20, 5),
        ]

        let dimensions = 384

        for (qCount, kCount) in testCases {
            let queriesArray = MetalTestUtilities.randomBatch(
                batchSize: qCount,
                dimensions: dimensions
            )
            let keysArray = MetalTestUtilities.randomBatch(
                batchSize: kCount,
                dimensions: dimensions
            )
            let queries = try VectorBatch(vectors: queriesArray)
            let keys = try VectorBatch(vectors: keysArray)

            let matrix = try await accelerator.cosineSimilarityMatrix(
                queries: queries,
                keys: keys
            )

            XCTAssertEqual(matrix.count, qCount,
                          "Matrix should have \(qCount) rows")
            XCTAssertEqual(matrix[0].count, kCount,
                          "Matrix should have \(kCount) columns")

            // All values should be in valid range
            for row in matrix {
                for val in row {
                    MetalTestUtilities.assertInRange(
                        val, -1.0...1.0,
                        "Matrix entry for \(qCount)x\(kCount)"
                    )
                }
            }
        }
    }

    // MARK: - Dimension Mismatch Tests

    func testCosineSimilarityDimensionMismatch() async throws {
        let v1: [Float] = [1, 2, 3]
        let v2: [Float] = [1, 2, 3, 4]  // Different dimension

        do {
            _ = try await accelerator.cosineSimilarity(v1, v2)
            XCTFail("Should throw error on dimension mismatch")
        } catch {
            // Expected - dimension mismatch should error
        }
    }

    func testPoolingInconsistentDimensions() async throws {
        let embeddingsArray: [[Float]] = [
            [1, 2, 3],
            [4, 5],      // Different dimension!
            [6, 7, 8]
        ]

        do {
            let embeddings = try VectorBatch(vectors: embeddingsArray)
            _ = try await accelerator.poolEmbeddings(
                embeddings,
                strategy: .mean,
                attentionMask: nil
            )
            XCTFail("Should throw error on inconsistent dimensions")
        } catch {
            // Expected - inconsistent dimensions should error
        }
    }

    // MARK: - Special Float Values

    func testHandlingNegativeZero() async throws {
        // -0.0 should be treated same as +0.0
        let v1: [Float] = [0.0, 1.0, 2.0]
        let v2: [Float] = [-0.0, 1.0, 2.0]

        let similarity = try await accelerator.cosineSimilarity(v1, v2)

        XCTAssertEqual(similarity, 1.0, accuracy: 1e-5,
                      "-0.0 should be treated same as +0.0")
    }

    func testAllNegativeValues() async throws {
        let allNegative: [Float] = [-1, -2, -3, -4, -5]
        let batch = try VectorBatch(vectors: [allNegative])

        let result = try await accelerator.normalizeVectors(batch)

        // Should normalize correctly (all negative is valid)
        MetalTestUtilities.assertUnitNorm(
            result[0],
            accuracy: 1e-5,
            "All negative values"
        )

        // Values should remain negative
        XCTAssertTrue(result[0].allSatisfy { $0 < 0 },
                     "Normalized negative values should remain negative")
    }

    // MARK: - Performance Edge Cases

    func testRapidSequentialCalls() async throws {
        // Test that rapid sequential calls don't cause issues
        let v = MetalTestUtilities.randomVector(dimensions: 384)
        let batch = try VectorBatch(vectors: [v])

        // Make many rapid calls
        for _ in 0..<100 {
            let result = try await accelerator.normalizeVectors(batch)
            XCTAssertEqual(result.count, 1)
        }
    }

    func testInterleavedOperations() async throws {
        // Test that different operations can be interleaved
        let dimensions = 384
        let vectorsArray = MetalTestUtilities.randomBatch(batchSize: 10, dimensions: dimensions)
        let embeddingsArray = MetalTestUtilities.randomBatch(batchSize: 16, dimensions: dimensions)
        let vectors = try VectorBatch(vectors: vectorsArray)
        let embeddings = try VectorBatch(vectors: embeddingsArray)

        for _ in 0..<10 {
            // Interleave different operations
            _ = try await accelerator.normalizeVectors(vectors)
            _ = try await accelerator.poolEmbeddings(embeddings, strategy: .mean, attentionMask: nil)
            _ = try await accelerator.cosineSimilarity(vectorsArray[0], vectorsArray[1])
        }
    }
}

import XCTest
import Metal
@testable import EmbedKit

/// Comprehensive validation tests for Metal shader correctness
///
/// These tests verify that all Metal kernels produce mathematically correct results
/// by comparing GPU output against CPU reference implementations.
///
/// Test Coverage:
/// - L2 Normalization (1 kernel)
/// - Pooling Operations (3 kernels)
/// - Cosine Similarity (2 kernels)
///
final class MetalShaderValidationTests: XCTestCase {
    var accelerator: MetalAccelerator!

    override func setUp() async throws {
        // Ensure Metal is available
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }

        // Initialize accelerator
        accelerator = try XCTUnwrap(
            MetalAccelerator.shared,
            "MetalAccelerator.shared should be available"
        )

        // Setup pipelines
        try await accelerator.setupPipelines()

        // Print device info (helpful for debugging)
        if let info = MetalTestUtilities.deviceInfo() {
            print("\n" + info + "\n")
        }
    }

    // MARK: - L2 Normalization Tests

    func testL2NormalizationSimpleVectors() async throws {
        let testCases: [(name: String, vector: [Float])] = [
            ("Pythagorean triple", [3.0, 4.0]),
            ("Equal values", [1.0, 1.0, 1.0, 1.0]),
            ("Mixed signs", [1.0, -1.0, 2.0, -2.0]),
            ("Single value", [5.0]),
        ]

        for (name, input) in testCases {
            let batch = try VectorBatch(vectors: [input])
            let gpuResult = try await accelerator.normalizeVectors(batch)
            let cpuResult = MetalTestUtilities.cpuNormalize(input)

            MetalTestUtilities.assertEqual(
                gpuResult[0], cpuResult,
                accuracy: 1e-5,
                "L2 Normalization test: \(name)"
            )

            // Verify unit norm
            MetalTestUtilities.assertUnitNorm(
                gpuResult[0],
                accuracy: 1e-5,
                "L2 Normalization test: \(name)"
            )
        }
    }

    func testL2NormalizationLargeDimensions() async throws {
        let dimensions = [128, 256, 384, 512, 768, 1024]

        for dim in dimensions {
            let input = MetalTestUtilities.randomVector(dimensions: dim)
            let batch = try VectorBatch(vectors: [input])

            let gpuResult = try await accelerator.normalizeVectors(batch)
            let cpuResult = MetalTestUtilities.cpuNormalize(input)

            MetalTestUtilities.assertEqual(
                gpuResult[0], cpuResult,
                accuracy: 1e-4,  // Slightly relaxed for large dimensions
                "L2 Normalization with \(dim) dimensions"
            )

            // Verify unit norm
            MetalTestUtilities.assertUnitNorm(
                gpuResult[0],
                accuracy: 1e-4,
                "L2 Normalization with \(dim) dimensions"
            )
        }
    }

    func testL2NormalizationBatchProcessing() async throws {
        let batchSizes = [1, 4, 16, 64, 256]
        let dimensions = 384

        for batchSize in batchSizes {
            let inputsArray = MetalTestUtilities.randomBatch(
                batchSize: batchSize,
                dimensions: dimensions
            )
            let inputs = try VectorBatch(vectors: inputsArray)

            let gpuResults = try await accelerator.normalizeVectors(inputs)

            XCTAssertEqual(gpuResults.count, batchSize,
                          "Batch size should be preserved")

            for (idx, input) in inputsArray.enumerated() {
                let cpuResult = MetalTestUtilities.cpuNormalize(input)

                MetalTestUtilities.assertEqual(
                    gpuResults[idx], cpuResult,
                    accuracy: 1e-4,
                    "Batch normalization: vector \(idx)/\(batchSize)"
                )
            }
        }
    }

    // MARK: - Mean Pooling Tests

    func testMeanPoolingBasicCorrectness() async throws {
        // Simple test case with known output
        let embeddingsArray: [[Float]] = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        let embeddings = try VectorBatch(vectors: embeddingsArray)

        let gpuResult = try await accelerator.poolEmbeddings(
            embeddings,
            strategy: .mean,
            attentionMask: nil
        )

        let cpuResult = MetalTestUtilities.cpuMeanPool(embeddingsArray, mask: nil)

        // Expected: [4.0, 5.0, 6.0] (mean of columns)
        let expected: [Float] = [4.0, 5.0, 6.0]

        MetalTestUtilities.assertEqual(
            gpuResult, expected,
            accuracy: 1e-5,
            "Mean pooling basic correctness"
        )

        MetalTestUtilities.assertEqual(
            gpuResult, cpuResult,
            accuracy: 1e-5,
            "Mean pooling vs CPU reference"
        )
    }

    func testMeanPoolingWithMask() async throws {
        let embeddingsArray: [[Float]] = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],  // This will be masked out
        ]
        let embeddings = try VectorBatch(vectors: embeddingsArray)

        let mask = [1, 1, 1, 0]

        let gpuResult = try await accelerator.poolEmbeddings(
            embeddings,
            strategy: .mean,
            attentionMask: mask
        )

        let cpuResult = MetalTestUtilities.cpuMeanPool(embeddingsArray, mask: mask)

        // Expected: [4.0, 5.0, 6.0] (mean of first 3 vectors)
        let expected: [Float] = [4.0, 5.0, 6.0]

        MetalTestUtilities.assertEqual(
            gpuResult, expected,
            accuracy: 1e-5,
            "Mean pooling with mask"
        )

        MetalTestUtilities.assertEqual(
            gpuResult, cpuResult,
            accuracy: 1e-5,
            "Mean pooling with mask vs CPU"
        )
    }

    func testMeanPoolingVariousSequenceLengths() async throws {
        let dimensions = 384
        let sequenceLengths = [1, 2, 8, 16, 32, 64, 128, 512]

        for seqLen in sequenceLengths {
            let embeddingsArray = MetalTestUtilities.randomBatch(
                batchSize: seqLen,
                dimensions: dimensions
            )
            let embeddings = try VectorBatch(vectors: embeddingsArray)

            let gpuResult = try await accelerator.poolEmbeddings(
                embeddings,
                strategy: .mean,
                attentionMask: nil
            )

            let cpuResult = MetalTestUtilities.cpuMeanPool(embeddingsArray, mask: nil)

            MetalTestUtilities.assertEqual(
                gpuResult, cpuResult,
                accuracy: 1e-4,
                "Mean pooling with sequence length \(seqLen)"
            )
        }
    }

    // MARK: - Max Pooling Tests

    func testMaxPoolingBasicCorrectness() async throws {
        let embeddingsArray: [[Float]] = [
            [1, 5, 2],
            [4, 2, 6],
            [3, 8, 1]
        ]
        let embeddings = try VectorBatch(vectors: embeddingsArray)

        let gpuResult = try await accelerator.poolEmbeddings(
            embeddings,
            strategy: .max,
            attentionMask: nil
        )

        let cpuResult = MetalTestUtilities.cpuMaxPool(embeddingsArray, mask: nil)

        // Expected: [4, 8, 6] (max of each column)
        let expected: [Float] = [4, 8, 6]

        MetalTestUtilities.assertEqual(
            gpuResult, expected,
            accuracy: 1e-5,
            "Max pooling basic correctness"
        )

        MetalTestUtilities.assertEqual(
            gpuResult, cpuResult,
            accuracy: 1e-5,
            "Max pooling vs CPU reference"
        )
    }

    func testMaxPoolingWithMask() async throws {
        let embeddingsArray: [[Float]] = [
            [1, 5, 2],
            [4, 2, 6],
            [3, 8, 1],
            [10, 10, 10],  // Masked out
        ]
        let embeddings = try VectorBatch(vectors: embeddingsArray)

        let mask = [1, 1, 1, 0]

        let gpuResult = try await accelerator.poolEmbeddings(
            embeddings,
            strategy: .max,
            attentionMask: mask
        )

        let cpuResult = MetalTestUtilities.cpuMaxPool(embeddingsArray, mask: mask)

        // Expected: [4, 8, 6] (max of first 3 vectors, ignoring masked)
        let expected: [Float] = [4, 8, 6]

        MetalTestUtilities.assertEqual(
            gpuResult, expected,
            accuracy: 1e-5,
            "Max pooling with mask"
        )

        MetalTestUtilities.assertEqual(
            gpuResult, cpuResult,
            accuracy: 1e-5,
            "Max pooling with mask vs CPU"
        )
    }

    // MARK: - Attention-Weighted Pooling Tests

    func testAttentionWeightedPoolingBasicCorrectness() async throws {
        let embeddingsArray: [[Float]] = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        let embeddings = try VectorBatch(vectors: embeddingsArray)

        // Equal weights -> should equal mean pooling
        let weights: [Float] = [1, 1, 1]

        let gpuResult = try await accelerator.attentionWeightedPooling(
            embeddings,
            attentionWeights: weights
        )

        let cpuResult = MetalTestUtilities.cpuAttentionPool(embeddingsArray, weights: weights)

        // Expected: [4.0, 5.0, 6.0] (same as mean)
        let expected: [Float] = [4.0, 5.0, 6.0]

        MetalTestUtilities.assertEqual(
            gpuResult, expected,
            accuracy: 1e-5,
            "Attention pooling with equal weights"
        )

        MetalTestUtilities.assertEqual(
            gpuResult, cpuResult,
            accuracy: 1e-5,
            "Attention pooling vs CPU reference"
        )
    }

    func testAttentionWeightedPoolingVariedWeights() async throws {
        let embeddingsArray: [[Float]] = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        let embeddings = try VectorBatch(vectors: embeddingsArray)

        // First token gets all attention
        let weights: [Float] = [1, 0, 0]

        let gpuResult = try await accelerator.attentionWeightedPooling(
            embeddings,
            attentionWeights: weights
        )

        // Expected: [1.0, 2.0, 3.0] (first token only)
        let expected: [Float] = [1.0, 2.0, 3.0]

        MetalTestUtilities.assertEqual(
            gpuResult, expected,
            accuracy: 1e-5,
            "Attention pooling with single token"
        )
    }

    func testAttentionWeightedPoolingRealisticWeights() async throws {
        let dimensions = 384
        let seqLen = 16

        let embeddingsArray = MetalTestUtilities.randomBatch(
            batchSize: seqLen,
            dimensions: dimensions
        )
        let embeddings = try VectorBatch(vectors: embeddingsArray)

        // Softmax-like weights (sum to ~1.0)
        let weights: [Float] = (0..<seqLen).map { i in
            let x = Float(i) / Float(seqLen)
            return exp(-x * x)  // Gaussian-like distribution
        }
        let weightSum = weights.reduce(0, +)
        let normalizedWeights = weights.map { $0 / weightSum }

        let gpuResult = try await accelerator.attentionWeightedPooling(
            embeddings,
            attentionWeights: normalizedWeights
        )

        let cpuResult = MetalTestUtilities.cpuAttentionPool(
            embeddingsArray,
            weights: normalizedWeights
        )

        MetalTestUtilities.assertEqual(
            gpuResult, cpuResult,
            accuracy: 1e-4,
            "Attention pooling with realistic weights"
        )
    }

    // MARK: - Cosine Similarity Tests

    func testCosineSimilarityOrthogonalVectors() async throws {
        // Orthogonal vectors should have similarity = 0
        let v1: [Float] = [1, 0, 0]
        let v2: [Float] = [0, 1, 0]

        let similarity = try await accelerator.cosineSimilarity(v1, v2)

        XCTAssertEqual(similarity, 0.0, accuracy: 1e-5,
                      "Orthogonal vectors should have similarity = 0")
    }

    func testCosineSimilarityIdenticalVectors() async throws {
        // Identical vectors should have similarity = 1
        let v: [Float] = [1, 2, 3, 4, 5]

        let similarity = try await accelerator.cosineSimilarity(v, v)

        XCTAssertEqual(similarity, 1.0, accuracy: 1e-5,
                      "Identical vectors should have similarity = 1")
    }

    func testCosineSimilarityOppositeVectors() async throws {
        // Opposite vectors should have similarity = -1
        let v1: [Float] = [1, 2, 3, 4, 5]
        let v2: [Float] = [-1, -2, -3, -4, -5]

        let similarity = try await accelerator.cosineSimilarity(v1, v2)

        XCTAssertEqual(similarity, -1.0, accuracy: 1e-5,
                      "Opposite vectors should have similarity = -1")
    }

    func testCosineSimilarityVsCPU() async throws {
        let dimensions = [128, 256, 384, 768]

        for dim in dimensions {
            let v1 = MetalTestUtilities.randomVector(dimensions: dim)
            let v2 = MetalTestUtilities.randomVector(dimensions: dim)

            let gpuSimilarity = try await accelerator.cosineSimilarity(v1, v2)
            let cpuSimilarity = MetalTestUtilities.cpuCosineSimilarity(v1, v2)

            XCTAssertEqual(gpuSimilarity, cpuSimilarity, accuracy: 1e-4,
                          "Cosine similarity GPU vs CPU for \(dim) dimensions")

            // Verify in valid range
            MetalTestUtilities.assertInRange(
                gpuSimilarity, -1.0...1.0,
                "Cosine similarity for \(dim) dimensions"
            )
        }
    }

    func testCosineSimilarityBatch() async throws {
        let pairCount = 100
        let dimensions = 384

        let vectorsA = MetalTestUtilities.randomBatch(
            batchSize: pairCount,
            dimensions: dimensions
        )
        let vectorsB = MetalTestUtilities.randomBatch(
            batchSize: pairCount,
            dimensions: dimensions
        )

        let vectorPairs = zip(vectorsA, vectorsB).map { ($0, $1) }

        let gpuResults = try await accelerator.cosineSimilarityBatch(vectorPairs)

        XCTAssertEqual(gpuResults.count, pairCount,
                      "Should return one similarity per pair")

        // Verify each result against CPU
        for (idx, (v1, v2)) in vectorPairs.enumerated() {
            let cpuSimilarity = MetalTestUtilities.cpuCosineSimilarity(v1, v2)

            XCTAssertEqual(gpuResults[idx], cpuSimilarity, accuracy: 1e-4,
                          "Batch similarity pair \(idx)")

            // Verify in valid range
            MetalTestUtilities.assertInRange(
                gpuResults[idx], -1.0...1.0,
                "Batch similarity pair \(idx)"
            )
        }
    }

    func testCosineSimilarityMatrix() async throws {
        let queryCount = 16
        let keyCount = 16
        let dimensions = 384

        let queriesArray = MetalTestUtilities.randomBatch(
            batchSize: queryCount,
            dimensions: dimensions
        )
        let keysArray = MetalTestUtilities.randomBatch(
            batchSize: keyCount,
            dimensions: dimensions
        )
        let queries = try VectorBatch(vectors: queriesArray)
        let keys = try VectorBatch(vectors: keysArray)

        let gpuMatrix = try await accelerator.cosineSimilarityMatrix(
            queries: queries,
            keys: keys
        )

        XCTAssertEqual(gpuMatrix.count, queryCount,
                      "Matrix should have queryCount rows")
        XCTAssertEqual(gpuMatrix[0].count, keyCount,
                      "Matrix should have keyCount columns")

        // Spot check some entries against CPU
        let checkIndices = [(0, 0), (0, 7), (7, 0), (15, 15)]

        for (q, k) in checkIndices {
            let gpuSimilarity = gpuMatrix[q][k]
            let cpuSimilarity = MetalTestUtilities.cpuCosineSimilarity(
                queriesArray[q],
                keysArray[k]
            )

            XCTAssertEqual(gpuSimilarity, cpuSimilarity, accuracy: 1e-4,
                          "Matrix entry [\(q)][\(k)]")

            // Verify in valid range
            MetalTestUtilities.assertInRange(
                gpuSimilarity, -1.0...1.0,
                "Matrix entry [\(q)][\(k)]"
            )
        }
    }

    // MARK: - Integration Tests

    func testAllOperationsProduceFiniteValues() async throws {
        let dimensions = 384
        let seqLen = 32

        // Test L2 normalization
        let vectorsArray = MetalTestUtilities.randomBatch(batchSize: 10, dimensions: dimensions)
        let vectors = try VectorBatch(vectors: vectorsArray)
        let normalized = try await accelerator.normalizeVectors(vectors)
        for vec in normalized.toArrays() {
            MetalTestUtilities.assertFinite(vec, "L2 normalization output")
        }

        // Test pooling
        let embeddingsArray = MetalTestUtilities.randomBatch(batchSize: seqLen, dimensions: dimensions)
        let embeddings = try VectorBatch(vectors: embeddingsArray)

        let meanPooled = try await accelerator.poolEmbeddings(
            embeddings, strategy: .mean, attentionMask: nil
        )
        MetalTestUtilities.assertFinite(meanPooled, "Mean pooling output")

        let maxPooled = try await accelerator.poolEmbeddings(
            embeddings, strategy: .max, attentionMask: nil
        )
        MetalTestUtilities.assertFinite(maxPooled, "Max pooling output")

        // Test similarity
        let v1 = MetalTestUtilities.randomVector(dimensions: dimensions)
        let v2 = MetalTestUtilities.randomVector(dimensions: dimensions)
        let similarity = try await accelerator.cosineSimilarity(v1, v2)
        XCTAssertTrue(similarity.isFinite, "Cosine similarity should be finite")
    }
}

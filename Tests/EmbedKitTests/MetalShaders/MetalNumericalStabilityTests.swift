import XCTest
import Metal
@testable import EmbedKit

/// Numerical stability tests for Metal shader operations
///
/// These tests verify that shaders handle extreme values robustly:
/// - Tiny values (1e-20, denormalized numbers)
/// - Huge values (1e+20, near overflow)
/// - Mixed magnitudes (1e-20 and 1e+20 in same vector)
/// - Zero vectors
/// - Near-zero values
///
/// The goal is to ensure no NaN, Inf, or catastrophic cancellation occurs.
///
final class MetalNumericalStabilityTests: XCTestCase {
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

    // MARK: - L2 Normalization Stability

    func testL2NormalizationZeroVector() async throws {
        let testCases: [(name: String, vector: [Float])] = [
            ("All zeros", [0, 0, 0, 0]),
            ("Single zero", [0]),
            ("Large zero vector", Array(repeating: 0, count: 384)),
        ]

        for (name, input) in testCases {
            let batch = try VectorBatch(vectors: [input])
            let result = try await accelerator.normalizeVectors(batch)

            // Zero vector should remain zero (not NaN)
            let arrays = result.toArrays()
            let norm = sqrt(arrays[0].reduce(0) { $0 + $1 * $1 })
            XCTAssertEqual(norm, 0.0, accuracy: 1e-6,
                          "Test: \(name) - should remain zero")

            // Verify all zeros
            XCTAssertTrue(arrays[0].allSatisfy { $0 == 0.0 },
                         "Test: \(name) - all elements should be zero")

            // No NaN or Inf
            MetalTestUtilities.assertFinite(arrays[0], "Test: \(name)")
        }
    }

    func testL2NormalizationTinyValues() async throws {
        let testCases: [(name: String, vector: [Float])] = [
            ("1e-20", Array(repeating: 1e-20, count: 128)),
            ("1e-15", Array(repeating: 1e-15, count: 128)),
            ("1e-10", Array(repeating: 1e-10, count: 128)),
            ("Denormalized", Array(repeating: Float.leastNormalMagnitude, count: 128)),
            ("Near denormalized", Array(repeating: Float.leastNormalMagnitude * 10, count: 128)),
        ]

        for (name, input) in testCases {
            let batch = try VectorBatch(vectors: [input])
            let result = try await accelerator.normalizeVectors(batch)
            let arrays = result.toArrays()

            // Should have unit norm (no underflow)
            let norm = result[0].l2Norm
            XCTAssertEqual(norm, 1.0, accuracy: 1e-4,
                          "Test: \(name) - should have unit norm")

            // No NaN or Inf
            MetalTestUtilities.assertFinite(arrays[0], "Test: \(name)")

            // Verify all values are similar (uniform input)
            let expectedValue = 1.0 / sqrt(Float(input.count))
            for val in result[0] {
                XCTAssertEqual(val, expectedValue, accuracy: 1e-3,
                              "Test: \(name) - uniform distribution")
            }
        }
    }

    func testL2NormalizationHugeValues() async throws {
        let testCases: [(name: String, vector: [Float])] = [
            ("1e+20", Array(repeating: 1e20, count: 128)),
            ("1e+15", Array(repeating: 1e15, count: 128)),
            ("1e+10", Array(repeating: 1e10, count: 128)),
            ("Near max", Array(repeating: Float.greatestFiniteMagnitude / 1000, count: 128)),
        ]

        for (name, input) in testCases {
            let batch = try VectorBatch(vectors: [input])
            let result = try await accelerator.normalizeVectors(batch)
            let arrays = result.toArrays()

            // Should have unit norm (no overflow)
            let norm = result[0].l2Norm
            XCTAssertEqual(norm, 1.0, accuracy: 1e-4,
                          "Test: \(name) - should have unit norm")

            // No NaN or Inf
            MetalTestUtilities.assertFinite(arrays[0], "Test: \(name)")

            // Verify all values are similar (uniform input)
            let expectedValue = 1.0 / sqrt(Float(input.count))
            for val in result[0] {
                XCTAssertEqual(val, expectedValue, accuracy: 1e-3,
                              "Test: \(name) - uniform distribution")
            }
        }
    }

    func testL2NormalizationMixedMagnitudes() async throws {
        let testCases: [(name: String, vector: [Float])] = [
            ("Tiny and huge",
             [1e-20, 1e20, 1e-20, 1e20]),
            ("Many tiny, one huge",
             [1e-20, 1e-20, 1e-20, 1e-20, 1e20]),
            ("Many huge, one tiny",
             [1e20, 1e20, 1e20, 1e20, 1e-20]),
            ("Extreme range",
             [Float.leastNormalMagnitude, 1.0, Float.greatestFiniteMagnitude / 1000]),
            ("Powers of 10",
             [1e-10, 1e-5, 1.0, 1e5, 1e10]),
        ]

        for (name, input) in testCases {
            let batch = try VectorBatch(vectors: [input])
            let result = try await accelerator.normalizeVectors(batch)
            let arrays = result.toArrays()

            // Should have unit norm
            let norm = result[0].l2Norm
            XCTAssertEqual(norm, 1.0, accuracy: 1e-3,
                          "Test: \(name) - should have unit norm")

            // No NaN or Inf
            MetalTestUtilities.assertFinite(arrays[0], "Test: \(name)")

            // Verify against CPU reference (which uses same stable algorithm)
            let cpuResult = MetalTestUtilities.cpuNormalize(input)
            MetalTestUtilities.assertEqual(
                result[0], cpuResult,
                accuracy: 1e-3,
                "Test: \(name) - vs CPU"
            )
        }
    }

    func testL2NormalizationNearZero() async throws {
        // Values near epsilon threshold
        let epsilon: Float = 1e-8
        let testCases: [(name: String, vector: [Float])] = [
            ("At epsilon", Array(repeating: epsilon, count: 128)),
            ("Below epsilon", Array(repeating: epsilon / 10, count: 128)),
            ("Mixed around epsilon",
             [epsilon * 2, epsilon / 2, epsilon, epsilon * 0.1]),
        ]

        for (name, input) in testCases {
            let batch = try VectorBatch(vectors: [input])
            let result = try await accelerator.normalizeVectors(batch)
            let arrays = result.toArrays()

            // No NaN or Inf
            MetalTestUtilities.assertFinite(arrays[0], "Test: \(name)")

            // Should either be zero or unit norm
            let norm = result[0].l2Norm
            if norm > 0.1 {
                // Non-zero result should have unit norm
                XCTAssertEqual(norm, 1.0, accuracy: 1e-3,
                              "Test: \(name) - if non-zero, should be unit norm")
            }
        }
    }

    // MARK: - Cosine Similarity Stability

    func testCosineSimilarityBothTiny() async throws {
        let testCases: [(name: String, scale: Float)] = [
            ("1e-20", 1e-20),
            ("1e-15", 1e-15),
            ("1e-10", 1e-10),
            ("Denormalized", Float.leastNormalMagnitude),
        ]

        for (name, scale) in testCases {
            let v1 = Array(repeating: scale, count: 384)
            let v2 = Array(repeating: scale, count: 384)

            let similarity = try await accelerator.cosineSimilarity(v1, v2)

            // Should be 1.0 (identical direction)
            XCTAssertEqual(similarity, 1.0, accuracy: 1e-3,
                          "Test: \(name) - identical tiny vectors")

            // Should be finite
            XCTAssertTrue(similarity.isFinite, "Test: \(name)")

            // Should be in valid range
            MetalTestUtilities.assertInRange(
                similarity, -1.0...1.0,
                "Test: \(name)"
            )
        }
    }

    func testCosineSimilarityBothHuge() async throws {
        let testCases: [(name: String, scale: Float)] = [
            ("1e+20", 1e20),
            ("1e+15", 1e15),
            ("1e+10", 1e10),
            ("Near max", Float.greatestFiniteMagnitude / 1000),
        ]

        for (name, scale) in testCases {
            let v1 = Array(repeating: scale, count: 384)
            let v2 = Array(repeating: scale, count: 384)

            let similarity = try await accelerator.cosineSimilarity(v1, v2)

            // Should be 1.0 (identical direction)
            XCTAssertEqual(similarity, 1.0, accuracy: 1e-3,
                          "Test: \(name) - identical huge vectors")

            // Should be finite
            XCTAssertTrue(similarity.isFinite, "Test: \(name)")

            // Should be in valid range
            MetalTestUtilities.assertInRange(
                similarity, -1.0...1.0,
                "Test: \(name)"
            )
        }
    }

    func testCosineSimilarityMixedTinyAndHuge() async throws {
        let testCases: [(name: String, v1Scale: Float, v2Scale: Float)] = [
            ("Tiny vs Huge", 1e-20, 1e20),
            ("Denorm vs Max", Float.leastNormalMagnitude, Float.greatestFiniteMagnitude / 1000),
            ("1e-15 vs 1e+15", 1e-15, 1e15),
        ]

        for (name, v1Scale, v2Scale) in testCases {
            let v1 = Array(repeating: v1Scale, count: 384)
            let v2 = Array(repeating: v2Scale, count: 384)

            let similarity = try await accelerator.cosineSimilarity(v1, v2)

            // Should be 1.0 (same direction, different magnitudes)
            XCTAssertEqual(similarity, 1.0, accuracy: 1e-2,
                          "Test: \(name) - same direction different scales")

            // Should be finite
            XCTAssertTrue(similarity.isFinite, "Test: \(name)")

            // Should be in valid range
            MetalTestUtilities.assertInRange(
                similarity, -1.0...1.0,
                "Test: \(name)"
            )
        }
    }

    func testCosineSimilarityOneZero() async throws {
        let testCases: [(name: String, nonZeroScale: Float)] = [
            ("Normal scale", 1.0),
            ("Tiny scale", 1e-20),
            ("Huge scale", 1e20),
        ]

        for (name, scale) in testCases {
            let v1 = Array(repeating: Float(0), count: 384)
            let v2 = Array(repeating: scale, count: 384)

            let similarity = try await accelerator.cosineSimilarity(v1, v2)

            // Zero vector vs any vector should be 0
            XCTAssertEqual(similarity, 0.0, accuracy: 1e-5,
                          "Test: \(name) - zero vs non-zero")

            // Should be finite
            XCTAssertTrue(similarity.isFinite, "Test: \(name)")
        }
    }

    func testCosineSimilarityBothZero() async throws {
        let v1 = Array(repeating: Float(0), count: 384)
        let v2 = Array(repeating: Float(0), count: 384)

        let similarity = try await accelerator.cosineSimilarity(v1, v2)

        // Zero vs zero should be 0 (not NaN)
        XCTAssertEqual(similarity, 0.0, accuracy: 1e-5,
                      "Zero vs zero should be 0")

        // Should be finite
        XCTAssertTrue(similarity.isFinite, "Zero vs zero")
    }

    func testCosineSimilarityMixedMagnitudesWithinVector() async throws {
        // Vectors with extreme internal magnitude variation
        let v1: [Float] = [1e-20, 1e20, 1e-10, 1e10, 1.0]
        let v2: [Float] = [1e-20, 1e20, 1e-10, 1e10, 1.0]

        let similarity = try await accelerator.cosineSimilarity(v1, v2)

        // Should be 1.0 (identical vectors)
        XCTAssertEqual(similarity, 1.0, accuracy: 1e-3,
                      "Mixed magnitudes within vector")

        // Should be finite
        XCTAssertTrue(similarity.isFinite, "Mixed magnitudes within vector")

        // Verify against CPU
        let cpuSimilarity = MetalTestUtilities.cpuCosineSimilarity(v1, v2)
        XCTAssertEqual(similarity, cpuSimilarity, accuracy: 1e-3,
                      "Mixed magnitudes within vector vs CPU")
    }

    // MARK: - Pooling Stability

    func testPoolingWithExtremeValues() async throws {
        let testCases: [(name: String, embeddings: [[Float]])] = [
            ("All tiny",
             [[1e-20, 1e-20, 1e-20], [1e-20, 1e-20, 1e-20]]),
            ("All huge",
             [[1e20, 1e20, 1e20], [1e20, 1e20, 1e20]]),
            ("Mixed tiny and huge",
             [[1e-20, 1e20, 1.0], [1e20, 1e-20, 1.0]]),
            ("With zeros",
             [[0, 1e20, 0], [1e20, 0, 0]]),
        ]

        for (name, embeddings) in testCases {
            let batch = try VectorBatch(vectors: embeddings)

            // Test mean pooling
            let meanResult = try await accelerator.poolEmbeddings(
                batch,
                strategy: .mean,
                attentionMask: nil
            )

            MetalTestUtilities.assertFinite(meanResult, "Mean pooling: \(name)")

            // Test max pooling
            let maxResult = try await accelerator.poolEmbeddings(
                batch,
                strategy: .max,
                attentionMask: nil
            )

            MetalTestUtilities.assertFinite(maxResult, "Max pooling: \(name)")

            // Verify against CPU reference
            let cpuMeanResult = MetalTestUtilities.cpuMeanPool(embeddings, mask: nil)
            MetalTestUtilities.assertEqual(
                meanResult, cpuMeanResult,
                accuracy: 1e-3,
                "Mean pooling vs CPU: \(name)"
            )

            let cpuMaxResult = MetalTestUtilities.cpuMaxPool(embeddings, mask: nil)
            MetalTestUtilities.assertEqual(
                maxResult, cpuMaxResult,
                accuracy: 1e-3,
                "Max pooling vs CPU: \(name)"
            )
        }
    }

    func testAttentionPoolingWithExtremeWeights() async throws {
        let embeddings: [[Float]] = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        let batch = try VectorBatch(vectors: embeddings)

        let testCases: [(name: String, weights: [Float])] = [
            ("Tiny weights", [1e-20, 1e-20, 1e-20]),
            ("Huge weights", [1e20, 1e20, 1e20]),
            ("Mixed weights", [1e-20, 1.0, 1e20]),
            ("One dominant weight", [1e-20, 1e20, 1e-20]),
        ]

        for (name, weights) in testCases {
            let result = try await accelerator.attentionWeightedPooling(
                batch,
                attentionWeights: weights
            )

            // Should be finite
            MetalTestUtilities.assertFinite(result, "Attention pooling: \(name)")

            // Verify against CPU reference
            let cpuResult = MetalTestUtilities.cpuAttentionPool(embeddings, weights: weights)
            MetalTestUtilities.assertEqual(
                result, cpuResult,
                accuracy: 1e-3,
                "Attention pooling vs CPU: \(name)"
            )
        }
    }

    // MARK: - Comprehensive Stability Test

    func testEndToEndStabilityPipeline() async throws {
        // Simulate a complete embedding pipeline with extreme values
        let dimensions = 384
        let seqLen = 32

        // Create embeddings with mixed magnitudes
        var embeddings: [[Float]] = []
        for i in 0..<seqLen {
            var embedding: [Float] = []
            for d in 0..<dimensions {
                let scale = Float(1 + i + d) / Float(seqLen * dimensions)
                let magnitude: Float
                if d % 3 == 0 {
                    magnitude = 1e-15 * scale  // Tiny
                } else if d % 3 == 1 {
                    magnitude = 1e15 * scale   // Huge
                } else {
                    magnitude = scale          // Normal
                }
                embedding.append(magnitude)
            }
            embeddings.append(embedding)
        }

        // Step 1: Normalize embeddings
        let batch = try VectorBatch(vectors: embeddings)
        let normalized = try await accelerator.normalizeVectors(batch)
        let normalizedArrays = normalized.toArrays()

        // Verify all normalized
        for (idx, vec) in normalizedArrays.enumerated() {
            MetalTestUtilities.assertFinite(vec, "Normalization step: vector \(idx)")
            let norm = sqrt(vec.reduce(0) { $0 + $1 * $1 })
            XCTAssertEqual(norm, 1.0, accuracy: 1e-3,
                          "Normalization step: vector \(idx) should have unit norm")
        }

        // Step 2: Pool embeddings
        let pooled = try await accelerator.poolEmbeddings(
            normalized,
            strategy: .mean,
            attentionMask: nil
        )

        MetalTestUtilities.assertFinite(pooled, "Pooling step")

        // Step 3: Compute self-similarity
        let similarity = try await accelerator.cosineSimilarity(pooled, pooled)

        XCTAssertTrue(similarity.isFinite, "Similarity step should be finite")
        XCTAssertEqual(similarity, 1.0, accuracy: 1e-3,
                      "Self-similarity should be 1.0")

        print("End-to-end stability test passed with mixed magnitudes (1e-15 to 1e+15)")
    }
}

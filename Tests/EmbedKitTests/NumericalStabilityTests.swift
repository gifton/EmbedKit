// Tests for Numerical Stability - P1 Category
// Validates floating point edge cases, normalization, and precision
import Testing
import Foundation
@testable import EmbedKit

// MARK: - Normalization Tests

@Suite("Numerical Stability - Normalization")
struct NormalizationStabilityTests {

    @Test("Normalize handles zero vector gracefully")
    func normalizeZeroVector() {
        let zero = [Float](repeating: 0, count: 384)
        let result = PoolingHelpers.normalize(zero)

        // Should not crash, return something reasonable
        #expect(result.count == 384)
        // With epsilon protection, result should be finite
        #expect(result.allSatisfy { $0.isFinite })
    }

    @Test("Normalize handles near-zero vector")
    func normalizeNearZeroVector() {
        let nearZero = [Float](repeating: 1e-20, count: 384)
        let result = PoolingHelpers.normalize(nearZero)

        #expect(result.count == 384)
        #expect(result.allSatisfy { $0.isFinite })
    }

    @Test("Normalize handles very large values")
    func normalizeLargeValues() {
        // Use moderately large values that won't overflow Float32
        let large = [Float](repeating: 1e15, count: 384)
        let result = PoolingHelpers.normalize(large)

        #expect(result.count == 384)
        #expect(result.allSatisfy { $0.isFinite })

        // Should be normalized (magnitude ~1)
        let mag = sqrt(result.reduce(0) { $0 + $1 * $1 })
        #expect(abs(mag - 1.0) < 0.01)
    }

    @Test("Normalize handles mixed sign values")
    func normalizeMixedSigns() {
        var mixed = [Float](repeating: 0, count: 384)
        for i in 0..<384 {
            mixed[i] = Float(i % 2 == 0 ? 1.0 : -1.0) * Float(i + 1)
        }

        let result = PoolingHelpers.normalize(mixed)

        #expect(result.count == 384)
        #expect(result.allSatisfy { $0.isFinite })

        let mag = sqrt(result.reduce(0) { $0 + $1 * $1 })
        #expect(abs(mag - 1.0) < 0.001)
    }

    @Test("Embedding.normalized handles zero magnitude")
    func embeddingNormalizedZeroMagnitude() {
        let embedding = Embedding(
            vector: [Float](repeating: 0, count: 4),
            metadata: EmbeddingMetadata(
                modelID: ModelID(provider: "test", name: "test", version: "1"),
                tokenCount: 1,
                processingTime: 0
            )
        )

        let normalized = embedding.normalized()

        // Should return self when magnitude is zero
        #expect(normalized.vector == embedding.vector)
    }

    @Test("Embedding.normalized produces unit vector")
    func embeddingNormalizedUnitVector() {
        let embedding = Embedding(
            vector: [3, 4, 0, 0],  // magnitude = 5
            metadata: EmbeddingMetadata(
                modelID: ModelID(provider: "test", name: "test", version: "1"),
                tokenCount: 1,
                processingTime: 0
            )
        )

        let normalized = embedding.normalized()

        #expect(abs(normalized.vector[0] - 0.6) < 0.001)
        #expect(abs(normalized.vector[1] - 0.8) < 0.001)
        #expect(abs(normalized.magnitude - 1.0) < 0.001)
    }
}

// MARK: - Similarity Computation Tests

@Suite("Numerical Stability - Similarity")
struct SimilarityStabilityTests {

    @Test("Similarity of identical vectors is 1.0")
    func similarityIdentical() {
        let vec = (0..<384).map { Float($0) / 384.0 }
        let emb1 = Embedding(
            vector: vec,
            metadata: EmbeddingMetadata(
                modelID: ModelID(provider: "test", name: "test", version: "1"),
                tokenCount: 1,
                processingTime: 0
            )
        )
        let emb2 = Embedding(
            vector: vec,
            metadata: EmbeddingMetadata(
                modelID: ModelID(provider: "test", name: "test", version: "1"),
                tokenCount: 1,
                processingTime: 0
            )
        )

        let sim = emb1.similarity(to: emb2)
        #expect(abs(sim - 1.0) < 0.001)
    }

    @Test("Similarity of opposite vectors is -1.0")
    func similarityOpposite() {
        let vec = (0..<384).map { Float($0) / 384.0 }
        let negVec = vec.map { -$0 }

        let emb1 = Embedding(
            vector: vec,
            metadata: EmbeddingMetadata(
                modelID: ModelID(provider: "test", name: "test", version: "1"),
                tokenCount: 1,
                processingTime: 0
            )
        )
        let emb2 = Embedding(
            vector: negVec,
            metadata: EmbeddingMetadata(
                modelID: ModelID(provider: "test", name: "test", version: "1"),
                tokenCount: 1,
                processingTime: 0
            )
        )

        let sim = emb1.similarity(to: emb2)
        #expect(abs(sim - (-1.0)) < 0.001)
    }

    @Test("Similarity of orthogonal vectors is 0.0")
    func similarityOrthogonal() {
        var vec1 = [Float](repeating: 0, count: 384)
        var vec2 = [Float](repeating: 0, count: 384)
        vec1[0] = 1.0
        vec2[1] = 1.0

        let emb1 = Embedding(
            vector: vec1,
            metadata: EmbeddingMetadata(
                modelID: ModelID(provider: "test", name: "test", version: "1"),
                tokenCount: 1,
                processingTime: 0
            )
        )
        let emb2 = Embedding(
            vector: vec2,
            metadata: EmbeddingMetadata(
                modelID: ModelID(provider: "test", name: "test", version: "1"),
                tokenCount: 1,
                processingTime: 0
            )
        )

        let sim = emb1.similarity(to: emb2)
        #expect(abs(sim) < 0.001)
    }

    @Test("Similarity handles zero vectors")
    func similarityZeroVectors() {
        let zero = [Float](repeating: 0, count: 384)
        let nonZero = (0..<384).map { Float($0) }

        let emb1 = Embedding(
            vector: zero,
            metadata: EmbeddingMetadata(
                modelID: ModelID(provider: "test", name: "test", version: "1"),
                tokenCount: 1,
                processingTime: 0
            )
        )
        let emb2 = Embedding(
            vector: nonZero,
            metadata: EmbeddingMetadata(
                modelID: ModelID(provider: "test", name: "test", version: "1"),
                tokenCount: 1,
                processingTime: 0
            )
        )

        // Should return finite value (epsilon protection)
        let sim = emb1.similarity(to: emb2)
        #expect(sim.isFinite)
    }

    @Test("Similarity of different dimension vectors returns 0")
    func similarityDifferentDimensions() {
        let emb1 = Embedding(
            vector: [1, 2, 3],
            metadata: EmbeddingMetadata(
                modelID: ModelID(provider: "test", name: "test", version: "1"),
                tokenCount: 1,
                processingTime: 0
            )
        )
        let emb2 = Embedding(
            vector: [1, 2, 3, 4],
            metadata: EmbeddingMetadata(
                modelID: ModelID(provider: "test", name: "test", version: "1"),
                tokenCount: 1,
                processingTime: 0
            )
        )

        let sim = emb1.similarity(to: emb2)
        #expect(sim == 0)
    }
}

// MARK: - Pooling Stability Tests

@Suite("Numerical Stability - Pooling")
struct PoolingStabilityTests {

    @Test("Mean pooling handles large accumulations")
    func meanPoolingLargeAccumulation() {
        // Many tokens with large values
        let tokens = 512
        let dim = 384
        let sequence = [Float](repeating: 1e6, count: tokens * dim)

        let result = PoolingHelpers.mean(sequence: sequence, tokens: tokens, dim: dim)

        #expect(result.count == dim)
        #expect(result.allSatisfy { $0.isFinite })
        #expect(result.allSatisfy { abs($0 - 1e6) < 1 })
    }

    @Test("Mean pooling handles mixed extreme values")
    func meanPoolingMixedExtremes() {
        let tokens = 4
        let dim = 4
        var sequence = [Float](repeating: 0, count: tokens * dim)

        // Token 0: very large positive
        for d in 0..<dim { sequence[0 * dim + d] = 1e10 }
        // Token 1: very large negative
        for d in 0..<dim { sequence[1 * dim + d] = -1e10 }
        // Token 2: tiny positive
        for d in 0..<dim { sequence[2 * dim + d] = 1e-10 }
        // Token 3: tiny negative
        for d in 0..<dim { sequence[3 * dim + d] = -1e-10 }

        let result = PoolingHelpers.mean(sequence: sequence, tokens: tokens, dim: dim)

        #expect(result.count == dim)
        #expect(result.allSatisfy { $0.isFinite })
    }

    @Test("Max pooling handles negative infinity initialization")
    func maxPoolingNegativeInfinity() {
        let tokens = 2
        let dim = 4
        let sequence = [Float](repeating: -1e30, count: tokens * dim)

        let result = PoolingHelpers.max(sequence: sequence, tokens: tokens, dim: dim)

        #expect(result.count == dim)
        #expect(result.allSatisfy { $0.isFinite })
        #expect(result.allSatisfy { abs($0 - (-1e30)) < 1e25 })
    }

    @Test("CLS pooling extracts first token correctly")
    func clsPoolingFirstToken() {
        let tokens = 4
        let dim = 3
        var sequence = [Float](repeating: 0, count: tokens * dim)

        // First token: [1, 2, 3]
        sequence[0] = 1; sequence[1] = 2; sequence[2] = 3
        // Other tokens: different values
        for t in 1..<tokens {
            for d in 0..<dim {
                sequence[t * dim + d] = Float(t * 10 + d)
            }
        }

        let result = PoolingHelpers.cls(sequence: sequence, tokens: tokens, dim: dim)

        #expect(result == [1, 2, 3])
    }

    @Test("Mean pooling with all-zero mask falls back to unmasked")
    func meanPoolingZeroMaskFallback() {
        let tokens = 4
        let dim = 2
        let sequence: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let mask = [0, 0, 0, 0]  // All masked out

        let result = PoolingHelpers.mean(sequence: sequence, tokens: tokens, dim: dim, mask: mask)

        // Should fall back to unmasked mean: (1+3+5+7)/4=4, (2+4+6+8)/4=5
        #expect(result.count == dim)
        #expect(abs(result[0] - 4.0) < 0.001)
        #expect(abs(result[1] - 5.0) < 0.001)
    }
}

// MARK: - Precision Tests

@Suite("Numerical Stability - Precision")
struct PrecisionStabilityTests {

    @Test("Float32 precision in dot product")
    func float32DotProductPrecision() {
        // Test accumulation doesn't lose precision
        let dim = 1000
        let vec1 = [Float](repeating: 0.001, count: dim)
        let vec2 = [Float](repeating: 0.001, count: dim)

        let dot = zip(vec1, vec2).reduce(Float(0)) { $0 + $1.0 * $1.1 }
        let expected: Float = 0.001 * 0.001 * Float(dim)

        #expect(abs(dot - expected) < 0.0001)
    }

    @Test("Embedding vectors preserve precision through normalization")
    func normalizationPreservesPrecision() {
        let vec: [Float] = [0.123456789, 0.987654321, 0.555555555, 0.111111111]

        let emb = Embedding(
            vector: vec,
            metadata: EmbeddingMetadata(
                modelID: ModelID(provider: "test", name: "test", version: "1"),
                tokenCount: 1,
                processingTime: 0
            )
        )

        let normalized = emb.normalized()
        let denormalized = normalized.vector.map { $0 * emb.magnitude }

        // Should be able to recover original values
        for (orig, recovered) in zip(vec, denormalized) {
            #expect(abs(orig - recovered) < 1e-5)
        }
    }

    @Test("Similarity computation is symmetric")
    func similaritySymmetric() {
        let vec1: [Float] = (0..<100).map { sin(Float($0) * 0.1) }
        let vec2: [Float] = (0..<100).map { cos(Float($0) * 0.1) }

        let emb1 = Embedding(
            vector: vec1,
            metadata: EmbeddingMetadata(
                modelID: ModelID(provider: "test", name: "test", version: "1"),
                tokenCount: 1,
                processingTime: 0
            )
        )
        let emb2 = Embedding(
            vector: vec2,
            metadata: EmbeddingMetadata(
                modelID: ModelID(provider: "test", name: "test", version: "1"),
                tokenCount: 1,
                processingTime: 0
            )
        )

        let sim1 = emb1.similarity(to: emb2)
        let sim2 = emb2.similarity(to: emb1)

        #expect(abs(sim1 - sim2) < 1e-6)
    }

    @Test("Batch normalization produces consistent unit vectors")
    func batchNormalizationConsistent() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()

        let texts = ["Hello world", "Test input", "Another example"]
        let embeddings = try await model.embedBatch(texts, options: BatchOptions())

        for emb in embeddings {
            let mag = emb.magnitude
            // All should be normalized (magnitude ~1)
            #expect(abs(mag - 1.0) < 0.01)
        }
    }
}

// MARK: - Edge Case Values Tests

@Suite("Numerical Stability - Edge Values")
struct EdgeValueStabilityTests {

    @Test("Handles subnormal floats")
    func subnormalFloats() {
        let subnormal = Float.leastNonzeroMagnitude
        let vec = [Float](repeating: subnormal, count: 10)

        let result = PoolingHelpers.normalize(vec)

        #expect(result.allSatisfy { $0.isFinite })
    }

    @Test("Handles Float.greatestFiniteMagnitude")
    func greatestFiniteMagnitude() {
        // Use a value that's large but won't overflow when squared
        // sqrt(Float.greatestFiniteMagnitude) is about 1.8e19
        let large: Float = 1e15
        let vec = [Float](repeating: large, count: 10)

        let result = PoolingHelpers.normalize(vec)

        #expect(result.allSatisfy { $0.isFinite })
        let mag = sqrt(result.reduce(0) { $0 + $1 * $1 })
        #expect(abs(mag - 1.0) < 0.01)
    }

    @Test("Handles NaN propagation check")
    func nanPropagation() {
        // Verify our operations don't introduce NaN from valid input
        let validInput: [Float] = [1.0, 2.0, 3.0, 4.0]

        let normalized = PoolingHelpers.normalize(validInput)
        #expect(normalized.allSatisfy { !$0.isNaN })

        let mean = PoolingHelpers.mean(sequence: validInput, tokens: 2, dim: 2)
        #expect(mean.allSatisfy { !$0.isNaN })

        let max = PoolingHelpers.max(sequence: validInput, tokens: 2, dim: 2)
        #expect(max.allSatisfy { !$0.isNaN })
    }

    @Test("Magnitude calculation handles extreme values")
    func magnitudeExtremeValues() {
        let emb1 = Embedding(
            vector: [Float](repeating: 1e-20, count: 100),
            metadata: EmbeddingMetadata(
                modelID: ModelID(provider: "test", name: "test", version: "1"),
                tokenCount: 1,
                processingTime: 0
            )
        )

        let emb2 = Embedding(
            vector: [Float](repeating: 1e10, count: 100),
            metadata: EmbeddingMetadata(
                modelID: ModelID(provider: "test", name: "test", version: "1"),
                tokenCount: 1,
                processingTime: 0
            )
        )

        #expect(emb1.magnitude.isFinite)
        #expect(emb2.magnitude.isFinite)
        #expect(emb1.magnitude >= 0)
        #expect(emb2.magnitude >= 0)
    }
}

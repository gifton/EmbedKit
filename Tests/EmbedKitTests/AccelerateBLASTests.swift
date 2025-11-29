// EmbedKit - Accelerate BLAS Tests

import Testing
import Foundation
@testable import EmbedKit

@Suite("Accelerate BLAS")
struct AccelerateBLASTests {

    // MARK: - Dot Product Tests

    @Test("Dot product of identical vectors equals sum of squares")
    func dotProductIdentical() {
        let v: [Float] = [1, 2, 3, 4, 5]
        let expected: Float = 1 + 4 + 9 + 16 + 25 // 55

        let result = AccelerateBLAS.dotProduct(v, v)
        #expect(abs(result - expected) < 0.0001)
    }

    @Test("Dot product of orthogonal vectors is zero")
    func dotProductOrthogonal() {
        let a: [Float] = [1, 0, 0]
        let b: [Float] = [0, 1, 0]

        let result = AccelerateBLAS.dotProduct(a, b)
        #expect(abs(result) < 0.0001)
    }

    @Test("Dot product with negative values")
    func dotProductNegative() {
        let a: [Float] = [1, -2, 3]
        let b: [Float] = [4, 5, -6]
        // 1*4 + (-2)*5 + 3*(-6) = 4 - 10 - 18 = -24
        let expected: Float = -24

        let result = AccelerateBLAS.dotProduct(a, b)
        #expect(abs(result - expected) < 0.0001)
    }

    // MARK: - Sum of Squares / Magnitude Tests

    @Test("Sum of squares calculation")
    func sumOfSquares() {
        let v: [Float] = [3, 4] // 9 + 16 = 25
        let result = AccelerateBLAS.sumOfSquares(v)
        #expect(abs(result - 25) < 0.0001)
    }

    @Test("Magnitude of 3-4-5 triangle")
    func magnitude345() {
        let v: [Float] = [3, 4] // sqrt(9 + 16) = 5
        let result = AccelerateBLAS.magnitude(v)
        #expect(abs(result - 5) < 0.0001)
    }

    @Test("Magnitude of unit vector is 1")
    func magnitudeUnit() {
        let v: [Float] = [1, 0, 0]
        let result = AccelerateBLAS.magnitude(v)
        #expect(abs(result - 1) < 0.0001)
    }

    // MARK: - Normalization Tests

    @Test("Normalize produces unit vector")
    func normalizeProducesUnit() {
        let v: [Float] = [3, 4]
        let normalized = AccelerateBLAS.normalize(v)

        let mag = AccelerateBLAS.magnitude(normalized)
        #expect(abs(mag - 1.0) < 0.0001)
    }

    @Test("Normalize preserves direction")
    func normalizePreservesDirection() {
        let v: [Float] = [3, 4]
        let normalized = AccelerateBLAS.normalize(v)

        // Direction should be same: [0.6, 0.8]
        #expect(abs(normalized[0] - 0.6) < 0.0001)
        #expect(abs(normalized[1] - 0.8) < 0.0001)
    }

    @Test("Normalize in-place returns original magnitude")
    func normalizeInPlaceReturnsMagnitude() {
        var v: [Float] = [3, 4]
        let originalMag = AccelerateBLAS.normalizeInPlace(&v)

        #expect(abs(originalMag - 5.0) < 0.0001)
        #expect(abs(AccelerateBLAS.magnitude(v) - 1.0) < 0.0001)
    }

    @Test("Normalize handles zero vector")
    func normalizeZeroVector() {
        let v: [Float] = [0, 0, 0]
        let normalized = AccelerateBLAS.normalize(v)

        // Should return the same zero vector
        for val in normalized {
            #expect(abs(val) < 0.0001)
        }
    }

    // MARK: - Cosine Similarity Tests

    @Test("Cosine similarity of identical vectors is 1")
    func cosineSimilarityIdentical() {
        let v: [Float] = [1, 2, 3, 4]
        let result = AccelerateBLAS.cosineSimilarity(v, v)
        #expect(abs(result - 1.0) < 0.0001)
    }

    @Test("Cosine similarity of opposite vectors is -1")
    func cosineSimilarityOpposite() {
        let a: [Float] = [1, 2, 3]
        let b: [Float] = [-1, -2, -3]
        let result = AccelerateBLAS.cosineSimilarity(a, b)
        #expect(abs(result - (-1.0)) < 0.0001)
    }

    @Test("Cosine similarity of orthogonal vectors is 0")
    func cosineSimilarityOrthogonal() {
        let a: [Float] = [1, 0]
        let b: [Float] = [0, 1]
        let result = AccelerateBLAS.cosineSimilarity(a, b)
        #expect(abs(result) < 0.0001)
    }

    @Test("Cosine distance is 1 minus similarity")
    func cosineDistanceRelation() {
        let a: [Float] = [1, 2, 3]
        let b: [Float] = [4, 5, 6]

        let similarity = AccelerateBLAS.cosineSimilarity(a, b)
        let distance = AccelerateBLAS.cosineDistance(a, b)

        #expect(abs(distance - (1.0 - similarity)) < 0.0001)
    }

    // MARK: - Euclidean Distance Tests

    @Test("Euclidean distance between identical vectors is 0")
    func euclideanDistanceIdentical() {
        let v: [Float] = [1, 2, 3, 4]
        let result = AccelerateBLAS.euclideanDistance(v, v)
        #expect(abs(result) < 0.0001)
    }

    @Test("Euclidean distance 3-4-5 triangle")
    func euclideanDistance345() {
        let a: [Float] = [0, 0]
        let b: [Float] = [3, 4]
        let result = AccelerateBLAS.euclideanDistance(a, b)
        #expect(abs(result - 5.0) < 0.0001)
    }

    @Test("Squared Euclidean distance")
    func euclideanDistanceSquared() {
        let a: [Float] = [0, 0]
        let b: [Float] = [3, 4]
        let result = AccelerateBLAS.euclideanDistanceSquared(a, b)
        #expect(abs(result - 25.0) < 0.0001)
    }

    // MARK: - Manhattan Distance Tests

    @Test("Manhattan distance between identical vectors is 0")
    func manhattanDistanceIdentical() {
        let v: [Float] = [1, 2, 3]
        let result = AccelerateBLAS.manhattanDistance(v, v)
        #expect(abs(result) < 0.0001)
    }

    @Test("Manhattan distance calculation")
    func manhattanDistanceBasic() {
        let a: [Float] = [0, 0, 0]
        let b: [Float] = [1, 2, 3]
        // |0-1| + |0-2| + |0-3| = 1 + 2 + 3 = 6
        let result = AccelerateBLAS.manhattanDistance(a, b)
        #expect(abs(result - 6.0) < 0.0001)
    }

    @Test("Manhattan distance with negative values")
    func manhattanDistanceNegative() {
        let a: [Float] = [-1, 2]
        let b: [Float] = [1, -3]
        // |(-1)-1| + |2-(-3)| = 2 + 5 = 7
        let result = AccelerateBLAS.manhattanDistance(a, b)
        #expect(abs(result - 7.0) < 0.0001)
    }

    // MARK: - Chebyshev Distance Tests

    @Test("Chebyshev distance between identical vectors is 0")
    func chebyshevDistanceIdentical() {
        let v: [Float] = [1, 2, 3]
        let result = AccelerateBLAS.chebyshevDistance(v, v)
        #expect(abs(result) < 0.0001)
    }

    @Test("Chebyshev distance calculation")
    func chebyshevDistanceBasic() {
        let a: [Float] = [0, 0, 0]
        let b: [Float] = [1, 5, 3]
        // max(|0-1|, |0-5|, |0-3|) = max(1, 5, 3) = 5
        let result = AccelerateBLAS.chebyshevDistance(a, b)
        #expect(abs(result - 5.0) < 0.0001)
    }

    @Test("Chebyshev distance with negative values")
    func chebyshevDistanceNegative() {
        let a: [Float] = [-3, 2]
        let b: [Float] = [4, -1]
        // max(|(-3)-4|, |2-(-1)|) = max(7, 3) = 7
        let result = AccelerateBLAS.chebyshevDistance(a, b)
        #expect(abs(result - 7.0) < 0.0001)
    }

    // MARK: - Vector Arithmetic Tests

    @Test("Vector addition")
    func vectorAddition() {
        let a: [Float] = [1, 2, 3]
        let b: [Float] = [4, 5, 6]
        let result = AccelerateBLAS.add(a, b)

        #expect(abs(result[0] - 5) < 0.0001)
        #expect(abs(result[1] - 7) < 0.0001)
        #expect(abs(result[2] - 9) < 0.0001)
    }

    @Test("Vector scaling")
    func vectorScaling() {
        let a: [Float] = [1, 2, 3]
        let result = AccelerateBLAS.scale(a, by: 2.5)

        #expect(abs(result[0] - 2.5) < 0.0001)
        #expect(abs(result[1] - 5.0) < 0.0001)
        #expect(abs(result[2] - 7.5) < 0.0001)
    }

    @Test("In-place addition")
    func inPlaceAddition() {
        var a: [Float] = [1, 2, 3]
        let b: [Float] = [4, 5, 6]
        AccelerateBLAS.addInPlace(&a, b)

        #expect(abs(a[0] - 5) < 0.0001)
        #expect(abs(a[1] - 7) < 0.0001)
        #expect(abs(a[2] - 9) < 0.0001)
    }

    // MARK: - Reduction Tests

    @Test("Max value in vector")
    func maxValue() {
        let v: [Float] = [1, 5, 3, -2, 4]
        let result = AccelerateBLAS.max(v)
        #expect(abs(result - 5.0) < 0.0001)
    }

    @Test("Min value in vector")
    func minValue() {
        let v: [Float] = [1, 5, 3, -2, 4]
        let result = AccelerateBLAS.min(v)
        #expect(abs(result - (-2.0)) < 0.0001)
    }

    @Test("Sum of vector")
    func sumOfVector() {
        let v: [Float] = [1, 2, 3, 4, 5]
        let result = AccelerateBLAS.sum(v)
        #expect(abs(result - 15.0) < 0.0001)
    }

    @Test("Mean of vector")
    func meanOfVector() {
        let v: [Float] = [1, 2, 3, 4, 5]
        let result = AccelerateBLAS.mean(v)
        #expect(abs(result - 3.0) < 0.0001)
    }

    // MARK: - Mean Pooling Tests

    @Test("Mean pooling basic")
    func meanPoolingBasic() {
        // 2 tokens, 3 dimensions
        let sequence: [Float] = [
            1, 2, 3,    // token 0
            4, 5, 6     // token 1
        ]
        let result = AccelerateBLAS.meanPool(sequence: sequence, tokens: 2, dim: 3)

        #expect(abs(result[0] - 2.5) < 0.0001) // (1+4)/2
        #expect(abs(result[1] - 3.5) < 0.0001) // (2+5)/2
        #expect(abs(result[2] - 4.5) < 0.0001) // (3+6)/2
    }

    @Test("Mean pooling with mask")
    func meanPoolingWithMask() {
        // 3 tokens, 2 dimensions
        let sequence: [Float] = [
            1, 2,       // token 0 (masked out)
            3, 4,       // token 1 (kept)
            5, 6        // token 2 (kept)
        ]
        let mask: [Int] = [0, 1, 1]
        let result = AccelerateBLAS.meanPool(sequence: sequence, tokens: 3, dim: 2, mask: mask)

        #expect(abs(result[0] - 4.0) < 0.0001) // (3+5)/2
        #expect(abs(result[1] - 5.0) < 0.0001) // (4+6)/2
    }

    // MARK: - Max Pooling Tests

    @Test("Max pooling basic")
    func maxPoolingBasic() {
        // 2 tokens, 3 dimensions
        let sequence: [Float] = [
            1, 5, 3,    // token 0
            4, 2, 6     // token 1
        ]
        let result = AccelerateBLAS.maxPool(sequence: sequence, tokens: 2, dim: 3)

        #expect(abs(result[0] - 4.0) < 0.0001) // max(1, 4)
        #expect(abs(result[1] - 5.0) < 0.0001) // max(5, 2)
        #expect(abs(result[2] - 6.0) < 0.0001) // max(3, 6)
    }

    @Test("Max pooling with mask")
    func maxPoolingWithMask() {
        // 3 tokens, 2 dimensions
        let sequence: [Float] = [
            10, 20,     // token 0 (masked out - has highest values)
            3, 4,       // token 1 (kept)
            5, 6        // token 2 (kept)
        ]
        let mask: [Int] = [0, 1, 1]
        let result = AccelerateBLAS.maxPool(sequence: sequence, tokens: 3, dim: 2, mask: mask)

        #expect(abs(result[0] - 5.0) < 0.0001) // max(3, 5)
        #expect(abs(result[1] - 6.0) < 0.0001) // max(4, 6)
    }

    // MARK: - Attention Pooling Tests

    @Test("Attention pooling with uniform weights equals mean")
    func attentionPoolingUniform() {
        // 3 tokens, 2 dimensions
        let sequence: [Float] = [
            1, 2,
            3, 4,
            5, 6
        ]
        let weights: [Float] = [1, 1, 1] // Uniform

        let attentionResult = AccelerateBLAS.attentionPool(sequence: sequence, weights: weights, tokens: 3, dim: 2)
        let meanResult = AccelerateBLAS.meanPool(sequence: sequence, tokens: 3, dim: 2)

        #expect(abs(attentionResult[0] - meanResult[0]) < 0.0001)
        #expect(abs(attentionResult[1] - meanResult[1]) < 0.0001)
    }

    @Test("Attention pooling with single weight")
    func attentionPoolingSingleWeight() {
        // 3 tokens, 2 dimensions
        let sequence: [Float] = [
            1, 2,
            3, 4,
            5, 6
        ]
        // All weight on token 1
        let weights: [Float] = [0, 1, 0]

        let result = AccelerateBLAS.attentionPool(sequence: sequence, weights: weights, tokens: 3, dim: 2)

        #expect(abs(result[0] - 3.0) < 0.0001)
        #expect(abs(result[1] - 4.0) < 0.0001)
    }

    // MARK: - Batch Distance Tests

    @Test("Batch cosine distance")
    func batchCosineDistance() {
        let query: [Float] = [1, 0, 0]
        let candidates: [[Float]] = [
            [1, 0, 0],      // identical
            [-1, 0, 0],     // opposite
            [0, 1, 0]       // orthogonal
        ]

        let distances = AccelerateBLAS.batchCosineDistance(query: query, candidates: candidates)

        #expect(abs(distances[0] - 0.0) < 0.0001)  // identical
        #expect(abs(distances[1] - 2.0) < 0.0001)  // opposite: 1 - (-1) = 2
        #expect(abs(distances[2] - 1.0) < 0.0001)  // orthogonal
    }

    @Test("Batch euclidean distance")
    func batchEuclideanDistance() {
        let query: [Float] = [0, 0]
        let candidates: [[Float]] = [
            [3, 4],     // distance = 5
            [0, 0],     // distance = 0
            [1, 0]      // distance = 1
        ]

        let distances = AccelerateBLAS.batchEuclideanDistance(query: query, candidates: candidates)

        #expect(abs(distances[0] - 5.0) < 0.0001)
        #expect(abs(distances[1] - 0.0) < 0.0001)
        #expect(abs(distances[2] - 1.0) < 0.0001)
    }

    @Test("Batch manhattan distance")
    func batchManhattanDistance() {
        let query: [Float] = [0, 0]
        let candidates: [[Float]] = [
            [1, 2],     // distance = 3
            [0, 0],     // distance = 0
            [3, 3]      // distance = 6
        ]

        let distances = AccelerateBLAS.batchManhattanDistance(query: query, candidates: candidates)

        #expect(abs(distances[0] - 3.0) < 0.0001)
        #expect(abs(distances[1] - 0.0) < 0.0001)
        #expect(abs(distances[2] - 6.0) < 0.0001)
    }

    @Test("Batch chebyshev distance")
    func batchChebyshevDistance() {
        let query: [Float] = [0, 0]
        let candidates: [[Float]] = [
            [1, 5],     // distance = 5
            [0, 0],     // distance = 0
            [3, 2]      // distance = 3
        ]

        let distances = AccelerateBLAS.batchChebyshevDistance(query: query, candidates: candidates)

        #expect(abs(distances[0] - 5.0) < 0.0001)
        #expect(abs(distances[1] - 0.0) < 0.0001)
        #expect(abs(distances[2] - 3.0) < 0.0001)
    }

    // MARK: - Edge Case Tests

    @Test("Operations on empty vectors")
    func emptyVectorOperations() {
        let empty: [Float] = []

        #expect(AccelerateBLAS.dotProduct(empty, empty) == 0)
        #expect(AccelerateBLAS.sumOfSquares(empty) == 0)
        #expect(AccelerateBLAS.magnitude(empty) == 0)
        #expect(AccelerateBLAS.sum(empty) == 0)
        #expect(AccelerateBLAS.mean(empty) == 0)
    }

    @Test("Single element vector")
    func singleElementVector() {
        let v: [Float] = [5]

        #expect(abs(AccelerateBLAS.dotProduct(v, v) - 25) < 0.0001)
        #expect(abs(AccelerateBLAS.magnitude(v) - 5) < 0.0001)
        #expect(abs(AccelerateBLAS.sum(v) - 5) < 0.0001)
        #expect(abs(AccelerateBLAS.mean(v) - 5) < 0.0001)
    }

    // MARK: - Performance Comparison Tests

    @Test("Large vector operations complete correctly", .tags(.performance))
    func largeVectorOperations() {
        // Create large vectors (384 dimensions - typical embedding size)
        let size = 384
        var a = [Float](repeating: 0, count: size)
        var b = [Float](repeating: 0, count: size)

        for i in 0..<size {
            a[i] = Float.random(in: -1...1)
            b[i] = Float.random(in: -1...1)
        }

        // Test all operations complete without error
        let dot = AccelerateBLAS.dotProduct(a, b)
        let cosine = AccelerateBLAS.cosineSimilarity(a, b)
        let euclidean = AccelerateBLAS.euclideanDistance(a, b)
        let manhattan = AccelerateBLAS.manhattanDistance(a, b)
        let chebyshev = AccelerateBLAS.chebyshevDistance(a, b)
        let normalized = AccelerateBLAS.normalize(a)

        // Basic sanity checks
        #expect(dot.isFinite)
        #expect(cosine >= -1.001 && cosine <= 1.001)
        #expect(euclidean >= 0)
        #expect(manhattan >= 0)
        #expect(chebyshev >= 0)
        #expect(abs(AccelerateBLAS.magnitude(normalized) - 1.0) < 0.001)
    }

    @Test("Batch operations on many candidates", .tags(.performance))
    func batchOperationsPerformance() {
        let querySize = 384
        let candidateCount = 100

        var query = [Float](repeating: 0, count: querySize)
        for i in 0..<querySize {
            query[i] = Float.random(in: -1...1)
        }

        var candidates: [[Float]] = []
        for _ in 0..<candidateCount {
            var candidate = [Float](repeating: 0, count: querySize)
            for i in 0..<querySize {
                candidate[i] = Float.random(in: -1...1)
            }
            candidates.append(candidate)
        }

        let distances = AccelerateBLAS.batchCosineDistance(query: query, candidates: candidates)

        #expect(distances.count == candidateCount)
        for d in distances {
            #expect(d >= -0.001 && d <= 2.001) // Cosine distance range [0, 2]
        }
    }
}

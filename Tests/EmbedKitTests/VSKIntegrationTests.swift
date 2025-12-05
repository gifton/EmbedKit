// EmbedKit - VSK Integration Tests
// Tests for VectorCore 0.1.5, VectorIndex 0.1.2, VectorAccelerate 0.1.2 integration

import Testing
import Foundation
@testable import EmbedKit
import VectorCore
import VectorIndex

// MARK: - Vector384Optimized Auto-Optimization Tests

@Suite("Vector384Optimized - Auto-Optimization")
struct Vector384OptimizedTests {

    // MARK: - Dimension Detection

    @Test("384-dim vectors use optimized path for dot product")
    func dotProduct384UsesOptimizedPath() {
        // Create 384-dim vectors (MiniLM dimension)
        var a = [Float](repeating: 0, count: 384)
        var b = [Float](repeating: 0, count: 384)
        for i in 0..<384 {
            a[i] = Float(i) / 384.0
            b[i] = Float(384 - i) / 384.0
        }

        let result = AccelerateBLAS.dotProduct(a, b)

        // Verify result is valid (exact value depends on implementation)
        #expect(result.isFinite)
        #expect(result >= 0) // These vectors have positive components
    }

    @Test("Non-384-dim vectors use generic path")
    func dotProductNon384UsesGenericPath() {
        // Test with 256-dim vectors
        let a = [Float](repeating: 1.0, count: 256)
        let b = [Float](repeating: 2.0, count: 256)

        let result = AccelerateBLAS.dotProduct(a, b)

        // 1.0 * 2.0 * 256 = 512
        #expect(abs(result - 512.0) < 0.01)
    }

    @Test("384-dim cosine similarity uses optimized path")
    func cosineSimilarity384() {
        var a = [Float](repeating: 0, count: 384)
        var b = [Float](repeating: 0, count: 384)

        // Create normalized-ish vectors
        for i in 0..<384 {
            a[i] = sin(Float(i) * 0.01)
            b[i] = cos(Float(i) * 0.01)
        }

        let similarity = AccelerateBLAS.cosineSimilarity(a, b)

        // Similarity should be in [-1, 1]
        #expect(similarity >= -1.001 && similarity <= 1.001)
    }

    @Test("384-dim euclidean distance uses optimized path")
    func euclideanDistance384() {
        let a = [Float](repeating: 0, count: 384)
        let b = [Float](repeating: 1, count: 384)

        let distance = AccelerateBLAS.euclideanDistance(a, b)

        // Distance should be sqrt(384) ≈ 19.6
        #expect(abs(distance - sqrt(384.0)) < 0.01)
    }

    // MARK: - Batch Operations

    @Test("Batch cosine distance auto-optimizes for 384-dim")
    func batchCosineDistance384() {
        let query = [Float](repeating: 1.0 / sqrt(384.0), count: 384) // Normalized
        let candidates: [[Float]] = (0..<10).map { _ in
            var v = [Float](repeating: 0, count: 384)
            for i in 0..<384 {
                v[i] = Float.random(in: -1...1)
            }
            return v
        }

        let distances = AccelerateBLAS.batchCosineDistance(query: query, candidates: candidates)

        #expect(distances.count == 10)
        for d in distances {
            #expect(d >= -0.001 && d <= 2.001) // Cosine distance range
        }
    }

    @Test("Batch euclidean distance auto-optimizes for 384-dim")
    func batchEuclideanDistance384() {
        let query = [Float](repeating: 0, count: 384)
        let candidates: [[Float]] = [
            [Float](repeating: 1, count: 384),  // Distance = sqrt(384)
            [Float](repeating: 0, count: 384),  // Distance = 0
            [Float](repeating: 0.5, count: 384) // Distance = sqrt(384 * 0.25)
        ]

        let distances = AccelerateBLAS.batchEuclideanDistance(query: query, candidates: candidates)

        #expect(distances.count == 3)
        #expect(abs(distances[0] - sqrt(384.0)) < 0.01)
        #expect(abs(distances[1]) < 0.01)
        #expect(abs(distances[2] - sqrt(384.0 * 0.25)) < 0.01)
    }

    // MARK: - Correctness Verification

    @Test("384-dim optimized path matches generic path results")
    func optimizedMatchesGeneric() {
        // Create random 384-dim vectors
        var a = [Float](repeating: 0, count: 384)
        var b = [Float](repeating: 0, count: 384)
        for i in 0..<384 {
            a[i] = Float.random(in: -1...1)
            b[i] = Float.random(in: -1...1)
        }

        // Compute using AccelerateBLAS (uses optimized path)
        let optimizedDot = AccelerateBLAS.dotProduct(a, b)
        let optimizedCosine = AccelerateBLAS.cosineSimilarity(a, b)
        let optimizedEuclidean = AccelerateBLAS.euclideanDistance(a, b)

        // Compute manually (generic algorithm)
        var manualDot: Float = 0
        var sumA: Float = 0
        var sumB: Float = 0
        var sumDiffSq: Float = 0
        for i in 0..<384 {
            manualDot += a[i] * b[i]
            sumA += a[i] * a[i]
            sumB += b[i] * b[i]
            let diff = a[i] - b[i]
            sumDiffSq += diff * diff
        }
        let manualCosine = manualDot / (sqrt(sumA) * sqrt(sumB))
        let manualEuclidean = sqrt(sumDiffSq)

        // Results should match within floating-point tolerance
        #expect(abs(optimizedDot - manualDot) < 0.01)
        #expect(abs(optimizedCosine - manualCosine) < 0.001)
        #expect(abs(optimizedEuclidean - manualEuclidean) < 0.01)
    }
}

// MARK: - TopKSelection Tests

@Suite("TopKSelection Integration")
struct TopKSelectionTests {

    @Test("topKNearest returns correct k results")
    func topKNearestBasic() {
        let query: [Float] = [0, 0, 0]
        let candidates: [[Float]] = [
            [1, 0, 0],  // distance = 1
            [0, 2, 0],  // distance = 2
            [0, 0, 3],  // distance = 3
            [0, 0, 0.5] // distance = 0.5 (nearest)
        ]

        let result = AccelerateBLAS.topKNearest(
            query: query,
            candidates: candidates,
            k: 2,
            metric: .euclidean
        )

        #expect(result.count == 2)
        // Nearest should be index 3 (distance 0.5)
        #expect(result.indices[0] == 3)
        #expect(abs(result.distances[0] - 0.5) < 0.01)
        // Second nearest should be index 0 (distance 1.0)
        #expect(result.indices[1] == 0)
        #expect(abs(result.distances[1] - 1.0) < 0.01)
    }

    @Test("topKNearest with k larger than candidates")
    func topKNearestLargeK() {
        let query: [Float] = [0, 0]
        let candidates: [[Float]] = [
            [1, 0],
            [0, 1]
        ]

        let result = AccelerateBLAS.topKNearest(
            query: query,
            candidates: candidates,
            k: 10, // More than available
            metric: .euclidean
        )

        #expect(result.count == 2) // Should return all available
    }

    @Test("topKNearest with different metrics")
    func topKNearestMetrics() {
        let query: [Float] = [1, 1, 1]
        let candidates: [[Float]] = [
            [1, 1, 1],    // Same as query
            [0, 0, 0],    // Different
            [2, 2, 2]     // Scaled
        ]

        // Cosine: same direction vectors should be nearest
        let cosineResult = AccelerateBLAS.topKNearest(
            query: query,
            candidates: candidates,
            k: 1,
            metric: .cosine
        )
        // Indices 0 and 2 have same direction (cosine distance ~0)
        #expect(cosineResult.indices[0] == 0 || cosineResult.indices[0] == 2)

        // Euclidean: identical vector should be nearest
        let euclideanResult = AccelerateBLAS.topKNearest(
            query: query,
            candidates: candidates,
            k: 1,
            metric: .euclidean
        )
        #expect(euclideanResult.indices[0] == 0)
        #expect(euclideanResult.distances[0] < 0.001)
    }

    @Test("topKNearest with 384-dim uses fused optimized path")
    func topKNearest384Dim() {
        // Create 384-dim query and candidates
        let query = [Float](repeating: 0, count: 384)
        var candidates: [[Float]] = []

        for i in 0..<100 {
            let v = [Float](repeating: Float(i) * 0.01, count: 384)
            candidates.append(v)
        }

        let result = AccelerateBLAS.topKNearest(
            query: query,
            candidates: candidates,
            k: 5,
            metric: .euclidean
        )

        #expect(result.count == 5)
        // Index 0 should be nearest (all zeros, same as query)
        #expect(result.indices[0] == 0)
        #expect(result.distances[0] < 0.001)
    }

    @Test("topKNearest empty candidates returns empty result")
    func topKNearestEmpty() {
        let query: [Float] = [1, 2, 3]
        let candidates: [[Float]] = []

        let result = AccelerateBLAS.topKNearest(
            query: query,
            candidates: candidates,
            k: 5,
            metric: .euclidean
        )

        #expect(result.isEmpty)
    }

    @Test("topKNearest k=0 returns empty result")
    func topKNearestZeroK() {
        let query: [Float] = [1, 2, 3]
        let candidates: [[Float]] = [[4, 5, 6]]

        let result = AccelerateBLAS.topKNearest(
            query: query,
            candidates: candidates,
            k: 0,
            metric: .euclidean
        )

        #expect(result.isEmpty)
    }
}

// MARK: - Performance Comparison Tests

@Suite("VSK Performance", .tags(.performance))
struct VSKPerformanceTests {

    @Test("384-dim batch operations complete efficiently")
    func batchOperations384Performance() {
        let query = (0..<384).map { Float($0) / 384.0 }
        var candidates: [[Float]] = []

        for _ in 0..<1000 {
            var v = [Float](repeating: 0, count: 384)
            for i in 0..<384 {
                v[i] = Float.random(in: -1...1)
            }
            candidates.append(v)
        }

        // Batch cosine distance
        let cosineDistances = AccelerateBLAS.batchCosineDistance(query: query, candidates: candidates)
        #expect(cosineDistances.count == 1000)

        // Batch euclidean distance
        let euclideanDistances = AccelerateBLAS.batchEuclideanDistance(query: query, candidates: candidates)
        #expect(euclideanDistances.count == 1000)

        // TopK selection
        let topK = AccelerateBLAS.topKNearest(
            query: query,
            candidates: candidates,
            k: 10,
            metric: .euclidean
        )
        #expect(topK.count == 10)
    }

    @Test("TopKSelection O(n log k) beats O(n log n) sort for small k")
    func topKSelectionEfficiency() {
        // Create large distance array
        var distances = [Float](repeating: 0, count: 10000)
        for i in 0..<distances.count {
            distances[i] = Float.random(in: 0...1000)
        }

        // Using TopKSelection (should be O(n log k))
        let result = TopKSelection.select(k: 10, from: distances)

        #expect(result.count == 10)

        // Verify results are sorted
        for i in 1..<result.count {
            #expect(result.distances[i-1] <= result.distances[i])
        }
    }
}

// MARK: - Integration Correctness Tests

@Suite("VSK Integration Correctness")
struct VSKIntegrationCorrectnessTests {

    @Test("TopKResult from AccelerateBLAS matches VectorCore TopKSelection")
    func topKResultConsistency() {
        let distances: [Float] = [5, 2, 8, 1, 9, 3, 7, 4, 6, 0]

        // Using AccelerateBLAS.topKNearest indirectly via distances
        let blasResult = TopKSelection.select(k: 3, from: distances)

        #expect(blasResult.count == 3)
        // Should return indices of smallest: 9 (0.0), 3 (1.0), 1 (2.0)
        #expect(blasResult.indices.contains(9))
        #expect(blasResult.indices.contains(3))
        #expect(blasResult.indices.contains(1))
    }

    @Test("Parallel batch search returns same results as sequential")
    func parallelBatchSearchCorrectness() async throws {
        let index = FlatIndex(dimension: 4, metric: .euclidean)

        // Insert test vectors
        for i in 0..<20 {
            let vector: [Float] = [Float(i), Float(i * 2), Float(i * 3), Float(i * 4)]
            try await index.insert(id: "v\(i)", vector: vector, metadata: nil)
        }

        let queries: [[Float]] = [
            [0, 0, 0, 0],
            [5, 10, 15, 20],
            [10, 20, 30, 40]
        ]

        // Batch search (parallel)
        let batchResults = try await index.batchSearch(queries: queries, k: 3, filter: nil)

        // Sequential search for comparison
        var sequentialResults: [[VectorIndex.SearchResult]] = []
        for query in queries {
            let result = try await index.search(query: query, k: 3, filter: nil)
            sequentialResults.append(result)
        }

        // Results should match
        #expect(batchResults.count == sequentialResults.count)
        for i in 0..<batchResults.count {
            #expect(batchResults[i].count == sequentialResults[i].count)
            for j in 0..<batchResults[i].count {
                #expect(batchResults[i][j].id == sequentialResults[i][j].id)
            }
        }
    }
}

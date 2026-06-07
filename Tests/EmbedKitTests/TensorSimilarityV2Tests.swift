// Tests for the V2 tensor similarity path after the audit-3.1/3.4/4.2 fixes.
//
// tensorSimilarityMatrixV2 no longer dispatches the incomplete Metal 4
// tensor_handle / single-tile path; it delegates to the proven buffer-based V1
// similarity (cosine/dot) and computes euclidean on a numerically stable CPU
// path. These tests verify correctness, full matrix population (no single-tile
// truncation), and overflow-free euclidean distances.
import Testing
import Foundation
@testable import EmbedKit

@Suite("Tensor Similarity V2")
struct TensorSimilarityV2Tests {

    @Test("Euclidean distance matrix matches reference for small inputs")
    func euclideanSmall() {
        // 2 queries × 2 keys, dim 2.
        let queries: [Float] = [0, 0,  3, 4]
        let keys: [Float]    = [0, 0,  1, 0]
        let m = MetalAccelerator.euclideanDistanceMatrix(
            queries: queries, keys: keys,
            queryBatchSize: 2, keyBatchSize: 2, dimensions: 2)

        #expect(m.count == 2 && m[0].count == 2)
        #expect(abs(m[0][0] - 0) < 1e-4)          // (0,0)-(0,0)
        #expect(abs(m[0][1] - 1) < 1e-4)          // (0,0)-(1,0)
        #expect(abs(m[1][0] - 5) < 1e-4)          // (3,4)-(0,0) = 5
        #expect(abs(m[1][1] - sqrt(Float(20))) < 1e-3) // (3,4)-(1,0)=√(4+16)
    }

    @Test("Euclidean distance stays finite for high-magnitude inputs")
    func euclideanNoOverflow() {
        // Magnitudes whose squared difference (1e40) exceeds Float32 max; the
        // scaled accumulation must keep the result finite (≈ 1e20).
        let queries: [Float] = [1e20, 0]
        let keys: [Float]    = [0, 0]
        let m = MetalAccelerator.euclideanDistanceMatrix(
            queries: queries, keys: keys,
            queryBatchSize: 1, keyBatchSize: 1, dimensions: 2)

        #expect(m[0][0].isFinite)
        #expect(abs(m[0][0] - 1e20) / 1e20 < 1e-3)
    }

    @Test("Euclidean zero-vectors yield zero distance")
    func euclideanZeros() {
        let m = MetalAccelerator.euclideanDistanceMatrix(
            queries: [0, 0, 0], keys: [0, 0, 0],
            queryBatchSize: 1, keyBatchSize: 1, dimensions: 3)
        #expect(m[0][0] == 0)
    }

    @Test("V2 cosine fully populates a matrix larger than one GPU tile")
    func cosineFullMatrixNoTruncation() async {
        // 80×48 exceeds the old single 64×32 tile; every entry must be computed
        // (the previous 1×1×1 dispatch would have left most of it uninitialized).
        let qN = 80, kN = 48, dim = 16
        func vec(_ seed: Int) -> [Float] {
            (0..<dim).map { Float(((seed * 7 + $0 * 3) % 11)) - 5 }
        }
        var queries: [Float] = []
        for q in 0..<qN { queries += vec(q) }
        var keys: [Float] = []
        for k in 0..<kN { keys += vec(k + 100) }

        let acc = await MetalAccelerator()
        guard let m = await acc.tensorSimilarityMatrixV2(
            queries: queries, keys: keys,
            queryBatchSize: qN, keyBatchSize: kN,
            dimensions: dim, normalized: false, metric: 0
        ) else {
            Issue.record("tensorSimilarityMatrixV2 returned nil")
            return
        }

        #expect(m.count == qN)
        #expect(m.allSatisfy { $0.count == kN })
        // Cosine values must be valid and the matrix must not be a degenerate
        // all-zero block (the single-tile truncation symptom).
        #expect(m.allSatisfy { $0.allSatisfy { $0 >= -1.0001 && $0 <= 1.0001 } })
        let nonZero = m.flatMap { $0 }.filter { abs($0) > 1e-6 }.count
        #expect(nonZero > qN * kN / 2, "Matrix looks truncated/uninitialized")
    }
}

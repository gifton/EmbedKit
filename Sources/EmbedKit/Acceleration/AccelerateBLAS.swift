// EmbedKit - Accelerate BLAS Operations
// High-performance CPU operations using Apple's Accelerate framework

import Foundation
import Accelerate
import VectorCore
import struct VectorCore.TopKResult

/// High-performance CPU operations using Apple's Accelerate framework.
///
/// This module provides SIMD-optimized alternatives to common vector operations
/// used throughout EmbedKit. These operations are significantly faster than
/// naive Swift loops, especially for large vectors (100+ elements).
///
/// ## Performance Notes
/// - vDSP operations use SIMD instructions (NEON on ARM, SSE/AVX on x86)
/// - For 384-dimensional vectors (MiniLM), uses VectorCore's Vector384Optimized (~75ns dot product)
/// - For small vectors (<16 elements), overhead may exceed benefits
/// - All operations are thread-safe and can be called from any context
///
/// ## Hot Path Optimization
/// The 384-dimension case is optimized using VectorCore's `Vector384Optimized` type,
/// which provides 2-3x speedup over generic vDSP operations for MiniLM embeddings.
public enum AccelerateBLAS {

    // MARK: - Dimension Detection

    /// The optimized dimension for MiniLM/Sentence-BERT models
    @usableFromInline
    static let optimizedDimension384 = 384

    // MARK: - Dot Product

    /// Computes the dot product of two vectors.
    ///
    /// Uses `vDSP_dotpr` for SIMD-optimized computation.
    /// For 384-dimensional vectors (MiniLM), automatically uses VectorCore's
    /// `Vector384Optimized` for ~75ns performance (2-3x faster).
    ///
    /// - Complexity: O(n) with SIMD parallelism
    /// - Parameters:
    ///   - a: First vector
    ///   - b: Second vector (must have same count as `a`)
    /// - Returns: The dot product `Σ(a[i] * b[i])`
    @inlinable
    public static func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "Vectors must have equal length")
        guard !a.isEmpty else { return 0 }

        // Auto-optimize for 384-dimensional vectors (MiniLM)
        if a.count == optimizedDimension384 {
            return dotProduct384(a, b)
        }

        var result: Float = 0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(a.count))
        return result
    }

    /// Computes the dot product using unsafe pointers (zero-copy).
    @inlinable
    public static func dotProduct(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        guard count > 0 else { return 0 }
        var result: Float = 0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(count))
        return result
    }

    // MARK: - Sum of Squares / Magnitude

    /// Computes the sum of squared elements (squared L2 norm).
    ///
    /// Uses `vDSP_svesq` for SIMD-optimized computation.
    /// - Parameter v: Input vector
    /// - Returns: `Σ(v[i]²)`
    @inlinable
    public static func sumOfSquares(_ v: [Float]) -> Float {
        guard !v.isEmpty else { return 0 }
        var result: Float = 0
        vDSP_svesq(v, 1, &result, vDSP_Length(v.count))
        return result
    }

    /// Computes the L2 magnitude (Euclidean norm) of a vector.
    ///
    /// - Parameter v: Input vector
    /// - Returns: `√(Σ(v[i]²))`
    @inlinable
    public static func magnitude(_ v: [Float]) -> Float {
        sqrt(sumOfSquares(v))
    }

    /// Computes the L2 magnitude using unsafe pointer (zero-copy).
    @inlinable
    public static func magnitude(_ v: UnsafePointer<Float>, count: Int) -> Float {
        guard count > 0 else { return 0 }
        var result: Float = 0
        vDSP_svesq(v, 1, &result, vDSP_Length(count))
        return sqrt(result)
    }

    // MARK: - Normalization

    /// L2-normalizes a vector in-place.
    ///
    /// Uses `vDSP_vsdiv` for SIMD-optimized division.
    /// - Parameter v: Vector to normalize (modified in-place)
    /// - Returns: The original magnitude before normalization
    @inlinable
    @discardableResult
    public static func normalizeInPlace(_ v: inout [Float]) -> Float {
        let mag = magnitude(v)
        guard mag > 1e-12 else { return mag }
        var divisor = mag
        vDSP_vsdiv(v, 1, &divisor, &v, 1, vDSP_Length(v.count))
        return mag
    }

    /// Returns an L2-normalized copy of the vector.
    ///
    /// - Parameter v: Input vector
    /// - Returns: Normalized vector with unit L2 norm
    @inlinable
    public static func normalize(_ v: [Float]) -> [Float] {
        let mag = magnitude(v)
        guard mag > 1e-12 else { return v }
        var result = [Float](repeating: 0, count: v.count)
        var divisor = mag
        vDSP_vsdiv(v, 1, &divisor, &result, 1, vDSP_Length(v.count))
        return result
    }

    // MARK: - Cosine Similarity

    /// Computes cosine similarity between two vectors.
    ///
    /// Formula: `(a · b) / (||a|| * ||b||)`
    /// For 384-dimensional vectors (MiniLM), automatically uses VectorCore's
    /// `Vector384Optimized` for improved performance.
    ///
    /// - Parameters:
    ///   - a: First vector
    ///   - b: Second vector (must have same count as `a`)
    /// - Returns: Cosine similarity in range [-1, 1]
    @inlinable
    public static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "Vectors must have equal length")
        guard !a.isEmpty else { return 0 }

        // Auto-optimize for 384-dimensional vectors (MiniLM)
        if a.count == optimizedDimension384 {
            return cosineSimilarity384(a, b)
        }

        let dot = dotProduct(a, b)
        let magA = magnitude(a)
        let magB = magnitude(b)

        guard magA > 1e-12 && magB > 1e-12 else { return 0 }
        return dot / (magA * magB)
    }

    /// Computes cosine distance (1 - cosine similarity).
    ///
    /// For 384-dimensional vectors (MiniLM), automatically uses the optimized path.
    @inlinable
    public static func cosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        1.0 - cosineSimilarity(a, b)
    }

    // MARK: - Euclidean Distance

    /// Computes squared Euclidean distance between two vectors.
    ///
    /// Uses `vDSP_distancesq` for SIMD-optimized computation.
    /// - Returns: `Σ((a[i] - b[i])²)`
    @inlinable
    public static func euclideanDistanceSquared(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "Vectors must have equal length")
        guard !a.isEmpty else { return 0 }

        var result: Float = 0
        vDSP_distancesq(a, 1, b, 1, &result, vDSP_Length(a.count))
        return result
    }

    /// Computes Euclidean distance between two vectors.
    ///
    /// For 384-dimensional vectors (MiniLM), automatically uses VectorCore's
    /// `Vector384Optimized` for ~90ns performance (2-3x faster).
    @inlinable
    public static func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        // Auto-optimize for 384-dimensional vectors (MiniLM)
        if a.count == optimizedDimension384 {
            return euclideanDistance384(a, b)
        }
        return sqrt(euclideanDistanceSquared(a, b))
    }

    // MARK: - Manhattan Distance

    /// Computes Manhattan (L1) distance between two vectors.
    ///
    /// Formula: `Σ|a[i] - b[i]|`
    @inlinable
    public static func manhattanDistance(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "Vectors must have equal length")
        guard !a.isEmpty else { return 0 }

        // Compute a - b, then absolute values, then sum
        var diff = [Float](repeating: 0, count: a.count)
        vDSP_vsub(b, 1, a, 1, &diff, 1, vDSP_Length(a.count))  // diff = a - b

        var absDiff = [Float](repeating: 0, count: a.count)
        vDSP_vabs(diff, 1, &absDiff, 1, vDSP_Length(a.count))

        var result: Float = 0
        vDSP_sve(absDiff, 1, &result, vDSP_Length(a.count))
        return result
    }

    // MARK: - Chebyshev Distance

    /// Computes Chebyshev (L∞) distance between two vectors.
    ///
    /// Formula: `max|a[i] - b[i]|`
    @inlinable
    public static func chebyshevDistance(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "Vectors must have equal length")
        guard !a.isEmpty else { return 0 }

        // Compute |a - b|, then find max
        var diff = [Float](repeating: 0, count: a.count)
        vDSP_vsub(b, 1, a, 1, &diff, 1, vDSP_Length(a.count))

        var absDiff = [Float](repeating: 0, count: a.count)
        vDSP_vabs(diff, 1, &absDiff, 1, vDSP_Length(a.count))

        var result: Float = 0
        vDSP_maxv(absDiff, 1, &result, vDSP_Length(a.count))
        return result
    }

    // MARK: - Vector Arithmetic

    /// Adds two vectors element-wise: `result = a + b`
    @inlinable
    public static func add(_ a: [Float], _ b: [Float]) -> [Float] {
        precondition(a.count == b.count, "Vectors must have equal length")
        var result = [Float](repeating: 0, count: a.count)
        vDSP_vadd(a, 1, b, 1, &result, 1, vDSP_Length(a.count))
        return result
    }

    /// Adds vector `b` to `a` in-place: `a += b`
    @inlinable
    public static func addInPlace(_ a: inout [Float], _ b: [Float]) {
        precondition(a.count == b.count, "Vectors must have equal length")
        vDSP_vadd(a, 1, b, 1, &a, 1, vDSP_Length(a.count))
    }

    /// Multiplies vector by scalar: `result = a * scalar`
    @inlinable
    public static func scale(_ a: [Float], by scalar: Float) -> [Float] {
        var result = [Float](repeating: 0, count: a.count)
        var s = scalar
        vDSP_vsmul(a, 1, &s, &result, 1, vDSP_Length(a.count))
        return result
    }

    /// Multiplies vector by scalar in-place: `a *= scalar`
    @inlinable
    public static func scaleInPlace(_ a: inout [Float], by scalar: Float) {
        var s = scalar
        vDSP_vsmul(a, 1, &s, &a, 1, vDSP_Length(a.count))
    }

    // MARK: - Reduction Operations

    /// Finds the maximum value in a vector.
    @inlinable
    public static func max(_ v: [Float]) -> Float {
        guard !v.isEmpty else { return -.greatestFiniteMagnitude }
        var result: Float = 0
        vDSP_maxv(v, 1, &result, vDSP_Length(v.count))
        return result
    }

    /// Finds the minimum value in a vector.
    @inlinable
    public static func min(_ v: [Float]) -> Float {
        guard !v.isEmpty else { return .greatestFiniteMagnitude }
        var result: Float = 0
        vDSP_minv(v, 1, &result, vDSP_Length(v.count))
        return result
    }

    /// Computes the sum of all elements.
    @inlinable
    public static func sum(_ v: [Float]) -> Float {
        guard !v.isEmpty else { return 0 }
        var result: Float = 0
        vDSP_sve(v, 1, &result, vDSP_Length(v.count))
        return result
    }

    /// Computes the mean of all elements.
    @inlinable
    public static func mean(_ v: [Float]) -> Float {
        guard !v.isEmpty else { return 0 }
        var result: Float = 0
        vDSP_meanv(v, 1, &result, vDSP_Length(v.count))
        return result
    }

    // MARK: - Pooling Operations

    /// Mean-pools a sequence of token embeddings (row-major layout).
    ///
    /// This is an optimized version using Accelerate for the accumulation.
    /// - Parameters:
    ///   - sequence: Flattened token embeddings [t0_d0, t0_d1, ..., t1_d0, ...]
    ///   - tokens: Number of tokens (rows)
    ///   - dim: Embedding dimension (columns)
    ///   - mask: Optional attention mask (1 = keep, 0 = ignore)
    /// - Returns: Pooled vector of length `dim`
    public static func meanPool(
        sequence: [Float],
        tokens: Int,
        dim: Int,
        mask: [Int]? = nil
    ) -> [Float] {
        precondition(tokens >= 1 && dim >= 1)
        precondition(sequence.count == tokens * dim, "sequence shape mismatch")

        var acc = [Float](repeating: 0, count: dim)
        var count: Int = 0

        sequence.withUnsafeBufferPointer { seqBuf in
            if let mask = mask {
                precondition(mask.count == tokens, "mask length mismatch")
                for t in 0..<tokens where mask[t] != 0 {
                    let base = seqBuf.baseAddress! + t * dim
                    // Add row to accumulator using vDSP
                    vDSP_vadd(acc, 1, base, 1, &acc, 1, vDSP_Length(dim))
                    count += 1
                }
            } else {
                for t in 0..<tokens {
                    let base = seqBuf.baseAddress! + t * dim
                    vDSP_vadd(acc, 1, base, 1, &acc, 1, vDSP_Length(dim))
                }
                count = tokens
            }
        }

        // Fallback to unmasked mean if no tokens selected
        if count == 0 {
            sequence.withUnsafeBufferPointer { seqBuf in
                for t in 0..<tokens {
                    let base = seqBuf.baseAddress! + t * dim
                    vDSP_vadd(acc, 1, base, 1, &acc, 1, vDSP_Length(dim))
                }
            }
            count = tokens
        }

        // Divide by count
        var invCount = 1.0 / Float(count)
        vDSP_vsmul(acc, 1, &invCount, &acc, 1, vDSP_Length(dim))
        return acc
    }

    /// Max-pools a sequence of token embeddings (row-major layout).
    ///
    /// - Parameters:
    ///   - sequence: Flattened token embeddings [t0_d0, t0_d1, ..., t1_d0, ...]
    ///   - tokens: Number of tokens (rows)
    ///   - dim: Embedding dimension (columns)
    ///   - mask: Optional attention mask (1 = keep, 0 = ignore)
    /// - Returns: Element-wise maximum across selected tokens
    public static func maxPool(
        sequence: [Float],
        tokens: Int,
        dim: Int,
        mask: [Int]? = nil
    ) -> [Float] {
        precondition(tokens >= 1 && dim >= 1)
        precondition(sequence.count == tokens * dim, "sequence shape mismatch")

        var acc = [Float](repeating: -.greatestFiniteMagnitude, count: dim)
        var selected = 0

        sequence.withUnsafeBufferPointer { seqBuf in
            if let mask = mask {
                precondition(mask.count == tokens, "mask length mismatch")
                for t in 0..<tokens where mask[t] != 0 {
                    let base = seqBuf.baseAddress! + t * dim
                    // Element-wise max using vDSP
                    vDSP_vmax(acc, 1, base, 1, &acc, 1, vDSP_Length(dim))
                    selected += 1
                }
            }
        }

        // Fallback to unmasked max
        if selected == 0 {
            sequence.withUnsafeBufferPointer { seqBuf in
                for t in 0..<tokens {
                    let base = seqBuf.baseAddress! + t * dim
                    vDSP_vmax(acc, 1, base, 1, &acc, 1, vDSP_Length(dim))
                }
            }
        }

        return acc
    }

    // MARK: - Batch Distance Computation

    /// Computes cosine distances from a query to multiple candidates.
    ///
    /// For 384-dimensional vectors (MiniLM), automatically uses VectorCore's
    /// `Vector384Optimized` for 2-3x faster batch computation.
    ///
    /// - Parameters:
    ///   - query: Query vector
    ///   - candidates: Array of candidate vectors (all must have same dimension)
    /// - Returns: Array of cosine distances
    public static func batchCosineDistance(
        query: [Float],
        candidates: [[Float]]
    ) -> [Float] {
        guard !candidates.isEmpty else { return [] }

        // Auto-optimize for 384-dimensional vectors (MiniLM)
        if query.count == optimizedDimension384 {
            return batchCosineDistance384(query: query, candidates: candidates)
        }

        let queryMag = magnitude(query)
        guard queryMag > 1e-12 else {
            return [Float](repeating: 1.0, count: candidates.count)
        }

        var distances = [Float](repeating: 0, count: candidates.count)

        for (i, candidate) in candidates.enumerated() {
            precondition(candidate.count == query.count, "Dimension mismatch at index \(i)")
            let dot = dotProduct(query, candidate)
            let candMag = magnitude(candidate)

            if candMag > 1e-12 {
                let similarity = dot / (queryMag * candMag)
                distances[i] = 1.0 - similarity
            } else {
                distances[i] = 1.0
            }
        }

        return distances
    }

    /// Computes Euclidean distances from a query to multiple candidates.
    ///
    /// For 384-dimensional vectors (MiniLM), automatically uses VectorCore's
    /// `Vector384Optimized` for 2-3x faster batch computation.
    public static func batchEuclideanDistance(
        query: [Float],
        candidates: [[Float]]
    ) -> [Float] {
        guard !candidates.isEmpty else { return [] }

        // Auto-optimize for 384-dimensional vectors (MiniLM)
        if query.count == optimizedDimension384 {
            return batchEuclideanDistance384(query: query, candidates: candidates)
        }

        var distances = [Float](repeating: 0, count: candidates.count)

        for (i, candidate) in candidates.enumerated() {
            precondition(candidate.count == query.count, "Dimension mismatch at index \(i)")
            distances[i] = euclideanDistance(query, candidate)
        }

        return distances
    }

    /// Computes Manhattan distances from a query to multiple candidates.
    public static func batchManhattanDistance(
        query: [Float],
        candidates: [[Float]]
    ) -> [Float] {
        guard !candidates.isEmpty else { return [] }

        var distances = [Float](repeating: 0, count: candidates.count)

        for (i, candidate) in candidates.enumerated() {
            precondition(candidate.count == query.count, "Dimension mismatch at index \(i)")
            distances[i] = manhattanDistance(query, candidate)
        }

        return distances
    }

    /// Computes Chebyshev distances from a query to multiple candidates.
    public static func batchChebyshevDistance(
        query: [Float],
        candidates: [[Float]]
    ) -> [Float] {
        guard !candidates.isEmpty else { return [] }

        var distances = [Float](repeating: 0, count: candidates.count)

        for (i, candidate) in candidates.enumerated() {
            precondition(candidate.count == query.count, "Dimension mismatch at index \(i)")
            distances[i] = chebyshevDistance(query, candidate)
        }

        return distances
    }

    // MARK: - Weighted Operations

    /// Computes attention-weighted pooling.
    ///
    /// - Parameters:
    ///   - sequence: Flattened token embeddings [t0_d0, t0_d1, ..., t1_d0, ...]
    ///   - weights: Attention weights per token (should sum to ~1)
    ///   - tokens: Number of tokens
    ///   - dim: Embedding dimension
    /// - Returns: Weighted sum of token embeddings
    public static func attentionPool(
        sequence: [Float],
        weights: [Float],
        tokens: Int,
        dim: Int
    ) -> [Float] {
        precondition(tokens >= 1 && dim >= 1)
        precondition(sequence.count == tokens * dim, "sequence shape mismatch")
        precondition(weights.count == tokens, "weights length mismatch")

        var acc = [Float](repeating: 0, count: dim)
        var weightSum: Float = 0

        sequence.withUnsafeBufferPointer { seqBuf in
            for t in 0..<tokens {
                let weight = weights[t]
                weightSum += weight

                let base = seqBuf.baseAddress! + t * dim

                // Scale row by weight and add to accumulator
                // acc += weight * row
                var w = weight
                var scaled = [Float](repeating: 0, count: dim)
                vDSP_vsmul(base, 1, &w, &scaled, 1, vDSP_Length(dim))
                vDSP_vadd(acc, 1, scaled, 1, &acc, 1, vDSP_Length(dim))
            }
        }

        // Normalize by weight sum (if not already normalized)
        if weightSum > 1e-12 && abs(weightSum - 1.0) > 1e-6 {
            var invSum = 1.0 / weightSum
            vDSP_vsmul(acc, 1, &invSum, &acc, 1, vDSP_Length(dim))
        }

        return acc
    }

    // MARK: - Vector384Optimized Hot Path (Internal)

    /// Internal: Computes dot product for 384-dimensional vectors using SIMD-optimized path.
    @usableFromInline
    internal static func dotProduct384(_ a: [Float], _ b: [Float]) -> Float {
        guard let vecA = try? Vector384Optimized(a),
              let vecB = try? Vector384Optimized(b) else {
            return dotProductGeneric(a, b)
        }
        return vecA.dotProduct(vecB)
    }

    /// Internal: Computes Euclidean distance for 384-dimensional vectors.
    @usableFromInline
    internal static func euclideanDistance384(_ a: [Float], _ b: [Float]) -> Float {
        guard let vecA = try? Vector384Optimized(a),
              let vecB = try? Vector384Optimized(b) else {
            return euclideanDistanceGeneric(a, b)
        }
        return vecA.euclideanDistance(to: vecB)
    }

    /// Internal: Computes cosine similarity for 384-dimensional vectors.
    @usableFromInline
    internal static func cosineSimilarity384(_ a: [Float], _ b: [Float]) -> Float {
        guard let vecA = try? Vector384Optimized(a),
              let vecB = try? Vector384Optimized(b) else {
            return cosineSimilarityGeneric(a, b)
        }
        return vecA.cosineSimilarity(to: vecB)
    }

    /// Internal: Generic vDSP dot product (non-384 path).
    @usableFromInline
    internal static func dotProductGeneric(_ a: [Float], _ b: [Float]) -> Float {
        guard !a.isEmpty else { return 0 }
        var result: Float = 0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(a.count))
        return result
    }

    /// Internal: Generic vDSP euclidean distance (non-384 path).
    @usableFromInline
    internal static func euclideanDistanceGeneric(_ a: [Float], _ b: [Float]) -> Float {
        guard !a.isEmpty else { return 0 }
        var result: Float = 0
        vDSP_distancesq(a, 1, b, 1, &result, vDSP_Length(a.count))
        return sqrt(result)
    }

    /// Internal: Generic vDSP cosine similarity (non-384 path).
    @usableFromInline
    internal static func cosineSimilarityGeneric(_ a: [Float], _ b: [Float]) -> Float {
        guard !a.isEmpty else { return 0 }
        let dot = dotProductGeneric(a, b)
        let magA = magnitude(a)
        let magB = magnitude(b)
        guard magA > 1e-12 && magB > 1e-12 else { return 0 }
        return dot / (magA * magB)
    }

    // MARK: - Batch Operations with 384-Dimension Optimization (Internal)

    /// Internal: Batch cosine distance for 384-dimensional vectors.
    @usableFromInline
    internal static func batchCosineDistance384(
        query: [Float],
        candidates: [[Float]]
    ) -> [Float] {
        guard let queryVec = try? Vector384Optimized(query) else {
            return batchCosineDistance(query: query, candidates: candidates)
        }

        var distances = [Float](repeating: 0, count: candidates.count)

        for (i, candidate) in candidates.enumerated() {
            if candidate.count == 384, let candVec = try? Vector384Optimized(candidate) {
                distances[i] = 1.0 - queryVec.cosineSimilarity(to: candVec)
            } else {
                distances[i] = cosineDistance(query, candidate)
            }
        }

        return distances
    }

    /// Internal: Batch Euclidean distance for 384-dimensional vectors.
    @usableFromInline
    internal static func batchEuclideanDistance384(
        query: [Float],
        candidates: [[Float]]
    ) -> [Float] {
        guard let queryVec = try? Vector384Optimized(query) else {
            return batchEuclideanDistance(query: query, candidates: candidates)
        }

        var distances = [Float](repeating: 0, count: candidates.count)

        for (i, candidate) in candidates.enumerated() {
            if candidate.count == 384, let candVec = try? Vector384Optimized(candidate) {
                distances[i] = queryVec.euclideanDistance(to: candVec)
            } else {
                distances[i] = euclideanDistance(query, candidate)
            }
        }

        return distances
    }

    /// Internal: Fused top-k Euclidean search for 384-dimensional vectors.
    @usableFromInline
    internal static func topKEuclidean384(
        query: [Float],
        candidates: [[Float]],
        k: Int
    ) -> TopKResult {
        guard let queryVec = try? Vector384Optimized(query) else {
            let distances = batchEuclideanDistance(query: query, candidates: candidates)
            return TopKSelection.select(k: k, from: distances)
        }

        var candidateVecs: [Vector384Optimized] = []
        candidateVecs.reserveCapacity(candidates.count)

        for candidate in candidates {
            if candidate.count == 384, let candVec = try? Vector384Optimized(candidate) {
                candidateVecs.append(candVec)
            } else {
                // Dimension mismatch - fallback to generic
                let distances = batchEuclideanDistance(query: query, candidates: candidates)
                return TopKSelection.select(k: k, from: distances)
            }
        }

        return TopKSelection.nearestEuclidean384(k: k, query: queryVec, candidates: candidateVecs)
    }

    // MARK: - Public API with Auto-Optimization

    /// Finds the k nearest candidates with automatic dimension optimization.
    ///
    /// For 384-dimensional vectors with Euclidean metric, automatically uses VectorCore's
    /// fused `TopKSelection.nearestEuclidean384()` for 2-3x performance improvement.
    /// Falls back to generic vDSP + heap selection for other dimensions.
    ///
    /// - Parameters:
    ///   - query: Query vector
    ///   - candidates: Array of candidate vectors
    ///   - k: Number of nearest neighbors to return
    ///   - metric: Distance metric to use
    /// - Returns: TopKResult with indices and distances of k nearest neighbors
    public static func topKNearest(
        query: [Float],
        candidates: [[Float]],
        k: Int,
        metric: SupportedDistanceMetric
    ) -> TopKResult {
        guard k > 0 && !candidates.isEmpty else { return TopKResult() }

        let actualK = Swift.min(k, candidates.count)

        // Use fused optimized path for 384-dim Euclidean
        if query.count == optimizedDimension384 && metric == .euclidean {
            return topKEuclidean384(query: query, candidates: candidates, k: actualK)
        }

        // Generic path: compute distances then select top-k
        let distances: [Float]
        switch metric {
        case .cosine:
            if query.count == optimizedDimension384 {
                distances = batchCosineDistance384(query: query, candidates: candidates)
            } else {
                distances = batchCosineDistance(query: query, candidates: candidates)
            }
        case .euclidean:
            if query.count == optimizedDimension384 {
                distances = batchEuclideanDistance384(query: query, candidates: candidates)
            } else {
                distances = batchEuclideanDistance(query: query, candidates: candidates)
            }
        case .dotProduct:
            distances = candidates.map { -dotProduct(query, $0) }
        case .manhattan:
            distances = batchManhattanDistance(query: query, candidates: candidates)
        case .chebyshev:
            distances = batchChebyshevDistance(query: query, candidates: candidates)
        }

        return TopKSelection.select(k: actualK, from: distances)
    }
}

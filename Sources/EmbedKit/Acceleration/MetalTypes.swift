// EmbedKit - Metal Types
//
// Parameter structures for Metal compute kernels. All structs are 16-byte aligned
// to match the Metal shader definitions in MetalCommon.h.

import Foundation

// MARK: - Kernel Names

/// Names of available Metal compute kernels
public enum MetalKernelName: String, CaseIterable, Sendable {
    case l2Normalize = "l2_normalize"
    case l2NormalizeBatchOptimized = "l2_normalize_batch_optimized"
    case meanPool = "mean_pool"
    case maxPool = "max_pool"
    case attentionWeightedPool = "attention_weighted_pool"
    case cosineSimilarity = "cosine_similarity"
    case cosineSimilarityBatch = "cosine_similarity_batch"
}

// MARK: - Parameter Structures

/// Parameters for pooling operations (mean, max, attention-weighted)
///
/// **Memory Layout**: 16 bytes total (4 x Int32)
/// Matches `PoolingParams` in MetalCommon.h
///
/// - Note: Padding fields ensure 16-byte alignment for optimal GPU memory access
@frozen
public struct PoolingParams: Sendable {
    public let sequenceLength: Int32
    public let dimensions: Int32
    private let _padding0: Int32
    private let _padding1: Int32

    public init(sequenceLength: Int, dimensions: Int) {
        self.sequenceLength = Int32(sequenceLength)
        self.dimensions = Int32(dimensions)
        self._padding0 = 0
        self._padding1 = 0
    }
}

/// Parameters for cosine similarity matrix calculations
///
/// **Memory Layout**: 16 bytes total (4 x Int32)
/// Matches `SimilarityParams` in MetalCommon.h
///
/// - Note: Computes similarity between `queryCount` queries and `keyCount` keys
@frozen
public struct SimilarityParams: Sendable {
    public let queryCount: Int32
    public let keyCount: Int32
    public let dimensions: Int32
    private let _padding0: Int32

    public init(queryCount: Int, keyCount: Int, dimensions: Int) {
        self.queryCount = Int32(queryCount)
        self.keyCount = Int32(keyCount)
        self.dimensions = Int32(dimensions)
        self._padding0 = 0
    }
}

/// Parameters for batch cosine similarity calculations
///
/// **Memory Layout**: 16 bytes total (4 x Int32)
/// Matches `BatchSimilarityParams` in MetalCommon.h
///
/// - Note: Processes `pairCount` independent vector pairs in parallel
@frozen
public struct BatchSimilarityParams: Sendable {
    public let pairCount: Int32
    public let dimensions: Int32
    private let _padding0: Int32
    private let _padding1: Int32

    public init(pairCount: Int, dimensions: Int) {
        self.pairCount = Int32(pairCount)
        self.dimensions = Int32(dimensions)
        self._padding0 = 0
        self._padding1 = 0
    }
}

// MARK: - Internal Vector Batch

/// Internal batch container for zero-copy GPU transfers
///
/// Uses flat row-major storage to eliminate the overhead of nested array allocations
/// and enable direct Metal buffer creation via pointer.
///
/// **Memory Layout (Row-Major)**:
/// ```
/// [v0_d0, v0_d1, ..., v0_dN, v1_d0, v1_d1, ..., v1_dN, ...]
/// ```
internal struct VectorBatch: Sendable {
    /// Flat row-major storage of all vectors
    var data: [Float]

    /// Number of vectors in the batch
    let count: Int

    /// Dimensions per vector
    let dimensions: Int

    /// Total number of elements (count * dimensions)
    var totalElements: Int { count * dimensions }

    /// Size in bytes for Metal buffer allocation
    var sizeInBytes: Int { data.count * MemoryLayout<Float>.size }

    /// Whether the batch is empty
    var isEmpty: Bool { count == 0 }

    /// Initialize from flat data with known count and dimensions
    ///
    /// - Parameters:
    ///   - data: Flat row-major float array
    ///   - count: Number of vectors
    ///   - dimensions: Dimensions per vector
    /// - Throws: `EmbedKitError.dimensionMismatch` if data.count != count * dimensions
    init(data: [Float], count: Int, dimensions: Int) throws {
        guard data.count == count * dimensions else {
            throw EmbedKitError.dimensionMismatch(
                expected: count * dimensions,
                got: data.count
            )
        }
        self.data = data
        self.count = count
        self.dimensions = dimensions
    }

    /// Initialize from array of vectors
    ///
    /// - Parameter vectors: Array of equal-length float vectors
    /// - Throws: `EmbedKitError.dimensionMismatch` if vectors have inconsistent dimensions
    init(vectors: [[Float]]) throws {
        guard let first = vectors.first else {
            self.data = []
            self.count = 0
            self.dimensions = 0
            return
        }
        let dims = first.count
        for (_, v) in vectors.enumerated() {
            guard v.count == dims else {
                throw EmbedKitError.dimensionMismatch(expected: dims, got: v.count)
            }
        }
        self.data = vectors.flatMap { $0 }
        self.count = vectors.count
        self.dimensions = dims
    }

    /// Access data via unsafe buffer pointer for zero-copy Metal transfer
    @inlinable
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try data.withUnsafeBufferPointer(body)
    }

    /// Convert back to array of vectors
    func toVectors() -> [[Float]] {
        guard dimensions > 0 else { return [] }
        var result: [[Float]] = []
        result.reserveCapacity(count)
        var offset = 0
        for _ in 0..<count {
            let end = offset + dimensions
            result.append(Array(data[offset..<end]))
            offset = end
        }
        return result
    }
}

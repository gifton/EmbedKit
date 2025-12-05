// EmbedKit - Index Configuration
// Configuration types for GPU-accelerated vector index

import Foundation
import VectorCore
import VectorAccelerate

// MARK: - Index Type

/// Type of GPU vector index to use.
///
/// EmbedKit uses VectorAccelerate's `AcceleratedVectorIndex` for all indexing,
/// providing GPU-accelerated search with two index types.
public enum IndexType: String, Sendable, CaseIterable, Codable {
    /// Flat (brute-force) GPU index.
    ///
    /// Best for:
    /// - Small to medium datasets (< 50K vectors)
    /// - When exact results are required
    /// - Sub-millisecond search latency
    ///
    /// Performance: ~0.30ms search on 5K vectors (128D)
    case flat

    /// IVF (Inverted File) GPU index with K-means clustering.
    ///
    /// Best for:
    /// - Large datasets (50K+ vectors)
    /// - When approximate results are acceptable
    /// - Memory-constrained environments
    ///
    /// Requires calling `train()` after inserting vectors.
    case ivf
}

// MARK: - Distance Metric

/// Re-export of VectorCore's distance metric for convenience.
public typealias DistanceMetric = SupportedDistanceMetric

// MARK: - Index Configuration

/// Configuration for creating a GPU-accelerated embedding index.
///
/// `IndexConfiguration` defines the parameters for VectorAccelerate's
/// `AcceleratedVectorIndex`, which provides GPU-first vector search.
///
/// ## Example Usage
/// ```swift
/// // Small dataset - exact search
/// let config = IndexConfiguration.flat(dimension: 384)
///
/// // Large dataset - approximate search
/// let config = IndexConfiguration.ivf(
///     dimension: 768,
///     nlist: 256,
///     nprobe: 16,
///     capacity: 100_000
/// )
/// ```
public struct IndexConfiguration: Sendable {

    // MARK: - Core Properties

    /// Type of GPU index to create.
    public let indexType: IndexType

    /// Vector dimension (must match embedding model output).
    public let dimension: Int

    /// Distance metric for similarity computation.
    public let metric: DistanceMetric

    /// Initial GPU buffer capacity (number of vectors).
    ///
    /// The buffer grows automatically (2x strategy) if exceeded,
    /// but pre-allocating reduces reallocations.
    public let capacity: Int

    /// Whether to store original text with embeddings.
    public let storeText: Bool

    // MARK: - IVF Parameters

    /// Number of clusters for IVF index.
    ///
    /// Only used when `indexType == .ivf`.
    /// Recommended: sqrt(n) to 4*sqrt(n) where n = expected vector count.
    public let nlist: Int?

    /// Number of clusters to probe during IVF search.
    ///
    /// Only used when `indexType == .ivf`.
    /// Higher values = better recall, slower search.
    /// Recommended: 1-20% of nlist.
    public let nprobe: Int?

    // MARK: - Factory Methods

    /// Create a flat index configuration for exact search.
    ///
    /// - Parameters:
    ///   - dimension: Vector dimension (e.g., 384 for MiniLM, 768 for BERT)
    ///   - metric: Distance metric (default: cosine)
    ///   - capacity: Initial capacity (default: 10,000)
    ///   - storeText: Whether to store original text (default: true)
    /// - Returns: Configuration for flat GPU index
    public static func flat(
        dimension: Int,
        metric: DistanceMetric = .cosine,
        capacity: Int = 10_000,
        storeText: Bool = true
    ) -> IndexConfiguration {
        IndexConfiguration(
            indexType: .flat,
            dimension: dimension,
            metric: metric,
            capacity: capacity,
            storeText: storeText,
            nlist: nil,
            nprobe: nil
        )
    }

    /// Create an IVF index configuration for approximate search.
    ///
    /// IVF uses K-means clustering to partition vectors. At search time,
    /// only `nprobe` clusters are searched instead of all vectors.
    ///
    /// **Important:** Call `train()` on the EmbeddingStore after inserting
    /// vectors to build the cluster centroids.
    ///
    /// - Parameters:
    ///   - dimension: Vector dimension
    ///   - nlist: Number of clusters (default: 256)
    ///   - nprobe: Clusters to search (default: 16)
    ///   - metric: Distance metric (default: cosine)
    ///   - capacity: Initial capacity (default: 100,000)
    ///   - storeText: Whether to store original text (default: false for large datasets)
    /// - Returns: Configuration for IVF GPU index
    public static func ivf(
        dimension: Int,
        nlist: Int = 256,
        nprobe: Int = 16,
        metric: DistanceMetric = .cosine,
        capacity: Int = 100_000,
        storeText: Bool = false
    ) -> IndexConfiguration {
        IndexConfiguration(
            indexType: .ivf,
            dimension: dimension,
            metric: metric,
            capacity: capacity,
            storeText: storeText,
            nlist: nlist,
            nprobe: nprobe
        )
    }

    /// Default configuration - flat index with cosine similarity.
    ///
    /// Suitable for most embedding use cases with < 50K vectors.
    public static func `default`(dimension: Int) -> IndexConfiguration {
        .flat(dimension: dimension)
    }

    /// Configuration for small datasets with exact search.
    public static func exact(dimension: Int) -> IndexConfiguration {
        .flat(dimension: dimension, capacity: 10_000, storeText: true)
    }

    /// Configuration for large datasets prioritizing speed.
    ///
    /// Uses IVF with automatic cluster sizing based on expected size.
    public static func scalable(
        dimension: Int,
        expectedSize: Int = 100_000
    ) -> IndexConfiguration {
        let nlist = max(16, min(Int(sqrt(Double(expectedSize)) * 2), 1024))
        let nprobe = max(1, nlist / 16)
        return .ivf(
            dimension: dimension,
            nlist: nlist,
            nprobe: nprobe,
            capacity: expectedSize,
            storeText: false
        )
    }

    // MARK: - Initialization

    public init(
        indexType: IndexType,
        dimension: Int,
        metric: DistanceMetric = .cosine,
        capacity: Int = 10_000,
        storeText: Bool = true,
        nlist: Int? = nil,
        nprobe: Int? = nil
    ) {
        self.indexType = indexType
        self.dimension = dimension
        self.metric = metric
        self.capacity = capacity
        self.storeText = storeText
        self.nlist = nlist
        self.nprobe = nprobe
    }

    // MARK: - VectorAccelerate Conversion

    /// Convert to VectorAccelerate's IndexConfiguration.
    internal func toVectorAccelerate() -> VectorAccelerate.IndexConfiguration {
        switch indexType {
        case .flat:
            return .flat(
                dimension: dimension,
                metric: metric,
                capacity: capacity
            )
        case .ivf:
            return .ivf(
                dimension: dimension,
                nlist: nlist ?? 256,
                nprobe: nprobe ?? 16,
                metric: metric,
                capacity: capacity
            )
        }
    }

    // MARK: - Validation

    /// Validate the configuration.
    /// - Throws: `IndexConfigurationError` if invalid
    public func validate() throws {
        guard dimension > 0 else {
            throw IndexConfigurationError.invalidDimension(dimension)
        }

        guard capacity > 0 else {
            throw IndexConfigurationError.invalidCapacity(capacity)
        }

        if indexType == .ivf {
            guard let nlist = nlist, nlist > 0 else {
                throw IndexConfigurationError.invalidIVFParameters("nlist must be > 0")
            }
            guard let nprobe = nprobe, nprobe > 0, nprobe <= nlist else {
                throw IndexConfigurationError.invalidIVFParameters("nprobe must be > 0 and <= nlist")
            }
        }
    }
}

// MARK: - Errors

/// Errors from index configuration validation.
public enum IndexConfigurationError: Error, LocalizedError, Sendable {
    case invalidDimension(Int)
    case invalidCapacity(Int)
    case invalidIVFParameters(String)

    public var errorDescription: String? {
        switch self {
        case .invalidDimension(let d):
            return "Invalid dimension: \(d). Dimension must be > 0."
        case .invalidCapacity(let c):
            return "Invalid capacity: \(c). Capacity must be > 0."
        case .invalidIVFParameters(let msg):
            return "Invalid IVF parameters: \(msg)"
        }
    }
}

// EmbedKit - Index Configuration
// Configuration types for vector index setup

import Foundation
import VectorCore
import VectorIndex

// MARK: - Index Type

/// Type of vector index to use.
public enum IndexType: String, Sendable, CaseIterable {
    /// Flat index - exact nearest neighbor search.
    /// Best for: small datasets (<10K vectors), when exact results required.
    case flat

    /// HNSW (Hierarchical Navigable Small World) index.
    /// Best for: medium to large datasets, fast approximate search.
    case hnsw

    /// IVF (Inverted File) index.
    /// Best for: very large datasets with acceptable accuracy trade-off.
    case ivf
}

// MARK: - Distance Metric

/// Re-export of VectorCore's distance metric for convenience.
public typealias DistanceMetric = SupportedDistanceMetric

// MARK: - Index Configuration

/// Configuration for creating an embedding index.
public struct IndexConfiguration: Sendable {
    /// Type of index to create.
    public let indexType: IndexType

    /// Vector dimension (must match embedding model output).
    public let dimension: Int

    /// Distance metric for similarity computation.
    public let metric: DistanceMetric

    /// Whether to store original text with embeddings.
    public let storeText: Bool

    /// HNSW-specific configuration.
    public let hnswConfig: HNSWConfiguration?

    /// IVF-specific configuration.
    public let ivfConfig: IVFConfiguration?

    /// Compute preference for GPU/CPU selection.
    public let computePreference: ComputePreference

    /// Default configuration for common use cases.
    public static func `default`(
        dimension: Int,
        computePreference: ComputePreference = .auto
    ) -> IndexConfiguration {
        IndexConfiguration(
            indexType: .hnsw,
            dimension: dimension,
            metric: .cosine,
            storeText: true,
            hnswConfig: .default,
            ivfConfig: nil,
            computePreference: computePreference
        )
    }

    /// Configuration for small datasets with exact search.
    public static func exact(
        dimension: Int,
        computePreference: ComputePreference = .auto
    ) -> IndexConfiguration {
        IndexConfiguration(
            indexType: .flat,
            dimension: dimension,
            metric: .cosine,
            storeText: true,
            hnswConfig: nil,
            ivfConfig: nil,
            computePreference: computePreference
        )
    }

    /// Configuration for large datasets prioritizing speed.
    public static func fast(
        dimension: Int,
        computePreference: ComputePreference = .auto
    ) -> IndexConfiguration {
        IndexConfiguration(
            indexType: .hnsw,
            dimension: dimension,
            metric: .cosine,
            storeText: false,
            hnswConfig: .fast,
            ivfConfig: nil,
            computePreference: computePreference
        )
    }

    /// Configuration for very large datasets.
    public static func scalable(
        dimension: Int,
        expectedSize: Int = 100_000,
        computePreference: ComputePreference = .auto
    ) -> IndexConfiguration {
        let nlist = max(16, min(expectedSize / 100, 1024))
        return IndexConfiguration(
            indexType: .ivf,
            dimension: dimension,
            metric: .cosine,
            storeText: false,
            hnswConfig: nil,
            ivfConfig: IVFConfiguration(nlist: nlist, nprobe: max(1, nlist / 16)),
            computePreference: computePreference
        )
    }

    public init(
        indexType: IndexType,
        dimension: Int,
        metric: DistanceMetric = .cosine,
        storeText: Bool = true,
        hnswConfig: HNSWConfiguration? = nil,
        ivfConfig: IVFConfiguration? = nil,
        computePreference: ComputePreference = .auto
    ) {
        self.indexType = indexType
        self.dimension = dimension
        self.metric = metric
        self.storeText = storeText
        self.hnswConfig = hnswConfig
        self.ivfConfig = ivfConfig
        self.computePreference = computePreference
    }
}

// MARK: - HNSW Configuration

/// Configuration for HNSW index.
public struct HNSWConfiguration: Sendable {
    /// Maximum number of connections per node per layer.
    public let m: Int

    /// Size of dynamic candidate list during construction.
    public let efConstruction: Int

    /// Size of dynamic candidate list during search.
    public let efSearch: Int

    /// Default balanced configuration.
    public static let `default` = HNSWConfiguration(
        m: 16,
        efConstruction: 200,
        efSearch: 64
    )

    /// Configuration prioritizing search speed.
    public static let fast = HNSWConfiguration(
        m: 12,
        efConstruction: 100,
        efSearch: 32
    )

    /// Configuration prioritizing accuracy.
    public static let accurate = HNSWConfiguration(
        m: 32,
        efConstruction: 400,
        efSearch: 128
    )

    public init(m: Int = 16, efConstruction: Int = 200, efSearch: Int = 64) {
        self.m = m
        self.efConstruction = efConstruction
        self.efSearch = efSearch
    }

    /// Convert to VectorIndex HNSW configuration.
    internal var toVectorIndex: HNSWIndex.Configuration {
        HNSWIndex.Configuration(
            m: m,
            efConstruction: efConstruction,
            efSearch: efSearch
        )
    }
}

// MARK: - IVF Configuration

/// Configuration for IVF index.
public struct IVFConfiguration: Sendable {
    /// Number of clusters (coarse centroids).
    public let nlist: Int

    /// Number of clusters to probe during search.
    public let nprobe: Int

    /// Default configuration.
    public static let `default` = IVFConfiguration(nlist: 256, nprobe: 8)

    public init(nlist: Int = 256, nprobe: Int = 8) {
        self.nlist = nlist
        self.nprobe = nprobe
    }

    /// Convert to VectorIndex IVF configuration.
    internal var toVectorIndex: IVFIndex.Configuration {
        IVFIndex.Configuration(nlist: nlist, nprobe: nprobe)
    }
}

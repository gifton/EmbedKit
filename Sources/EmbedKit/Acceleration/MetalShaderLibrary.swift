import Foundation

/// Centralized Metal shader library containing all GPU kernels for embedding operations
///
/// This struct provides a single source of truth for all Metal shaders used in EmbedKit,
/// making shader management easier and more maintainable.
public struct MetalShaderLibrary {

    /// Names of available compute kernels
    public enum KernelName: String, CaseIterable {
        case l2Normalize = "l2_normalize"
        case l2NormalizeBatchOptimized = "l2_normalize_batch_optimized"  // Batch-optimized for improved throughput
        case meanPool = "mean_pool"
        case maxPool = "max_pool"
        case cosineSimilarity = "cosine_similarity"
        case cosineSimilarityBatch = "cosine_similarity_batch"
        case attentionWeightedPool = "attention_weighted_pool"
    }

    /// Metal shader source - DEPRECATED
    /// Shaders are now compiled to metallib. Run ./Scripts/CompileMetalShaders.sh
    /// This string is only kept for emergency fallback and will be removed in future versions.
    public static let source = """
    #error Metal shaders must be compiled to metallib. Run: ./Scripts/CompileMetalShaders.sh
    """
}

// MARK: - Supporting Parameter Structures

/// Parameters for pooling operations
/// 16-byte aligned for optimal GPU memory access
@frozen
public struct PoolingParams {
    public let sequenceLength: Int32
    public let dimensions: Int32
    private let _padding0: Int32 = 0  // Explicit padding to 16 bytes
    private let _padding1: Int32 = 0

    public init(sequenceLength: Int, dimensions: Int) {
        self.sequenceLength = Int32(sequenceLength)
        self.dimensions = Int32(dimensions)
    }
}

/// Parameters for similarity calculations
/// 16-byte aligned for optimal GPU memory access
@frozen
public struct SimilarityParams {
    public let queryCount: Int32
    public let keyCount: Int32
    public let dimensions: Int32
    private let _padding0: Int32 = 0  // Explicit padding to 16 bytes

    public init(queryCount: Int, keyCount: Int, dimensions: Int) {
        self.queryCount = Int32(queryCount)
        self.keyCount = Int32(keyCount)
        self.dimensions = Int32(dimensions)
    }
}

/// Parameters for batch similarity calculations
/// 16-byte aligned for optimal GPU memory access
@frozen
public struct BatchSimilarityParams {
    public let pairCount: Int32
    public let dimensions: Int32
    private let _padding0: Int32 = 0  // Explicit padding to 16 bytes
    private let _padding1: Int32 = 0

    public init(pairCount: Int, dimensions: Int) {
        self.pairCount = Int32(pairCount)
        self.dimensions = Int32(dimensions)
    }
}

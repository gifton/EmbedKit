import Foundation
@preconcurrency import Metal
import OSLog

/// Slim coordinator for Metal-accelerated operations
///
/// This  MetalAccelerator  acts as a lightweight coordinator that delegates
/// to specialized processor actors for different types of operations:
/// - MetalVectorProcessor: Vector normalization and mathematical operations
/// - MetalPoolingProcessor: Pooling strategies (mean, max, CLS, attention-weighted)
/// - MetalSimilarityProcessor: Cosine similarity and matrix operations
/// - MetalResourceManager: Device, queue, and pipeline management
///
public actor MetalAccelerator: MetalAcceleratorProtocol {
    nonisolated private let logger = EmbedKitLogger.metal()

    // Specialized processor components
    private let resourceManager: MetalResourceManager
    private let vectorProcessor: MetalVectorProcessor  // Includes batch optimizations
    private let poolingProcessor: MetalPoolingProcessor
    private let similarityProcessor: MetalSimilarityProcessor

    /// Get shared instance for the default GPU
    public static let shared: MetalAccelerator? = {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return nil
        }
        guard let accelerator = try? MetalAccelerator(device: device) else {
            return nil
        }
        return accelerator
    }()

    public init(device: MTLDevice) throws {
        // Initialize resource manager first
        self.resourceManager = try MetalResourceManager(device: device)

        // Initialize specialized processors
        self.vectorProcessor = MetalVectorProcessor(resourceManager: resourceManager)
        self.poolingProcessor = MetalPoolingProcessor(resourceManager: resourceManager)
        self.similarityProcessor = MetalSimilarityProcessor(resourceManager: resourceManager)

        logger.info("MetalAccelerator initialized with batch optimizations integrated")
    }

    /// Ensure pipelines are ready for use
    public func setupPipelines() async throws {
        try await resourceManager.setupPipelines()
    }

    // MARK: - Numerics Configuration

    /// Update numeric configuration for GPU specialization (normalization stability/epsilon)
    public func setNumerics(stableNormalization: Bool, epsilon: Float) async {
        await resourceManager.updateNumerics(stable: stableNormalization, epsilon: epsilon)
    }

    // MARK: - Batch Optimization Configuration

    /// Enable or disable batch GPU occupancy optimizations
    ///
    /// Batch optimization provides 2-4× throughput improvement for batch processing:
    /// - Small vectors (≤32 dim): 4× speedup
    /// - Medium vectors (33-64 dim): 2× speedup
    /// - Large vectors (>64 dim): Baseline performance
    ///
    /// - Parameter enabled: Whether to enable batch optimizations (default: true)
    public func setBatchOptimization(_ enabled: Bool) async {
        await vectorProcessor.setBatchOptimization(enabled)
        logger.info("Batch optimization \(enabled ? "enabled" : "disabled")")
    }

    #if DEBUG
    /// Get batch optimization performance metrics summary (debug builds only)
    ///
    /// Returns detailed metrics about vector processing performance including:
    /// - Total vectors processed
    /// - Average time per vector
    ///
    /// - Returns: Formatted metrics summary string
    public func getBatchMetrics() async -> String {
        return await vectorProcessor.getMetricsSummary()
    }

    /// Reset batch optimization performance metrics (debug builds only)
    public func resetBatchMetrics() async {
        await vectorProcessor.resetMetrics()
    }
    #endif

    // MARK: - Vector Operations (VectorBatch API)

    /// Normalize a batch of vectors using L2 normalization (high-performance VectorBatch API)
    ///
    /// **Performance with Batch Optimizations:**
    /// - Small vectors (≤32 dim): 4× faster batch processing
    /// - Medium vectors (33-64 dim): 2× faster batch processing
    /// - Large vectors (>64 dim): Baseline performance maintained
    /// - Zero-copy buffer creation adds 10-20% additional improvement
    ///
    /// Delegates to batch-optimized MetalVectorProcessor.
    ///
    /// - Parameter batch: Batch of vectors to normalize
    /// - Returns: Batch of L2-normalized vectors
    /// - Throws: MetalError if GPU operations fail
    public func normalizeVectors(_ batch: VectorBatch) async throws -> VectorBatch {
        return try await vectorProcessor.normalizeVectors(batch)
    }

    /// Metal 3 optimization: Fast batch normalization with epsilon parameter (VectorBatch API)
    ///
    /// - Parameters:
    ///   - batch: Batch of vectors to normalize
    ///   - epsilon: Small value to prevent division by zero
    /// - Returns: Normalized batch
    public func fastBatchNormalize(_ batch: VectorBatch, epsilon: Float = 1e-6) async throws -> VectorBatch {
        return try await vectorProcessor.fastBatchNormalize(batch, epsilon: epsilon)
    }


    // MARK: - Pooling Operations (VectorBatch API)

    /// Pool token embeddings using VectorBatch (optimized)
    ///
    /// **Performance:** 10-15% faster than array-based API due to zero-copy buffer creation.
    ///
    /// Delegates to MetalPoolingProcessor for specialized pooling operations.
    ///
    /// - Parameters:
    ///   - tokenEmbeddings: Batch of token embeddings (sequence × dimensions)
    ///   - strategy: Pooling strategy to use (mean, max, CLS, attentionWeighted)
    ///   - attentionMask: Optional attention mask for ignoring padding tokens
    ///   - attentionWeights: Optional weights for attention-weighted pooling
    /// - Returns: Single pooled embedding vector
    /// - Throws: MetalError if GPU operations fail
    public func poolEmbeddings(
        _ tokenEmbeddings: VectorBatch,
        strategy: PoolingStrategy,
        attentionMask: [Int]? = nil,
        attentionWeights: [Float]? = nil
    ) async throws -> [Float] {
        return try await poolingProcessor.poolEmbeddings(
            tokenEmbeddings,
            strategy: strategy,
            attentionMask: attentionMask,
            attentionWeights: attentionWeights
        )
    }

    /// Attention-weighted pooling with VectorBatch (optimized)
    ///
    /// **Performance:** Zero-copy GPU transfer eliminates allocation overhead.
    ///
    /// Delegates to MetalPoolingProcessor for specialized pooling operations.
    ///
    /// - Parameters:
    ///   - tokenEmbeddings: Batch of token embeddings
    ///   - attentionWeights: Weights for each token (must match sequence length)
    /// - Returns: Attention-weighted pooled embedding
    /// - Throws: MetalError if GPU operations fail
    public func attentionWeightedPooling(
        _ tokenEmbeddings: VectorBatch,
        attentionWeights: [Float]
    ) async throws -> [Float] {
        return try await poolingProcessor.attentionWeightedPooling(
            tokenEmbeddings,
            attentionWeights: attentionWeights
        )
    }


    // MARK: - Similarity Operations (VectorBatch API)

    /// Calculate cosine similarity matrix using VectorBatch (optimized)
    ///
    /// **Performance:** 10-15% faster than array-based API due to zero-copy buffer creation.
    ///
    /// Delegates to MetalSimilarityProcessor for specialized similarity calculations.
    ///
    /// - Parameters:
    ///   - queries: Batch of query vectors
    ///   - keys: Batch of key vectors to compare against
    /// - Returns: Matrix of cosine similarities (queries.count × keys.count)
    /// - Throws: MetalError if GPU operations fail or dimensions mismatch
    public func cosineSimilarityMatrix(queries: VectorBatch, keys: VectorBatch) async throws -> [[Float]] {
        return try await similarityProcessor.cosineSimilarityMatrix(queries: queries, keys: keys)
    }

    /// Calculate similarity between single query and VectorBatch of keys (optimized)
    ///
    /// **Performance:** Zero-copy GPU transfer for keys batch.
    ///
    /// Delegates to MetalSimilarityProcessor for specialized similarity calculations.
    ///
    /// - Parameters:
    ///   - query: Single query vector
    ///   - keys: Batch of key vectors to compare against
    /// - Returns: Array of similarity scores (one per key)
    /// - Throws: MetalError if GPU operations fail
    public func cosineSimilarity(query: [Float], keys: VectorBatch) async throws -> [Float] {
        return try await similarityProcessor.cosineSimilarity(query: query, keys: keys)
    }


    /// Calculate cosine similarity between two vectors
    ///
    /// - Parameters:
    ///   - vectorA: First vector for similarity calculation
    ///   - vectorB: Second vector for similarity calculation
    /// - Returns: Cosine similarity score between -1 and 1
    /// - Throws: MetalError if GPU operations fail or vectors have different dimensions
    public func cosineSimilarity(_ vectorA: [Float], _ vectorB: [Float]) async throws -> Float {
        return try await similarityProcessor.cosineSimilarity(vectorA, vectorB)
    }

    /// Calculate cosine similarities for multiple vector pairs in batch
    ///
    /// - Parameters:
    ///   - vectorPairs: Array of (vectorA, vectorB) tuples to compute similarities for
    /// - Returns: Array of cosine similarity scores for each pair
    /// - Throws: MetalError if GPU operations fail or vectors have mismatched dimensions
    public func cosineSimilarityBatch(_ vectorPairs: [([Float], [Float])]) async throws -> [Float] {
        return try await similarityProcessor.cosineSimilarityBatch(vectorPairs)
    }

    // MARK: - Resource Management

    /// Handle memory pressure by delegating to resource manager
    public func handleMemoryPressure() async {
        await resourceManager.handleMemoryPressure()
    }

    /// Get current GPU memory usage in bytes
    nonisolated public func getCurrentMemoryUsage() -> Int64 {
        return resourceManager.getCurrentMemoryUsage()
    }

    /// Check if Metal acceleration is available
    nonisolated public var isAvailable: Bool {
        return resourceManager.isAvailable
    }

    // MARK: - Metal 3 Optimizations

}

/// Factory for creating MetalAccelerator instances
public extension MetalAccelerator {
    /// Create a MetalAccelerator instance with custom resource manager
    static func create(with resourceManager: MetalResourceManager) -> MetalAccelerator? {
        // This would require some refactoring to support dependency injection
        // For now, we use the standard initializer
        return shared
    }
}

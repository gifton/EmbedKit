import Foundation

/// Protocol defining the interface for Metal acceleration operations
///
/// This protocol allows for both real Metal implementations and mock implementations
/// for testing environments where Metal is not available.
public protocol MetalAcceleratorProtocol: Actor {
    /// Check if acceleration is available
    var isAvailable: Bool { get }

    /// Normalize vectors using L2 normalization with VectorBatch
    func normalizeVectors(_ batch: VectorBatch) async throws -> VectorBatch

    /// Fast batch normalization with epsilon parameter using VectorBatch
    func fastBatchNormalize(_ batch: VectorBatch, epsilon: Float) async throws -> VectorBatch

    /// Pool token embeddings using the specified strategy
    func poolEmbeddings(
        _ tokenEmbeddings: VectorBatch,
        strategy: PoolingStrategy,
        attentionMask: [Int]?,
        attentionWeights: [Float]?
    ) async throws -> [Float]

    /// Attention-weighted pooling implementation
    func attentionWeightedPooling(
        _ tokenEmbeddings: VectorBatch,
        attentionWeights: [Float]
    ) async throws -> [Float]

    /// Calculate cosine similarity matrix between two sets of vectors
    func cosineSimilarityMatrix(queries: VectorBatch, keys: VectorBatch) async throws -> [[Float]]

    /// Calculate similarity between a single query and multiple keys
    func cosineSimilarity(query: [Float], keys: VectorBatch) async throws -> [Float]

    /// Calculate cosine similarity between two vectors
    func cosineSimilarity(_ vectorA: [Float], _ vectorB: [Float]) async throws -> Float

    /// Handle memory pressure
    func handleMemoryPressure() async
}

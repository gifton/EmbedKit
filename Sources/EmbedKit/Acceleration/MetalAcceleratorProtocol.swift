import Foundation

/// Protocol defining the interface for Metal acceleration operations
///
/// This protocol allows for both real Metal implementations and mock implementations
/// for testing environments where Metal is not available.
public protocol MetalAcceleratorProtocol: Actor {
    /// Check if acceleration is available
    var isAvailable: Bool { get }
    
    /// Normalize vectors using L2 normalization
    func normalizeVectors(_ vectors: [[Float]]) async throws -> [[Float]]
    
    /// Fast batch normalization with epsilon parameter
    func fastBatchNormalize(_ vectors: [[Float]], epsilon: Float) async throws -> [[Float]]
    
    /// Pool token embeddings using the specified strategy
    func poolEmbeddings(
        _ tokenEmbeddings: [[Float]],
        strategy: PoolingStrategy,
        attentionMask: [Int]?
    ) async throws -> [Float]
    
    /// Attention-weighted pooling implementation
    func attentionWeightedPooling(
        _ tokenEmbeddings: [[Float]],
        attentionWeights: [Float]
    ) async throws -> [Float]
    
    /// Calculate cosine similarity matrix between two sets of vectors
    func cosineSimilarityMatrix(queries: [[Float]], keys: [[Float]]) async throws -> [[Float]]
    
    /// Calculate similarity between a single query and multiple keys
    func cosineSimilarity(query: [Float], keys: [[Float]]) async throws -> [Float]
    
    /// Calculate cosine similarity between two vectors
    func cosineSimilarity(_ vectorA: [Float], _ vectorB: [Float]) async throws -> Float
    
    /// Calculate cosine similarities for multiple vector pairs in batch
    func cosineSimilarityBatch(_ vectorPairs: [([Float], [Float])]) async throws -> [Float]
    
    /// Handle memory pressure
    func handleMemoryPressure() async
}
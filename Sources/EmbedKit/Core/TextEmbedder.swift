import Foundation

/// A vector representation of text in high-dimensional space
///
/// `EmbeddingVector` is the fundamental output type of all text embedding operations.
/// It represents text as a dense numerical vector that captures semantic meaning,
/// enabling mathematical operations like similarity comparisons.
///
/// Design decisions:
/// - Immutable by design to ensure thread safety and prevent accidental modifications
/// - Conforms to `Collection` for seamless integration with Swift's algorithms
/// - Stores values as `Float` (32-bit) rather than `Double` for memory efficiency,
///   as embedding models typically don't require 64-bit precision
/// - Marked as `Sendable` for safe concurrent access across actor boundaries
public struct EmbeddingVector: Collection, Sendable {
    public typealias Element = Float
    public typealias Index = Int
    
    private let values: [Float]
    
    public init(_ values: [Float]) {
        self.values = values
    }
    
    public var startIndex: Int { values.startIndex }
    public var endIndex: Int { values.endIndex }
    
    public subscript(index: Int) -> Float {
        values[index]
    }
    
    public func index(after i: Int) -> Int {
        values.index(after: i)
    }
    
    /// The dimensionality of the embedding vector
    public var dimensions: Int { values.count }
    
    /// The underlying array of float values
    public var array: [Float] { values }
    
    /// Compute cosine similarity with another vector
    ///
    /// Cosine similarity measures the angle between two vectors, ranging from -1 to 1:
    /// - 1.0: Vectors point in the same direction (semantically identical)
    /// - 0.0: Vectors are orthogonal (unrelated)
    /// - -1.0: Vectors point in opposite directions (semantically opposite)
    ///
    /// Implementation note: Uses manual loop instead of Accelerate framework here
    /// for simplicity, but production code could benefit from SIMD optimization
    ///
    /// - Parameter other: The vector to compare with
    /// - Returns: Cosine similarity score, or 0 if dimensions mismatch
    public func cosineSimilarity(with other: EmbeddingVector) -> Float {
        guard dimensions == other.dimensions else { return 0 }
        
        var dotProduct: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        
        for i in 0..<dimensions {
            dotProduct += self[i] * other[i]
            normA += self[i] * self[i]
            normB += other[i] * other[i]
        }
        
        guard normA > 0 && normB > 0 else { return 0 }
        return dotProduct / (sqrt(normA) * sqrt(normB))
    }
}

/// Configuration for text embedding generation
///
/// This configuration struct encapsulates all parameters that affect how embeddings
/// are generated. It's designed to be immutable and `Sendable` for safe concurrent use.
///
/// Design rationale:
/// - Provides sensible defaults based on common transformer models (BERT-like)
/// - Allows fine-tuning for specific use cases without modifying core logic
/// - Separates configuration from implementation for better testability
public struct EmbeddingConfiguration: Sendable {
    /// Maximum sequence length for input text
    public let maxSequenceLength: Int
    
    /// Whether to normalize embeddings to unit length
    public let normalizeEmbeddings: Bool
    
    /// Pooling strategy for token embeddings
    public let poolingStrategy: PoolingStrategy
    
    /// Batch size for processing multiple texts
    public let batchSize: Int
    
    /// Whether to use GPU acceleration when available
    public let useGPUAcceleration: Bool
    
    public init(
        maxSequenceLength: Int = 512,
        normalizeEmbeddings: Bool = true,
        poolingStrategy: PoolingStrategy = .mean,
        batchSize: Int = 32,
        useGPUAcceleration: Bool = true
    ) {
        self.maxSequenceLength = maxSequenceLength
        self.normalizeEmbeddings = normalizeEmbeddings
        self.poolingStrategy = poolingStrategy
        self.batchSize = batchSize
        self.useGPUAcceleration = useGPUAcceleration
    }
}

/// Strategy for pooling token embeddings into a single vector
///
/// Transformer models output one embedding per token. This enum defines
/// how to combine these into a single vector representing the entire text.
///
/// Performance implications:
/// - `cls`: Fastest, O(1) - just returns first token
/// - `mean`: O(n) - requires averaging all tokens
/// - `max`: O(n) - requires comparing all tokens
/// - `attentionWeighted`: O(n) - most expensive but potentially most accurate
public enum PoolingStrategy: String, CaseIterable, Sendable {
    /// Average all token embeddings
    case mean
    /// Use only the CLS token embedding
    case cls
    /// Take the maximum value across all tokens
    case max
    /// Average pooling with attention weights
    case attentionWeighted
}

/// Errors that can occur during embedding generation
public enum EmbeddingError: LocalizedError, Equatable {
    case modelNotLoaded
    case tokenizationFailed(String)
    case inferenceFailed(String)
    case invalidInput(String)
    case dimensionMismatch(expected: Int, actual: Int)
    case resourceUnavailable(String)
    
    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "The embedding model is not loaded"
        case .tokenizationFailed(let details):
            return "Failed to tokenize input: \(details)"
        case .inferenceFailed(let details):
            return "Model inference failed: \(details)"
        case .invalidInput(let details):
            return "Invalid input: \(details)"
        case .dimensionMismatch(let expected, let actual):
            return "Dimension mismatch: expected \(expected), got \(actual)"
        case .resourceUnavailable(let resource):
            return "Resource unavailable: \(resource)"
        }
    }
}

/// Protocol for text embedding generation
///
/// Core abstraction for all embedding implementations. Designed as an actor to:
/// - Ensure thread-safe access to model resources
/// - Prevent concurrent model loading/unloading issues
/// - Manage memory pressure from large models
///
/// Implementation guidance:
/// - Models should lazy-load on first use or explicit `loadModel()` call
/// - Batch operations should be optimized over sequential processing
/// - Memory management is critical - models can be 100MB-1GB+
///
/// PipelineKit integration:
/// - Handlers wrap TextEmbedder instances for command processing
/// - Middleware can inject caching, monitoring, and acceleration
public protocol TextEmbedder: Actor {
    /// The configuration for this embedder
    var configuration: EmbeddingConfiguration { get }
    
    /// The number of dimensions in the embedding space
    var dimensions: Int { get }
    
    /// Unique identifier for the model being used
    var modelIdentifier: String { get }
    
    /// Whether the model is currently loaded and ready
    var isReady: Bool { get async }
    
    /// Generate an embedding for a single text
    /// - Parameter text: The input text to embed
    /// - Returns: The embedding vector
    func embed(_ text: String) async throws -> EmbeddingVector
    
    /// Generate embeddings for multiple texts
    /// - Parameter texts: Array of texts to embed
    /// - Returns: Array of embedding vectors
    func embed(batch texts: [String]) async throws -> [EmbeddingVector]
    
    /// Load the model if not already loaded
    func loadModel() async throws
    
    /// Unload the model to free memory
    func unloadModel() async throws
    
    /// Warm up the model with a sample input
    func warmup() async throws
}

/// Extension providing default implementations
public extension TextEmbedder {
    /// Default batch implementation using sequential processing
    func embed(batch texts: [String]) async throws -> [EmbeddingVector] {
        var results: [EmbeddingVector] = []
        results.reserveCapacity(texts.count)
        
        for text in texts {
            let embedding = try await embed(text)
            results.append(embedding)
        }
        
        return results
    }
    
    /// Default warmup implementation
    func warmup() async throws {
        _ = try await embed("warmup")
    }
}
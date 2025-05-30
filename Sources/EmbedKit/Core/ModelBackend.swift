import Foundation
import CoreML

/// Token representation for model input
public struct TokenizedInput: Sendable {
    /// Token IDs for the model
    public let tokenIds: [Int]
    
    /// Attention mask (1 for real tokens, 0 for padding)
    public let attentionMask: [Int]
    
    /// Token type IDs (for models that use them)
    public let tokenTypeIds: [Int]?
    
    /// Original text length before tokenization
    public let originalLength: Int
    
    public init(
        tokenIds: [Int],
        attentionMask: [Int],
        tokenTypeIds: [Int]? = nil,
        originalLength: Int
    ) {
        self.tokenIds = tokenIds
        self.attentionMask = attentionMask
        self.tokenTypeIds = tokenTypeIds
        self.originalLength = originalLength
    }
}

/// Output from model inference
public struct ModelOutput: Sendable {
    /// Raw embeddings for each token
    public let tokenEmbeddings: [[Float]]
    
    /// Attention weights if available
    public let attentionWeights: [[Float]]?
    
    /// Model-specific metadata
    public let metadata: [String: String]
    
    public init(
        tokenEmbeddings: [[Float]],
        attentionWeights: [[Float]]? = nil,
        metadata: [String: String] = [:]
    ) {
        self.tokenEmbeddings = tokenEmbeddings
        self.attentionWeights = attentionWeights
        self.metadata = metadata
    }
}

/// Protocol for different model backend implementations
public protocol ModelBackend: Actor {
    /// Unique identifier for this backend
    var identifier: String { get }
    
    /// Whether the model is currently loaded
    var isLoaded: Bool { get }
    
    /// Model metadata
    var metadata: ModelMetadata? { get }
    
    /// Load a model from the specified URL
    /// - Parameter url: URL to the model file
    func loadModel(from url: URL) async throws
    
    /// Unload the current model
    func unloadModel() async throws
    
    /// Generate embeddings for tokenized input
    /// - Parameter input: Tokenized input
    /// - Returns: Model output with embeddings
    func generateEmbeddings(for input: TokenizedInput) async throws -> ModelOutput
    
    /// Generate embeddings for a batch of inputs
    /// - Parameter inputs: Array of tokenized inputs
    /// - Returns: Array of model outputs
    func generateEmbeddings(for inputs: [TokenizedInput]) async throws -> [ModelOutput]
    
    /// Get the expected input dimensions
    func inputDimensions() async -> (sequence: Int, features: Int)?
    
    /// Get the output embedding dimensions
    func outputDimensions() async -> Int?
}

/// Metadata about a loaded model
public struct ModelMetadata: Sendable {
    /// Model name
    public let name: String
    
    /// Model version
    public let version: String
    
    /// Embedding dimensions
    public let embeddingDimensions: Int
    
    /// Maximum sequence length
    public let maxSequenceLength: Int
    
    /// Vocabulary size
    public let vocabularySize: Int
    
    /// Model type (e.g., "bert", "sentence-transformers")
    public let modelType: String
    
    /// Additional metadata
    public let additionalInfo: [String: String]
    
    public init(
        name: String,
        version: String,
        embeddingDimensions: Int,
        maxSequenceLength: Int,
        vocabularySize: Int,
        modelType: String,
        additionalInfo: [String: String] = [:]
    ) {
        self.name = name
        self.version = version
        self.embeddingDimensions = embeddingDimensions
        self.maxSequenceLength = maxSequenceLength
        self.vocabularySize = vocabularySize
        self.modelType = modelType
        self.additionalInfo = additionalInfo
    }
}

/// Default implementation for batch processing
public extension ModelBackend {
    func generateEmbeddings(for inputs: [TokenizedInput]) async throws -> [ModelOutput] {
        var results: [ModelOutput] = []
        results.reserveCapacity(inputs.count)
        
        for input in inputs {
            let output = try await generateEmbeddings(for: input)
            results.append(output)
        }
        
        return results
    }
}

/// Configuration for model backends
public struct ModelBackendConfiguration: Sendable {
    /// Whether to use memory mapping for large models
    public let useMemoryMapping: Bool
    
    /// Whether to use compute units (GPU/Neural Engine)
    public let computeUnits: MLComputeUnits
    
    /// Model loading timeout in seconds
    public let loadingTimeout: TimeInterval
    
    /// Maximum memory usage in bytes (0 for unlimited)
    public let maxMemoryUsage: Int
    
    public init(
        useMemoryMapping: Bool = true,
        computeUnits: MLComputeUnits = .all,
        loadingTimeout: TimeInterval = 30,
        maxMemoryUsage: Int = 0
    ) {
        self.useMemoryMapping = useMemoryMapping
        self.computeUnits = computeUnits
        self.loadingTimeout = loadingTimeout
        self.maxMemoryUsage = maxMemoryUsage
    }
}
import Foundation
import CoreML

/// Token representation for model input
///
/// Represents text after tokenization, ready for model consumption.
/// Design follows the standard transformer input format (BERT/RoBERTa style).
///
/// Memory considerations:
/// - Token arrays can be large (512-4096 tokens per sequence)
/// - Consider using ContiguousArray for better cache locality in production
/// - Attention mask could be compressed as bit vector for memory savings
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
///
/// Encapsulates raw model outputs before pooling/post-processing.
/// Designed to preserve all information from the model for flexible processing.
///
/// Performance note: Token embeddings can be memory-intensive
/// (e.g., 512 tokens × 768 dims × 4 bytes = 1.5MB per sequence)
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
///
/// Abstraction layer for different ML frameworks (CoreML, Metal Performance Shaders, etc.)
/// Actor-based design ensures thread-safe model access and prevents race conditions
/// during model loading/unloading.
///
/// Implementation considerations:
/// - Models are expensive resources - implement proper lifecycle management
/// - Batch processing should be atomic to prevent partial failures
/// - Memory-mapped models can reduce RAM usage but may impact latency
///
/// Why actor instead of class:
/// - Automatic synchronization for model state
/// - Prevents concurrent model mutations
/// - Natural fit for async/await patterns in Swift
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
///
/// Centralizes all backend-specific settings that affect performance and resource usage.
///
/// Key trade-offs:
/// - Memory mapping: Lower RAM usage vs potential disk I/O latency
/// - Compute units: GPU/ANE faster but limited memory vs CPU flexibility
/// - Loading timeout: Prevents hanging but may fail on slower devices
///
/// Default values chosen for optimal performance on modern iOS/macOS devices
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

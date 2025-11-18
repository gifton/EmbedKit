// EmbedKit - Clean Multi-Model Architecture
// No legacy, no migration, pure design

import Foundation
import CoreML

// ============================================================================
// MARK: - Core Model Protocol
// ============================================================================

/// The fundamental protocol every embedding model must implement
public protocol EmbeddingModel: Actor {

    /// Unique identifier for this model instance
    var id: ModelID { get }

    /// Model's native dimensions
    var dimensions: Int { get }

    /// Compute device being used
    var device: ComputeDevice { get }

    /// Generate embedding for single text
    func embed(_ text: String) async throws -> Embedding

    /// Batch processing with custom options
    func embedBatch(
        _ texts: [String],
        options: BatchOptions
    ) async throws -> [Embedding]

    /// Preload model resources
    func warmup() async throws

    /// Release model resources
    func release() async throws

    /// Get current model metrics
    var metrics: ModelMetrics { get async }
}

// ============================================================================
// MARK: - Model Identification
// ============================================================================

public struct ModelID: Hashable, Codable, CustomStringConvertible {
    public let provider: String     // "apple", "openai", "local"
    public let name: String         // "text-embedding"
    public let version: String      // "1.0.0"
    public let variant: String?     // "base", "large", "xl"

    public var description: String {
        let v = variant.map { "-\($0)" } ?? ""
        return "\(provider)/\(name)\(v)@\(version)"
    }

    public init(provider: String, name: String, version: String, variant: String? = nil) {
        self.provider = provider
        self.name = name
        self.version = version
        self.variant = variant
    }
}

// ============================================================================
// MARK: - Embedding Types
// ============================================================================

public struct Embedding: Sendable {
    public let vector: [Float]
    public let metadata: EmbeddingMetadata

    public var dimensions: Int { vector.count }

    /// Computed properties for common operations
    public var magnitude: Float {
        sqrt(vector.reduce(0) { $0 + $1 * $1 })
    }

    public var isNormalized: Bool {
        abs(magnitude - 1.0) < 1e-6
    }

    /// Normalize to unit vector
    public func normalized() -> Embedding {
        let mag = magnitude
        guard mag > 0 else { return self }
        return Embedding(
            vector: vector.map { $0 / mag },
            metadata: metadata
        )
    }

    /// Cosine similarity with another embedding
    public func similarity(to other: Embedding) -> Float {
        guard dimensions == other.dimensions else { return 0 }

        let dotProduct = zip(vector, other.vector)
            .reduce(0) { $0 + $1.0 * $1.1 }

        return dotProduct / (magnitude * other.magnitude)
    }
}

public struct EmbeddingMetadata: Codable, Sendable {
    public let modelID: ModelID
    public let tokenCount: Int
    public let processingTime: TimeInterval
    public let normalized: Bool
    public let poolingStrategy: PoolingStrategy
    public let truncated: Bool
    public let custom: [String: String]

    public init(
        modelID: ModelID,
        tokenCount: Int,
        processingTime: TimeInterval,
        normalized: Bool = true,
        poolingStrategy: PoolingStrategy = .mean,
        truncated: Bool = false,
        custom: [String: String] = [:]
    ) {
        self.modelID = modelID
        self.tokenCount = tokenCount
        self.processingTime = processingTime
        self.normalized = normalized
        self.poolingStrategy = poolingStrategy
        self.truncated = truncated
        self.custom = custom
    }
}

// ============================================================================
// MARK: - Configuration Types
// ============================================================================

/// User-configurable options for embedding generation
public struct EmbeddingConfiguration: Sendable {
    // Tokenization
    public var maxTokens: Int = 512
    public var truncationStrategy: TruncationStrategy = .end
    public var paddingStrategy: PaddingStrategy = .none

    // Processing
    public var poolingStrategy: PoolingStrategy = .mean
    public var normalizeOutput: Bool = true
    public var includeSpecialTokens: Bool = true

    // Performance
    public var batchSize: Int = 32
    public var useMixedPrecision: Bool = false
    public var cacheTokenization: Bool = true

    // Device
    public var preferredDevice: ComputeDevice = .auto
    public var allowFallback: Bool = true

    public init() {}
}

public enum TruncationStrategy: String, CaseIterable, Sendable {
    case none       // Error if too long
    case end        // Truncate from end
    case start      // Truncate from start
    case middle     // Remove from middle
}

public enum PaddingStrategy: String, CaseIterable, Sendable {
    case none       // No padding
    case max        // Pad to max length
    case batch      // Pad to max in batch
}

public enum PoolingStrategy: String, CaseIterable, Sendable {
    case cls        // Use [CLS] token
    case mean       // Average all tokens
    case max        // Max pooling
    case meanSqrt   // Mean with sqrt length normalization
    case attention  // Attention-weighted
    case last       // Last token (GPT-style)
}

public enum ComputeDevice: String, CaseIterable, Sendable {
    case cpu
    case gpu
    case ane        // Apple Neural Engine
    case auto       // Let system decide
}

// ============================================================================
// MARK: - Tokenizer Protocol
// ============================================================================

public protocol Tokenizer: Sendable {
    /// Tokenize text into tokens and IDs
    func encode(_ text: String, config: TokenizerConfig) async throws -> TokenizedText

    /// Decode token IDs back to text
    func decode(_ ids: [Int]) async throws -> String

    /// Get vocabulary size
    var vocabularySize: Int { get }

    /// Special tokens used
    var specialTokens: SpecialTokens { get }
}

public struct TokenizedText: Sendable {
    public let ids: [Int]
    public let tokens: [String]
    public let attentionMask: [Int]
    public let typeIds: [Int]?
    public let specialTokenMask: [Bool]
    public let offsets: [Range<String.Index>]?

    public var length: Int { ids.count }
}

public struct TokenizerConfig: Sendable {
    public var maxLength: Int = 512
    public var truncation: TruncationStrategy = .end
    public var padding: PaddingStrategy = .none
    public var addSpecialTokens: Bool = true
    public var returnOffsets: Bool = false

    public init() {}
}

public struct SpecialTokens: Sendable {
    public let cls: Token?
    public let sep: Token?
    public let pad: Token?
    public let unk: Token?
    public let mask: Token?
    public let bos: Token?
    public let eos: Token?

    public struct Token: Sendable {
        public let text: String
        public let id: Int
    }
}

// ============================================================================
// MARK: - Model Backend Protocol
// ============================================================================

public protocol ModelBackend: Actor {
    associatedtype Input
    associatedtype Output

    /// Process input through the model
    func process(_ input: Input, options: ProcessingOptions) async throws -> Output

    /// Batch processing
    func processBatch(_ inputs: [Input], options: ProcessingOptions) async throws -> [Output]

    /// Load model into memory
    func load() async throws

    /// Unload model from memory
    func unload() async throws

    /// Check if model is loaded
    var isLoaded: Bool { get }

    /// Get memory usage
    var memoryUsage: Int64 { get }
}

public struct ProcessingOptions: Sendable {
    public var device: ComputeDevice = .auto
    public var precision: ComputePrecision = .float32
    public var enableProfiling: Bool = false

    public init() {}
}

public enum ComputePrecision: String, CaseIterable, Sendable {
    case float16
    case float32
    case mixed
}

// ============================================================================
// MARK: - Batch Processing
// ============================================================================

public struct BatchOptions: Sendable {
    public var maxBatchSize: Int = 32
    public var dynamicBatching: Bool = true
    public var sortByLength: Bool = true  // Optimize padding
    public var timeout: TimeInterval?

    public init() {}
}

// ============================================================================
// MARK: - Model Metrics
// ============================================================================

public struct ModelMetrics: Sendable {
    public let totalRequests: Int
    public let totalTokensProcessed: Int
    public let averageLatency: TimeInterval
    public let p95Latency: TimeInterval
    public let throughput: Double  // tokens/second
    public let cacheHitRate: Double
    public let memoryUsage: Int64
    public let lastUsed: Date
}

// ============================================================================
// MARK: - Error Types
// ============================================================================

public enum EmbedKitError: LocalizedError, Sendable {
    case modelNotFound(ModelID)
    case modelLoadFailed(String)
    case tokenizationFailed(String)
    case dimensionMismatch(expected: Int, got: Int)
    case deviceNotAvailable(ComputeDevice)
    case inputTooLong(length: Int, max: Int)
    case batchSizeExceeded(size: Int, max: Int)
    case processingTimeout
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let id):
            return "Model not found: \(id)"
        case .modelLoadFailed(let reason):
            return "Failed to load model: \(reason)"
        case .tokenizationFailed(let reason):
            return "Tokenization failed: \(reason)"
        case .dimensionMismatch(let expected, let got):
            return "Dimension mismatch: expected \(expected), got \(got)"
        case .deviceNotAvailable(let device):
            return "Device not available: \(device)"
        case .inputTooLong(let length, let max):
            return "Input too long: \(length) tokens (max: \(max))"
        case .batchSizeExceeded(let size, let max):
            return "Batch size exceeded: \(size) (max: \(max))"
        case .processingTimeout:
            return "Processing timeout"
        case .invalidConfiguration(let reason):
            return "Invalid configuration: \(reason)"
        }
    }
}
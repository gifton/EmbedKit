import Foundation
import CoreML

// MARK: - Core Protocol Definitions

/// Protocol for text embedding generation
public protocol TextEmbedderProtocol: Actor {
    /// Generate embeddings for a batch of texts
    func embed(_ texts: [String]) async throws -> [Embedding]
    
    /// Generate embedding for a single text
    func embed(_ text: String) async throws -> Embedding
    
    /// Get the embedding dimension
    var embeddingDimension: Int { get async }
    
    /// Get the model identifier
    var modelIdentifier: ModelIdentifier { get async }
    
    /// Check if the embedder is ready
    var isReady: Bool { get async }
}

/// Protocol for tokenization
public protocol TokenizerProtocol: Sendable {
    /// Maximum sequence length supported
    var maxSequenceLength: Int { get }
    
    /// Tokenize text into token IDs
    func tokenize(_ text: String) async throws -> [Int]
    
    /// Batch tokenize multiple texts
    func tokenizeBatch(_ texts: [String]) async throws -> [[Int]]
    
    /// Convert tokens back to text
    func detokenize(_ tokens: [Int]) async throws -> String
    
    /// Get vocabulary size
    var vocabularySize: Int { get }
}

/// Protocol for pooling strategies
public protocol PoolingStrategyProtocol: Sendable {
    /// Pool token embeddings into a single embedding
    func pool(tokenEmbeddings: [[Float]], attentionMask: [Float]?) async throws -> [Float]
    
    /// Get the pooling strategy type
    var type: PoolingStrategy { get }
}

/// Protocol for model loading
public protocol ModelLoaderProtocol: Actor {
    /// Load a model from URL
    func loadModel(from url: URL) async throws -> MLModel
    
    /// Load a model with configuration
    func loadModel(from url: URL, configuration: MLModelConfiguration) async throws -> MLModel
    
    /// Validate model compatibility
    func validateModel(_ model: MLModel) async throws
}

/// Protocol for model management
public protocol ModelManagerProtocol: Actor {
    /// Load a model by identifier
    func loadModel(_ identifier: ModelIdentifier) async throws -> MLModel
    
    /// Download a model if needed
    func downloadModel(_ identifier: ModelIdentifier) async throws -> URL
    
    /// Get active model version
    func getActiveModelVersion(for identifier: ModelIdentifier) async throws -> ModelVersion?
    
    /// Check if model is available locally
    func isModelAvailable(_ identifier: ModelIdentifier) async -> Bool
}

/// Protocol for caching
public protocol CacheProtocol: Actor {
    associatedtype Key: Hashable & Sendable
    associatedtype Value: Sendable
    
    /// Get value from cache
    func get(_ key: Key) async -> Value?
    
    /// Set value in cache
    func set(_ key: Key, value: Value) async
    
    /// Remove value from cache
    func remove(_ key: Key) async
    
    /// Clear all cache entries
    func clear() async
    
    /// Get cache statistics
    func getStatistics() async -> CacheStatistics
}

/// Protocol for persistent storage
public protocol PersistentStorageProtocol: Actor {
    /// Save data to persistent storage
    func save<T: Codable>(_ data: T, forKey key: String) async throws
    
    /// Load data from persistent storage
    func load<T: Codable>(_ type: T.Type, forKey key: String) async throws -> T?
    
    /// Delete data from persistent storage
    func delete(forKey key: String) async throws
    
    /// Check if key exists
    func exists(forKey key: String) async -> Bool
}

/// Metric unit for telemetry
public enum MetricUnit: String, Sendable, CaseIterable {
    case count
    case bytes
    case milliseconds
    case seconds
    case percentage
    case custom
}

/// Protocol for telemetry
public protocol TelemetryProtocol: Actor {
    /// Record an event
    func recordEvent(_ name: String, properties: [String: Any]) async
    
    /// Record a metric
    func recordMetric(_ name: String, value: Double, unit: MetricUnit) async
    
    /// Track error
    func trackError(_ error: Error, context: [String: Any]) async
    
    /// Flush telemetry data
    func flush() async
}


/// Protocol for model registry
public protocol ModelRegistryProtocol: Sendable {
    /// Get model information
    func getModelInfo(for identifier: ModelIdentifier) async throws -> ModelInfo
    
    /// Get download URL for model
    func getDownloadURL(for identifier: ModelIdentifier) async throws -> URL
    
    /// List available models
    func listAvailableModels() async throws -> [ModelInfo]
}

/// Protocol for download delegates
public protocol DownloadDelegateProtocol: Actor {
    /// Download started
    func downloadDidStart(url: URL) async
    
    /// Download progress
    func downloadDidProgress(bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpected: Int64) async
    
    /// Download completed
    func downloadDidComplete(url: URL, localURL: URL) async
    
    /// Download failed
    func downloadDidFail(url: URL, error: Error) async
}

// MARK: - Factory Protocols

/// Protocol for creating tokenizers
public protocol TokenizerFactoryProtocol: Sendable {
    /// Create tokenizer for model
    func createTokenizer(for modelIdentifier: ModelIdentifier) async throws -> any TokenizerProtocol
}

/// Protocol for creating pooling strategies
public protocol PoolingStrategyFactoryProtocol: Sendable {
    /// Create pooling strategy
    func createPoolingStrategy(_ type: PoolingStrategy) -> any PoolingStrategyProtocol
}

/// Protocol for creating embedders
public protocol TextEmbedderFactoryProtocol: Actor {
    /// Create text embedder
    func createEmbedder(
        model: MLModel,
        tokenizer: any TokenizerProtocol,
        poolingStrategy: any PoolingStrategyProtocol,
        configuration: EmbedKitConfig
    ) async throws -> any TextEmbedderProtocol
}

// MARK: - Pipeline Protocols

/// Protocol for embedding pipeline
public protocol EmbeddingPipelineProtocol: Actor {
    /// Process texts through the pipeline
    func process(_ texts: [String]) async throws -> [Embedding]
    
    /// Add middleware to the pipeline
    func addMiddleware(_ middleware: any PipelineMiddlewareProtocol) async
    
    /// Get pipeline metrics
    func getMetrics() async -> PipelineMetrics
}

/// Protocol for pipeline middleware
public protocol PipelineMiddlewareProtocol: Sendable {
    /// Process texts before embedding
    func processBefore(_ texts: [String]) async throws -> [String]
    
    /// Process embeddings after generation
    func processAfter(_ embeddings: [Embedding]) async throws -> [Embedding]
    
    /// Middleware priority (higher executes first)
    var priority: Int { get }
}

// MARK: - Configuration Protocols

/// Protocol for configuration providers
public protocol ConfigurationProviderProtocol: Sendable {
    /// Get configuration
    func getConfiguration() async throws -> EmbedKitConfig
    
    /// Validate configuration
    func validateConfiguration(_ config: EmbedKitConfig) async throws
}

/// Protocol for environment-based configuration
public protocol EnvironmentConfigurationProtocol: ConfigurationProviderProtocol {
    /// Environment variables prefix
    var prefix: String { get }
    
    /// Required environment variables
    var requiredVariables: [String] { get }
}

// MARK: - Supporting Types

/// Embedding representation
public struct Embedding: Sendable, Codable {
    public let vector: [Float]
    public let metadata: [String: String]
    
    public init(vector: [Float], metadata: [String: String] = [:]) {
        self.vector = vector
        self.metadata = metadata
    }
}

/// Model information
public struct ModelInfo: Sendable, Codable {
    public let identifier: ModelIdentifier
    public let name: String
    public let description: String
    public let version: String
    public let size: Int64
    public let downloadURL: URL
    public let checksum: String?
    public let requirements: ModelRequirements
}

/// Model requirements
public struct ModelRequirements: Sendable, Codable {
    public let minimumOSVersion: String
    public let hardwareRequirements: [String]
    public let memoryRequirement: Int64
}

/// Cache statistics
public struct CacheStatistics: Sendable {
    public let hitCount: Int
    public let missCount: Int
    public let evictionCount: Int
    public let currentSize: Int
    public let maxSize: Int
    
    public var hitRate: Double {
        let total = hitCount + missCount
        return total > 0 ? Double(hitCount) / Double(total) : 0
    }
}

/// Pipeline metrics
public struct PipelineMetrics: Sendable {
    public let totalProcessed: Int
    public let averageLatency: TimeInterval
    public let throughput: Double
    public let errorRate: Double
}

// MARK: - Error Protocol

/// Protocol for EmbedKit errors
public protocol EmbedKitError: Error, Sendable {
    /// Error code
    var code: String { get }
    
    /// Error context
    var context: [String: Any] { get }
    
    /// Suggested recovery action
    var recoveryAction: String? { get }
}
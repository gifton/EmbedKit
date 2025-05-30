import Foundation
// import PipelineKit  // Temporarily disabled due to build issues

// Temporary Command protocol for standalone development
public protocol Command: Sendable {
    associatedtype Result: Sendable
}

/// Command for generating text embeddings
public struct EmbedTextCommand: Command {
    public typealias Result = EmbeddingVector
    
    /// The text to embed
    public let text: String
    
    /// Optional configuration overrides
    public let configuration: EmbeddingConfiguration?
    
    /// Optional model identifier to use (nil uses default)
    public let modelIdentifier: String?
    
    /// Metadata to pass through the pipeline
    public let metadata: [String: String]
    
    public init(
        text: String,
        configuration: EmbeddingConfiguration? = nil,
        modelIdentifier: String? = nil,
        metadata: [String: String] = [:]
    ) {
        self.text = text
        self.configuration = configuration
        self.modelIdentifier = modelIdentifier
        self.metadata = metadata
    }
}

/// Command for batch text embedding
public struct EmbedBatchCommand: Command {
    public typealias Result = [EmbeddingVector]
    
    /// The texts to embed
    public let texts: [String]
    
    /// Optional configuration overrides
    public let configuration: EmbeddingConfiguration?
    
    /// Optional model identifier to use (nil uses default)
    public let modelIdentifier: String?
    
    /// Metadata to pass through the pipeline
    public let metadata: [String: String]
    
    public init(
        texts: [String],
        configuration: EmbeddingConfiguration? = nil,
        modelIdentifier: String? = nil,
        metadata: [String: String] = [:]
    ) {
        self.texts = texts
        self.configuration = configuration
        self.modelIdentifier = modelIdentifier
        self.metadata = metadata
    }
}

/// Command for streaming text embedding
public struct EmbedStreamCommand<S: AsyncSequence & Sendable>: Command where S.Element == String {
    public typealias Result = AsyncStream<Swift.Result<EmbeddingVector, Error>>
    
    /// The async sequence of texts to embed
    public let texts: S
    
    /// Optional configuration overrides
    public let configuration: EmbeddingConfiguration?
    
    /// Optional model identifier to use (nil uses default)
    public let modelIdentifier: String?
    
    /// Maximum concurrent embeddings
    public let maxConcurrency: Int
    
    /// Metadata to pass through the pipeline
    public let metadata: [String: String]
    
    public init(
        texts: S,
        configuration: EmbeddingConfiguration? = nil,
        modelIdentifier: String? = nil,
        maxConcurrency: Int = 10,
        metadata: [String: String] = [:]
    ) {
        self.texts = texts
        self.configuration = configuration
        self.modelIdentifier = modelIdentifier
        self.maxConcurrency = maxConcurrency
        self.metadata = metadata
    }
}

/// Command to load an embedding model
public struct LoadEmbeddingModelCommand: Command {
    public typealias Result = ModelMetadata
    
    /// URL to the model file
    public let modelURL: URL
    
    /// Model identifier for future reference
    public let identifier: String
    
    /// Backend configuration
    public let backendConfiguration: ModelBackendConfiguration?
    
    public init(
        modelURL: URL,
        identifier: String,
        backendConfiguration: ModelBackendConfiguration? = nil
    ) {
        self.modelURL = modelURL
        self.identifier = identifier
        self.backendConfiguration = backendConfiguration
    }
}

/// Command to unload an embedding model
public struct UnloadEmbeddingModelCommand: Command {
    public typealias Result = Void
    
    /// Model identifier to unload
    public let identifier: String
    
    public init(identifier: String) {
        self.identifier = identifier
    }
}

/// Command to warm up the embedding model
public struct WarmupEmbeddingModelCommand: Command {
    public typealias Result = TimeInterval
    
    /// Optional model identifier (nil uses default)
    public let modelIdentifier: String?
    
    /// Number of warmup iterations
    public let iterations: Int
    
    public init(
        modelIdentifier: String? = nil,
        iterations: Int = 3
    ) {
        self.modelIdentifier = modelIdentifier
        self.iterations = iterations
    }
}

/// Command to get embedding model information
public struct GetEmbeddingModelInfoCommand: Command {
    public typealias Result = EmbeddingModelInfo
    
    /// Optional model identifier (nil uses default)
    public let modelIdentifier: String?
    
    public init(modelIdentifier: String? = nil) {
        self.modelIdentifier = modelIdentifier
    }
}

/// Information about an embedding model
public struct EmbeddingModelInfo: Sendable {
    public let identifier: String
    public let dimensions: Int
    public let metadata: ModelMetadata?
    public let isReady: Bool
    public let cacheStatistics: CacheStatistics?
    
    public init(
        identifier: String,
        dimensions: Int,
        metadata: ModelMetadata?,
        isReady: Bool,
        cacheStatistics: CacheStatistics? = nil
    ) {
        self.identifier = identifier
        self.dimensions = dimensions
        self.metadata = metadata
        self.isReady = isReady
        self.cacheStatistics = cacheStatistics
    }
}

/// Cache statistics for monitoring
public struct CacheStatistics: Sendable {
    public let hits: Int
    public let misses: Int
    public let evictions: Int
    public let currentSize: Int
    public let maxSize: Int
    
    public var hitRate: Double {
        let total = hits + misses
        return total > 0 ? Double(hits) / Double(total) : 0
    }
    
    public init(
        hits: Int,
        misses: Int,
        evictions: Int,
        currentSize: Int,
        maxSize: Int
    ) {
        self.hits = hits
        self.misses = misses
        self.evictions = evictions
        self.currentSize = currentSize
        self.maxSize = maxSize
    }
}
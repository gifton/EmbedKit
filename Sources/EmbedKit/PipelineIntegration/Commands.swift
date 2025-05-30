import Foundation
import PipelineKit

// MARK: - Embedding Commands

/// Command to embed a single text
public struct EmbedTextCommand: Command, ValidatableCommand {
    public typealias Result = EmbeddingResult
    
    public let text: String
    public let modelIdentifier: String?
    public let useCache: Bool
    public let normalize: Bool
    
    public init(
        text: String,
        modelIdentifier: String? = nil,
        useCache: Bool = true,
        normalize: Bool = true
    ) {
        self.text = text
        self.modelIdentifier = modelIdentifier
        self.useCache = useCache
        self.normalize = normalize
    }
    
    public func validate() throws {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ValidationError.custom("Text cannot be empty")
        }
        
        guard text.count <= 10_000 else {
            throw ValidationError.custom("Text exceeds maximum length of 10,000 characters")
        }
    }
}

/// Command to embed multiple texts in batch
public struct BatchEmbedCommand: Command, ValidatableCommand {
    public typealias Result = BatchEmbeddingResult
    
    public let texts: [String]
    public let modelIdentifier: String?
    public let useCache: Bool
    public let normalize: Bool
    public let batchSize: Int
    
    public init(
        texts: [String],
        modelIdentifier: String? = nil,
        useCache: Bool = true,
        normalize: Bool = true,
        batchSize: Int = 32
    ) {
        self.texts = texts
        self.modelIdentifier = modelIdentifier
        self.useCache = useCache
        self.normalize = normalize
        self.batchSize = batchSize
    }
    
    public func validate() throws {
        guard !texts.isEmpty else {
            throw ValidationError.custom("Texts array cannot be empty")
        }
        
        guard texts.count <= 1000 else {
            throw ValidationError.custom("Batch size exceeds maximum of 1000 texts")
        }
        
        for (index, text) in texts.enumerated() {
            guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw ValidationError.custom("Text at index \(index) is empty")
            }
            
            guard text.count <= 10_000 else {
                throw ValidationError.custom("Text at index \(index) exceeds maximum length")
            }
        }
        
        guard batchSize > 0 && batchSize <= 100 else {
            throw ValidationError.custom("Batch size must be between 1 and 100")
        }
    }
}

/// Command to stream embeddings for a large collection of texts
public struct StreamEmbedCommand: Command {
    public typealias Result = AsyncThrowingStream<StreamingEmbeddingResult, Error>
    
    public let textSource: AsyncTextSource
    public let modelIdentifier: String?
    public let maxConcurrency: Int
    public let bufferSize: Int
    
    public init(
        textSource: AsyncTextSource,
        modelIdentifier: String? = nil,
        maxConcurrency: Int = 10,
        bufferSize: Int = 1000
    ) {
        self.textSource = textSource
        self.modelIdentifier = modelIdentifier
        self.maxConcurrency = maxConcurrency
        self.bufferSize = bufferSize
    }
}

// MARK: - Model Management Commands

/// Command to load a specific embedding model
public struct LoadModelCommand: Command, ValidatableCommand {
    public typealias Result = ModelLoadResult
    
    public let modelIdentifier: String
    public let preload: Bool
    public let useGPU: Bool
    
    public init(
        modelIdentifier: String,
        preload: Bool = true,
        useGPU: Bool = true
    ) {
        self.modelIdentifier = modelIdentifier
        self.preload = preload
        self.useGPU = useGPU
    }
    
    public func validate() throws {
        guard !modelIdentifier.isEmpty else {
            throw ValidationError.custom("Model identifier cannot be empty")
        }
    }
}

/// Command to swap the current model with a new one
public struct SwapModelCommand: Command, ValidatableCommand {
    public typealias Result = ModelSwapResult
    
    public let newModelIdentifier: String
    public let unloadCurrent: Bool
    public let warmupAfterSwap: Bool
    
    public init(
        newModelIdentifier: String,
        unloadCurrent: Bool = true,
        warmupAfterSwap: Bool = true
    ) {
        self.newModelIdentifier = newModelIdentifier
        self.unloadCurrent = unloadCurrent
        self.warmupAfterSwap = warmupAfterSwap
    }
    
    public func validate() throws {
        guard !newModelIdentifier.isEmpty else {
            throw ValidationError.custom("New model identifier cannot be empty")
        }
    }
}

/// Command to unload the current model
public struct UnloadModelCommand: Command {
    public typealias Result = ModelUnloadResult
    
    public let clearCache: Bool
    
    public init(clearCache: Bool = true) {
        self.clearCache = clearCache
    }
}

// MARK: - Cache Management Commands

/// Command to clear the embedding cache
public struct ClearCacheCommand: Command {
    public typealias Result = CacheClearResult
    
    public let modelIdentifier: String?
    
    public init(modelIdentifier: String? = nil) {
        self.modelIdentifier = modelIdentifier
    }
}

/// Command to preload embeddings into cache
public struct PreloadCacheCommand: Command, ValidatableCommand {
    public typealias Result = CachePreloadResult
    
    public let texts: [String]
    public let modelIdentifier: String?
    
    public init(texts: [String], modelIdentifier: String? = nil) {
        self.texts = texts
        self.modelIdentifier = modelIdentifier
    }
    
    public func validate() throws {
        guard !texts.isEmpty else {
            throw ValidationError.custom("Texts array cannot be empty for preloading")
        }
        
        guard texts.count <= 10_000 else {
            throw ValidationError.custom("Preload batch exceeds maximum of 10,000 texts")
        }
    }
}

// MARK: - Command Results

/// Result of a single text embedding operation
public struct EmbeddingResult: Sendable {
    public let embedding: EmbeddingVector
    public let modelIdentifier: String
    public let duration: TimeInterval
    public let fromCache: Bool
    
    public init(
        embedding: EmbeddingVector,
        modelIdentifier: String,
        duration: TimeInterval,
        fromCache: Bool
    ) {
        self.embedding = embedding
        self.modelIdentifier = modelIdentifier
        self.duration = duration
        self.fromCache = fromCache
    }
}

/// Result of a batch embedding operation
public struct BatchEmbeddingResult: Sendable {
    public let embeddings: [EmbeddingVector]
    public let modelIdentifier: String
    public let totalDuration: TimeInterval
    public let averageDuration: TimeInterval
    public let cacheHitRate: Double
    
    public init(
        embeddings: [EmbeddingVector],
        modelIdentifier: String,
        totalDuration: TimeInterval,
        averageDuration: TimeInterval,
        cacheHitRate: Double
    ) {
        self.embeddings = embeddings
        self.modelIdentifier = modelIdentifier
        self.totalDuration = totalDuration
        self.averageDuration = averageDuration
        self.cacheHitRate = cacheHitRate
    }
}

/// Result of a streaming embedding operation
public struct StreamingEmbeddingResult: Sendable {
    public let embedding: EmbeddingVector
    public let text: String
    public let index: Int
    public let modelIdentifier: String
    public let timestamp: Date
    
    public init(
        embedding: EmbeddingVector,
        text: String,
        index: Int,
        modelIdentifier: String,
        timestamp: Date
    ) {
        self.embedding = embedding
        self.text = text
        self.index = index
        self.modelIdentifier = modelIdentifier
        self.timestamp = timestamp
    }
}

/// Result of a model load operation
public struct ModelLoadResult: Sendable {
    public let modelIdentifier: String
    public let loadDuration: TimeInterval
    public let modelSize: Int64
    public let success: Bool
    public let error: String?
    
    public init(
        modelIdentifier: String,
        loadDuration: TimeInterval,
        modelSize: Int64,
        success: Bool,
        error: String? = nil
    ) {
        self.modelIdentifier = modelIdentifier
        self.loadDuration = loadDuration
        self.modelSize = modelSize
        self.success = success
        self.error = error
    }
}

/// Result of a model swap operation
public struct ModelSwapResult: Sendable {
    public let previousModel: String?
    public let newModel: String
    public let swapDuration: TimeInterval
    public let warmupDuration: TimeInterval?
    
    public init(
        previousModel: String?,
        newModel: String,
        swapDuration: TimeInterval,
        warmupDuration: TimeInterval? = nil
    ) {
        self.previousModel = previousModel
        self.newModel = newModel
        self.swapDuration = swapDuration
        self.warmupDuration = warmupDuration
    }
}

/// Result of a model unload operation
public struct ModelUnloadResult: Sendable {
    public let modelIdentifier: String?
    public let freedMemory: Int64
    public let cacheCleared: Bool
    
    public init(
        modelIdentifier: String?,
        freedMemory: Int64,
        cacheCleared: Bool
    ) {
        self.modelIdentifier = modelIdentifier
        self.freedMemory = freedMemory
        self.cacheCleared = cacheCleared
    }
}

/// Result of a cache clear operation
public struct CacheClearResult: Sendable {
    public let entriesCleared: Int
    public let memoryFreed: Int64
    
    public init(entriesCleared: Int, memoryFreed: Int64) {
        self.entriesCleared = entriesCleared
        self.memoryFreed = memoryFreed
    }
}

/// Result of a cache preload operation
public struct CachePreloadResult: Sendable {
    public let textsProcessed: Int
    public let duration: TimeInterval
    public let averageEmbeddingTime: TimeInterval
    
    public init(
        textsProcessed: Int,
        duration: TimeInterval,
        averageEmbeddingTime: TimeInterval
    ) {
        self.textsProcessed = textsProcessed
        self.duration = duration
        self.averageEmbeddingTime = averageEmbeddingTime
    }
}

// MARK: - Async Text Source Protocol

/// Protocol for async text sources used in streaming
public protocol AsyncTextSource: AsyncSequence, Sendable where Element == String {}

/// Concrete implementation of AsyncTextSource
public struct ArrayTextSource: AsyncTextSource {
    private let texts: [String]
    
    public init(_ texts: [String]) {
        self.texts = texts
    }
    
    public func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(texts: texts)
    }
    
    public struct AsyncIterator: AsyncIteratorProtocol {
        private var index = 0
        private let texts: [String]
        
        init(texts: [String]) {
            self.texts = texts
        }
        
        public mutating func next() async -> String? {
            guard index < texts.count else { return nil }
            let text = texts[index]
            index += 1
            return text
        }
    }
}
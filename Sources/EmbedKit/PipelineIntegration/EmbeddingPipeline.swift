import Foundation
import PipelineKit

// MARK: - Embedding Pipeline

/// A comprehensive pipeline for embedding operations with all middleware integrated
public actor EmbeddingPipeline {
    private let pipeline: any Pipeline
    private let commandBus: CommandBus
    private let embedder: any TextEmbedder
    private let cache: EmbeddingCache
    private let telemetry: TelemetrySystem
    private let modelManager: EmbeddingModelManager
    private let logger = EmbedKitLogger.embeddings()
    
    /// Configuration for the embedding pipeline
    public struct Configuration: Sendable {
        public let enableCache: Bool
        public let enableGPUAcceleration: Bool
        public let enableRateLimiting: Bool
        public let enableMonitoring: Bool
        public let maxTextLength: Int
        public let maxBatchSize: Int
        public let requestsPerSecond: Double
        public let telemetryConfiguration: TelemetryConfiguration
        
        public init(
            enableCache: Bool = true,
            enableGPUAcceleration: Bool = true,
            enableRateLimiting: Bool = true,
            enableMonitoring: Bool = true,
            maxTextLength: Int = 10_000,
            maxBatchSize: Int = 1000,
            requestsPerSecond: Double = 100,
            telemetryConfiguration: TelemetryConfiguration = TelemetryConfiguration()
        ) {
            self.enableCache = enableCache
            self.enableGPUAcceleration = enableGPUAcceleration
            self.enableRateLimiting = enableRateLimiting
            self.enableMonitoring = enableMonitoring
            self.maxTextLength = maxTextLength
            self.maxBatchSize = maxBatchSize
            self.requestsPerSecond = requestsPerSecond
            self.telemetryConfiguration = telemetryConfiguration
        }
    }
    
    /// Initialize the embedding pipeline with configuration
    public init(
        embedder: any TextEmbedder,
        modelManager: EmbeddingModelManager,
        configuration: Configuration = Configuration()
    ) async throws {
        self.embedder = embedder
        self.modelManager = modelManager
        self.cache = EmbeddingCache(maxEntries: 10_000, maxMemoryMB: 100)
        let telemetryConfig = configuration.telemetryConfiguration
        self.telemetry = TelemetrySystem(configuration: telemetryConfig)
        
        // Create handlers
        let embedTextHandler = EmbedTextHandler(
            embedder: embedder,
            cache: cache,
            telemetry: telemetry
        )
        
        let batchEmbedHandler = BatchEmbedHandler(
            embedder: embedder,
            cache: cache,
            telemetry: telemetry
        )
        
        let streamEmbedHandler = StreamEmbedHandler(
            embedder: embedder,
            cache: cache,
            telemetry: telemetry
        )
        
        let loadModelHandler = LoadModelHandler(
            modelManager: modelManager,
            telemetry: telemetry
        )
        
        let swapModelHandler = SwapModelHandler(
            modelManager: modelManager,
            telemetry: telemetry
        )
        
        let unloadModelHandler = UnloadModelHandler(
            modelManager: modelManager,
            cache: cache
        )
        
        let clearCacheHandler = ClearCacheHandler(cache: cache)
        let preloadCacheHandler = PreloadCacheHandler(embedder: embedder, cache: cache)
        
        // Build command bus using fluent API
        self.commandBus = try await CommandBusBuilder()
            .with(EmbedTextCommand.self, handler: embedTextHandler)
            .with(BatchEmbedCommand.self, handler: batchEmbedHandler)
            .with(StreamEmbedCommand.self, handler: streamEmbedHandler)
            .with(LoadModelCommand.self, handler: loadModelHandler)
            .with(SwapModelCommand.self, handler: swapModelHandler)
            .with(UnloadModelCommand.self, handler: unloadModelHandler)
            .with(ClearCacheCommand.self, handler: clearCacheHandler)
            .with(PreloadCacheCommand.self, handler: preloadCacheHandler)
            .build()
        
        // Create a simple command bus based pipeline
        // This uses the command bus directly rather than trying to wrap it in a ContextAwarePipeline
        self.pipeline = CommandBusPipelineWrapper(commandBus: commandBus)
        
        logger.success("Embedding pipeline initialized")
    }
    
    // MARK: - Embedding Operations
    
    /// Embed a single text
    public func embed(
        _ text: String,
        modelIdentifier: ModelIdentifier? = nil,
        useCache: Bool = true
    ) async throws -> EmbeddingResult {
        let command = EmbedTextCommand(
            text: text,
            modelIdentifier: modelIdentifier,
            useCache: useCache
        )
        
        return try await pipeline.execute(
            command,
            metadata: DefaultCommandMetadata()
        )
    }
    
    /// Embed multiple texts in batch
    public func embedBatch(
        _ texts: [String],
        modelIdentifier: ModelIdentifier? = nil,
        useCache: Bool = true,
        batchSize: Int = 32
    ) async throws -> BatchEmbeddingResult {
        let command = BatchEmbedCommand(
            texts: texts,
            modelIdentifier: modelIdentifier,
            useCache: useCache,
            batchSize: batchSize
        )
        
        return try await pipeline.execute(
            command,
            metadata: DefaultCommandMetadata()
        )
    }
    
    /// Stream embeddings for a large collection of texts
    public func streamEmbeddings(
        from source: any AsyncTextSource,
        modelIdentifier: ModelIdentifier? = nil,
        maxConcurrency: Int = 10,
        bufferSize: Int = 1000
    ) async throws -> AsyncThrowingStream<StreamingEmbeddingResult, Error> {
        let command = StreamEmbedCommand(
            textSource: source,
            modelIdentifier: modelIdentifier,
            maxConcurrency: maxConcurrency,
            bufferSize: bufferSize
        )
        
        return try await pipeline.execute(
            command,
            metadata: DefaultCommandMetadata()
        )
    }
    
    // MARK: - Model Management
    
    /// Load a specific model
    public func loadModel(
        _ identifier: ModelIdentifier,
        preload: Bool = true,
        useGPU: Bool = true
    ) async throws -> ModelLoadResult {
        let command = LoadModelCommand(
            modelIdentifier: identifier,
            preload: preload,
            useGPU: useGPU
        )
        
        return try await pipeline.execute(
            command,
            metadata: DefaultCommandMetadata()
        )
    }
    
    /// Swap the current model with a new one
    public func swapModel(
        to newIdentifier: ModelIdentifier,
        unloadCurrent: Bool = true,
        warmupAfterSwap: Bool = true
    ) async throws -> ModelSwapResult {
        let command = SwapModelCommand(
            newModelIdentifier: newIdentifier,
            unloadCurrent: unloadCurrent,
            warmupAfterSwap: warmupAfterSwap
        )
        
        return try await pipeline.execute(
            command,
            metadata: DefaultCommandMetadata()
        )
    }
    
    /// Unload the current model
    public func unloadModel(clearCache: Bool = true) async throws -> ModelUnloadResult {
        let command = UnloadModelCommand(clearCache: clearCache)
        
        return try await pipeline.execute(
            command,
            metadata: DefaultCommandMetadata()
        )
    }
    
    // MARK: - Cache Management
    
    /// Clear the embedding cache
    public func clearCache(modelIdentifier: ModelIdentifier? = nil) async throws -> CacheClearResult {
        let command = ClearCacheCommand(modelIdentifier: modelIdentifier)
        
        return try await pipeline.execute(
            command,
            metadata: DefaultCommandMetadata()
        )
    }
    
    /// Preload embeddings into cache
    public func preloadCache(
        texts: [String],
        modelIdentifier: ModelIdentifier? = nil
    ) async throws -> CachePreloadResult {
        let command = PreloadCacheCommand(
            texts: texts,
            modelIdentifier: modelIdentifier
        )
        
        return try await pipeline.execute(
            command,
            metadata: DefaultCommandMetadata()
        )
    }
    
    // MARK: - Statistics and Monitoring
    
    /// Get current pipeline statistics
    public func getStatistics() async -> PipelineStatistics {
        let cacheStats = await cache.statistics()
        let telemetryMetrics = await telemetry.exportMetrics()
        let systemMetrics = await telemetry.getSystemMetrics()
        
        return PipelineStatistics(
            cacheStatistics: cacheStats,
            systemMetrics: systemMetrics,
            telemetryData: telemetryMetrics,
            currentModel: (await embedder.modelIdentifier).rawValue,
            isReady: await embedder.isReady
        )
    }
    
    /// Get cache statistics
    public func getCacheStatistics() async -> CacheStatistics {
        return await cache.statistics()
    }
    
    /// Get telemetry data
    public func getTelemetryData() async -> Data? {
        return await telemetry.exportMetrics()
    }
    
    /// Reset telemetry data
    public func resetTelemetry() async {
        await telemetry.reset()
    }
}

// MARK: - Pipeline Statistics

/// Comprehensive statistics about the pipeline
public struct PipelineStatistics: Sendable {
    public let cacheStatistics: CacheStatistics
    public let systemMetrics: SystemMetrics
    public let telemetryData: Data?
    public let currentModel: String
    public let isReady: Bool
    
    public init(
        cacheStatistics: CacheStatistics,
        systemMetrics: SystemMetrics,
        telemetryData: Data?,
        currentModel: String,
        isReady: Bool
    ) {
        self.cacheStatistics = cacheStatistics
        self.systemMetrics = systemMetrics
        self.telemetryData = telemetryData
        self.currentModel = currentModel
        self.isReady = isReady
    }
}

// MARK: - Command Bus Pipeline Wrapper

/// A simple wrapper that makes CommandBus conform to Pipeline protocol
private struct CommandBusPipelineWrapper: Pipeline {
    private let commandBus: CommandBus
    
    init(commandBus: CommandBus) {
        self.commandBus = commandBus
    }
    
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata
    ) async throws -> T.Result {
        return try await commandBus.send(command, metadata: metadata)
    }
    
    func addMiddleware(_ middleware: any Middleware) async throws {
        // Command bus based pipeline doesn't support adding middleware after creation
        // Middleware should be added to individual handlers during command bus construction
    }
}

// MARK: - Pipeline Factory removed - using the one in OperatorExamples.swift

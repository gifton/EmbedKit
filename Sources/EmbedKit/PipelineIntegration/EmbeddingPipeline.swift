import Foundation
import PipelineKit

// MARK: - Embedding Pipeline

/// A comprehensive pipeline for embedding operations with all middleware integrated
public actor EmbeddingPipeline {
    private let pipeline: ContextAwarePipeline
    private let commandBus: CommandBus
    private let embedder: any TextEmbedder
    private let cache: EmbeddingCache
    private let telemetry: TelemetrySystem
    private let modelManager: EmbeddingModelManager
    private let logger = EmbedKitLogger.embeddings()
    
    /// Configuration for the embedding pipeline
    public struct Configuration {
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
        self.telemetry = TelemetrySystem(configuration: configuration.telemetryConfiguration)
        
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
        
        // Build command bus
        let busBuilder = CommandBusBuilder()
        await busBuilder.register(embedTextHandler, for: EmbedTextCommand.self)
        await busBuilder.register(batchEmbedHandler, for: BatchEmbedCommand.self)
        await busBuilder.register(streamEmbedHandler, for: StreamEmbedCommand.self)
        await busBuilder.register(loadModelHandler, for: LoadModelCommand.self)
        await busBuilder.register(swapModelHandler, for: SwapModelCommand.self)
        await busBuilder.register(unloadModelHandler, for: UnloadModelCommand.self)
        await busBuilder.register(clearCacheHandler, for: ClearCacheCommand.self)
        await busBuilder.register(preloadCacheHandler, for: PreloadCacheCommand.self)
        
        self.commandBus = try await busBuilder.build()
        
        // Build pipeline with middleware
        let pipelineBuilder = ContextAwarePipelineBuilder(bus: commandBus)
        
        // Add validation middleware (always enabled)
        _ = await pipelineBuilder.with(
            EmbeddingValidationMiddleware(
                maxTextLength: configuration.maxTextLength,
                maxBatchSize: configuration.maxBatchSize
            )
        )
        
        // Add telemetry middleware
        _ = await pipelineBuilder.with(TelemetryMiddleware(telemetry: telemetry))
        
        // Add cache middleware if enabled
        if configuration.enableCache {
            _ = await pipelineBuilder.with(EmbeddingCacheMiddleware(cache: cache))
        }
        
        // Add GPU acceleration if enabled
        if configuration.enableGPUAcceleration {
            if let metalMiddleware = try? MetalAccelerationMiddleware() {
                _ = await pipelineBuilder.with(metalMiddleware)
            }
        }
        
        // Add rate limiting if enabled
        if configuration.enableRateLimiting {
            _ = await pipelineBuilder.with(
                EmbeddingRateLimitMiddleware(
                    requestsPerSecond: configuration.requestsPerSecond
                )
            )
        }
        
        // Add monitoring if enabled
        if configuration.enableMonitoring {
            _ = await pipelineBuilder.with(
                EmbeddingMonitoringMiddleware(telemetry: telemetry)
            )
        }
        
        self.pipeline = try await pipelineBuilder.build()
        
        logger.success("Embedding pipeline initialized")
    }
    
    // MARK: - Embedding Operations
    
    /// Embed a single text
    public func embed(
        _ text: String,
        modelIdentifier: String? = nil,
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
        modelIdentifier: String? = nil,
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
        from source: AsyncTextSource,
        modelIdentifier: String? = nil,
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
        _ identifier: String,
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
        to newIdentifier: String,
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
    public func clearCache(modelIdentifier: String? = nil) async throws -> CacheClearResult {
        let command = ClearCacheCommand(modelIdentifier: modelIdentifier)
        
        return try await pipeline.execute(
            command,
            metadata: DefaultCommandMetadata()
        )
    }
    
    /// Preload embeddings into cache
    public func preloadCache(
        texts: [String],
        modelIdentifier: String? = nil
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
            currentModel: embedder.modelIdentifier,
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

// MARK: - Pipeline Factory

/// Factory for creating pre-configured pipelines
public struct EmbeddingPipelineFactory {
    
    /// Create a high-performance pipeline optimized for throughput
    public static func highPerformance(
        embedder: any TextEmbedder,
        modelManager: EmbeddingModelManager
    ) async throws -> EmbeddingPipeline {
        let configuration = EmbeddingPipeline.Configuration(
            enableCache: true,
            enableGPUAcceleration: true,
            enableRateLimiting: false, // No rate limiting for max performance
            enableMonitoring: false, // Minimal monitoring
            maxTextLength: 10_000,
            maxBatchSize: 1000,
            requestsPerSecond: 1000
        )
        
        return try await EmbeddingPipeline(
            embedder: embedder,
            modelManager: modelManager,
            configuration: configuration
        )
    }
    
    /// Create a balanced pipeline with all features enabled
    public static func balanced(
        embedder: any TextEmbedder,
        modelManager: EmbeddingModelManager
    ) async throws -> EmbeddingPipeline {
        return try await EmbeddingPipeline(
            embedder: embedder,
            modelManager: modelManager
        )
    }
    
    /// Create a development pipeline with extensive monitoring
    public static func development(
        embedder: any TextEmbedder,
        modelManager: EmbeddingModelManager
    ) async throws -> EmbeddingPipeline {
        let configuration = EmbeddingPipeline.Configuration(
            enableCache: true,
            enableGPUAcceleration: true,
            enableRateLimiting: true,
            enableMonitoring: true,
            maxTextLength: 10_000,
            maxBatchSize: 100,
            requestsPerSecond: 10,
            telemetryConfiguration: TelemetryConfiguration(
                logMetrics: true,
                logEvents: true
            )
        )
        
        return try await EmbeddingPipeline(
            embedder: embedder,
            modelManager: modelManager,
            configuration: configuration
        )
    }
    
    /// Create a minimal pipeline with only essential features
    public static func minimal(
        embedder: any TextEmbedder,
        modelManager: EmbeddingModelManager
    ) async throws -> EmbeddingPipeline {
        let configuration = EmbeddingPipeline.Configuration(
            enableCache: false,
            enableGPUAcceleration: false,
            enableRateLimiting: false,
            enableMonitoring: false
        )
        
        return try await EmbeddingPipeline(
            embedder: embedder,
            modelManager: modelManager,
            configuration: configuration
        )
    }
}
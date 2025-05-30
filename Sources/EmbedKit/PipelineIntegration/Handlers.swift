import Foundation
import PipelineKit

// MARK: - Embedding Handlers

/// Handler for single text embedding commands
public actor EmbedTextHandler: CommandHandler {
    public typealias CommandType = EmbedTextCommand
    
    private let embedder: any TextEmbedder
    private let cache: EmbeddingCache
    private let telemetry: TelemetrySystem
    private let logger = EmbedKitLogger.embeddings()
    
    public init(
        embedder: any TextEmbedder,
        cache: EmbeddingCache,
        telemetry: TelemetrySystem
    ) {
        self.embedder = embedder
        self.cache = cache
        self.telemetry = telemetry
    }
    
    public func handle(_ command: EmbedTextCommand) async throws -> EmbeddingResult {
        let timer = await telemetry.startTimer("embed_text")
        defer {
            Task {
                await timer.stop(tags: ["model": embedder.modelIdentifier])
            }
        }
        
        let modelId = command.modelIdentifier ?? embedder.modelIdentifier
        
        // Check cache if enabled
        if command.useCache {
            if let cachedEmbedding = await cache.get(text: command.text, modelIdentifier: modelId) {
                logger.cache("Cache hit", hitRate: nil, size: nil)
                await telemetry.incrementCounter("embedding.cache_hits")
                
                return EmbeddingResult(
                    embedding: cachedEmbedding,
                    modelIdentifier: modelId,
                    duration: 0,
                    fromCache: true
                )
            } else {
                logger.cache("Cache miss", hitRate: nil, size: nil)
                await telemetry.incrementCounter("embedding.cache_misses")
            }
        }
        
        // Ensure embedder is ready
        if await !embedder.isReady {
            logger.model("Loading model", modelId: modelId)
            try await embedder.loadModel()
        }
        
        // Generate embedding
        let startTime = CFAbsoluteTimeGetCurrent()
        logger.processing("text", progress: 0.5)
        
        let embedding = try await embedder.embed(command.text)
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        
        logger.performance("Text embedding", duration: duration)
        
        // Store in cache if enabled
        if command.useCache {
            await cache.set(
                text: command.text,
                modelIdentifier: modelId,
                embedding: embedding
            )
        }
        
        // Record telemetry
        await telemetry.recordEmbeddingOperation(
            operation: "embed_single",
            duration: duration,
            inputLength: command.text.count,
            outputDimensions: embedding.dimensions
        )
        
        return EmbeddingResult(
            embedding: embedding,
            modelIdentifier: modelId,
            duration: duration,
            fromCache: false
        )
    }
}

/// Handler for batch embedding commands
public actor BatchEmbedHandler: CommandHandler {
    public typealias CommandType = BatchEmbedCommand
    
    private let embedder: any TextEmbedder
    private let cache: EmbeddingCache
    private let telemetry: TelemetrySystem
    private let logger = EmbedKitLogger.embeddings()
    
    public init(
        embedder: any TextEmbedder,
        cache: EmbeddingCache,
        telemetry: TelemetrySystem
    ) {
        self.embedder = embedder
        self.cache = cache
        self.telemetry = telemetry
    }
    
    public func handle(_ command: BatchEmbedCommand) async throws -> BatchEmbeddingResult {
        let timer = await telemetry.startTimer("embed_batch")
        defer {
            Task {
                await timer.stop(tags: [
                    "model": embedder.modelIdentifier,
                    "batch_size": String(command.texts.count)
                ])
            }
        }
        
        let modelId = command.modelIdentifier ?? embedder.modelIdentifier
        logger.start("batch embedding", details: "\(command.texts.count) texts")
        
        // Ensure embedder is ready
        if await !embedder.isReady {
            logger.model("Loading model", modelId: modelId)
            try await embedder.loadModel()
        }
        
        var embeddings: [EmbeddingVector] = []
        var cacheHits = 0
        var totalDuration: TimeInterval = 0
        
        // Process in batches
        for batchStart in stride(from: 0, to: command.texts.count, by: command.batchSize) {
            let batchEnd = min(batchStart + command.batchSize, command.texts.count)
            let batchTexts = Array(command.texts[batchStart..<batchEnd])
            
            var batchEmbeddings: [EmbeddingVector] = []
            var textsToEmbed: [(String, Int)] = []
            
            // Check cache for each text
            for (index, text) in batchTexts.enumerated() {
                if command.useCache,
                   let cachedEmbedding = await cache.get(text: text, modelIdentifier: modelId) {
                    batchEmbeddings.append(cachedEmbedding)
                    cacheHits += 1
                } else {
                    textsToEmbed.append((text, index))
                    batchEmbeddings.append(EmbeddingVector([])) // Placeholder
                }
            }
            
            // Generate embeddings for cache misses
            if !textsToEmbed.isEmpty {
                let startTime = CFAbsoluteTimeGetCurrent()
                let newEmbeddings = try await embedder.embed(batch: textsToEmbed.map { $0.0 })
                let duration = CFAbsoluteTimeGetCurrent() - startTime
                totalDuration += duration
                
                // Insert new embeddings and update cache
                for (i, (text, originalIndex)) in textsToEmbed.enumerated() {
                    batchEmbeddings[originalIndex] = newEmbeddings[i]
                    
                    if command.useCache {
                        await cache.set(
                            text: text,
                            modelIdentifier: modelId,
                            embedding: newEmbeddings[i]
                        )
                    }
                }
            }
            
            embeddings.append(contentsOf: batchEmbeddings)
            
            let progress = Double(batchEnd) / Double(command.texts.count)
            logger.processing("batch", progress: progress)
        }
        
        let cacheHitRate = Double(cacheHits) / Double(command.texts.count)
        logger.cache("Batch processing complete", hitRate: cacheHitRate, size: nil)
        
        // Record telemetry
        await telemetry.recordEmbeddingOperation(
            operation: "embed_batch",
            duration: totalDuration,
            inputLength: command.texts.map { $0.count }.reduce(0, +) / command.texts.count,
            outputDimensions: embeddings.first?.dimensions ?? 0,
            batchSize: command.texts.count
        )
        
        logger.complete("batch embedding", result: "\(embeddings.count) embeddings generated")
        
        return BatchEmbeddingResult(
            embeddings: embeddings,
            modelIdentifier: modelId,
            totalDuration: totalDuration,
            averageDuration: totalDuration / Double(command.texts.count - cacheHits),
            cacheHitRate: cacheHitRate
        )
    }
}

/// Handler for streaming embedding commands
public actor StreamEmbedHandler: CommandHandler {
    public typealias CommandType = StreamEmbedCommand
    
    private let embedder: any TextEmbedder
    private let cache: EmbeddingCache
    private let telemetry: TelemetrySystem
    private let logger = EmbedKitLogger.streaming()
    
    public init(
        embedder: any TextEmbedder,
        cache: EmbeddingCache,
        telemetry: TelemetrySystem
    ) {
        self.embedder = embedder
        self.cache = cache
        self.telemetry = telemetry
    }
    
    public func handle(_ command: StreamEmbedCommand) async throws -> AsyncThrowingStream<StreamingEmbeddingResult, Error> {
        let modelId = command.modelIdentifier ?? embedder.modelIdentifier
        
        // Ensure embedder is ready
        if await !embedder.isReady {
            logger.model("Loading model for streaming", modelId: modelId)
            try await embedder.loadModel()
        }
        
        // Create streaming embedder
        let streamingEmbedder = StreamingEmbedder(
            embedder: embedder,
            configuration: StreamingEmbedder<any TextEmbedder>.StreamingConfiguration(
                maxConcurrency: command.maxConcurrency,
                inputBufferSize: command.bufferSize,
                outputBufferSize: command.bufferSize,
                preserveOrder: true
            )
        )
        
        logger.start("streaming embeddings", details: "concurrency: \(command.maxConcurrency)")
        
        // Transform the text source into embedding results
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    var index = 0
                    let stream = streamingEmbedder.embedTextStream(command.textSource)
                    
                    for try await result in stream {
                        let embeddingResult = StreamingEmbeddingResult(
                            embedding: result.embedding,
                            text: result.text,
                            index: index,
                            modelIdentifier: modelId,
                            timestamp: result.timestamp
                        )
                        
                        continuation.yield(embeddingResult)
                        index += 1
                        
                        // Update cache if available
                        await cache.set(
                            text: result.text,
                            modelIdentifier: modelId,
                            embedding: result.embedding
                        )
                        
                        // Record telemetry periodically
                        if index % 100 == 0 {
                            await telemetry.recordEmbeddingOperation(
                                operation: "stream_batch",
                                duration: result.duration ?? 0,
                                inputLength: result.text.count,
                                outputDimensions: result.embedding.dimensions,
                                batchSize: 100
                            )
                        }
                    }
                    
                    logger.complete("streaming", result: "\(index) embeddings processed")
                    continuation.finish()
                } catch {
                    logger.error("Streaming failed", error: error)
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}

// MARK: - Model Management Handlers

/// Handler for model loading commands
public actor LoadModelHandler: CommandHandler {
    public typealias CommandType = LoadModelCommand
    
    private let modelManager: EmbeddingModelManager
    private let telemetry: TelemetrySystem
    private let logger = EmbedKitLogger.modelManagement()
    
    public init(
        modelManager: EmbeddingModelManager,
        telemetry: TelemetrySystem
    ) {
        self.modelManager = modelManager
        self.telemetry = telemetry
    }
    
    public func handle(_ command: LoadModelCommand) async throws -> ModelLoadResult {
        logger.start("loading model", details: command.modelIdentifier)
        let startTime = CFAbsoluteTimeGetCurrent()
        
        do {
            // Load the model
            let embedder = try await modelManager.loadModel(
                identifier: command.modelIdentifier,
                configuration: EmbeddingConfiguration(useGPUAcceleration: command.useGPU)
            )
            
            // Preload/warmup if requested
            if command.preload {
                logger.thinking("warming up model")
                try await embedder.warmup()
            }
            
            let loadDuration = CFAbsoluteTimeGetCurrent() - startTime
            let modelSize = await modelManager.getModelSize(command.modelIdentifier) ?? 0
            
            // Record telemetry
            await telemetry.recordModelLoad(
                modelId: command.modelIdentifier,
                loadDuration: loadDuration,
                modelSize: Int(modelSize),
                success: true
            )
            
            logger.success("Model loaded", context: "duration: \(loadDuration)s")
            
            return ModelLoadResult(
                modelIdentifier: command.modelIdentifier,
                loadDuration: loadDuration,
                modelSize: modelSize,
                success: true
            )
        } catch {
            let loadDuration = CFAbsoluteTimeGetCurrent() - startTime
            
            await telemetry.recordModelLoad(
                modelId: command.modelIdentifier,
                loadDuration: loadDuration,
                modelSize: 0,
                success: false
            )
            
            logger.error("Model load failed", error: error)
            
            return ModelLoadResult(
                modelIdentifier: command.modelIdentifier,
                loadDuration: loadDuration,
                modelSize: 0,
                success: false,
                error: error.localizedDescription
            )
        }
    }
}

/// Handler for model swap commands
public actor SwapModelHandler: CommandHandler {
    public typealias CommandType = SwapModelCommand
    
    private let modelManager: EmbeddingModelManager
    private let telemetry: TelemetrySystem
    private let logger = EmbedKitLogger.modelManagement()
    
    public init(
        modelManager: EmbeddingModelManager,
        telemetry: TelemetrySystem
    ) {
        self.modelManager = modelManager
        self.telemetry = telemetry
    }
    
    public func handle(_ command: SwapModelCommand) async throws -> ModelSwapResult {
        logger.start("swapping model", details: "to \(command.newModelIdentifier)")
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Get current model ID
        let previousModel = await modelManager.currentModelIdentifier
        
        // Load new model
        let newEmbedder = try await modelManager.loadModel(
            identifier: command.newModelIdentifier,
            configuration: EmbeddingConfiguration()
        )
        
        // Unload previous model if requested
        if command.unloadCurrent, let previousModel = previousModel {
            logger.info("Unloading previous model", context: previousModel)
            try await modelManager.unloadModel(previousModel)
        }
        
        let swapDuration = CFAbsoluteTimeGetCurrent() - startTime
        var warmupDuration: TimeInterval? = nil
        
        // Warmup new model if requested
        if command.warmupAfterSwap {
            let warmupStart = CFAbsoluteTimeGetCurrent()
            try await newEmbedder.warmup()
            warmupDuration = CFAbsoluteTimeGetCurrent() - warmupStart
            logger.performance("Model warmup", duration: warmupDuration!)
        }
        
        // Set as current model
        await modelManager.setCurrentModel(command.newModelIdentifier)
        
        logger.complete("model swap", result: "from \(previousModel ?? "none") to \(command.newModelIdentifier)")
        
        return ModelSwapResult(
            previousModel: previousModel,
            newModel: command.newModelIdentifier,
            swapDuration: swapDuration,
            warmupDuration: warmupDuration
        )
    }
}

/// Handler for model unload commands
public actor UnloadModelHandler: CommandHandler {
    public typealias CommandType = UnloadModelCommand
    
    private let modelManager: EmbeddingModelManager
    private let cache: EmbeddingCache
    private let logger = EmbedKitLogger.modelManagement()
    
    public init(
        modelManager: EmbeddingModelManager,
        cache: EmbeddingCache
    ) {
        self.modelManager = modelManager
        self.cache = cache
    }
    
    public func handle(_ command: UnloadModelCommand) async throws -> ModelUnloadResult {
        let modelId = await modelManager.currentModelIdentifier
        logger.start("unloading model", details: modelId)
        
        let initialMemory = ProcessInfo.processInfo.physicalMemory
        
        // Unload current model
        if let modelId = modelId {
            try await modelManager.unloadModel(modelId)
        }
        
        // Clear cache if requested
        if command.clearCache {
            await cache.clear()
        }
        
        let freedMemory = Int64(ProcessInfo.processInfo.physicalMemory) - Int64(initialMemory)
        
        logger.complete("model unload", result: "freed \(freedMemory / 1024 / 1024)MB")
        
        return ModelUnloadResult(
            modelIdentifier: modelId,
            freedMemory: freedMemory,
            cacheCleared: command.clearCache
        )
    }
}

// MARK: - Cache Management Handlers

/// Handler for cache clear commands
public actor ClearCacheHandler: CommandHandler {
    public typealias CommandType = ClearCacheCommand
    
    private let cache: EmbeddingCache
    private let logger = EmbedKitLogger.cache()
    
    public init(cache: EmbeddingCache) {
        self.cache = cache
    }
    
    public func handle(_ command: ClearCacheCommand) async throws -> CacheClearResult {
        logger.start("clearing cache")
        
        let stats = await cache.statistics()
        let initialSize = stats.currentSize
        
        await cache.clear()
        
        logger.complete("cache cleared", result: "\(initialSize) entries removed")
        
        return CacheClearResult(
            entriesCleared: initialSize,
            memoryFreed: Int64(initialSize * 1024) // Approximate
        )
    }
}

/// Handler for cache preload commands
public actor PreloadCacheHandler: CommandHandler {
    public typealias CommandType = PreloadCacheCommand
    
    private let embedder: any TextEmbedder
    private let cache: EmbeddingCache
    private let logger = EmbedKitLogger.cache()
    
    public init(
        embedder: any TextEmbedder,
        cache: EmbeddingCache
    ) {
        self.embedder = embedder
        self.cache = cache
    }
    
    public func handle(_ command: PreloadCacheCommand) async throws -> CachePreloadResult {
        let modelId = command.modelIdentifier ?? embedder.modelIdentifier
        logger.start("preloading cache", details: "\(command.texts.count) texts")
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Ensure embedder is ready
        if await !embedder.isReady {
            try await embedder.loadModel()
        }
        
        // Generate embeddings for all texts
        let embeddings = try await embedder.embed(batch: command.texts)
        
        // Store in cache
        await cache.preload(
            texts: command.texts,
            modelIdentifier: modelId,
            embeddings: embeddings
        )
        
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        let averageTime = duration / Double(command.texts.count)
        
        logger.complete("cache preload", result: "\(command.texts.count) embeddings cached")
        
        return CachePreloadResult(
            textsProcessed: command.texts.count,
            duration: duration,
            averageEmbeddingTime: averageTime
        )
    }
}
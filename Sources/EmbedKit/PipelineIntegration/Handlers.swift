import Foundation
import PipelineKit
#if canImport(Darwin)
import Darwin
#endif

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
                await timer.stop(tags: ["model": embedder.modelIdentifier.rawValue])
            }
        }
        
        let modelId: ModelIdentifier
        if let commandModelId = command.modelIdentifier {
            modelId = commandModelId
        } else {
            modelId = await embedder.modelIdentifier
        }
        
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
            logger.model("Loading model", modelId: modelId.rawValue)
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
                    "model": embedder.modelIdentifier.rawValue,
                    "batch_size": String(command.texts.count)
                ])
            }
        }
        
        let modelId: ModelIdentifier
        if let commandModelId = command.modelIdentifier {
            modelId = commandModelId
        } else {
            modelId = await embedder.modelIdentifier
        }
        logger.start("batch embedding", details: "\(command.texts.count) texts")
        
        // Ensure embedder is ready
        if await !embedder.isReady {
            logger.model("Loading model", modelId: modelId.rawValue)
            try await embedder.loadModel()
        }
        
        var embeddings: [EmbeddingVector] = []
        var cacheHits = 0
        var totalDuration: TimeInterval = 0
        
        // Process in batches
        for batchStart in stride(from: 0, to: command.texts.count, by: command.batchSize) {
            let batchEnd = min(batchStart + command.batchSize, command.texts.count)
            let batchTexts = Array(command.texts[batchStart..<batchEnd])
            
            var batchEmbeddings: [EmbeddingVector?] = Array(repeating: nil, count: batchTexts.count)
            var textsToEmbed: [(String, Int)] = []
            
            // Check cache for each text
            for (index, text) in batchTexts.enumerated() {
                if command.useCache,
                   let cachedEmbedding = await cache.get(text: text, modelIdentifier: modelId) {
                    batchEmbeddings[index] = cachedEmbedding
                    cacheHits += 1
                } else {
                    textsToEmbed.append((text, index))
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
            
            // Convert optionals to non-optionals (all should be filled at this point)
            let filledBatchEmbeddings = batchEmbeddings.compactMap { $0 }
            
            embeddings.append(contentsOf: filledBatchEmbeddings)
            
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
        let modelId: ModelIdentifier
        if let commandModelId = command.modelIdentifier {
            modelId = commandModelId
        } else {
            modelId = await embedder.modelIdentifier
        }
        
        // Ensure embedder is ready
        if await !embedder.isReady {
            logger.model("Loading model for streaming", modelId: modelId.rawValue)
            try await embedder.loadModel()
        }
        
        logger.start("streaming embeddings", details: "concurrency: \(command.maxConcurrency)")
        
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    var index = 0
                    
                    // Special handling for ArrayTextSource to avoid type erasure issues
                    if let arraySource = command.textSource as? ArrayTextSource {
                        for try await text in arraySource {
                            let startTime = Date()
                            let embedding = try await embedder.embed(text)
                            let duration = Date().timeIntervalSince(startTime)
                            
                            let embeddingResult = StreamingEmbeddingResult(
                                embedding: embedding,
                                text: text,
                                index: index,
                                modelIdentifier: modelId,
                                timestamp: startTime
                            )
                            
                            continuation.yield(embeddingResult)
                            index += 1
                            
                            // Update cache if available
                            await cache.set(
                                text: text,
                                modelIdentifier: modelId,
                                embedding: embedding
                            )
                            
                            // Record telemetry
                            if index % 10 == 0 {
                                await telemetry.recordEmbeddingOperation(
                                    operation: "stream_batch",
                                    duration: duration,
                                    inputLength: text.count,
                                    outputDimensions: embedding.dimensions,
                                    batchSize: 10
                                )
                            }
                        }
                    } else {
                        // Fallback for other AsyncTextSource implementations
                        continuation.finish(throwing: ContextualEmbeddingError.configurationError(
                            context: ErrorContext(operation: .streaming),
                            issue: .incompatible
                        ))
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
        logger.start("loading model", details: command.modelIdentifier.rawValue)
        let startTime = CFAbsoluteTimeGetCurrent()
        
        do {
            // Load the model
            let url = Bundle.main.url(forResource: command.modelIdentifier.rawValue, withExtension: "mlmodelc") ?? URL(fileURLWithPath: command.modelIdentifier.rawValue)
            let _ = try await modelManager.loadModel(
                from: url,
                identifier: command.modelIdentifier,
                configuration: ModelBackendConfiguration(computeUnits: command.useGPU ? .all : .cpuOnly)
            )
            
            // Get the embedder
            guard let embedder = await modelManager.getModel(identifier: command.modelIdentifier) else {
                throw ContextualEmbeddingError.modelNotLoaded(
                    context: ErrorContext(
                        operation: .modelLoading,
                        modelIdentifier: command.modelIdentifier,
                        sourceLocation: SourceLocation()
                    )
                )
            }
            
            // Preload/warmup if requested
            if command.preload {
                logger.thinking("warming up model")
                try await embedder.warmup()
            }
            
            let loadDuration = CFAbsoluteTimeGetCurrent() - startTime
            // ModelMetadata doesn't include file size, using a default value
            let modelSize: Int64 = 0 // Size would need to be tracked separately
            
            // Record telemetry
            await telemetry.recordModelLoad(
                modelId: command.modelIdentifier.rawValue,
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
                modelId: command.modelIdentifier.rawValue,
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
        
        // Get current model ID before loading the new one
        let previousModel = await modelManager.getCurrentModel()
        
        // Check if the new model is already the current model
        if let currentModel = previousModel, currentModel == command.newModelIdentifier {
            logger.info("Model already loaded", context: command.newModelIdentifier.rawValue)
            return ModelSwapResult(
                previousModel: currentModel,
                newModel: command.newModelIdentifier,
                swapDuration: 0,
                warmupDuration: nil
            )
        }
        
        // Load new model
        let url = Bundle.main.url(forResource: command.newModelIdentifier.rawValue, withExtension: "mlmodelc") ?? URL(fileURLWithPath: command.newModelIdentifier.rawValue)
        let _ = try await modelManager.loadModel(
            from: url,
            identifier: command.newModelIdentifier,
            configuration: nil
        )
        
        // Get the embedder
        guard let newEmbedder = await modelManager.getModel(identifier: command.newModelIdentifier) else {
            throw ContextualEmbeddingError.modelNotLoaded(
                context: ErrorContext(
                    operation: .modelLoading,
                    modelIdentifier: command.newModelIdentifier,
                    sourceLocation: SourceLocation()
                )
            )
        }
        
        // Unload previous model if requested and it exists
        if command.unloadCurrent, let previousModel = previousModel {
            logger.info("Unloading previous model", context: previousModel.rawValue)
            try await modelManager.unloadModel(identifier: previousModel)
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
        
        // Log completion with proper model tracking
        logger.complete("model swap", result: "from \(previousModel?.rawValue ?? "none") to \(command.newModelIdentifier.rawValue)")
        
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
        // Determine which model to unload
        let modelToUnload: ModelIdentifier?
        
        if let specificModel = command.modelIdentifier {
            // Unload specific model
            modelToUnload = specificModel
            logger.start("unloading model", details: specificModel.rawValue)
        } else {
            // Unload current/default model
            modelToUnload = await modelManager.getCurrentModel()
            logger.start("unloading model", details: modelToUnload?.rawValue ?? "current model")
        }
        
        // Track memory before unloading
        var info = mach_task_basic_info()
        let initialMemory = getMemoryUsage(&info)
        
        var unloadedModel: ModelIdentifier? = nil
        
        // Unload the model if we identified one
        if let modelId = modelToUnload {
            // Check if the model is actually loaded
            let loadedModels = await modelManager.loadedModels()
            if loadedModels.contains(modelId) {
                try await modelManager.unloadModel(identifier: modelId)
                unloadedModel = modelId
                logger.info("Model unloaded", context: modelId.rawValue)
            } else {
                logger.warning("Model not loaded", context: modelId.rawValue)
            }
        } else {
            logger.warning("No model to unload")
        }
        
        // Clear cache if requested
        if command.clearCache {
            await cache.clear()
            logger.info("Cache cleared")
        }
        
        // Measure memory after unloading
        // Note: This is a best-effort approach as Swift doesn't guarantee immediate memory release
        var infoAfter = mach_task_basic_info()
        let finalMemory = getMemoryUsage(&infoAfter)
        let freedMemory = max(0, initialMemory - finalMemory)
        
        logger.complete("model unload", result: "freed ~\(freedMemory / 1024 / 1024)MB")
        
        return ModelUnloadResult(
            modelIdentifier: unloadedModel,
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
        let modelId: ModelIdentifier
        if let commandModelId = command.modelIdentifier {
            modelId = commandModelId
        } else {
            modelId = await embedder.modelIdentifier
        }
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

// MARK: - Memory Measurement Helpers

#if canImport(Darwin)
/// Get current memory usage of the process in bytes
private func getMemoryUsage(_ info: inout mach_task_basic_info) -> Int64 {
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
            task_info(
                mach_task_self_,
                task_flavor_t(MACH_TASK_BASIC_INFO),
                $0,
                &count
            )
        }
    }
    
    return result == KERN_SUCCESS ? Int64(info.resident_size) : 0
}
#else
/// Fallback memory measurement for non-Darwin platforms
private func getMemoryUsage(_ info: inout mach_task_basic_info) -> Int64 {
    // On non-Darwin platforms, we can't get accurate memory usage
    // Return 0 to indicate unknown
    return 0
}
#endif
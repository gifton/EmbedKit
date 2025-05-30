import Foundation
import PipelineKit

// MARK: - Embedding Cache Middleware

/// Middleware that integrates with EmbedKit's caching system
public struct EmbeddingCacheMiddleware: ContextAwareMiddleware {
    private let cache: EmbeddingCache
    private let logger = EmbedKitLogger.cache()
    
    public init(cache: EmbeddingCache) {
        self.cache = cache
    }
    
    public func execute<T: Command>(
        _ command: T,
        context: CommandContext,
        next: @Sendable (T, CommandContext) async throws -> T.Result
    ) async throws -> T.Result {
        // Only handle embedding commands
        guard command is EmbedTextCommand || command is BatchEmbedCommand else {
            return try await next(command, context)
        }
        
        // Store cache instance in context for handlers
        await context.set(cache, for: EmbeddingCacheKey.self)
        
        // Track cache performance
        let initialStats = await cache.statistics()
        
        let result = try await next(command, context)
        
        // Log cache performance
        let finalStats = await cache.statistics()
        let hitRate = finalStats.hits > 0 ? Double(finalStats.hits) / Double(finalStats.hits + finalStats.misses) : 0
        
        logger.cache(
            "Cache stats",
            hitRate: hitRate,
            size: finalStats.currentSize
        )
        
        return result
    }
}

// MARK: - Metal Acceleration Middleware

/// Middleware for GPU optimization using Metal
public struct MetalAccelerationMiddleware: ContextAwareMiddleware {
    private let metalAccelerator: MetalAccelerator
    private let logger = EmbedKitLogger.metal()
    
    public init() throws {
        self.metalAccelerator = try MetalAccelerator()
    }
    
    public func execute<T: Command>(
        _ command: T,
        context: CommandContext,
        next: @Sendable (T, CommandContext) async throws -> T.Result
    ) async throws -> T.Result {
        // Check if Metal acceleration should be enabled
        let shouldUseGPU = await shouldUseMetalAcceleration(for: command)
        
        if shouldUseGPU {
            logger.info("Metal acceleration enabled for command")
            await context.set(metalAccelerator, for: MetalAcceleratorKey.self)
            await context.set(true, for: UseGPUKey.self)
        }
        
        // Monitor GPU memory if enabled
        if shouldUseGPU {
            let initialMemory = metalAccelerator.getCurrentMemoryUsage()
            logger.memory("Initial GPU", bytes: initialMemory)
            
            let result = try await next(command, context)
            
            let finalMemory = metalAccelerator.getCurrentMemoryUsage()
            let memoryDelta = finalMemory - initialMemory
            logger.memory("GPU delta", bytes: memoryDelta)
            
            return result
        } else {
            return try await next(command, context)
        }
    }
    
    private func shouldUseMetalAcceleration<T: Command>(for command: T) async -> Bool {
        // Enable for batch operations and streaming
        if command is BatchEmbedCommand || command is StreamEmbedCommand {
            return metalAccelerator.isAvailable
        }
        
        // Enable for single embeddings with GPU flag
        if let embedCommand = command as? EmbedTextCommand {
            return metalAccelerator.isAvailable && embedCommand.normalize
        }
        
        return false
    }
}

// MARK: - Embedding Validation Middleware

/// Middleware for input validation specific to embeddings
public struct EmbeddingValidationMiddleware: Middleware {
    private let maxTextLength: Int
    private let maxBatchSize: Int
    private let logger = EmbedKitLogger.embeddings()
    
    public init(
        maxTextLength: Int = 10_000,
        maxBatchSize: Int = 1000
    ) {
        self.maxTextLength = maxTextLength
        self.maxBatchSize = maxBatchSize
    }
    
    public func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        // Validate embedding-specific constraints
        if let embedCommand = command as? EmbedTextCommand {
            try validateText(embedCommand.text)
        } else if let batchCommand = command as? BatchEmbedCommand {
            try validateBatch(batchCommand.texts)
        }
        
        // Add validation metadata
        var enrichedMetadata = metadata
        if let validatable = command as? ValidatableCommand {
            do {
                try validatable.validate()
                enrichedMetadata = DefaultCommandMetadata(
                    commandId: metadata.commandId,
                    userId: metadata.userId,
                    correlationId: metadata.correlationId,
                    timestamp: metadata.timestamp,
                    tags: metadata.tags.merging(["validation": "passed"], uniquingKeysWith: { _, new in new })
                )
            } catch {
                logger.warning("Validation failed", context: error.localizedDescription)
                throw error
            }
        }
        
        return try await next(command, enrichedMetadata)
    }
    
    private func validateText(_ text: String) throws {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw EmbeddingError.invalidInput("Text cannot be empty")
        }
        
        guard text.count <= maxTextLength else {
            throw EmbeddingError.invalidInput("Text exceeds maximum length of \(maxTextLength) characters")
        }
        
        // Check for potentially problematic content
        if text.contains("\0") {
            throw EmbeddingError.invalidInput("Text contains null characters")
        }
    }
    
    private func validateBatch(_ texts: [String]) throws {
        guard !texts.isEmpty else {
            throw EmbeddingError.invalidInput("Batch cannot be empty")
        }
        
        guard texts.count <= maxBatchSize else {
            throw EmbeddingError.invalidInput("Batch size exceeds maximum of \(maxBatchSize)")
        }
        
        for (index, text) in texts.enumerated() {
            do {
                try validateText(text)
            } catch {
                throw EmbeddingError.invalidInput("Invalid text at index \(index): \(error.localizedDescription)")
            }
        }
    }
}

// MARK: - Telemetry Middleware

/// Middleware that integrates with EmbedKit's telemetry system
public struct TelemetryMiddleware: ContextAwareMiddleware {
    private let telemetry: TelemetrySystem
    private let logger = EmbedKitLogger.telemetry()
    
    public init(telemetry: TelemetrySystem) {
        self.telemetry = telemetry
    }
    
    public func execute<T: Command>(
        _ command: T,
        context: CommandContext,
        next: @Sendable (T, CommandContext) async throws -> T.Result
    ) async throws -> T.Result {
        let commandName = String(describing: type(of: command))
        let timer = await telemetry.startTimer("command.\(commandName)")
        
        // Record command start
        await telemetry.incrementCounter("commands.started", tags: ["command": commandName])
        
        // Store telemetry in context for handlers
        await context.set(telemetry, for: TelemetrySystemKey.self)
        
        do {
            let result = try await next(command, context)
            
            // Record success
            await telemetry.incrementCounter("commands.succeeded", tags: ["command": commandName])
            await timer.stop(tags: ["command": commandName, "status": "success"])
            
            // Record command-specific metrics
            await recordCommandMetrics(command: command, result: result)
            
            return result
        } catch {
            // Record failure
            await telemetry.incrementCounter("commands.failed", tags: ["command": commandName])
            await timer.stop(tags: ["command": commandName, "status": "failure"])
            
            await telemetry.recordEvent(
                TelemetryEvent(
                    name: "command_failed",
                    description: "Command \(commandName) failed: \(error.localizedDescription)",
                    severity: .error,
                    metadata: ["command": commandName, "error": String(describing: error)]
                )
            )
            
            logger.error("Command failed", error: error, context: commandName)
            throw error
        }
    }
    
    private func recordCommandMetrics<T: Command>(command: T, result: T.Result) async {
        switch command {
        case let embedCommand as EmbedTextCommand:
            if let embedResult = result as? EmbeddingResult {
                await telemetry.recordGauge(
                    "embedding.dimensions",
                    value: Double(embedResult.embedding.dimensions),
                    tags: ["model": embedResult.modelIdentifier]
                )
                
                if embedResult.fromCache {
                    await telemetry.incrementCounter("embedding.cache_hits")
                }
            }
            
        case let batchCommand as BatchEmbedCommand:
            if let batchResult = result as? BatchEmbeddingResult {
                await telemetry.recordHistogram(
                    "batch.size",
                    value: Double(batchResult.embeddings.count)
                )
                
                await telemetry.recordGauge(
                    "batch.cache_hit_rate",
                    value: batchResult.cacheHitRate
                )
                
                await telemetry.recordTiming(
                    "batch.average_time",
                    duration: batchResult.averageDuration
                )
            }
            
        case let loadCommand as LoadModelCommand:
            if let loadResult = result as? ModelLoadResult {
                await telemetry.recordGauge(
                    "model.size_bytes",
                    value: Double(loadResult.modelSize),
                    tags: ["model": loadResult.modelIdentifier]
                )
            }
            
        default:
            break
        }
    }
}

// MARK: - Rate Limiting Middleware

/// Middleware for rate limiting embedding requests
public struct EmbeddingRateLimitMiddleware: Middleware {
    private let rateLimiter: RateLimiter
    private let logger = EmbedKitLogger.embeddings()
    
    public init(requestsPerSecond: Double = 100, burstSize: Int = 200) {
        self.rateLimiter = RateLimiter(
            strategy: .tokenBucket(refillRate: requestsPerSecond, capacity: Double(burstSize)),
            scope: .global
        )
    }
    
    public func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        // Calculate cost based on command type
        let cost = calculateCost(for: command)
        
        // Check rate limit
        let status = await rateLimiter.checkLimit(
            for: metadata.userId ?? "anonymous",
            cost: cost
        )
        
        switch status {
        case .allowed:
            logger.debug("Rate limit check passed", context: "cost: \(cost)")
            return try await next(command, metadata)
            
        case .limited(let retryAfter):
            logger.warning("Rate limit exceeded", context: "retry after: \(retryAfter)s")
            throw RateLimitError.limitExceeded(retryAfter: retryAfter)
            
        case .blocked:
            logger.error("Request blocked by rate limiter")
            throw RateLimitError.blocked
        }
    }
    
    private func calculateCost<T: Command>(for command: T) -> Double {
        switch command {
        case let embedCommand as EmbedTextCommand:
            // Cost based on text length
            return Double(embedCommand.text.count) / 1000.0
            
        case let batchCommand as BatchEmbedCommand:
            // Higher cost for batch operations
            return Double(batchCommand.texts.count)
            
        case is StreamEmbedCommand:
            // Fixed cost for streaming (per connection)
            return 10.0
            
        case is LoadModelCommand, is SwapModelCommand:
            // High cost for model operations
            return 50.0
            
        default:
            return 1.0
        }
    }
}

// MARK: - Context Keys

struct EmbeddingCacheKey: ContextKey {
    typealias Value = EmbeddingCache
}

struct MetalAcceleratorKey: ContextKey {
    typealias Value = MetalAccelerator
}

struct UseGPUKey: ContextKey {
    typealias Value = Bool
}

struct TelemetrySystemKey: ContextKey {
    typealias Value = TelemetrySystem
}

struct CurrentModelKey: ContextKey {
    typealias Value = String
}

// MARK: - Monitoring Middleware

/// Middleware for comprehensive monitoring of embedding operations
public struct EmbeddingMonitoringMiddleware: ContextAwareMiddleware {
    private let telemetry: TelemetrySystem
    private let logger = EmbedKitLogger.embeddings()
    private let alertThresholds: AlertThresholds
    
    public struct AlertThresholds {
        let maxLatency: TimeInterval
        let minCacheHitRate: Double
        let maxMemoryUsageMB: Double
        
        public init(
            maxLatency: TimeInterval = 5.0,
            minCacheHitRate: Double = 0.7,
            maxMemoryUsageMB: Double = 1000
        ) {
            self.maxLatency = maxLatency
            self.minCacheHitRate = minCacheHitRate
            self.maxMemoryUsageMB = maxMemoryUsageMB
        }
    }
    
    public init(
        telemetry: TelemetrySystem,
        alertThresholds: AlertThresholds = AlertThresholds()
    ) {
        self.telemetry = telemetry
        self.alertThresholds = alertThresholds
    }
    
    public func execute<T: Command>(
        _ command: T,
        context: CommandContext,
        next: @Sendable (T, CommandContext) async throws -> T.Result
    ) async throws -> T.Result {
        let startTime = CFAbsoluteTimeGetCurrent()
        let systemMetrics = await telemetry.getSystemMetrics()
        
        // Monitor memory before execution
        if systemMetrics.memoryUsage > alertThresholds.maxMemoryUsageMB {
            logger.warning(
                "High memory usage detected",
                context: "\(Int(systemMetrics.memoryUsage))MB"
            )
            
            await telemetry.recordEvent(
                TelemetryEvent(
                    name: "high_memory_usage",
                    description: "Memory usage exceeds threshold",
                    severity: .warning,
                    metadata: ["usage_mb": String(systemMetrics.memoryUsage)]
                )
            )
        }
        
        let result = try await next(command, context)
        
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        
        // Check latency threshold
        if duration > alertThresholds.maxLatency {
            logger.warning(
                "High latency detected",
                context: "\(duration)s"
            )
            
            await telemetry.recordEvent(
                TelemetryEvent(
                    name: "high_latency",
                    description: "Command execution exceeds latency threshold",
                    severity: .warning,
                    metadata: [
                        "command": String(describing: type(of: command)),
                        "duration": String(duration)
                    ]
                )
            )
        }
        
        // Monitor cache performance for embedding commands
        if let cache = await context[EmbeddingCacheKey.self] {
            let stats = await cache.statistics()
            let hitRate = Double(stats.hits) / Double(stats.hits + stats.misses)
            
            if hitRate < alertThresholds.minCacheHitRate && stats.hits + stats.misses > 100 {
                logger.warning(
                    "Low cache hit rate",
                    context: "\(Int(hitRate * 100))%"
                )
            }
        }
        
        return result
    }
}
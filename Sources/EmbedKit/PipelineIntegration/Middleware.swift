import Foundation
import PipelineKit
@preconcurrency import Metal

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
        let _ = await cache.statistics() // Track for side effects
        
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
        if let shared = MetalAccelerator.shared {
            self.metalAccelerator = shared
        } else {
            guard let device = MTLCreateSystemDefaultDevice() else {
                throw MetalError.deviceNotAvailable
            }
            self.metalAccelerator = try MetalAccelerator(device: device)
        }
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
        if let validatable = command as? any ValidatableCommand {
            do {
                try validatable.validate()
                // CommandMetadata protocol doesn't support tags, so we'll use the original metadata
                enrichedMetadata = metadata
            } catch {
                logger.warning("Validation failed", context: error.localizedDescription)
                throw error
            }
        }
        
        return try await next(command, enrichedMetadata)
    }
    
    private func validateText(_ text: String) throws {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ContextualEmbeddingError.invalidInput(
                context: ErrorContext(
                    operation: .validation,
                    sourceLocation: SourceLocation()
                ),
                reason: .empty
            )
        }
        
        guard text.count <= maxTextLength else {
            throw ContextualEmbeddingError.invalidInput(
                context: ErrorContext(
                    operation: .validation,
                    metadata: ErrorMetadata()
                        .with(key: "maxLength", value: String(maxTextLength))
                        .with(key: "actualLength", value: String(text.count)),
                    sourceLocation: SourceLocation()
                ),
                reason: .tooLong
            )
        }
        
        // Check for potentially problematic content
        if text.contains("\0") {
            throw ContextualEmbeddingError.invalidInput(
                context: ErrorContext(
                    operation: .validation,
                    metadata: ErrorMetadata()
                        .with(key: "issue", value: "null characters"),
                    sourceLocation: SourceLocation()
                ),
                reason: .invalidCharacters
            )
        }
    }
    
    private func validateBatch(_ texts: [String]) throws {
        guard !texts.isEmpty else {
            throw ContextualEmbeddingError.invalidInput(
                context: ErrorContext(
                    operation: .validation,
                    metadata: ErrorMetadata()
                        .with(key: "batchSize", value: "0"),
                    sourceLocation: SourceLocation()
                ),
                reason: .empty
            )
        }
        
        guard texts.count <= maxBatchSize else {
            throw ContextualEmbeddingError.invalidInput(
                context: ErrorContext(
                    operation: .validation,
                    metadata: ErrorMetadata()
                        .with(key: "maxBatchSize", value: String(maxBatchSize))
                        .with(key: "actualBatchSize", value: String(texts.count)),
                    sourceLocation: SourceLocation()
                ),
                reason: .tooLong
            )
        }
        
        for (index, text) in texts.enumerated() {
            do {
                try validateText(text)
            } catch {
                throw ContextualEmbeddingError.invalidInput(
                    context: ErrorContext(
                        operation: .validation,
                        metadata: ErrorMetadata()
                            .with(key: "index", value: String(index))
                            .with(key: "error", value: error.localizedDescription),
                        sourceLocation: SourceLocation()
                    ),
                    reason: .malformed,
                    underlyingError: error
                )
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
        case _ as EmbedTextCommand:
            if let embedResult = result as? EmbeddingResult {
                await telemetry.recordGauge(
                    "embedding.dimensions",
                    value: Double(embedResult.embedding.dimensions),
                    tags: ["model": embedResult.modelIdentifier.rawValue]
                )
                
                if embedResult.fromCache {
                    await telemetry.incrementCounter("embedding.cache_hits")
                }
            }
            
        case _ as BatchEmbedCommand:
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
            
        case _ as LoadModelCommand:
            if let loadResult = result as? ModelLoadResult {
                await telemetry.recordGauge(
                    "model.size_bytes",
                    value: Double(loadResult.modelSize),
                    tags: ["model": loadResult.modelIdentifier.rawValue]
                )
            }
            
        default:
            break
        }
    }
}

// MARK: - Rate Limiting

/// Token bucket rate limiter implementation
actor RateLimiter {
    private var tokens: Double
    private let capacity: Double
    private let refillRate: Double // tokens per second
    private var lastRefillTime: Date
    
    init(capacity: Double, refillRate: Double) {
        self.capacity = capacity
        self.refillRate = refillRate
        self.tokens = capacity
        self.lastRefillTime = Date()
    }
    
    /// Attempts to consume the specified number of tokens
    /// Returns true if tokens were available, false if rate limit exceeded
    func tryConsume(_ count: Double) async -> Bool {
        // Refill tokens based on elapsed time
        let now = Date()
        let elapsed = now.timeIntervalSince(lastRefillTime)
        let tokensToAdd = elapsed * refillRate
        
        tokens = min(capacity, tokens + tokensToAdd)
        lastRefillTime = now
        
        // Check if we have enough tokens
        if tokens >= count {
            tokens -= count
            return true
        }
        
        return false
    }
    
    /// Waits until the specified number of tokens are available
    func waitAndConsume(_ count: Double) async throws {
        while true {
            if await tryConsume(count) {
                return
            }
            
            // Calculate wait time
            let tokensNeeded = count - tokens
            let waitTime = tokensNeeded / refillRate
            
            // Wait with a maximum of 1 second at a time to allow for cancellation
            let actualWaitTime = min(waitTime, 1.0)
            try await Task.sleep(nanoseconds: UInt64(actualWaitTime * 1_000_000_000))
        }
    }
    
    /// Gets current token count (for monitoring)
    func getCurrentTokens() async -> Double {
        // Refill before reporting
        let now = Date()
        let elapsed = now.timeIntervalSince(lastRefillTime)
        let tokensToAdd = elapsed * refillRate
        return min(capacity, tokens + tokensToAdd)
    }
}

// MARK: - Rate Limiting Middleware

/// Middleware for rate limiting embedding requests
public struct EmbeddingRateLimitMiddleware: Middleware {
    private let rateLimiter: RateLimiter
    private let logger = EmbedKitLogger.embeddings()
    private let waitForTokens: Bool
    
    public init(requestsPerSecond: Double = 100, burstSize: Int = 200, waitForTokens: Bool = true) {
        self.rateLimiter = RateLimiter(
            capacity: Double(burstSize),
            refillRate: requestsPerSecond
        )
        self.waitForTokens = waitForTokens
    }
    
    public func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        // Calculate cost based on command type
        let cost = calculateCost(for: command)
        
        // Check current token count for monitoring
        let currentTokens = await rateLimiter.getCurrentTokens()
        logger.debug("Rate limiter state", context: "tokens: \(currentTokens), cost: \(cost)")
        
        // Apply rate limiting
        if waitForTokens {
            // Wait until tokens are available
            do {
                try await rateLimiter.waitAndConsume(cost)
                logger.debug("Rate limit check passed after waiting", context: "cost: \(cost)")
            } catch {
                logger.error("Rate limiting interrupted", error: error)
                throw ContextualEmbeddingError.resourceUnavailable(
                    context: ErrorContext(
                        operation: .validation,
                        metadata: ErrorMetadata()
                            .with(key: "reason", value: "rate_limit_interrupted")
                            .with(key: "cost", value: "\(cost)")
                            .with(key: "error", value: error.localizedDescription),
                        sourceLocation: SourceLocation()
                    ),
                    resource: .network
                )
            }
        } else {
            // Fail fast if no tokens available
            let consumed = await rateLimiter.tryConsume(cost)
            if !consumed {
                logger.warning("Rate limit exceeded", context: "cost: \(cost), tokens: \(currentTokens)")
                throw ContextualEmbeddingError.resourceUnavailable(
                    context: ErrorContext(
                        operation: .validation,
                        metadata: ErrorMetadata()
                            .with(key: "reason", value: "rate_limit_exceeded")
                            .with(key: "cost", value: "\(cost)")
                            .with(key: "availableTokens", value: "\(currentTokens)")
                            .with(key: "retryAfter", value: "\((cost - currentTokens) / 100.0)"),
                        sourceLocation: SourceLocation()
                    ),
                    resource: .network
                )
            }
            logger.debug("Rate limit check passed", context: "cost: \(cost)")
        }
        
        // Execute the command
        let startTime = CFAbsoluteTimeGetCurrent()
        do {
            let result = try await next(command, metadata)
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            
            // Log successful execution
            logger.debug("Command executed within rate limit", 
                        context: "type: \(type(of: command)), cost: \(cost), duration: \(duration)s")
            
            return result
        } catch {
            // Log failed execution but don't refund tokens (to prevent abuse)
            logger.error("Command failed after rate limit check", error: error)
            throw error
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
    
    public struct AlertThresholds: Sendable {
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
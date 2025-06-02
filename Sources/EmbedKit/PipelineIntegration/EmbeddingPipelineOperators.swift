import Foundation
import PipelineKit

// MARK: - Helper Functions

/// Creates an embedding pipeline builder for the given handler
public func embeddingPipeline<H: CommandHandler>(for handler: H) -> EmbeddingPipelineBuilder<H> {
    EmbeddingPipelineBuilder(handler: handler)
}

/// Creates a middleware-priority tuple for use with <++ operator
public func middleware<M: Middleware>(_ middleware: M, priority: ExecutionPriority) -> (M, ExecutionPriority) {
    (middleware, priority)
}

// MARK: - EmbeddingPipeline Operator Extensions

/// Operator support for EmbeddingPipeline to enable fluent pipeline construction
public extension EmbeddingPipeline {
    
    // MARK: - Builder Creation
    
    /// Creates a pipeline builder for the specified handler
    static func builder<H: CommandHandler>(for handler: H) -> EmbeddingPipelineBuilder<H> {
        EmbeddingPipelineBuilder(handler: handler)
    }
}

// MARK: - Execution Priority

/// Priority levels for middleware execution order
public enum ExecutionPriority: Int, Sendable, Comparable {
    case authentication = 1000
    case authorization = 900
    case validation = 800
    case preExecution = 700
    case caching = 600
    case processing = 500
    case postExecution = 400
    case monitoring = 300
    case cleanup = 200
    
    public static func < (lhs: ExecutionPriority, rhs: ExecutionPriority) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

// MARK: - Pipeline Builder with Operator Support

/// A builder that supports operator syntax for constructing embedding pipelines
public struct EmbeddingPipelineBuilder<H: CommandHandler> {
    private let handler: H
    private var middlewares: [any Middleware] = []
    private var prioritizedMiddlewares: [(middleware: any Middleware, priority: ExecutionPriority)] = []
    private var errorHandler: (@Sendable (Error) async throws -> Void)?
    private var conditions: [(condition: @Sendable (CommandMetadata) async -> Bool, middleware: any Middleware)] = []
    
    init(handler: H) {
        self.handler = handler
    }
    
    /// Builds the final pipeline
    public func build() async throws -> any Pipeline {
        let builder = PipelineBuilder(handler: handler)
        
        // Add prioritized middleware first (sorted by priority)
        let sortedPrioritized = prioritizedMiddlewares.sorted { 
            $0.priority.rawValue > $1.priority.rawValue 
        }
        
        for (middleware, _) in sortedPrioritized {
            await builder.with(middleware)
        }
        
        // Add regular middleware
        for middleware in middlewares {
            await builder.with(middleware)
        }
        
        // Apply conditional middleware
        for (condition, middleware) in conditions {
            let conditionalMiddleware = ConditionalMiddlewareWrapper(
                middleware: middleware,
                condition: condition
            )
            await builder.with(conditionalMiddleware)
        }
        
        // Build the pipeline
        let basePipeline = try await builder.build()
        
        // Apply error handler if present
        if let errorHandler = errorHandler {
            return ErrorHandlingPipelineWrapper(
                pipeline: basePipeline,
                errorHandler: errorHandler
            )
        }
        
        return basePipeline
    }
}

// MARK: - Basic Middleware Addition Operator

public extension EmbeddingPipelineBuilder {
    /// Adds middleware with default priority
    static func <+ (builder: EmbeddingPipelineBuilder, middleware: any Middleware) -> EmbeddingPipelineBuilder {
        var newBuilder = builder
        newBuilder.middlewares.append(middleware)
        return newBuilder
    }
}

// MARK: - Prioritized Middleware Addition Operator

public extension EmbeddingPipelineBuilder {
    /// Adds middleware with specified priority
    static func <++ (builder: EmbeddingPipelineBuilder, prioritized: (any Middleware, ExecutionPriority)) -> EmbeddingPipelineBuilder {
        var newBuilder = builder
        newBuilder.prioritizedMiddlewares.append(prioritized)
        return newBuilder
    }
}

// MARK: - Additional Methods

public extension EmbeddingPipelineBuilder {
    /// Adds middleware with default priority
    func addMiddleware(_ middleware: any Middleware) -> EmbeddingPipelineBuilder {
        var newBuilder = self
        newBuilder.middlewares.append(middleware)
        return newBuilder
    }
    
    /// Adds middleware with specified priority
    func addPrioritizedMiddleware(_ middleware: any Middleware, priority: ExecutionPriority) -> EmbeddingPipelineBuilder {
        var newBuilder = self
        newBuilder.prioritizedMiddlewares.append((middleware, priority))
        return newBuilder
    }
    
    /// Adds conditional middleware
    func when(_ condition: @escaping @Sendable (CommandMetadata) async -> Bool, use middleware: any Middleware) -> EmbeddingPipelineBuilder {
        var newBuilder = self
        newBuilder.conditions.append((condition, middleware))
        return newBuilder
    }
    
    /// Adds error handler
    func withErrorHandler(_ handler: @escaping @Sendable (Error) async throws -> Void) -> EmbeddingPipelineBuilder {
        var newBuilder = self
        newBuilder.errorHandler = handler
        return newBuilder
    }
}

// MARK: - Pipeline Composition Operators

// MARK: - Global Pipeline Operators

/// Error handling operator
public func |! (pipeline: any Pipeline, errorHandler: @escaping @Sendable (Error) async throws -> Void) -> any Pipeline {
    ErrorHandlingPipelineWrapper(pipeline: pipeline, errorHandler: errorHandler)
}

// MARK: - Pipeline Wrapper Types


/// Wrapper for error handling in pipelines
struct ErrorHandlingPipelineWrapper: Pipeline {
    private let pipeline: any Pipeline
    private let errorHandler: @Sendable (Error) async throws -> Void
    
    init(pipeline: any Pipeline, errorHandler: @escaping @Sendable (Error) async throws -> Void) {
        self.pipeline = pipeline
        self.errorHandler = errorHandler
    }
    
    func execute<T: Command>(_ command: T, metadata: CommandMetadata) async throws -> T.Result {
        do {
            return try await pipeline.execute(command, metadata: metadata)
        } catch {
            try await errorHandler(error)
            throw error
        }
    }
}

// MARK: - Default CommandMetadata Implementation

/// Default implementation of CommandMetadata for testing and examples
public struct DefaultCommandMetadata: CommandMetadata {
    public let id: UUID
    public let timestamp: Date
    public let correlationId: String?
    public let userId: String?
    private let metadata: [String: any Sendable]
    
    public init(
        userId: String? = nil,
        correlationId: String? = UUID().uuidString,
        timestamp: Date = Date(),
        metadata: [String: any Sendable] = [:]
    ) {
        self.id = UUID()
        self.userId = userId
        self.correlationId = correlationId
        self.timestamp = timestamp
        self.metadata = metadata
    }
    
    public func get<T>(_ key: String, as type: T.Type) -> T? {
        metadata[key] as? T
    }
}

// MARK: - Helper Types

/// Conditional middleware wrapper that conforms to PipelineKit's Middleware protocol
struct ConditionalMiddlewareWrapper: Middleware {
    let middleware: any Middleware
    let condition: @Sendable (CommandMetadata) async -> Bool
    
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        if await condition(metadata) {
            return try await middleware.execute(command, metadata: metadata, next: next)
        } else {
            return try await next(command, metadata)
        }
    }
}

// MARK: - Convenience Extensions

public extension EmbeddingPipeline {
    /// Creates a simple embedding pipeline demonstrating the fluent API
    /// This example shows how to build a pipeline with various middleware components
    static func createWithOperators() async throws -> any Pipeline {
        // Create real dependencies
        let embedder = MockTextEmbedder()
        let cache = EmbeddingCache()
        let telemetrySystem = TelemetrySystem()
        
        // Create the handler with dependencies
        let handler = EmbedTextHandler(
            embedder: embedder,
            cache: cache,
            telemetry: telemetrySystem
        )
        
        // Create middleware instances
        let validationMiddleware = EmbeddingValidationMiddleware(
            maxTextLength: 10_000,
            maxBatchSize: 100
        )
        let cacheMiddleware = EmbeddingCacheMiddleware(cache: cache)
        let telemetryMiddleware = TelemetryMiddleware(telemetry: telemetrySystem)
        
        // Build pipeline using the fluent API (operators have been replaced)
        return try await EmbeddingPipelineBuilder(handler: handler)
            .addPrioritizedMiddleware(validationMiddleware, priority: .validation)
            .addMiddleware(cacheMiddleware)
            .addPrioritizedMiddleware(telemetryMiddleware, priority: .postExecution)
            .build()
    }
    
    /// Creates a batch processing pipeline optimized for high-throughput operations
    /// This example demonstrates pipeline configuration for batch processing scenarios
    static func createBatchPipelineWithOperators() async throws -> any Pipeline {
        // Create dependencies for batch processing
        let embedder = MockTextEmbedder()
        let cache = EmbeddingCache()
        let telemetrySystem = TelemetrySystem()
        
        // Create batch handler
        let handler = BatchEmbedHandler(
            embedder: embedder,
            cache: cache,
            telemetry: telemetrySystem
        )
        
        // Create specialized batch middleware
        let batchOptimizer = BatchOptimizationMiddleware()
        let progressReporter = ProgressReportingMiddleware()
        let validationMiddleware = EmbeddingValidationMiddleware(
            maxTextLength: 10_000,
            maxBatchSize: 1000  // Higher limit for batch operations
        )
        
        // Add rate limiting for batch operations
        let rateLimitMiddleware = EmbeddingRateLimitMiddleware(
            requestsPerSecond: 100,
            burstSize: 200
        )
        
        // Build the pipeline with appropriate middleware ordering
        return try await EmbeddingPipelineBuilder(handler: handler)
            .addPrioritizedMiddleware(rateLimitMiddleware, priority: .authorization)
            .addPrioritizedMiddleware(validationMiddleware, priority: .validation)
            .addPrioritizedMiddleware(batchOptimizer, priority: .preExecution)
            .addPrioritizedMiddleware(progressReporter, priority: .postExecution)
            .build()
    }
    
    /// Creates a streaming pipeline for processing large text collections
    /// This example shows how to configure a pipeline for streaming scenarios
    static func createStreamingPipeline() async throws -> any Pipeline {
        // Create dependencies
        let embedder = MockTextEmbedder()
        let cache = EmbeddingCache()
        let telemetrySystem = TelemetrySystem()
        
        // Create stream handler
        let handler = StreamEmbedHandler(
            embedder: embedder,
            cache: cache,
            telemetry: telemetrySystem
        )
        
        // Create streaming-specific middleware
        let backpressureMiddleware = BackpressureMiddleware(maxBufferSize: 1000)
        let chunkingMiddleware = ChunkingMiddleware(chunkSize: 100)
        let streamMonitoringMiddleware = StreamMonitoringMiddleware()
        
        // Build pipeline optimized for streaming
        return try await EmbeddingPipelineBuilder(handler: handler)
            .addPrioritizedMiddleware(backpressureMiddleware, priority: .preExecution)
            .addMiddleware(chunkingMiddleware)
            .addPrioritizedMiddleware(streamMonitoringMiddleware, priority: .monitoring)
            .withErrorHandler { error in
                print("[StreamPipeline] Error: \(error)")
            }
            .build()
    }
    
    /// Creates a minimal pipeline with just the essentials
    /// Useful for testing or when maximum performance is needed
    static func createMinimalPipeline() async throws -> any Pipeline {
        // Create minimal dependencies
        let embedder = MockTextEmbedder()
        let cache = EmbeddingCache()
        let telemetrySystem = TelemetrySystem()
        
        let handler = EmbedTextHandler(
            embedder: embedder,
            cache: cache,
            telemetry: telemetrySystem
        )
        
        // Build with no middleware - just the handler
        return try await EmbeddingPipelineBuilder(handler: handler).build()
    }
    
    /// Creates a development pipeline with extensive logging and debugging
    static func createDevelopmentPipeline() async throws -> any Pipeline {
        // Create dependencies
        let embedder = MockTextEmbedder()
        let cache = EmbeddingCache()
        let telemetrySystem = TelemetrySystem()
        
        let handler = EmbedTextHandler(
            embedder: embedder,
            cache: cache,
            telemetry: telemetrySystem
        )
        
        // Create debugging middleware
        let loggingMiddleware = LoggingMiddleware(logLevel: .debug)
        let timingMiddleware = TimingMiddleware()
        let validationMiddleware = EmbeddingValidationMiddleware(
            maxTextLength: 5_000,  // Stricter limits for development
            maxBatchSize: 10
        )
        
        // Build with extensive middleware for debugging
        return try await EmbeddingPipelineBuilder(handler: handler)
            .addPrioritizedMiddleware(loggingMiddleware, priority: .authentication) // Log early
            .addPrioritizedMiddleware(timingMiddleware, priority: .preExecution)
            .addPrioritizedMiddleware(validationMiddleware, priority: .validation)
            .addPrioritizedMiddleware(
                TelemetryMiddleware(telemetry: telemetrySystem),
                priority: .postExecution
            )
            .withErrorHandler { error in
                print("[DevPipeline] ❌ Error: \(error)")
                print("[DevPipeline] Stack trace: \(Thread.callStackSymbols)")
            }
            .build()
    }
}

// MARK: - Mock Middleware Types (for examples and testing)

/// Mock caching middleware that simulates cache operations
struct CachingMiddleware: Middleware {
    // Using a simple dictionary for thread-safe caching
    private let cacheActor = CacheActor()
    
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        // Simple cache key based on command type and metadata
        let cacheKey = "\(type(of: command))-\(metadata.id)"
        
        // Check cache first
        if await cacheActor.hasValue(for: cacheKey) {
            print("[CachingMiddleware] Cache hit for \(cacheKey)")
            // In a real implementation, we'd deserialize and return the cached result
        }
        
        // Execute the command
        let result = try await next(command, metadata)
        
        // Cache the result (in a real implementation, we'd serialize it)
        await cacheActor.setValue(true, for: cacheKey)
        print("[CachingMiddleware] Cached result for \(cacheKey)")
        
        return result
    }
}

/// Actor to manage cache state in a thread-safe way
private actor CacheActor {
    private var cache: [String: Bool] = [:]
    
    func hasValue(for key: String) -> Bool {
        cache[key] != nil
    }
    
    func setValue(_ value: Bool, for key: String) {
        cache[key] = value
    }
}

/// Mock validation middleware that performs basic command validation
struct ValidationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        // Perform validation based on command type
        if let embedCommand = command as? EmbedTextCommand {
            guard !embedCommand.text.isEmpty else {
                throw ContextualEmbeddingError.invalidInput(
                    context: ErrorContext(
                        operation: .validation,
                        metadata: ErrorMetadata().with(key: "reason", value: "Empty text"),
                        sourceLocation: SourceLocation()
                    ),
                    reason: .empty
                )
            }
            guard embedCommand.text.count <= 10_000 else {
                throw ContextualEmbeddingError.invalidInput(
                    context: ErrorContext(
                        operation: .validation,
                        metadata: ErrorMetadata().with(key: "textLength", value: "\(embedCommand.text.count)"),
                        sourceLocation: SourceLocation()
                    ),
                    reason: .tooLong
                )
            }
        }
        
        print("[ValidationMiddleware] Validation passed for \(type(of: command))")
        return try await next(command, metadata)
    }
}

/// Middleware that optimizes batch operations
struct BatchOptimizationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        // Apply optimizations for batch commands
        if let batchCommand = command as? BatchEmbedCommand {
            print("[BatchOptimizer] Optimizing batch of \(batchCommand.texts.count) texts")
            
            // In a real implementation, we might:
            // - Sort texts by length for better GPU utilization
            // - Group similar texts together
            // - Pre-allocate memory buffers
            // - Enable GPU acceleration if available
        }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let result = try await next(command, metadata)
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        
        print("[BatchOptimizer] Batch completed in \(String(format: "%.3f", duration))s")
        return result
    }
}

/// Middleware that reports progress for long-running operations
struct ProgressReportingMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        let operationId = metadata.id
        print("[Progress] Starting operation \(operationId)")
        
        // In a real implementation, we might:
        // - Start a background task to report progress
        // - Send progress updates to a UI
        // - Log intermediate results
        
        let result = try await next(command, metadata)
        
        print("[Progress] Completed operation \(operationId)")
        return result
    }
}

// MARK: - Additional Middleware for Streaming and Development

/// Middleware that handles backpressure in streaming scenarios
struct BackpressureMiddleware: Middleware {
    let maxBufferSize: Int
    
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        print("[Backpressure] Monitoring buffer (max: \(maxBufferSize))")
        // In a real implementation, would manage buffer sizes and apply backpressure
        return try await next(command, metadata)
    }
}

/// Middleware that chunks large operations into smaller pieces
struct ChunkingMiddleware: Middleware {
    let chunkSize: Int
    
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        if command is StreamEmbedCommand {
            print("[Chunking] Processing stream in chunks of \(chunkSize)")
        }
        return try await next(command, metadata)
    }
}

/// Middleware that monitors streaming operations
struct StreamMonitoringMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        if command is StreamEmbedCommand {
            print("[StreamMonitor] Monitoring stream health")
        }
        return try await next(command, metadata)
    }
}

/// Middleware that logs all operations for debugging
struct LoggingMiddleware: Middleware {
    enum LogLevel {
        case debug, info, warning, error
    }
    
    let logLevel: LogLevel
    
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        print("[Logger:\(logLevel)] Executing \(type(of: command)) - ID: \(metadata.id)")
        
        do {
            let result = try await next(command, metadata)
            print("[Logger:\(logLevel)] Success for \(metadata.id)")
            return result
        } catch {
            print("[Logger:\(logLevel)] Error for \(metadata.id): \(error)")
            throw error
        }
    }
}

/// Middleware that measures execution time
struct TimingMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        let startTime = CFAbsoluteTimeGetCurrent()
        let result = try await next(command, metadata)
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        
        print("[Timing] \(type(of: command)) took \(String(format: "%.3f", duration))s")
        return result
    }
}

// MARK: - Additional Mock Types for Examples

struct SimpleMockTextEmbedder {
    init() throws {}
}

struct MockEmbeddingCache {}

struct MockTelemetrySystem {}

// Note: Real implementations of these types are defined in their respective files:
// - TelemetryMiddleware is in Middleware.swift
// - EmbeddingValidationMiddleware is in Middleware.swift
// - EmbeddingCacheMiddleware is in Middleware.swift
// - EmbeddingRateLimitMiddleware is in Middleware.swift

struct MockEmbedBatchHandler: CommandHandler {
    typealias CommandType = MockEmbedTextCommand
    
    func handle(_ command: MockEmbedTextCommand) async throws -> MockEmbedResult {
        MockEmbedResult(text: command.text)
    }
}

struct MockEmbedTextCommand: Command {
    typealias Result = MockEmbedResult
    let text: String
}

struct MockEmbedResult: Sendable {
    let text: String
    let embedding: [Float]
    
    init(text: String) {
        self.text = text
        self.embedding = Array(repeating: 0.1, count: 384) // Mock embedding
    }
}

struct MockEmbedTextHandler: CommandHandler {
    typealias CommandType = MockEmbedTextCommand
    
    func handle(_ command: MockEmbedTextCommand) async throws -> MockEmbedResult {
        MockEmbedResult(text: command.text)
    }
}
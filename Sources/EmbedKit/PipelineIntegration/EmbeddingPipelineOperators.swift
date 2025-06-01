import Foundation
import PipelineKit

// MARK: - EmbeddingPipeline Operator Extensions

/// Operator support for EmbeddingPipeline to enable fluent pipeline construction
public extension EmbeddingPipeline {
    
    // MARK: - Builder Creation
    
    /// Creates a pipeline builder for the specified handler
    static func builder<H: CommandHandler>(for handler: H) -> EmbeddingPipelineBuilder<H> {
        EmbeddingPipelineBuilder(handler: handler)
    }
}

// MARK: - Pipeline Builder with Operator Support

/// A builder that supports operator syntax for constructing embedding pipelines
public struct EmbeddingPipelineBuilder<H: CommandHandler> {
    private let handler: H
    private var middlewares: [any Middleware] = []
    private var prioritizedMiddlewares: [(middleware: any Middleware, priority: ExecutionPriority)] = []
    private var errorHandler: ((Error) async throws -> Void)?
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
        
        for (middleware, priority) in sortedPrioritized {
            await builder.with(middleware, order: priority)
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
        var pipeline = try await builder.build()
        
        // Apply error handler if present
        if let errorHandler = errorHandler {
            pipeline = ErrorHandlingPipelineWrapper(
                pipeline: pipeline,
                errorHandler: errorHandler
            )
        }
        
        return pipeline
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
    /// Adds conditional middleware
    func when(_ condition: @escaping @Sendable (CommandMetadata) async -> Bool, use middleware: any Middleware) -> EmbeddingPipelineBuilder {
        var newBuilder = self
        newBuilder.conditions.append((condition, middleware))
        return newBuilder
    }
    
    /// Adds error handler
    func withErrorHandler(_ handler: @escaping (Error) async throws -> Void) -> EmbeddingPipelineBuilder {
        var newBuilder = self
        newBuilder.errorHandler = handler
        return newBuilder
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
    /// Creates a simple embedding pipeline with operator syntax
    static func createWithOperators() async throws -> any Pipeline {
        let handler = EmbedTextHandler()
        let cache = CachingMiddleware()
        let validation = ValidationMiddleware()
        let telemetry = TelemetryMiddleware()
        
        return try await builder(for: handler)
            <++ (validation, .validation)
            <+ cache
            <++ (telemetry, .postExecution)
            .build()
    }
    
    /// Creates a batch processing pipeline with operators
    static func createBatchPipelineWithOperators() async throws -> any Pipeline {
        let handler = EmbedBatchHandler()
        let optimizer = BatchOptimizationMiddleware()
        let progress = ProgressReportingMiddleware()
        
        return try await builder(for: handler)
            <++ (optimizer, .preExecution)
            <++ (progress, .postExecution)
            .build()
    }
}

// MARK: - Mock Middleware Types (for compilation)

// These would normally come from your actual implementation
struct CachingMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}

struct ValidationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}

struct TelemetryMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}

struct BatchOptimizationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}

struct ProgressReportingMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}
import Foundation
import PipelineKit

// MARK: - Pipeline Factory for EmbedKit Integration

/// Factory for creating common embedding pipelines
public struct EmbeddingPipelineFactory {
    
    private init() {} // Prevent instantiation
    
    /// Create a pipeline for document indexing (embed + store)
    /// Note: This returns a basic configuration. Use with specific handlers to create a complete pipeline.
    public static func createIndexingConfiguration(
        embedder: TextEmbedder,
        telemetry: TelemetrySystem? = nil
    ) -> [any Middleware] {
        var middlewares: [any Middleware] = []
        
        // Add validation middleware
        middlewares.append(ValidationMiddleware())
        
        // Add telemetry if provided
        if let telemetry = telemetry {
            middlewares.append(TelemetryMiddleware(telemetry: telemetry))
        }
        
        // Add embedding validation middleware
        middlewares.append(EmbeddingValidationMiddleware())
        
        // Add caching middleware (defined in EmbeddingPipelineOperators.swift)
        middlewares.append(CachingMiddleware())
        
        return middlewares
    }
    
    /// Create a pipeline configuration for search operations
    public static func createSearchConfiguration(
        embedder: TextEmbedder,
        telemetry: TelemetrySystem? = nil
    ) -> [any Middleware] {
        var middlewares: [any Middleware] = []
        
        // Add validation
        middlewares.append(ValidationMiddleware())
        
        // Add telemetry
        if let telemetry = telemetry {
            middlewares.append(TelemetryMiddleware(telemetry: telemetry))
        }
        
        // Add query embedding
        middlewares.append(QueryEmbeddingMiddleware(embedder: embedder))
        
        // Add result ranking
        middlewares.append(ResultRankingMiddleware())
        
        return middlewares
    }
    
    /// Create a pipeline configuration for batch operations
    public static func createBatchConfiguration(
        embedder: TextEmbedder,
        maxConcurrency: Int = 10,
        telemetry: TelemetrySystem? = nil
    ) -> [any Middleware] {
        var middlewares: [any Middleware] = []
        
        // Add validation
        middlewares.append(ValidationMiddleware())
        
        // Add telemetry
        if let telemetry = telemetry {
            middlewares.append(TelemetryMiddleware(telemetry: telemetry))
        }
        
        // Add batch processing
        middlewares.append(BatchProcessingMiddleware(maxConcurrency: maxConcurrency))
        
        // Add embedding validation
        middlewares.append(EmbeddingValidationMiddleware())
        
        // Add progress tracking
        middlewares.append(ProgressTrackingMiddleware())
        
        return middlewares
    }
    
    /// Create a pipeline configuration for streaming operations
    public static func createStreamingConfiguration(
        embedder: TextEmbedder,
        bufferSize: Int = 100,
        telemetry: TelemetrySystem? = nil
    ) -> [any Middleware] {
        var middlewares: [any Middleware] = []
        
        // Add telemetry
        if let telemetry = telemetry {
            middlewares.append(TelemetryMiddleware(telemetry: telemetry))
        }
        
        // Add streaming support
        middlewares.append(StreamingMiddleware(bufferSize: bufferSize))
        
        // Add embedding validation
        middlewares.append(EmbeddingValidationMiddleware())
        
        // Add backpressure handling
        middlewares.append(BackpressureMiddleware(maxBufferSize: bufferSize))
        
        return middlewares
    }
    
    /// Creates a high-performance pipeline optimized for throughput
    public static func highPerformance(
        embedder: any TextEmbedder,
        modelManager: EmbeddingModelManager
    ) async throws -> EmbeddingPipeline {
        let configuration = EmbeddingPipeline.Configuration(
            enableCache: true,
            enableGPUAcceleration: true,
            enableRateLimiting: false,
            enableMonitoring: true,
            maxBatchSize: 1000,
            requestsPerSecond: 1000
        )
        
        return try await EmbeddingPipeline(
            embedder: embedder,
            modelManager: modelManager,
            configuration: configuration
        )
    }
    
    /// Creates a minimal pipeline for basic embedding operations
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
    
    /// Creates a balanced pipeline with moderate performance and monitoring
    public static func balanced(
        embedder: any TextEmbedder,
        modelManager: EmbeddingModelManager
    ) async throws -> EmbeddingPipeline {
        let configuration = EmbeddingPipeline.Configuration(
            enableCache: true,
            enableGPUAcceleration: true,
            enableRateLimiting: true,
            enableMonitoring: true,
            maxBatchSize: 100,
            requestsPerSecond: 100
        )
        
        return try await EmbeddingPipeline(
            embedder: embedder,
            modelManager: modelManager,
            configuration: configuration
        )
    }
}

// MARK: - Specialized Middleware

/// Middleware for handling query embedding
public struct QueryEmbeddingMiddleware: Middleware {
    private let embedder: TextEmbedder
    
    public init(embedder: TextEmbedder) {
        self.embedder = embedder
    }
    
    public func execute<C: Command>(
        _ command: C,
        metadata: CommandMetadata,
        next: @Sendable (C, CommandMetadata) async throws -> C.Result
    ) async throws -> C.Result {
        // Handle SearchDocumentsCommand specially
        if let searchCommand = command as? SearchDocumentsCommand {
            // Embed the query text
            let _ = try await embedder.embed(searchCommand.query)
            
            // Create an internal command with the embedded query
            // This would be handled by the vector store handler
            // For now, pass through
            return try await next(command, metadata)
        }
        
        return try await next(command, metadata)
    }
}

/// Middleware for result ranking and re-ranking
public struct ResultRankingMiddleware: Middleware {
    
    public init() {}
    
    public func execute<C: Command>(
        _ command: C,
        metadata: CommandMetadata,
        next: @Sendable (C, CommandMetadata) async throws -> C.Result
    ) async throws -> C.Result {
        let result = try await next(command, metadata)
        
        // If the result is search results, apply re-ranking
        if let searchResults = result as? [SearchResult] {
            // Apply any re-ranking logic here
            // For now, just ensure they're sorted by score
            let ranked = searchResults.sorted { $0.score > $1.score }
            return ranked as! C.Result
        }
        
        return result
    }
}

/// Middleware for batch processing with concurrency control
public struct BatchProcessingMiddleware: Middleware {
    private let maxConcurrency: Int
    
    public init(maxConcurrency: Int = 10) {
        self.maxConcurrency = maxConcurrency
    }
    
    public func execute<C: Command>(
        _ command: C,
        metadata: CommandMetadata,
        next: @Sendable (C, CommandMetadata) async throws -> C.Result
    ) async throws -> C.Result {
        // Handle batch commands with concurrency control
        if command is BatchIndexDocumentsCommand {
            // Implement concurrent processing logic
            // For now, pass through
            return try await next(command, metadata)
        }
        
        return try await next(command, metadata)
    }
}

/// Middleware for progress tracking
public struct ProgressTrackingMiddleware: Middleware {
    
    public init() {}
    
    public func execute<C: Command>(
        _ command: C,
        metadata: CommandMetadata,
        next: @Sendable (C, CommandMetadata) async throws -> C.Result
    ) async throws -> C.Result {
        let startTime = Date()
        
        let result = try await next(command, metadata)
        
        let duration = Date().timeIntervalSince(startTime)
        
        // Log progress or emit events
        if let batchResult = result as? BatchIndexResult {
            print("Batch operation completed: \(batchResult.successful.count) successful, \(batchResult.failed.count) failed in \(duration)s")
        }
        
        return result
    }
}

/// Middleware for streaming operations
public struct StreamingMiddleware: Middleware {
    private let bufferSize: Int
    
    public init(bufferSize: Int = 100) {
        self.bufferSize = bufferSize
    }
    
    public func execute<C: Command>(
        _ command: C,
        metadata: CommandMetadata,
        next: @Sendable (C, CommandMetadata) async throws -> C.Result
    ) async throws -> C.Result {
        // Handle streaming commands
        if command is StreamIndexDocumentsCommand {
            // Set up buffering for streaming
            // For now, pass through
            return try await next(command, metadata)
        }
        
        return try await next(command, metadata)
    }
}

/// Middleware for handling backpressure
public struct BackpressureMiddleware: Middleware {
    private let maxBufferSize: Int
    
    public init(maxBufferSize: Int = 1000) {
        self.maxBufferSize = maxBufferSize
    }
    
    public func execute<C: Command>(
        _ command: C,
        metadata: CommandMetadata,
        next: @Sendable (C, CommandMetadata) async throws -> C.Result
    ) async throws -> C.Result {
        // Implement backpressure handling for streaming
        return try await next(command, metadata)
    }
}

// MARK: - Pipeline Builder Helpers

/// Helper to create a configured pipeline builder
public struct PipelineBuilderHelper {
    
    /// Create a pipeline builder configured for document indexing
    public static func createIndexingBuilder<H: CommandHandler & Sendable>(
        handler: H,
        embedder: TextEmbedder,
        telemetry: TelemetrySystem? = nil
    ) -> EmbeddingPipelineBuilder<H> {
        var builder = EmbeddingPipelineBuilder(handler: handler)
        
        builder = builder.addMiddleware(ValidationMiddleware())
        
        if let telemetry = telemetry {
            builder = builder.addMiddleware(TelemetryMiddleware(telemetry: telemetry))
        }
        
        builder = builder
            .addMiddleware(EmbeddingValidationMiddleware())
            .addMiddleware(CachingMiddleware())
        
        return builder
    }
    
    /// Create a pipeline builder configured for search operations
    public static func createSearchBuilder<H: CommandHandler & Sendable>(
        handler: H,
        embedder: TextEmbedder,
        telemetry: TelemetrySystem? = nil
    ) -> EmbeddingPipelineBuilder<H> {
        var builder = EmbeddingPipelineBuilder(handler: handler)
        
        builder = builder.addMiddleware(ValidationMiddleware())
        
        if let telemetry = telemetry {
            builder = builder.addMiddleware(TelemetryMiddleware(telemetry: telemetry))
        }
        
        builder = builder
            .addMiddleware(QueryEmbeddingMiddleware(embedder: embedder))
            .addMiddleware(ResultRankingMiddleware())
        
        return builder
    }
}
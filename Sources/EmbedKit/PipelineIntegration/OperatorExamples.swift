import Foundation
import PipelineKit

// MARK: - Operator Syntax Examples for EmbedKit Pipeline Integration

/// This file demonstrates various operator syntax patterns for building
/// embedding pipelines using PipelineKit's expressive operator syntax.

// MARK: - Basic Operator Usage

/// Example 1: Simple pipeline with operator syntax
func createBasicPipeline() async throws -> any Pipeline {
    // Create dependencies
    let embedder = CoreMLTextEmbedder(
        modelIdentifier: "all-MiniLM-L6-v2",
        configuration: Configuration()
    )
    let cache = EmbeddingCache()
    let telemetry = TelemetrySystem()
    
    let handler = EmbedTextHandler(
        embedder: embedder,
        cache: cache,
        telemetry: telemetry
    )
    
    // Using fluent API instead of operators to avoid precedence issues
    let builder = PipelineBuilder(handler: handler)
    await builder.with(MockCachingMiddleware())
    await builder.with(MockValidationMiddleware())
    let pipeline = try await builder.build()
    
    return pipeline
}

/// Example 2: Pipeline with prioritized middleware
func createPrioritizedPipeline() async throws -> any Pipeline {
    // Create dependencies
    let embedder = CoreMLTextEmbedder(
        modelIdentifier: "all-MiniLM-L6-v2",
        configuration: Configuration()
    )
    let cache = EmbeddingCache()
    let telemetry = TelemetrySystem()
    
    let handler = EmbedTextHandler(
        embedder: embedder,
        cache: cache,
        telemetry: telemetry
    )
    
    // Using fluent API for prioritized middleware instead of operators
    let pipeline = try await EmbeddingPipelineBuilder(handler: handler)
        .addPrioritizedMiddleware(MockAuthenticationMiddleware(), priority: ExecutionPriority.authentication)
        .addPrioritizedMiddleware(MockValidationMiddleware(), priority: ExecutionPriority.validation)
        .addMiddleware(MockCachingMiddleware())  // Default priority
        .addPrioritizedMiddleware(MockMonitoringMiddleware(), priority: ExecutionPriority.postExecution)
        .build()
    
    return pipeline
}

// MARK: - Pipeline Composition Operators

/// Example 3: Composing pipelines with |> operator
func createComposedPipeline() async throws -> any Pipeline {
    let preprocessingPipeline = try await createPreprocessingPipeline()
    let embeddingPipeline = try await createEmbeddingPipeline()
    let postprocessingPipeline = try await createPostprocessingPipeline()
    
    // Sequential composition using |> operator
    let composedPipeline = preprocessingPipeline
        |> embeddingPipeline
        |> postprocessingPipeline
    
    return composedPipeline
}

/// Example 4: Parallel pipeline composition with || operator
func createParallelPipeline() async throws -> any Pipeline {
    let cpuPipeline = try await createCPUPipeline()
    let gpuPipeline = try await createGPUPipeline()
    
    // Parallel execution (simulated with sequential for now)
    let parallelPipeline = cpuPipeline |> gpuPipeline
    
    return parallelPipeline
}

/// Example 5: Error handling with |! operator
func createErrorHandlingPipeline() async throws -> any Pipeline {
    let mainPipeline = try await createMainPipeline()
    
    // Add error handling using wrapper instead of |! operator
    let safePipeline = ErrorHandlingPipelineWrapper(
        pipeline: mainPipeline,
        errorHandler: { error in
            print("Pipeline error: \(error)")
            // Additional error handling logic
        }
    )
    
    return safePipeline
}

/// Example 6: Conditional pipeline with |? operator
func createConditionalPipeline() async throws -> any Pipeline {
    let basePipeline = try await createBasePipeline()
    
    // Conditional execution based on runtime conditions
    let conditionalPipeline = basePipeline |? { 
        // Check if GPU is available
        await checkGPUAvailability()
    }
    
    return conditionalPipeline
}

// MARK: - Advanced Operator Patterns

/// Example 7: Complex production pipeline with all operators
func createProductionPipeline() async throws -> any Pipeline {
    // Create dependencies
    let embedder = CoreMLTextEmbedder(
        modelIdentifier: "all-MiniLM-L6-v2",
        configuration: Configuration()
    )
    let cache = EmbeddingCache()
    let telemetry = TelemetrySystem()
    
    let handler = EmbedTextHandler(
        embedder: embedder,
        cache: cache,
        telemetry: telemetry
    )
    
    // Build complex pipeline using fluent API to avoid operator issues
    let pipeline = try await EmbeddingPipelineBuilder(handler: handler)
        // Critical security layer
        .addPrioritizedMiddleware(MockAuthenticationMiddleware(), priority: ExecutionPriority.authentication)
        .addPrioritizedMiddleware(MockRateLimitingMiddleware(requestsPerSecond: 100), priority: .authorization)
        
        // Validation layer
        .addPrioritizedMiddleware(MockValidationMiddleware(), priority: ExecutionPriority.validation)
        .addMiddleware(MockSanitizationMiddleware())
        
        // Processing layer
        .addMiddleware(MockCachingMiddleware())
        .addMiddleware(MockGPUAccelerationMiddleware())
        
        // Monitoring layer
        .addPrioritizedMiddleware(MockMonitoringMiddleware(), priority: ExecutionPriority.postExecution)
        
        .build()
    
    // Add error handling
    let safePipeline = ErrorHandlingPipelineWrapper(
        pipeline: pipeline,
        errorHandler: { error in
            await logError(error)
        }
    )
    
    return safePipeline
}

/// Example 8: Using EmbedKit's custom builder with operators
func createEmbedKitPipeline() async throws -> any Pipeline {
    // Create dependencies
    let embedder = CoreMLTextEmbedder(
        modelIdentifier: "all-MiniLM-L6-v2",
        configuration: Configuration()
    )
    let cache = EmbeddingCache()
    let telemetry = TelemetrySystem()
    
    let handler = EmbedTextHandler(
        embedder: embedder,
        cache: cache,
        telemetry: telemetry
    )
    
    return try await EmbeddingPipelineBuilder(handler: handler)
        .addPrioritizedMiddleware(MockValidationMiddleware(), priority: ExecutionPriority.validation)
        .addMiddleware(MockCachingMiddleware())
        .addPrioritizedMiddleware(MockTelemetryMiddleware(), priority: ExecutionPriority.postExecution)
        .addMiddleware(MockGPUAccelerationMiddleware())  // Always add GPU acceleration
        .withErrorHandler { error in
            print("Embedding error: \(error)")
        }
        .build()
}

// MARK: - Pipeline Factory Examples

struct EmbeddingPipelineFactory {
    
    /// Creates a pipeline optimized for single text embedding
    static func singleTextPipeline() async throws -> any Pipeline {
        // Create dependencies
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: "all-MiniLM-L6-v2",
            configuration: Configuration()
        )
        let cache = EmbeddingCache()
        let telemetry = TelemetrySystem()
        
        let handler = EmbedTextHandler(
            embedder: embedder,
            cache: cache,
            telemetry: telemetry
        )
        
        return try await EmbeddingPipelineBuilder(handler: handler)
            .addMiddleware(MockValidationMiddleware())
            .addMiddleware(MockCachingMiddleware())
            .addMiddleware(MockTelemetryMiddleware())
            .build()
    }
    
    /// Creates a pipeline optimized for batch processing
    static func batchProcessingPipeline() async throws -> any Pipeline {
        // Create dependencies
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: "all-MiniLM-L6-v2",
            configuration: Configuration()
        )
        let cache = EmbeddingCache()
        let telemetry = TelemetrySystem()
        
        let batchHandler = BatchEmbedHandler(
            embedder: embedder,
            cache: cache,
            telemetry: telemetry
        )
        
        return try await EmbeddingPipelineBuilder(handler: batchHandler)
            .addPrioritizedMiddleware(MockBatchOptimizationMiddleware(), priority: .preExecution)
            .addMiddleware(MockParallelProcessingMiddleware())
            .addPrioritizedMiddleware(MockProgressReportingMiddleware(), priority: ExecutionPriority.postExecution)
            .build()
    }
    
    /// Creates a pipeline for streaming embeddings
    static func streamingPipeline() async throws -> any Pipeline {
        // Create dependencies
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: "all-MiniLM-L6-v2",
            configuration: Configuration()
        )
        let cache = EmbeddingCache()
        let telemetry = TelemetrySystem()
        
        let streamHandler = StreamEmbedHandler(
            embedder: embedder,
            cache: cache,
            telemetry: telemetry
        )
        
        let pipeline = try await EmbeddingPipelineBuilder(handler: streamHandler)
            .addPrioritizedMiddleware(MockBackpressureMiddleware(), priority: .preExecution)
            .addMiddleware(MockChunkingMiddleware())
            .addMiddleware(MockStreamMonitoringMiddleware())
            .build()
        
        return ErrorHandlingPipelineWrapper(
            pipeline: pipeline,
            errorHandler: { error in
                print("Stream error: \(error)")
            }
        )
    }
    
    /// Creates a high-performance pipeline optimized for throughput
    static func highPerformance(
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
    static func minimal(
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
    static func balanced(
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

// MARK: - Helper Functions

private func createPreprocessingPipeline() async throws -> any Pipeline {
    let embedder = CoreMLTextEmbedder(
        modelIdentifier: "all-MiniLM-L6-v2",
        configuration: Configuration()
    )
    let cache = EmbeddingCache()
    let telemetry = TelemetrySystem()
    let handler = EmbedTextHandler(embedder: embedder, cache: cache, telemetry: telemetry)
    
    return try await EmbeddingPipelineBuilder(handler: handler)
        .addMiddleware(MockTextPreprocessingMiddleware())
        .addMiddleware(MockTokenizationMiddleware())
        .build()
}

private func createEmbeddingPipeline() async throws -> any Pipeline {
    let embedder = CoreMLTextEmbedder(
        modelIdentifier: "all-MiniLM-L6-v2",
        configuration: Configuration()
    )
    let cache = EmbeddingCache()
    let telemetry = TelemetrySystem()
    let handler = EmbedTextHandler(embedder: embedder, cache: cache, telemetry: telemetry)
    
    return try await EmbeddingPipelineBuilder(handler: handler)
        .addMiddleware(MockEmbeddingGenerationMiddleware())
        .addMiddleware(MockNormalizationMiddleware())
        .build()
}

private func createPostprocessingPipeline() async throws -> any Pipeline {
    let embedder = CoreMLTextEmbedder(
        modelIdentifier: "all-MiniLM-L6-v2",
        configuration: Configuration()
    )
    let cache = EmbeddingCache()
    let telemetry = TelemetrySystem()
    let handler = EmbedTextHandler(embedder: embedder, cache: cache, telemetry: telemetry)
    
    return try await EmbeddingPipelineBuilder(handler: handler)
        .addMiddleware(MockPostProcessingMiddleware())
        .build()
}

private func createCPUPipeline() async throws -> any Pipeline {
    let embedder = CoreMLTextEmbedder(
        modelIdentifier: "all-MiniLM-L6-v2",
        configuration: Configuration()
    )
    let cache = EmbeddingCache()
    let telemetry = TelemetrySystem()
    let handler = BatchEmbedHandler(embedder: embedder, cache: cache, telemetry: telemetry)
    
    return try await EmbeddingPipelineBuilder(handler: handler)
        .addMiddleware(MockCPUOptimizationMiddleware())
        .build()
}

private func createGPUPipeline() async throws -> any Pipeline {
    let embedder = CoreMLTextEmbedder(
        modelIdentifier: "all-MiniLM-L6-v2",
        configuration: Configuration()
    )
    let cache = EmbeddingCache()
    let telemetry = TelemetrySystem()
    let handler = BatchEmbedHandler(embedder: embedder, cache: cache, telemetry: telemetry)
    
    return try await EmbeddingPipelineBuilder(handler: handler)
        .addMiddleware(MockGPUAccelerationMiddleware())
        .build()
}

private func createMainPipeline() async throws -> any Pipeline {
    let embedder = CoreMLTextEmbedder(
        modelIdentifier: "all-MiniLM-L6-v2",
        configuration: Configuration()
    )
    let cache = EmbeddingCache()
    let telemetry = TelemetrySystem()
    let handler = EmbedTextHandler(embedder: embedder, cache: cache, telemetry: telemetry)
    
    return try await EmbeddingPipelineBuilder(handler: handler).build()
}

private func createBasePipeline() async throws -> any Pipeline {
    let embedder = CoreMLTextEmbedder(
        modelIdentifier: "all-MiniLM-L6-v2",
        configuration: Configuration()
    )
    let cache = EmbeddingCache()
    let telemetry = TelemetrySystem()
    let handler = EmbedTextHandler(embedder: embedder, cache: cache, telemetry: telemetry)
    
    return try await EmbeddingPipelineBuilder(handler: handler).build()
}

private func checkGPUAvailability() async -> Bool {
    // Check if GPU is available
    return true
}

private func logError(_ error: Error) async {
    print("Error logged: \(error)")
}

// MARK: - Mock Middleware Implementations

struct AuthenticationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}

struct RateLimitingMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}

struct SanitizationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}

struct MonitoringMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}

struct GPUAccelerationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}

struct MockParallelProcessingMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}

struct MockBackpressureMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}

struct MockChunkingMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}

struct MockStreamMonitoringMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}

struct TextPreprocessingMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}

struct TokenizationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}

struct EmbeddingGenerationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}

struct NormalizationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}

struct MockPostProcessingMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}

struct MockCPUOptimizationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}

struct MockBatchOptimizationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}

struct MockProgressReportingMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}
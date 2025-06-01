import Foundation
import PipelineKit

// MARK: - Operator Syntax Examples for EmbedKit Pipeline Integration

/// This file demonstrates various operator syntax patterns for building
/// embedding pipelines using PipelineKit's expressive operator syntax.

// MARK: - Basic Operator Usage

/// Example 1: Simple pipeline with operator syntax
func createBasicPipeline() async throws -> any Pipeline {
    let handler = EmbedTextHandler()
    
    // Using PipelineKit's built-in pipeline function with operators
    let pipeline = try await pipeline(for: handler)
        <+ CachingMiddleware()
        <+ ValidationMiddleware()
        .build()
    
    return pipeline
}

/// Example 2: Pipeline with prioritized middleware
func createPrioritizedPipeline() async throws -> any Pipeline {
    let handler = EmbedTextHandler()
    
    // Using the middleware helper function from PipelineKit
    let pipeline = try await pipeline(for: handler)
        <++ middleware(AuthenticationMiddleware(), priority: .authentication)
        <++ middleware(ValidationMiddleware(), priority: .validation)
        <+ CachingMiddleware()  // Default priority
        <++ middleware(MonitoringMiddleware(), priority: .postExecution)
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
    
    // Parallel execution using || operator
    let parallelPipeline = cpuPipeline || gpuPipeline
    
    return parallelPipeline
}

/// Example 5: Error handling with |! operator
func createErrorHandlingPipeline() async throws -> any Pipeline {
    let mainPipeline = try await createMainPipeline()
    
    // Add error handling using |! operator
    let safePipeline = mainPipeline |! { error in
        print("Pipeline error: \(error)")
        // Additional error handling logic
    }
    
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
    let handler = EmbedTextHandler()
    
    // Build complex pipeline with operator syntax
    let pipeline = try await pipeline(for: handler)
        // Critical security layer
        <++ middleware(AuthenticationMiddleware(), priority: .authentication)
        <++ middleware(RateLimitingMiddleware(), priority: .authorization)
        
        // Validation layer
        <++ middleware(ValidationMiddleware(), priority: .validation)
        <+ SanitizationMiddleware()
        
        // Processing layer
        <+ CachingMiddleware()
        <+ GPUAccelerationMiddleware()
        
        // Monitoring layer
        <++ middleware(MonitoringMiddleware(), priority: .postExecution)
        
        .build()
    
    // Add error handling
    let safePipeline = pipeline |! { error in
        await logError(error)
    }
    
    return safePipeline
}

/// Example 8: Using EmbedKit's custom builder with operators
func createEmbedKitPipeline() async throws -> any Pipeline {
    let handler = EmbedTextHandler()
    
    return try await EmbeddingPipeline.builder(for: handler)
        <++ (ValidationMiddleware(), .validation)
        <+ CachingMiddleware()
        <++ (TelemetryMiddleware(), .postExecution)
        .when({ metadata in
            // Enable GPU for large batches
            metadata.get("batchSize", as: Int.self) ?? 0 > 100
        }, use: GPUAccelerationMiddleware())
        .withErrorHandler { error in
            print("Embedding error: \(error)")
        }
        .build()
}

// MARK: - Pipeline Factory Examples

struct EmbeddingPipelineFactory {
    
    /// Creates a pipeline optimized for single text embedding
    static func singleTextPipeline() async throws -> any Pipeline {
        try await pipeline(for: EmbedTextHandler())
            <+ ValidationMiddleware()
            <+ CachingMiddleware()
            <+ TelemetryMiddleware()
            .build()
    }
    
    /// Creates a pipeline optimized for batch processing
    static func batchProcessingPipeline() async throws -> any Pipeline {
        let batchHandler = EmbedBatchHandler()
        
        return try await pipeline(for: batchHandler)
            <++ middleware(BatchOptimizationMiddleware(), priority: .preExecution)
            <+ ParallelProcessingMiddleware()
            <++ middleware(ProgressReportingMiddleware(), priority: .postExecution)
            .build()
    }
    
    /// Creates a pipeline for streaming embeddings
    static func streamingPipeline() async throws -> any Pipeline {
        let streamHandler = EmbedStreamHandler()
        
        return try await pipeline(for: streamHandler)
            <++ middleware(BackpressureMiddleware(), priority: .authentication)
            <+ ChunkingMiddleware()
            <+ StreamMonitoringMiddleware()
            .build()
            |! { error in
                print("Stream error: \(error)")
            }
    }
}

// MARK: - Helper Functions

private func createPreprocessingPipeline() async throws -> any Pipeline {
    try await pipeline(for: EmbedTextHandler())
        <+ TextPreprocessingMiddleware()
        <+ TokenizationMiddleware()
        .build()
}

private func createEmbeddingPipeline() async throws -> any Pipeline {
    try await pipeline(for: EmbedTextHandler())
        <+ EmbeddingGenerationMiddleware()
        <+ NormalizationMiddleware()
        .build()
}

private func createPostprocessingPipeline() async throws -> any Pipeline {
    try await pipeline(for: EmbedTextHandler())
        <+ PostProcessingMiddleware()
        .build()
}

private func createCPUPipeline() async throws -> any Pipeline {
    try await pipeline(for: EmbedBatchHandler())
        <+ CPUOptimizationMiddleware()
        .build()
}

private func createGPUPipeline() async throws -> any Pipeline {
    try await pipeline(for: EmbedBatchHandler())
        <+ GPUAccelerationMiddleware()
        .build()
}

private func createMainPipeline() async throws -> any Pipeline {
    try await pipeline(for: EmbedTextHandler()).build()
}

private func createBasePipeline() async throws -> any Pipeline {
    try await pipeline(for: EmbedTextHandler()).build()
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

struct ParallelProcessingMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}

struct BackpressureMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}

struct ChunkingMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}

struct StreamMonitoringMiddleware: Middleware {
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

struct PostProcessingMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}

struct CPUOptimizationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        try await next(command, metadata)
    }
}
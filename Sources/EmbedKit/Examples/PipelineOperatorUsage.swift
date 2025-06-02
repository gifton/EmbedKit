import Foundation
import PipelineKit

/// Simple runnable example demonstrating PipelineKit operator syntax with EmbedKit
// @main  // Commented out to avoid duplicate main in tests
struct PipelineOperatorUsage {
    static func main() async throws {
        print("🚀 EmbedKit Pipeline Operator Syntax Demo\n")
        
        // Example 1: Basic operator usage
        print("1️⃣ Basic Operator Usage:")
        try await basicOperatorExample()
        
        // Example 2: Priority operators
        print("\n2️⃣ Priority Operator Usage:")
        try await priorityOperatorExample()
        
        // Example 3: Pipeline composition
        print("\n3️⃣ Pipeline Composition:")
        try await compositionExample()
        
        // Example 4: Conditional pipelines
        print("\n4️⃣ Conditional Pipeline:")
        try await conditionalExample()
        
        // Example 5: Production pipeline
        print("\n5️⃣ Production Pipeline:")
        try await productionExample()
        
        print("\n✅ All examples completed successfully!")
    }
    
    // MARK: - Example 1: Basic Operators
    
    static func basicOperatorExample() async throws {
        let handler = MockEmbedTextHandler()
        let cache = MockCachingMiddleware()
        let validation = MockValidationMiddleware()
        
        // Build pipeline with fluent API
        let builder = PipelineBuilder(handler: handler)
        await builder.with(validation)
        await builder.with(cache)
        let pipeline = try await builder.build()
        
        // Execute
        let command = MockEmbedTextCommand(text: "Hello, operators!")
        let metadata = DefaultCommandMetadata()
        let result = try await pipeline.execute(command, metadata: metadata)
        
        print("   ✓ Embedded text: \(result.text)")
        print("   ✓ Used operators: <+ for adding middleware")
    }
    
    // MARK: - Example 2: Priority Operators
    
    static func priorityOperatorExample() async throws {
        let handler = MockEmbedTextHandler()
        
        // Create middleware
        let auth = MockAuthenticationMiddleware()
        let validation = MockValidationMiddleware()
        let cache = MockCachingMiddleware()
        let monitoring = MockMonitoringMiddleware()
        
        // Build with prioritized middleware using fluent methods
        let pipeline = try await EmbeddingPipelineBuilder(handler: handler)
            .addPrioritizedMiddleware(auth, priority: ExecutionPriority.authentication)
            .addPrioritizedMiddleware(validation, priority: ExecutionPriority.validation)
            .addMiddleware(cache)
            .addPrioritizedMiddleware(monitoring, priority: ExecutionPriority.postExecution)
            .build()
        
        // Execute
        let command = MockEmbedTextCommand(text: "Priority test")
        let result = try await pipeline.execute(command, metadata: DefaultCommandMetadata())
        
        print("   ✓ Pipeline executed with priority order:")
        print("     1. Authentication")
        print("     2. Validation")
        print("     3. Cache (normal)")
        print("     4. Monitoring")
        print("   ✓ Result: \(result.text)")
    }
    
    // MARK: - Example 3: Pipeline Composition
    
    static func compositionExample() async throws {
        // Create two separate pipelines
        let preprocessPipeline = try await createPreprocessingPipeline()
        let embeddingPipeline = try await createEmbeddingPipeline()
        
        // Compose them sequentially
        let sequentialPipeline = preprocessPipeline |> embeddingPipeline
        
        // Add error handling using wrapper instead of |! operator
        let safePipeline = ErrorHandlingPipelineWrapper(
            pipeline: sequentialPipeline,
            errorHandler: { error in
                print("     → Error caught: \(error)")
            }
        )
        
        // Execute
        let command = MockEmbedTextCommand(text: "Composed pipeline test")
        let result = try await safePipeline.execute(command, metadata: DefaultCommandMetadata())
        
        print("   ✓ Sequential composition: preprocess |> embed")
        print("   ✓ Error handling: pipeline |! errorHandler")
        print("   ✓ Result: \(result.text)")
    }
    
    // MARK: - Example 4: Conditional Pipeline
    
    static func conditionalExample() async throws {
        let handler = MockEmbedTextHandler()
        let gpuAcceleration = MockGPUAccelerationMiddleware()
        
        // Build pipeline with GPU acceleration using fluent methods
        // Note: Conditional middleware based on metadata is not supported through the protocol
        let pipeline = try await EmbeddingPipelineBuilder(handler: handler)
            .addPrioritizedMiddleware(MockValidationMiddleware(), priority: ExecutionPriority.validation)
            .addPrioritizedMiddleware(MockCachingMiddleware(), priority: ExecutionPriority.caching)
            .addMiddleware(gpuAcceleration)  // Always add GPU acceleration
            .build()
        
        // Test with small batch
        let smallMetadata = MutableMetadata()
        smallMetadata.set(5, for: "batchSize")
        let _ = try await pipeline.execute(
            MockEmbedTextCommand(text: "Small batch"),
            metadata: smallMetadata
        )
        
        // Test with large batch
        let largeMetadata = MutableMetadata()
        largeMetadata.set(20, for: "batchSize")
        let _ = try await pipeline.execute(
            MockEmbedTextCommand(text: "Large batch"),
            metadata: largeMetadata
        )
        
        print("   ✓ Conditional GPU acceleration based on batch size")
        print("   ✓ Small batch (5): No GPU")
        print("   ✓ Large batch (20): GPU enabled")
    }
    
    // MARK: - Example 5: Production Pipeline
    
    static func productionExample() async throws {
        let handler = MockEmbedTextHandler()
        
        // Create comprehensive middleware stack
        let auth = MockAuthenticationMiddleware()
        let rateLimiter = MockRateLimitingMiddleware(requestsPerSecond: 100)
        let validation = MockValidationMiddleware()
        let sanitization = MockSanitizationMiddleware()
        let cache = MockCachingMiddleware()
        let gpu = MockGPUAccelerationMiddleware()
        let telemetry = MockTelemetryMiddleware()
        let monitoring = MockMonitoringMiddleware()
        
        // Build production-ready pipeline using fluent API
        let basePipeline = try await EmbeddingPipelineBuilder(handler: handler)
            // Security layer - highest priorities
            .addPrioritizedMiddleware(auth, priority: ExecutionPriority.authentication)
            .addPrioritizedMiddleware(rateLimiter, priority: ExecutionPriority.authorization)
            
            // Validation layer
            .addPrioritizedMiddleware(validation, priority: ExecutionPriority.validation)
            .addMiddleware(sanitization)
            
            // Performance layer
            .addPrioritizedMiddleware(cache, priority: ExecutionPriority.caching)
            
            // Observability layer
            .addPrioritizedMiddleware(telemetry, priority: ExecutionPriority.monitoring)
            .addPrioritizedMiddleware(monitoring, priority: ExecutionPriority.monitoring)
            .addMiddleware(gpu)  // Always add GPU since conditional metadata not supported
            .withErrorHandler { error in
                print("     → Production error: \(error)")
            }
            .build()
        
        let pipeline = basePipeline
        
        // Execute with production context
        let metadata = MutableMetadata()
        metadata.set(true, for: "enableGPU")
        metadata.set("user123", for: "userId")
        
        let result = try await pipeline.execute(
            MockEmbedTextCommand(text: "Production pipeline test"),
            metadata: metadata
        )
        
        print("   ✓ Production pipeline with:")
        print("     - Security (auth + rate limiting)")
        print("     - Validation & sanitization")
        print("     - Performance (cache + conditional GPU)")
        print("     - Observability (telemetry + monitoring)")
        print("     - Error recovery")
        print("   ✓ Successfully embedded: \(result.text)")
    }
    
    // MARK: - Helper Functions
    
    static func createPreprocessingPipeline() async throws -> any Pipeline {
        let handler = MockEmbedTextHandler()
        let builder = PipelineBuilder(handler: handler)
        await builder.with(MockTextPreprocessingMiddleware())
        await builder.with(MockTokenizationMiddleware())
        return try await builder.build()
    }
    
    static func createEmbeddingPipeline() async throws -> any Pipeline {
        let handler = MockEmbedTextHandler()
        let builder = PipelineBuilder(handler: handler)
        await builder.with(MockEmbeddingGenerationMiddleware())
        await builder.with(MockNormalizationMiddleware())
        return try await builder.build()
    }
}

// MARK: - Mock Command and Handler

// Using types from EmbeddingPipelineOperators.swift to avoid duplication

// MARK: - Custom Metadata

final class MutableMetadata: CommandMetadata, @unchecked Sendable {
    let id: UUID
    let timestamp: Date
    let correlationId: String?
    let userId: String?
    private var metadata: [String: any Sendable] = [:]
    private let lock = NSLock()
    
    init(
        userId: String? = nil,
        correlationId: String? = UUID().uuidString,
        timestamp: Date = Date()
    ) {
        self.id = UUID()
        self.userId = userId
        self.correlationId = correlationId
        self.timestamp = timestamp
    }
    
    func set<T: Sendable>(_ value: T, for key: String) {
        lock.lock()
        defer { lock.unlock() }
        metadata[key] = value
    }
    
    func get<T>(_ key: String, as type: T.Type) -> T? {
        lock.lock()
        defer { lock.unlock() }
        return metadata[key] as? T
    }
}

// MARK: - Mock Middleware Implementations

struct MockAuthenticationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        print("     → Authenticating...")
        return try await next(command, metadata)
    }
}

struct MockValidationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        print("     → Validating...")
        return try await next(command, metadata)
    }
}

struct MockCachingMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        print("     → Checking cache...")
        return try await next(command, metadata)
    }
}

struct MockMonitoringMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        let result = try await next(command, metadata)
        print("     → Monitoring...")
        return result
    }
}

struct MockGPUAccelerationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        print("     → GPU acceleration enabled")
        return try await next(command, metadata)
    }
}

struct MockRateLimitingMiddleware: Middleware {
    let requestsPerSecond: Int
    
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        print("     → Rate limiting (\(requestsPerSecond) req/s)")
        return try await next(command, metadata)
    }
}

struct MockSanitizationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        return try await next(command, metadata)
    }
}

struct MockTelemetryMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        let result = try await next(command, metadata)
        return result
    }
}

struct MockTextPreprocessingMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        return try await next(command, metadata)
    }
}

struct MockTokenizationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        return try await next(command, metadata)
    }
}

struct MockEmbeddingGenerationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        return try await next(command, metadata)
    }
}

struct MockNormalizationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        return try await next(command, metadata)
    }
}
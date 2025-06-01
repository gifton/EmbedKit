import Foundation
import PipelineKit
@preconcurrency import EmbedKit

/// Simple runnable example demonstrating PipelineKit operator syntax with EmbedKit
@main
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
        let cache = CachingMiddleware()
        let validation = ValidationMiddleware()
        
        // Build pipeline with <+ operator
        let pipeline = try await pipeline(for: handler)
            <+ validation
            <+ cache
            .build()
        
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
        let auth = AuthenticationMiddleware()
        let validation = ValidationMiddleware()
        let cache = CachingMiddleware()
        let monitoring = MonitoringMiddleware()
        
        // Build with prioritized middleware using <++
        let pipeline = try await pipeline(for: handler)
            <++ middleware(auth, priority: .authentication)        // Runs first
            <++ middleware(validation, priority: .validation)       // Runs second
            <+ cache                                               // Normal priority
            <++ middleware(monitoring, priority: .postExecution)   // Runs last
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
        
        // Add error handling
        let safePipeline = sequentialPipeline |! { error in
            print("     → Error caught: \(error)")
        }
        
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
        let gpuAcceleration = GPUAccelerationMiddleware()
        
        // Build pipeline with conditional GPU acceleration using EmbedKit's builder
        let pipeline = try await EmbeddingPipeline.builder(for: handler)
            <+ ValidationMiddleware()
            <+ CachingMiddleware()
            .when({ metadata in
                // Enable GPU for large batches
                metadata.get("batchSize", as: Int.self) ?? 0 > 10
            }, use: gpuAcceleration)
            .build()
        
        // Test with small batch
        var smallMetadata = DefaultCommandMetadata()
        smallMetadata.set(5, for: "batchSize")
        let smallResult = try await pipeline.execute(
            MockEmbedTextCommand(text: "Small batch"),
            metadata: smallMetadata
        )
        
        // Test with large batch
        var largeMetadata = DefaultCommandMetadata()
        largeMetadata.set(20, for: "batchSize")
        let largeResult = try await pipeline.execute(
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
        let auth = AuthenticationMiddleware()
        let rateLimiter = RateLimitingMiddleware(requestsPerSecond: 100)
        let validation = ValidationMiddleware()
        let sanitization = SanitizationMiddleware()
        let cache = CachingMiddleware()
        let gpu = GPUAccelerationMiddleware()
        let telemetry = TelemetryMiddleware()
        let monitoring = MonitoringMiddleware()
        
        // Build production-ready pipeline
        let pipeline = try await EmbeddingPipeline.builder(for: handler)
            // Security layer
            <++ (auth, .authentication)
            <++ (rateLimiter, .authorization)
            
            // Validation layer
            <++ (validation, .validation)
            <+ sanitization
            
            // Performance layer
            <+ cache
            .when({ metadata in
                metadata.get("enableGPU", as: Bool.self) ?? false
            }, use: gpu)
            
            // Observability layer
            <++ (telemetry, .postExecution)
            <++ (monitoring, .postExecution)
            
            .withErrorHandler { error in
                print("     → Production error: \(error)")
            }
            .build()
        
        // Execute with production context
        var metadata = DefaultCommandMetadata()
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
        return try await pipeline(for: handler)
            <+ TextPreprocessingMiddleware()
            <+ TokenizationMiddleware()
            .build()
    }
    
    static func createEmbeddingPipeline() async throws -> any Pipeline {
        let handler = MockEmbedTextHandler()
        return try await pipeline(for: handler)
            <+ EmbeddingGenerationMiddleware()
            <+ NormalizationMiddleware()
            .build()
    }
}

// MARK: - Mock Command and Handler

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
        print("     → Processing: \(command.text)")
        return MockEmbedResult(text: command.text)
    }
}

// MARK: - Mock Middleware Implementations

struct AuthenticationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        print("     → Authenticating...")
        return try await next(command, metadata)
    }
}

struct ValidationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        print("     → Validating...")
        return try await next(command, metadata)
    }
}

struct CachingMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        print("     → Checking cache...")
        return try await next(command, metadata)
    }
}

struct MonitoringMiddleware: Middleware {
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

struct GPUAccelerationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        print("     → GPU acceleration enabled")
        return try await next(command, metadata)
    }
}

struct RateLimitingMiddleware: Middleware {
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

struct SanitizationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        return try await next(command, metadata)
    }
}

struct TelemetryMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        let result = try await next(command, metadata)
        return result
    }
}

struct TextPreprocessingMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        return try await next(command, metadata)
    }
}

struct TokenizationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        return try await next(command, metadata)
    }
}

struct EmbeddingGenerationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        return try await next(command, metadata)
    }
}

struct NormalizationMiddleware: Middleware {
    func execute<T: Command>(
        _ command: T,
        metadata: CommandMetadata,
        next: @Sendable (T, CommandMetadata) async throws -> T.Result
    ) async throws -> T.Result {
        return try await next(command, metadata)
    }
}
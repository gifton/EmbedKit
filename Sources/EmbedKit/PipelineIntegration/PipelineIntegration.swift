import Foundation
import PipelineKit

/// Main module for PipelineKit integration
/// 
/// This module provides seamless integration between EmbedKit and PipelineKit,
/// enabling command-driven, middleware-enhanced text embedding operations.
public enum PipelineIntegration {
    
    /// The current version of the PipelineKit integration
    public static let version = "1.0.0"
    
    /// Check if PipelineKit integration is available
    public static var isAvailable: Bool {
        return true
    }
    
    /// Create a default embedding pipeline with recommended settings
    public static func createDefaultPipeline(
        modelIdentifier: String = "all-MiniLM-L6-v2"
    ) async throws -> EmbeddingPipeline {
        let modelManager = DefaultEmbeddingModelManager()
        // Create mock embedder for example (in production, load actual model)
        let embedder = MockTextEmbedder(dimensions: 384)
        try await embedder.loadModel()
        
        return try await EmbeddingPipelineFactory.balanced(
            embedder: embedder,
            modelManager: modelManager
        )
    }
    
    /// Create a custom pipeline with specific configuration
    public static func createCustomPipeline(
        embedder: any TextEmbedder,
        modelManager: EmbeddingModelManager,
        configuration: EmbeddingPipeline.Configuration
    ) async throws -> EmbeddingPipeline {
        return try await EmbeddingPipeline(
            embedder: embedder,
            modelManager: modelManager,
            configuration: configuration
        )
    }
    
    /// Register EmbedKit handlers with an existing command bus using fluent API
    public static func registerHandlers(
        with busBuilder: CommandBusBuilder,
        embedder: any TextEmbedder,
        modelManager: EmbeddingModelManager,
        cache: EmbeddingCache,
        telemetry: TelemetrySystem
    ) async throws -> CommandBusBuilder {
        // Register all handlers using fluent API
        return try await busBuilder
            .with(EmbedTextCommand.self, handler: EmbedTextHandler(embedder: embedder, cache: cache, telemetry: telemetry))
            .with(BatchEmbedCommand.self, handler: BatchEmbedHandler(embedder: embedder, cache: cache, telemetry: telemetry))
            .with(StreamEmbedCommand.self, handler: StreamEmbedHandler(embedder: embedder, cache: cache, telemetry: telemetry))
            .with(LoadModelCommand.self, handler: LoadModelHandler(modelManager: modelManager, telemetry: telemetry))
            .with(SwapModelCommand.self, handler: SwapModelHandler(modelManager: modelManager, telemetry: telemetry))
            .with(UnloadModelCommand.self, handler: UnloadModelHandler(modelManager: modelManager, cache: cache))
            .with(ClearCacheCommand.self, handler: ClearCacheHandler(cache: cache))
            .with(PreloadCacheCommand.self, handler: PreloadCacheHandler(embedder: embedder, cache: cache))
    }
    
    /// Create a middleware stack for embedding operations
    public static func createMiddlewareStack(
        cache: EmbeddingCache,
        telemetry: TelemetrySystem,
        configuration: MiddlewareConfiguration = .default
    ) throws -> [any Middleware] {
        var middleware: [any Middleware] = []
        
        // Always add validation
        middleware.append(
            EmbeddingValidationMiddleware(
                maxTextLength: configuration.maxTextLength,
                maxBatchSize: configuration.maxBatchSize
            )
        )
        
        // Add telemetry if enabled
        if configuration.enableTelemetry {
            middleware.append(TelemetryMiddleware(telemetry: telemetry))
        }
        
        // Add cache if enabled
        if configuration.enableCache {
            middleware.append(EmbeddingCacheMiddleware(cache: cache))
        }
        
        // Add GPU acceleration if enabled and available
        if configuration.enableGPUAcceleration {
            if let metalMiddleware = try? MetalAccelerationMiddleware() {
                middleware.append(metalMiddleware)
            }
        }
        
        // Add rate limiting if enabled
        if configuration.enableRateLimiting {
            middleware.append(
                EmbeddingRateLimitMiddleware(
                    requestsPerSecond: configuration.requestsPerSecond,
                    burstSize: configuration.burstSize
                )
            )
        }
        
        // Add monitoring if enabled
        if configuration.enableMonitoring {
            middleware.append(
                EmbeddingMonitoringMiddleware(
                    telemetry: telemetry,
                    alertThresholds: configuration.alertThresholds
                )
            )
        }
        
        return middleware
    }
}

// MARK: - Middleware Configuration

/// Configuration for middleware stack
public struct MiddlewareConfiguration: Sendable {
    public let enableCache: Bool
    public let enableGPUAcceleration: Bool
    public let enableRateLimiting: Bool
    public let enableMonitoring: Bool
    public let enableTelemetry: Bool
    public let maxTextLength: Int
    public let maxBatchSize: Int
    public let requestsPerSecond: Double
    public let burstSize: Int
    public let alertThresholds: EmbeddingMonitoringMiddleware.AlertThresholds
    
    public init(
        enableCache: Bool = true,
        enableGPUAcceleration: Bool = true,
        enableRateLimiting: Bool = true,
        enableMonitoring: Bool = true,
        enableTelemetry: Bool = true,
        maxTextLength: Int = 10_000,
        maxBatchSize: Int = 1000,
        requestsPerSecond: Double = 100,
        burstSize: Int = 200,
        alertThresholds: EmbeddingMonitoringMiddleware.AlertThresholds = .init()
    ) {
        self.enableCache = enableCache
        self.enableGPUAcceleration = enableGPUAcceleration
        self.enableRateLimiting = enableRateLimiting
        self.enableMonitoring = enableMonitoring
        self.enableTelemetry = enableTelemetry
        self.maxTextLength = maxTextLength
        self.maxBatchSize = maxBatchSize
        self.requestsPerSecond = requestsPerSecond
        self.burstSize = burstSize
        self.alertThresholds = alertThresholds
    }
    
    /// Default configuration with all features enabled
    public static let `default` = MiddlewareConfiguration()
    
    /// High-performance configuration with minimal overhead
    public static let highPerformance = MiddlewareConfiguration(
        enableCache: true,
        enableGPUAcceleration: true,
        enableRateLimiting: false,
        enableMonitoring: false,
        enableTelemetry: false,
        requestsPerSecond: 1000,
        burstSize: 2000
    )
    
    /// Development configuration with extensive logging
    public static let development = MiddlewareConfiguration(
        enableCache: true,
        enableGPUAcceleration: true,
        enableRateLimiting: true,
        enableMonitoring: true,
        enableTelemetry: true,
        maxTextLength: 5_000,
        maxBatchSize: 100,
        requestsPerSecond: 10,
        burstSize: 20
    )
    
    /// Minimal configuration for testing
    public static let minimal = MiddlewareConfiguration(
        enableCache: false,
        enableGPUAcceleration: false,
        enableRateLimiting: false,
        enableMonitoring: false,
        enableTelemetry: false
    )
}

// MARK: - Integration Helpers

public extension PipelineIntegration {
    
    /// Quick start helper to create and configure a pipeline
    static func quickStart(
        modelIdentifier: String = "all-MiniLM-L6-v2",
        configuration: EmbeddingPipeline.Configuration = .init()
    ) async throws -> (pipeline: EmbeddingPipeline, cleanup: () async -> Void) {
        let logger = EmbedKitLogger.embeddings()
        logger.start("Quick start pipeline setup", details: modelIdentifier)
        
        let pipeline = try await createDefaultPipeline(modelIdentifier: modelIdentifier)
        
        let cleanup: () async -> Void = {
            logger.info("Cleaning up pipeline resources")
            _ = try? await pipeline.unloadModel(clearCache: true)
        }
        
        logger.success("Pipeline ready")
        return (pipeline, cleanup)
    }
    
    /// Validate integration setup
    static func validateSetup() async throws {
        let logger = EmbedKitLogger.custom("Integration")
        logger.start("Validating PipelineKit integration")
        
        // Check if we can create basic components
        do {
            let modelManager = DefaultEmbeddingModelManager()
            let _ = EmbeddingCache()
            let _ = TelemetrySystem()
            
            logger.success("Core components created")
            
            // Try to create a minimal pipeline
            let _ = try await modelManager.loadModel(
                from: URL(fileURLWithPath: "/tmp/models/all-MiniLM-L6-v2.mlmodel"),
                identifier: ModelIdentifier.miniLM_L6_v2,
                configuration: nil
            )
            
            // Get the actual embedder from the model manager
            guard let embedder = await modelManager.getModel(identifier: ModelIdentifier.miniLM_L6_v2) else {
                throw ContextualEmbeddingError.modelNotLoaded(
                    context: ErrorContext(
                        operation: .modelLoading,
                        modelIdentifier: ModelIdentifier.miniLM_L6_v2,
                        metadata: ErrorMetadata()
                            .with(key: "reason", value: "Model not found in manager"),
                        sourceLocation: SourceLocation()
                    )
                )
            }
            
            let pipeline = try await createCustomPipeline(
                embedder: embedder,
                modelManager: modelManager,
                configuration: .init(
                    enableCache: false,
                    enableGPUAcceleration: false,
                    enableRateLimiting: false,
                    enableMonitoring: false
                )
            )
            
            // Test basic embedding
            let testResult = try await pipeline.embed("Integration test")
            
            logger.success(
                "Integration validated",
                context: "dimensions: \(testResult.embedding.dimensions)"
            )
            
            // Cleanup
            _ = try await pipeline.unloadModel(clearCache: true)
            
        } catch {
            logger.error("Integration validation failed", error: error)
            throw error
        }
    }
}

// MARK: - Public Exports

// Re-export commonly used types for convenience
public typealias EmbedKitCommand = Command
public typealias EmbedKitHandler = CommandHandler
public typealias EmbedKitMiddleware = Middleware
public typealias EmbedKitPipeline = Pipeline

// Re-export result types
public typealias EmbedKitResult<T: Sendable> = Result<T, Error>
public typealias EmbedKitAsyncResult<T: Sendable> = AsyncThrowingStream<T, Error>

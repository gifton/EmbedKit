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
        let modelManager = EmbeddingModelManager()
        let embedder = try await modelManager.loadModel(
            identifier: modelIdentifier,
            configuration: EmbeddingConfiguration()
        )
        
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
    
    /// Register EmbedKit handlers with an existing command bus
    public static func registerHandlers(
        with busBuilder: CommandBusBuilder,
        embedder: any TextEmbedder,
        modelManager: EmbeddingModelManager,
        cache: EmbeddingCache,
        telemetry: TelemetrySystem
    ) async {
        // Register embedding handlers
        await busBuilder.register(
            EmbedTextHandler(embedder: embedder, cache: cache, telemetry: telemetry),
            for: EmbedTextCommand.self
        )
        
        await busBuilder.register(
            BatchEmbedHandler(embedder: embedder, cache: cache, telemetry: telemetry),
            for: BatchEmbedCommand.self
        )
        
        await busBuilder.register(
            StreamEmbedHandler(embedder: embedder, cache: cache, telemetry: telemetry),
            for: StreamEmbedCommand.self
        )
        
        // Register model management handlers
        await busBuilder.register(
            LoadModelHandler(modelManager: modelManager, telemetry: telemetry),
            for: LoadModelCommand.self
        )
        
        await busBuilder.register(
            SwapModelHandler(modelManager: modelManager, telemetry: telemetry),
            for: SwapModelCommand.self
        )
        
        await busBuilder.register(
            UnloadModelHandler(modelManager: modelManager, cache: cache),
            for: UnloadModelCommand.self
        )
        
        // Register cache management handlers
        await busBuilder.register(
            ClearCacheHandler(cache: cache),
            for: ClearCacheCommand.self
        )
        
        await busBuilder.register(
            PreloadCacheHandler(embedder: embedder, cache: cache),
            for: PreloadCacheCommand.self
        )
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
public struct MiddlewareConfiguration {
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
            let modelManager = EmbeddingModelManager()
            let cache = EmbeddingCache()
            let telemetry = TelemetrySystem()
            
            logger.success("Core components created")
            
            // Try to create a minimal pipeline
            let embedder = try await modelManager.loadModel(
                identifier: "all-MiniLM-L6-v2",
                configuration: EmbeddingConfiguration()
            )
            
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
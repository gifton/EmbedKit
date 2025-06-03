/// EmbedKit - On-device text embedding generation for Apple platforms
///
/// EmbedKit provides high-performance text embedding generation with hardware acceleration,
/// designed to work seamlessly with PipelineKit's command-based architecture.
///
/// ## Key Features
/// - Actor-based thread-safe API
/// - Core ML and Metal acceleration
/// - Smart caching with LRU eviction
/// - Streaming support for large datasets
/// - PipelineKit integration
/// - Production-ready error handling and graceful degradation
/// - Comprehensive telemetry and monitoring

// MARK: - Core Protocols and Types
@_exported import Foundation

// MARK: - Core Embedding Types
public typealias EmbedKit_EmbeddingVector = EmbeddingVector
public typealias EmbedKit_TextEmbedder = TextEmbedder
public typealias EmbedKit_Configuration = Configuration
public typealias EmbedKit_ModelIdentifier = ModelIdentifier

// MARK: - Error Handling
public typealias EmbedKit_ErrorContext = ErrorContext
public typealias EmbedKit_ContextualError = ContextualError
public typealias EmbedKit_GracefulDegradationManager = GracefulDegradationManager

// MARK: - Telemetry and Monitoring
public typealias EmbedKit_TelemetrySystem = TelemetrySystem
public typealias EmbedKit_TelemetryEvent = TelemetryEvent
public typealias EmbedKit_MetricSummary = MetricSummary

// MARK: - Global Services
/// Global telemetry system instance
public let telemetry = TelemetrySystem()

/// Global graceful degradation manager
public let degradationManager = GracefulDegradationManager()

// MARK: - Convenience API
/// High-level API for common embedding operations
public enum EmbedKit {
    /// Create a production-ready text embedder with error handling and telemetry
    public static func createEmbedder(
        modelIdentifier: ModelIdentifier = .default,
        configuration: Configuration = Configuration()
    ) -> CoreMLTextEmbedder {
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: modelIdentifier, 
            configuration: configuration
        )
        
        Task {
            await telemetry.recordEvent(TelemetryEvent(
                name: "embedder_created",
                description: "Created embedder with model \(modelIdentifier.rawValue)",
                severity: .info,
                metadata: ["model": modelIdentifier.rawValue]
            ))
        }
        
        return embedder
    }
    
    /// Create a streaming embedder with full production features
    public static func createStreamingEmbedder<Embedder: TextEmbedder>(
        embedder: Embedder,
        configuration: StreamingEmbedder<Embedder>.StreamingConfiguration = StreamingEmbedder<Embedder>.StreamingConfiguration()
    ) -> StreamingEmbedder<Embedder> {
        return StreamingEmbedder(embedder: embedder, configuration: configuration)
    }
    
    /// Create a hot-swappable model manager
    public static func createModelManager(
        registry: ModelVersionRegistry = ModelVersionRegistry(),
        maxConcurrentModels: Int = 3
    ) -> HotSwappableModelManager {
        return HotSwappableModelManager(
            registry: registry,
            maxConcurrentModels: maxConcurrentModels
        )
    }
    
    /// Get current system health status
    public static func getHealthStatus() async -> HealthStatus {
        let systemMetrics = await telemetry.getSystemMetrics()
        let degradationStatus = await degradationManager.getDegradationStatus()
        
        return HealthStatus(
            errorCount: 0, // No longer tracking via ErrorHandlingSystem
            memoryUsage: systemMetrics.memoryUsage,
            degradationLevel: degradationStatus.values.max() ?? .normal,
            timestamp: Date()
        )
    }
}

/// System health status
public struct HealthStatus {
    public let errorCount: Int
    public let memoryUsage: Double
    public let degradationLevel: GracefulDegradationManager.DegradationLevel
    public let timestamp: Date
    
    public var isHealthy: Bool {
        errorCount < 10 && 
        memoryUsage < 0.8 && 
        degradationLevel.rawValue <= GracefulDegradationManager.DegradationLevel.reduced.rawValue
    }
    
    public init(errorCount: Int, memoryUsage: Double, degradationLevel: GracefulDegradationManager.DegradationLevel, timestamp: Date) {
        self.errorCount = errorCount
        self.memoryUsage = memoryUsage
        self.degradationLevel = degradationLevel
        self.timestamp = timestamp
    }
}
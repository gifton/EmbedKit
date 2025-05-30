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
public typealias EmbedKit_EmbeddingConfiguration = EmbeddingConfiguration

// MARK: - Error Handling
public typealias EmbedKit_ErrorHandlingSystem = ErrorHandlingSystem
public typealias EmbedKit_GracefulDegradationManager = GracefulDegradationManager
public typealias EmbedKit_ErrorContext = ErrorContext

// MARK: - Telemetry and Monitoring
public typealias EmbedKit_TelemetrySystem = TelemetrySystem
public typealias EmbedKit_TelemetryEvent = TelemetryEvent
public typealias EmbedKit_MetricSummary = MetricSummary

// MARK: - Global Services
/// Global error handling system instance
public let errorHandler = ErrorHandlingSystem()

/// Global telemetry system instance
public let telemetry = TelemetrySystem()

/// Global graceful degradation manager
public let degradationManager = GracefulDegradationManager()

// MARK: - Convenience API
/// High-level API for common embedding operations
public enum EmbedKit {
    /// Create a production-ready text embedder with error handling and telemetry
    public static func createEmbedder(
        modelIdentifier: String = "all-MiniLM-L6-v2",
        configuration: EmbeddingConfiguration = EmbeddingConfiguration()
    ) -> CoreMLTextEmbedder {
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: modelIdentifier, 
            configuration: configuration
        )
        
        Task {
            await telemetry.recordEvent(TelemetryEvent(
                name: "embedder_created",
                description: "Created embedder with model \(modelIdentifier)",
                severity: .info,
                metadata: ["model": modelIdentifier]
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
        let errorStats = await errorHandler.getErrorStatistics()
        let systemMetrics = await telemetry.getSystemMetrics()
        let degradationStatus = await degradationManager.getDegradationStatus()
        
        return HealthStatus(
            errorCount: errorStats.recentErrorCount,
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
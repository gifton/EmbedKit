import Foundation

// MARK: - Unified Configuration System

/// Unified configuration for all EmbedKit operations.
///
/// This configuration system consolidates all settings and provides:
/// - Type-safe model configuration
/// - Resource management settings
/// - Performance tuning options
/// - Monitoring and telemetry settings
///
/// ## Example
///
/// ```swift
/// let config = Configuration(
///     model: .miniLM_L6_v2,
///     resources: .balanced,
///     performance: .optimized,
///     monitoring: .enabled
/// )
/// ```
public struct Configuration: Sendable {
    /// Model configuration
    public let model: ModelConfiguration
    
    /// Resource management configuration
    public let resources: ResourceConfiguration
    
    /// Performance configuration
    public let performance: PerformanceConfiguration
    
    /// Monitoring and telemetry configuration
    public let monitoring: MonitoringConfiguration
    
    /// Cache configuration
    public let cache: CacheConfiguration
    
    /// Error handling configuration
    public let errorHandling: ErrorHandlingConfiguration
    
    // MARK: - Initialization
    
    public init(
        model: ModelConfiguration,
        resources: ResourceConfiguration = ResourceConfiguration(),
        performance: PerformanceConfiguration = PerformanceConfiguration(),
        monitoring: MonitoringConfiguration = MonitoringConfiguration(),
        cache: CacheConfiguration = CacheConfiguration(),
        errorHandling: ErrorHandlingConfiguration = ErrorHandlingConfiguration()
    ) {
        self.model = model
        self.resources = resources
        self.performance = performance
        self.monitoring = monitoring
        self.cache = cache
        self.errorHandling = errorHandling
    }
    
    // MARK: - Factory Methods
    
    /// Create default configuration for a specific model
    public static func `default`(for modelIdentifier: ModelIdentifier) -> Configuration {
        Configuration(
            model: ModelConfiguration.custom(
                identifier: modelIdentifier,
                maxSequenceLength: 512
            )
        )
    }
    
    /// Create high-performance configuration for a specific model
    public static func highPerformance(
        for modelIdentifier: ModelIdentifier,
        maxSequenceLength: Int = 256
    ) -> Configuration {
        Configuration(
            model: ModelConfiguration.highPerformance(
                identifier: modelIdentifier,
                maxSequenceLength: maxSequenceLength
            ),
            resources: .unlimited,
            performance: .maximum,
            cache: .aggressive
        )
    }
    
    /// Create memory-optimized configuration for a specific model
    public static func memoryOptimized(
        for modelIdentifier: ModelIdentifier,
        maxSequenceLength: Int = 128
    ) -> Configuration {
        Configuration(
            model: ModelConfiguration.memoryOptimized(
                identifier: modelIdentifier,
                maxSequenceLength: maxSequenceLength
            ),
            resources: .constrained,
            performance: .balanced,
            cache: .minimal
        )
    }
    
    /// Create production configuration for miniLM model
    public static let productionMiniLM = Configuration(
        model: ModelConfiguration.miniLM_L6_v2(),
        resources: .managed,
        performance: .optimized,
        monitoring: .comprehensive,
        errorHandling: .strict
    )
}
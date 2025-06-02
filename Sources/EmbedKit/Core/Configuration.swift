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
        model: ModelConfiguration = ModelConfiguration(),
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
    
    // MARK: - Presets
    
    /// Default balanced configuration
    public static let `default` = Configuration()
    
    /// High-performance configuration
    public static let highPerformance = Configuration(
        model: .highPerformance,
        resources: .unlimited,
        performance: .maximum,
        cache: .aggressive
    )
    
    /// Memory-optimized configuration
    public static let memoryOptimized = Configuration(
        model: .memoryOptimized,
        resources: .constrained,
        performance: .balanced,
        cache: .minimal
    )
    
    /// Production configuration with full monitoring
    public static let production = Configuration(
        model: .production,
        resources: .managed,
        performance: .optimized,
        monitoring: .comprehensive,
        errorHandling: .strict
    )
}

// MARK: - Model Configuration

/// Configuration specific to model loading and inference
public struct ModelConfiguration: Sendable {
    /// Model identifier
    public let identifier: ModelIdentifier
    
    /// Maximum sequence length for input
    public let maxSequenceLength: Int
    
    /// Whether to normalize embeddings
    public let normalizeEmbeddings: Bool
    
    /// Pooling strategy for embeddings
    public let poolingStrategy: PoolingStrategy
    
    /// Model loading options
    public let loadingOptions: LoadingOptions
    
    /// Compute units to use
    public let computeUnits: ComputeUnits
    
    public init(
        identifier: ModelIdentifier = .default,
        maxSequenceLength: Int = 512,
        normalizeEmbeddings: Bool = true,
        poolingStrategy: PoolingStrategy = .mean,
        loadingOptions: LoadingOptions = LoadingOptions(),
        computeUnits: ComputeUnits = .auto
    ) {
        self.identifier = identifier
        self.maxSequenceLength = maxSequenceLength
        self.normalizeEmbeddings = normalizeEmbeddings
        self.poolingStrategy = poolingStrategy
        self.loadingOptions = loadingOptions
        self.computeUnits = computeUnits
    }
    
    // MARK: - Presets
    
    public static let `default` = ModelConfiguration()
    
    public static let highPerformance = ModelConfiguration(
        maxSequenceLength: 256,
        computeUnits: .cpuAndGPU
    )
    
    public static let memoryOptimized = ModelConfiguration(
        maxSequenceLength: 128,
        loadingOptions: LoadingOptions(preloadWeights: false),
        computeUnits: .cpuOnly
    )
    
    public static let production = ModelConfiguration(
        identifier: .miniLM_L6_v2,
        maxSequenceLength: 512,
        normalizeEmbeddings: true,
        poolingStrategy: .mean,
        loadingOptions: LoadingOptions(
            preloadWeights: true,
            enableOptimizations: true,
            verifyIntegrity: true
        ),
        computeUnits: .auto
    )
}

/// Model loading options
public struct LoadingOptions: Sendable {
    /// Whether to preload model weights
    public let preloadWeights: Bool
    
    /// Whether to enable CoreML optimizations
    public let enableOptimizations: Bool
    
    /// Whether to verify model integrity
    public let verifyIntegrity: Bool
    
    /// Custom model URL (if not using bundled model)
    public let customModelURL: URL?
    
    public init(
        preloadWeights: Bool = true,
        enableOptimizations: Bool = true,
        verifyIntegrity: Bool = false,
        customModelURL: URL? = nil
    ) {
        self.preloadWeights = preloadWeights
        self.enableOptimizations = enableOptimizations
        self.verifyIntegrity = verifyIntegrity
        self.customModelURL = customModelURL
    }
}

/// Compute units for model execution
public enum ComputeUnits: String, Sendable {
    case cpuOnly = "cpu_only"
    case cpuAndGPU = "cpu_and_gpu"
    case cpuAndNeuralEngine = "cpu_and_neural_engine"
    case all = "all"
    case auto = "auto"
}

// MARK: - Resource Configuration

/// Configuration for resource management
public struct ResourceConfiguration: Sendable {
    /// Maximum memory usage in bytes (nil = unlimited)
    public let maxMemoryUsage: Int?
    
    /// Maximum concurrent operations
    public let maxConcurrentOperations: Int
    
    /// Batch size for processing
    public let batchSize: Int
    
    /// Memory pressure handling
    public let memoryPressureHandling: MemoryPressureHandling
    
    /// Resource monitoring interval
    public let monitoringInterval: TimeInterval
    
    public init(
        maxMemoryUsage: Int? = nil,
        maxConcurrentOperations: Int = 4,
        batchSize: Int = 32,
        memoryPressureHandling: MemoryPressureHandling = .adaptive,
        monitoringInterval: TimeInterval = 1.0
    ) {
        self.maxMemoryUsage = maxMemoryUsage
        self.maxConcurrentOperations = maxConcurrentOperations
        self.batchSize = batchSize
        self.memoryPressureHandling = memoryPressureHandling
        self.monitoringInterval = monitoringInterval
    }
    
    // MARK: - Presets
    
    public static let balanced = ResourceConfiguration()
    
    public static let unlimited = ResourceConfiguration(
        maxMemoryUsage: nil,
        maxConcurrentOperations: ProcessInfo.processInfo.processorCount,
        batchSize: 64
    )
    
    public static let constrained = ResourceConfiguration(
        maxMemoryUsage: 256 * 1024 * 1024, // 256MB
        maxConcurrentOperations: 2,
        batchSize: 16,
        memoryPressureHandling: .aggressive
    )
    
    public static let managed = ResourceConfiguration(
        maxMemoryUsage: 512 * 1024 * 1024, // 512MB
        maxConcurrentOperations: 4,
        batchSize: 32,
        memoryPressureHandling: .adaptive,
        monitoringInterval: 0.5
    )
}

/// Memory pressure handling strategies
public enum MemoryPressureHandling: String, Sendable {
    case ignore = "ignore"
    case adaptive = "adaptive"
    case aggressive = "aggressive"
    case critical = "critical"
}

// MARK: - Performance Configuration

/// Configuration for performance optimization
public struct PerformanceConfiguration: Sendable {
    /// Whether to use Metal acceleration
    public let useMetalAcceleration: Bool
    
    /// Whether to enable prefetching
    public let enablePrefetching: Bool
    
    /// Whether to use streaming for large inputs
    public let autoStreamingThreshold: Int?
    
    /// Task priority
    public let taskPriority: TaskPriority
    
    /// Performance monitoring
    public let enablePerformanceMetrics: Bool
    
    public init(
        useMetalAcceleration: Bool = true,
        enablePrefetching: Bool = true,
        autoStreamingThreshold: Int? = 1000,
        taskPriority: TaskPriority = .userInitiated,
        enablePerformanceMetrics: Bool = false
    ) {
        self.useMetalAcceleration = useMetalAcceleration
        self.enablePrefetching = enablePrefetching
        self.autoStreamingThreshold = autoStreamingThreshold
        self.taskPriority = taskPriority
        self.enablePerformanceMetrics = enablePerformanceMetrics
    }
    
    // MARK: - Presets
    
    public static let balanced = PerformanceConfiguration()
    
    public static let maximum = PerformanceConfiguration(
        useMetalAcceleration: true,
        enablePrefetching: true,
        autoStreamingThreshold: 500,
        taskPriority: .high,
        enablePerformanceMetrics: true
    )
    
    public static let optimized = PerformanceConfiguration(
        useMetalAcceleration: true,
        enablePrefetching: true,
        autoStreamingThreshold: 1000,
        taskPriority: .userInitiated,
        enablePerformanceMetrics: true
    )
}

// MARK: - Cache Configuration

/// Configuration for caching behavior
public struct CacheConfiguration: Sendable {
    /// Maximum number of cached embeddings
    public let maxCacheSize: Int
    
    /// Cache TTL in seconds
    public let ttl: TimeInterval
    
    /// Whether to persist cache to disk
    public let persistToDisk: Bool
    
    /// Cache eviction policy
    public let evictionPolicy: CacheEvictionPolicy
    
    /// Whether to use memory-aware caching
    public let memoryAware: Bool
    
    public init(
        maxCacheSize: Int = 1000,
        ttl: TimeInterval = 3600,
        persistToDisk: Bool = false,
        evictionPolicy: CacheEvictionPolicy = .lru,
        memoryAware: Bool = true
    ) {
        self.maxCacheSize = maxCacheSize
        self.ttl = ttl
        self.persistToDisk = persistToDisk
        self.evictionPolicy = evictionPolicy
        self.memoryAware = memoryAware
    }
    
    // MARK: - Presets
    
    public static let `default` = CacheConfiguration()
    
    public static let aggressive = CacheConfiguration(
        maxCacheSize: 5000,
        ttl: 7200,
        persistToDisk: true,
        evictionPolicy: .lfu
    )
    
    public static let minimal = CacheConfiguration(
        maxCacheSize: 100,
        ttl: 300,
        persistToDisk: false,
        evictionPolicy: .lru,
        memoryAware: true
    )
}

/// Cache eviction policies
public enum CacheEvictionPolicy: String, Sendable {
    case lru = "lru" // Least Recently Used
    case lfu = "lfu" // Least Frequently Used
    case fifo = "fifo" // First In First Out
    case ttl = "ttl" // Time To Live based
}

// MARK: - Monitoring Configuration

/// Configuration for monitoring and telemetry
public struct MonitoringConfiguration: Sendable {
    /// Whether monitoring is enabled
    public let enabled: Bool
    
    /// Telemetry level
    public let telemetryLevel: TelemetryLevel
    
    /// Metrics to collect
    public let metrics: Set<MonitoringMetricType>
    
    /// Export interval for metrics
    public let exportInterval: TimeInterval
    
    /// Custom metric handlers
    public let customHandlers: [String: Bool]
    
    public init(
        enabled: Bool = true,
        telemetryLevel: TelemetryLevel = .standard,
        metrics: Set<MonitoringMetricType> = [MonitoringMetricType.performance, MonitoringMetricType.errors, MonitoringMetricType.usage],
        exportInterval: TimeInterval = 60.0,
        customHandlers: [String: Bool] = [:]
    ) {
        self.enabled = enabled
        self.telemetryLevel = telemetryLevel
        self.metrics = metrics
        self.exportInterval = exportInterval
        self.customHandlers = customHandlers
    }
    
    // MARK: - Presets
    
    public static let disabled = MonitoringConfiguration(enabled: false)
    
    public static let enabled = MonitoringConfiguration()
    
    public static let comprehensive = MonitoringConfiguration(
        enabled: true,
        telemetryLevel: TelemetryLevel.detailed,
        metrics: [MonitoringMetricType.performance, MonitoringMetricType.memory, MonitoringMetricType.errors, MonitoringMetricType.usage, MonitoringMetricType.cache, MonitoringMetricType.model],
        exportInterval: 30.0
    )
}

/// Telemetry levels
public enum TelemetryLevel: String, Sendable {
    case minimal = "minimal"
    case standard = "standard"
    case detailed = "detailed"
    case debug = "debug"
}

/// Types of metrics to collect for monitoring
public struct MonitoringMetricType: OptionSet, Sendable, Hashable {
    public let rawValue: Int
    
    public init(rawValue: Int) {
        self.rawValue = rawValue
    }
    
    public static let performance = MonitoringMetricType(rawValue: 1 << 0)
    public static let memory = MonitoringMetricType(rawValue: 1 << 1)
    public static let errors = MonitoringMetricType(rawValue: 1 << 2)
    public static let usage = MonitoringMetricType(rawValue: 1 << 3)
    public static let cache = MonitoringMetricType(rawValue: 1 << 4)
    public static let model = MonitoringMetricType(rawValue: 1 << 5)
    
    public static let standard: MonitoringMetricType = [.performance, .errors, .usage]
    public static let all: MonitoringMetricType = [.performance, .memory, .errors, .usage, .cache, .model]
}

// MARK: - Error Handling Configuration

/// Configuration for error handling behavior
public struct ErrorHandlingConfiguration: Sendable {
    /// Maximum retry attempts
    public let maxRetries: Int
    
    /// Retry delay strategy
    public let retryStrategy: RetryStrategy
    
    /// Circuit breaker threshold
    public let circuitBreakerThreshold: Int
    
    /// Recovery timeout for circuit breaker
    public let recoveryTimeout: TimeInterval
    
    /// Whether to use fallback values
    public let enableFallbacks: Bool
    
    /// Error reporting
    public let errorReporting: ErrorReporting
    
    public init(
        maxRetries: Int = 3,
        retryStrategy: RetryStrategy = .exponentialBackoff,
        circuitBreakerThreshold: Int = 5,
        recoveryTimeout: TimeInterval = 60.0,
        enableFallbacks: Bool = true,
        errorReporting: ErrorReporting = .standard
    ) {
        self.maxRetries = maxRetries
        self.retryStrategy = retryStrategy
        self.circuitBreakerThreshold = circuitBreakerThreshold
        self.recoveryTimeout = recoveryTimeout
        self.enableFallbacks = enableFallbacks
        self.errorReporting = errorReporting
    }
    
    // MARK: - Presets
    
    public static let `default` = ErrorHandlingConfiguration()
    
    public static let strict = ErrorHandlingConfiguration(
        maxRetries: 1,
        retryStrategy: .immediate,
        circuitBreakerThreshold: 3,
        recoveryTimeout: 120.0,
        enableFallbacks: false,
        errorReporting: .detailed
    )
    
    public static let lenient = ErrorHandlingConfiguration(
        maxRetries: 5,
        retryStrategy: .exponentialBackoff,
        circuitBreakerThreshold: 10,
        recoveryTimeout: 30.0,
        enableFallbacks: true,
        errorReporting: .minimal
    )
}

/// Retry strategies
public enum RetryStrategy: String, Sendable {
    case immediate = "immediate"
    case fixedDelay = "fixed_delay"
    case exponentialBackoff = "exponential_backoff"
    case custom = "custom"
}

/// Error reporting levels
public enum ErrorReporting: String, Sendable {
    case none = "none"
    case minimal = "minimal"
    case standard = "standard"
    case detailed = "detailed"
}


// MARK: - Configuration Validation

extension Configuration {
    /// Validates the configuration and returns any issues
    public func validate() -> [ConfigurationIssue] {
        var issues: [ConfigurationIssue] = []
        
        // Validate model configuration
        if model.maxSequenceLength < 1 {
            issues.append(.invalid("maxSequenceLength must be positive"))
        }
        
        // Validate resource configuration
        if resources.batchSize < 1 {
            issues.append(.invalid("batchSize must be positive"))
        }
        
        if resources.maxConcurrentOperations < 1 {
            issues.append(.invalid("maxConcurrentOperations must be positive"))
        }
        
        // Validate cache configuration
        if cache.maxCacheSize < 0 {
            issues.append(.invalid("maxCacheSize cannot be negative"))
        }
        
        if cache.ttl < 0 {
            issues.append(.invalid("cache TTL cannot be negative"))
        }
        
        // Check for incompatible settings
        if performance.useMetalAcceleration && model.computeUnits == .cpuOnly {
            issues.append(.incompatible("Metal acceleration requested but compute units set to CPU only"))
        }
        
        return issues
    }
}

/// Configuration validation issues
public enum ConfigurationIssue: Equatable {
    case invalid(String)
    case missing(String)
    case incompatible(String)
    case outOfRange(String)
    
    public var description: String {
        switch self {
        case .invalid(let message):
            return "Invalid configuration: \(message)"
        case .missing(let message):
            return "Missing configuration: \(message)"
        case .incompatible(let message):
            return "Incompatible configuration: \(message)"
        case .outOfRange(let message):
            return "Configuration out of range: \(message)"
        }
    }
}
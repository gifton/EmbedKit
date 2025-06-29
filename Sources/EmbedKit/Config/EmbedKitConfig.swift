import Foundation

// MARK: - Unified Configuration System

/// Unified configuration for all EmbedKit operations with fluent builder pattern
///
/// This configuration system simplifies setup and provides:
/// - Single point of configuration
/// - Fluent builder API
/// - Preset configurations
/// - Environment variable support
/// - Swift 6 compliant Sendable conformance
///
/// ## Example Usage
///
/// ```swift
/// // Using preset
/// let config = EmbedKitConfig.production()
///
/// // Using builder
/// let config = EmbedKitConfig.builder()
///     .model(.miniLM_L6_v2)
///     .maxSequenceLength(256)
///     .enableCache()
///     .metalAcceleration(true)
///     .build()
///
/// // From environment
/// let config = try EmbedKitConfig.fromEnvironment()
/// ```
public struct EmbedKitConfig: Sendable {
    // MARK: - Core Properties
    
    /// Model identifier
    public let modelIdentifier: ModelIdentifier
    
    /// Maximum sequence length for tokenization
    public let maxSequenceLength: Int
    
    /// Batch size for processing
    public let batchSize: Int
    
    /// Pooling strategy for embeddings
    public let poolingStrategy: PoolingStrategy
    
    // MARK: - Performance Settings
    
    /// Enable Metal acceleration
    public let useMetalAcceleration: Bool
    
    /// Number of concurrent operations
    public let maxConcurrentOperations: Int
    
    /// Memory limit in bytes (nil for unlimited)
    public let memoryLimit: Int?
    
    // MARK: - Cache Settings
    
    /// Enable caching
    public let cacheEnabled: Bool
    
    /// Maximum cache size in bytes
    public let maxCacheSize: Int
    
    /// Cache time-to-live in seconds
    public let cacheTTL: TimeInterval
    
    // MARK: - Storage Settings
    
    /// Custom storage directory
    public let storageDirectory: URL?
    
    /// Enable persistent storage
    public let persistentStorageEnabled: Bool
    
    // MARK: - Network Settings
    
    /// Allow model downloads
    public let allowModelDownloads: Bool
    
    /// Download timeout in seconds
    public let downloadTimeout: TimeInterval
    
    /// Maximum retry attempts
    public let maxRetryAttempts: Int
    
    // MARK: - Security Settings
    
    /// Verify model signatures
    public let verifyModelSignatures: Bool
    
    /// Enable secure enclave usage
    public let useSecureEnclave: Bool
    
    // MARK: - Monitoring Settings
    
    /// Enable telemetry
    public let telemetryEnabled: Bool
    
    /// Telemetry sampling rate (0.0 to 1.0)
    public let telemetrySamplingRate: Double
    
    /// Enable performance logging
    public let performanceLoggingEnabled: Bool
    
    // MARK: - Error Handling
    
    /// Error recovery strategy
    public let errorRecoveryStrategy: ErrorRecoveryStrategy
    
    /// Maximum error retry attempts
    public let errorRetryLimit: Int
    
    // MARK: - Initialization
    
    private init(
        modelIdentifier: ModelIdentifier,
        maxSequenceLength: Int,
        batchSize: Int,
        poolingStrategy: PoolingStrategy,
        useMetalAcceleration: Bool,
        maxConcurrentOperations: Int,
        memoryLimit: Int?,
        cacheEnabled: Bool,
        maxCacheSize: Int,
        cacheTTL: TimeInterval,
        storageDirectory: URL?,
        persistentStorageEnabled: Bool,
        allowModelDownloads: Bool,
        downloadTimeout: TimeInterval,
        maxRetryAttempts: Int,
        verifyModelSignatures: Bool,
        useSecureEnclave: Bool,
        telemetryEnabled: Bool,
        telemetrySamplingRate: Double,
        performanceLoggingEnabled: Bool,
        errorRecoveryStrategy: ErrorRecoveryStrategy,
        errorRetryLimit: Int
    ) {
        self.modelIdentifier = modelIdentifier
        self.maxSequenceLength = maxSequenceLength
        self.batchSize = batchSize
        self.poolingStrategy = poolingStrategy
        self.useMetalAcceleration = useMetalAcceleration
        self.maxConcurrentOperations = maxConcurrentOperations
        self.memoryLimit = memoryLimit
        self.cacheEnabled = cacheEnabled
        self.maxCacheSize = maxCacheSize
        self.cacheTTL = cacheTTL
        self.storageDirectory = storageDirectory
        self.persistentStorageEnabled = persistentStorageEnabled
        self.allowModelDownloads = allowModelDownloads
        self.downloadTimeout = downloadTimeout
        self.maxRetryAttempts = maxRetryAttempts
        self.verifyModelSignatures = verifyModelSignatures
        self.useSecureEnclave = useSecureEnclave
        self.telemetryEnabled = telemetryEnabled
        self.telemetrySamplingRate = telemetrySamplingRate
        self.performanceLoggingEnabled = performanceLoggingEnabled
        self.errorRecoveryStrategy = errorRecoveryStrategy
        self.errorRetryLimit = errorRetryLimit
    }
    
    // MARK: - Builder
    
    /// Configuration builder for fluent API
    public final class Builder: Sendable {
        private var modelIdentifier: ModelIdentifier = .miniLM_L6_v2
        private var maxSequenceLength: Int = 512
        private var batchSize: Int = 32
        private var poolingStrategy: PoolingStrategy = .mean
        private var useMetalAcceleration: Bool = true
        private var maxConcurrentOperations: Int = 4
        private var memoryLimit: Int? = nil
        private var cacheEnabled: Bool = true
        private var maxCacheSize: Int = 100 * 1024 * 1024 // 100MB
        private var cacheTTL: TimeInterval = 3600 // 1 hour
        private var storageDirectory: URL? = nil
        private var persistentStorageEnabled: Bool = true
        private var allowModelDownloads: Bool = true
        private var downloadTimeout: TimeInterval = 300 // 5 minutes
        private var maxRetryAttempts: Int = 3
        private var verifyModelSignatures: Bool = true
        private var useSecureEnclave: Bool = false
        private var telemetryEnabled: Bool = false
        private var telemetrySamplingRate: Double = 0.1
        private var performanceLoggingEnabled: Bool = false
        private var errorRecoveryStrategy: ErrorRecoveryStrategy = .retry
        private var errorRetryLimit: Int = 3
        
        public init() {}
        
        @discardableResult
        public func model(_ identifier: ModelIdentifier) -> Builder {
            self.modelIdentifier = identifier
            return self
        }
        
        @discardableResult
        public func maxSequenceLength(_ length: Int) -> Builder {
            self.maxSequenceLength = length
            return self
        }
        
        @discardableResult
        public func batchSize(_ size: Int) -> Builder {
            self.batchSize = size
            return self
        }
        
        @discardableResult
        public func poolingStrategy(_ strategy: PoolingStrategy) -> Builder {
            self.poolingStrategy = strategy
            return self
        }
        
        @discardableResult
        public func metalAcceleration(_ enabled: Bool) -> Builder {
            self.useMetalAcceleration = enabled
            return self
        }
        
        @discardableResult
        public func maxConcurrentOperations(_ count: Int) -> Builder {
            self.maxConcurrentOperations = count
            return self
        }
        
        @discardableResult
        public func memoryLimit(_ bytes: Int?) -> Builder {
            self.memoryLimit = bytes
            return self
        }
        
        @discardableResult
        public func enableCache(_ enabled: Bool = true) -> Builder {
            self.cacheEnabled = enabled
            return self
        }
        
        @discardableResult
        public func cacheSize(_ bytes: Int) -> Builder {
            self.maxCacheSize = bytes
            return self
        }
        
        @discardableResult
        public func cacheTTL(_ seconds: TimeInterval) -> Builder {
            self.cacheTTL = seconds
            return self
        }
        
        @discardableResult
        public func storageDirectory(_ url: URL?) -> Builder {
            self.storageDirectory = url
            return self
        }
        
        @discardableResult
        public func persistentStorage(_ enabled: Bool) -> Builder {
            self.persistentStorageEnabled = enabled
            return self
        }
        
        @discardableResult
        public func allowDownloads(_ enabled: Bool) -> Builder {
            self.allowModelDownloads = enabled
            return self
        }
        
        @discardableResult
        public func downloadTimeout(_ seconds: TimeInterval) -> Builder {
            self.downloadTimeout = seconds
            return self
        }
        
        @discardableResult
        public func maxRetries(_ count: Int) -> Builder {
            self.maxRetryAttempts = count
            return self
        }
        
        @discardableResult
        public func verifySignatures(_ enabled: Bool) -> Builder {
            self.verifyModelSignatures = enabled
            return self
        }
        
        @discardableResult
        public func secureEnclave(_ enabled: Bool) -> Builder {
            self.useSecureEnclave = enabled
            return self
        }
        
        @discardableResult
        public func telemetry(_ enabled: Bool) -> Builder {
            self.telemetryEnabled = enabled
            return self
        }
        
        @discardableResult
        public func telemetrySamplingRate(_ rate: Double) -> Builder {
            self.telemetrySamplingRate = max(0.0, min(1.0, rate))
            return self
        }
        
        @discardableResult
        public func performanceLogging(_ enabled: Bool) -> Builder {
            self.performanceLoggingEnabled = enabled
            return self
        }
        
        @discardableResult
        public func errorRecovery(_ strategy: ErrorRecoveryStrategy) -> Builder {
            self.errorRecoveryStrategy = strategy
            return self
        }
        
        @discardableResult
        public func errorRetryLimit(_ limit: Int) -> Builder {
            self.errorRetryLimit = limit
            return self
        }
        
        public func build() -> EmbedKitConfig {
            EmbedKitConfig(
                modelIdentifier: modelIdentifier,
                maxSequenceLength: maxSequenceLength,
                batchSize: batchSize,
                poolingStrategy: poolingStrategy,
                useMetalAcceleration: useMetalAcceleration,
                maxConcurrentOperations: maxConcurrentOperations,
                memoryLimit: memoryLimit,
                cacheEnabled: cacheEnabled,
                maxCacheSize: maxCacheSize,
                cacheTTL: cacheTTL,
                storageDirectory: storageDirectory,
                persistentStorageEnabled: persistentStorageEnabled,
                allowModelDownloads: allowModelDownloads,
                downloadTimeout: downloadTimeout,
                maxRetryAttempts: maxRetryAttempts,
                verifyModelSignatures: verifyModelSignatures,
                useSecureEnclave: useSecureEnclave,
                telemetryEnabled: telemetryEnabled,
                telemetrySamplingRate: telemetrySamplingRate,
                performanceLoggingEnabled: performanceLoggingEnabled,
                errorRecoveryStrategy: errorRecoveryStrategy,
                errorRetryLimit: errorRetryLimit
            )
        }
    }
    
    /// Create a new builder
    public static func builder() -> Builder {
        Builder()
    }
}

// MARK: - Preset Configurations

extension EmbedKitConfig {
    /// Production configuration optimized for reliability
    public static func production() -> EmbedKitConfig {
        builder()
            .model(.miniLM_L6_v2)
            .maxSequenceLength(512)
            .batchSize(32)
            .metalAcceleration(true)
            .enableCache(true)
            .cacheSize(200 * 1024 * 1024) // 200MB
            .persistentStorage(true)
            .verifySignatures(true)
            .telemetry(true)
            .telemetrySamplingRate(0.1)
            .errorRecovery(.retry)
            .errorRetryLimit(3)
            .build()
    }
    
    /// Development configuration for testing
    public static func development() -> EmbedKitConfig {
        builder()
            .model(.miniLM_L6_v2)
            .maxSequenceLength(256)
            .batchSize(16)
            .metalAcceleration(true)
            .enableCache(true)
            .cacheSize(50 * 1024 * 1024) // 50MB
            .persistentStorage(false)
            .verifySignatures(false)
            .telemetry(true)
            .telemetrySamplingRate(1.0)
            .performanceLogging(true)
            .errorRecovery(.fail)
            .build()
    }
    
    /// High performance configuration
    public static func highPerformance() -> EmbedKitConfig {
        builder()
            .model(.miniLM_L6_v2)
            .maxSequenceLength(256)
            .batchSize(64)
            .metalAcceleration(true)
            .maxConcurrentOperations(8)
            .enableCache(true)
            .cacheSize(500 * 1024 * 1024) // 500MB
            .persistentStorage(true)
            .telemetry(false)
            .performanceLogging(false)
            .build()
    }
    
    /// Memory constrained configuration
    public static func memoryConstrained() -> EmbedKitConfig {
        builder()
            .model(.miniLM_L6_v2)
            .maxSequenceLength(128)
            .batchSize(8)
            .metalAcceleration(false)
            .maxConcurrentOperations(2)
            .memoryLimit(100 * 1024 * 1024) // 100MB
            .enableCache(false)
            .persistentStorage(false)
            .telemetry(false)
            .build()
    }
    
    /// Testing configuration with minimal features
    public static func testing() -> EmbedKitConfig {
        builder()
            .model(.miniLM_L6_v2)
            .maxSequenceLength(128)
            .batchSize(1)
            .metalAcceleration(false)
            .enableCache(false)
            .persistentStorage(false)
            .allowDownloads(false)
            .verifySignatures(false)
            .telemetry(false)
            .build()
    }
}

// MARK: - Environment Variable Support

extension EmbedKitConfig {
    /// Environment variable keys
    private enum EnvironmentKey: String {
        case modelIdentifier = "EMBEDKIT_MODEL"
        case maxSequenceLength = "EMBEDKIT_MAX_SEQUENCE_LENGTH"
        case batchSize = "EMBEDKIT_BATCH_SIZE"
        case useMetalAcceleration = "EMBEDKIT_USE_METAL"
        case cacheEnabled = "EMBEDKIT_CACHE_ENABLED"
        case maxCacheSize = "EMBEDKIT_CACHE_SIZE"
        case telemetryEnabled = "EMBEDKIT_TELEMETRY_ENABLED"
        case verifySignatures = "EMBEDKIT_VERIFY_SIGNATURES"
        case storageDirectory = "EMBEDKIT_STORAGE_DIR"
    }
    
    /// Create configuration from environment variables
    public static func fromEnvironment() throws -> EmbedKitConfig {
        let builder = EmbedKitConfig.builder()
        
        // Model identifier
        if let modelString = ProcessInfo.processInfo.environment[EnvironmentKey.modelIdentifier.rawValue] {
            if let model = ModelIdentifier(rawValue: modelString) {
                builder.model(model)
            } else {
                throw ConfigurationError.invalidEnvironmentValue(
                    key: EnvironmentKey.modelIdentifier.rawValue,
                    value: modelString
                )
            }
        }
        
        // Numeric values
        if let seqLength = ProcessInfo.processInfo.environment[EnvironmentKey.maxSequenceLength.rawValue],
           let length = Int(seqLength) {
            builder.maxSequenceLength(length)
        }
        
        if let batchSizeString = ProcessInfo.processInfo.environment[EnvironmentKey.batchSize.rawValue],
           let size = Int(batchSizeString) {
            builder.batchSize(size)
        }
        
        if let cacheSize = ProcessInfo.processInfo.environment[EnvironmentKey.maxCacheSize.rawValue],
           let size = Int(cacheSize) {
            builder.cacheSize(size)
        }
        
        // Boolean values
        if let metalString = ProcessInfo.processInfo.environment[EnvironmentKey.useMetalAcceleration.rawValue] {
            builder.metalAcceleration(metalString.lowercased() == "true" || metalString == "1")
        }
        
        if let cacheString = ProcessInfo.processInfo.environment[EnvironmentKey.cacheEnabled.rawValue] {
            builder.enableCache(cacheString.lowercased() == "true" || cacheString == "1")
        }
        
        if let telemetryString = ProcessInfo.processInfo.environment[EnvironmentKey.telemetryEnabled.rawValue] {
            builder.telemetry(telemetryString.lowercased() == "true" || telemetryString == "1")
        }
        
        if let verifyString = ProcessInfo.processInfo.environment[EnvironmentKey.verifySignatures.rawValue] {
            builder.verifySignatures(verifyString.lowercased() == "true" || verifyString == "1")
        }
        
        // URL values
        if let storagePath = ProcessInfo.processInfo.environment[EnvironmentKey.storageDirectory.rawValue] {
            builder.storageDirectory(URL(fileURLWithPath: storagePath))
        }
        
        return builder.build()
    }
}

// MARK: - Supporting Types

/// Error recovery strategies
public enum ErrorRecoveryStrategy: String, Sendable {
    /// Retry the operation with exponential backoff
    case retry
    /// Fail immediately
    case fail
    /// Use fallback/default values
    case fallback
    /// Log and continue
    case logAndContinue
}

/// Configuration errors
public enum ConfigurationError: Error, Sendable {
    case invalidEnvironmentValue(key: String, value: String)
    case missingRequiredValue(key: String)
    case validationFailed(message: String)
}

// MARK: - Configuration Validation

extension EmbedKitConfig {
    /// Validate configuration
    public func validate() throws {
        // Validate sequence length
        guard maxSequenceLength > 0 && maxSequenceLength <= 2048 else {
            throw ConfigurationError.validationFailed(
                message: "maxSequenceLength must be between 1 and 2048"
            )
        }
        
        // Validate batch size
        guard batchSize > 0 && batchSize <= 256 else {
            throw ConfigurationError.validationFailed(
                message: "batchSize must be between 1 and 256"
            )
        }
        
        // Validate concurrent operations
        guard maxConcurrentOperations > 0 && maxConcurrentOperations <= 32 else {
            throw ConfigurationError.validationFailed(
                message: "maxConcurrentOperations must be between 1 and 32"
            )
        }
        
        // Validate cache settings
        if cacheEnabled {
            guard maxCacheSize > 0 else {
                throw ConfigurationError.validationFailed(
                    message: "maxCacheSize must be positive when cache is enabled"
                )
            }
            guard cacheTTL > 0 else {
                throw ConfigurationError.validationFailed(
                    message: "cacheTTL must be positive when cache is enabled"
                )
            }
        }
        
        // Validate telemetry settings
        guard telemetrySamplingRate >= 0.0 && telemetrySamplingRate <= 1.0 else {
            throw ConfigurationError.validationFailed(
                message: "telemetrySamplingRate must be between 0.0 and 1.0"
            )
        }
        
        // Validate memory limit
        if let limit = memoryLimit, limit <= 0 {
            throw ConfigurationError.validationFailed(
                message: "memoryLimit must be positive if specified"
            )
        }
    }
}
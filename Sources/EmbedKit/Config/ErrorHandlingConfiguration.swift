import Foundation

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
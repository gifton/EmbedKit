import Foundation
import OSLog

/// Production-ready error handling system with graceful degradation
public actor ErrorHandlingSystem {
    private let logger = Logger(subsystem: "EmbedKit", category: "ErrorHandling")
    
    private var errorCounts: [String: Int] = [:]
    private var lastErrors: [String: Date] = [:]
    private var circuitBreakers: [String: CircuitBreaker] = [:]
    
    public init() {}
    
    /// Handle an error with context and determine appropriate response
    public func handleError(
        _ error: Error,
        context: ErrorContext,
        operation: String
    ) async -> ErrorHandlingResult {
        let errorKey = "\(operation):\(type(of: error))"
        
        // Update error tracking
        errorCounts[errorKey, default: 0] += 1
        lastErrors[errorKey] = Date()
        
        // Check circuit breaker status
        let circuitBreaker = circuitBreakers[operation] ?? CircuitBreaker(
            failureThreshold: context.failureThreshold,
            recoveryTimeout: context.recoveryTimeout
        )
        circuitBreakers[operation] = circuitBreaker
        
        await circuitBreaker.recordFailure()
        
        // Log the error with appropriate level
        let logLevel = determineLogLevel(error: error, context: context)
        logError(error: error, context: context, operation: operation, level: logLevel)
        
        // Determine handling strategy
        let strategy = await determineStrategy(
            error: error,
            context: context,
            circuitBreaker: circuitBreaker
        )
        
        return ErrorHandlingResult(
            strategy: strategy,
            shouldRetry: await shouldRetry(error: error, context: context, circuitBreaker: circuitBreaker),
            fallbackValue: context.fallbackProvider?(),
            errorInfo: ErrorInfo(
                error: error,
                operation: operation,
                timestamp: Date(),
                errorCount: errorCounts[errorKey] ?? 0
            )
        )
    }
    
    /// Handle successful operation (for circuit breaker recovery)
    public func recordSuccess(operation: String) async {
        if let circuitBreaker = circuitBreakers[operation] {
            await circuitBreaker.recordSuccess()
        }
    }
    
    /// Get error statistics for monitoring
    public func getErrorStatistics() async -> ErrorStatistics {
        let now = Date()
        let recentErrors = lastErrors.filter { now.timeIntervalSince($0.value) < 3600 } // Last hour
        
        return ErrorStatistics(
            totalErrorTypes: errorCounts.count,
            recentErrorCount: recentErrors.count,
            topErrors: errorCounts.sorted { $0.value > $1.value }.prefix(5).map { ($0.key, $0.value) },
            circuitBreakerStates: await getCircuitBreakerStates()
        )
    }
    
    /// Reset error tracking for an operation
    public func resetErrorTracking(operation: String) async {
        let keysToRemove = errorCounts.keys.filter { $0.hasPrefix(operation) }
        for key in keysToRemove {
            errorCounts.removeValue(forKey: key)
            lastErrors.removeValue(forKey: key)
        }
        circuitBreakers.removeValue(forKey: operation)
    }
    
    // MARK: - Private Methods
    
    private func determineLogLevel(error: Error, context: ErrorContext) -> OSLogType {
        if error is CriticalError || context.isCritical {
            return .error
        } else if error is RecoverableError {
            return .info
        } else if context.isExpected {
            return .debug
        } else {
            return .default
        }
    }
    
    private func logError(error: Error, context: ErrorContext, operation: String, level: OSLogType) {
        let message = "Error in \(operation): \(error.localizedDescription)"
        
        switch level {
        case .error:
            logger.error("\(message)")
        case .info:
            logger.info("\(message)")
        case .debug:
            logger.debug("\(message)")
        default:
            logger.log("\(message)")
        }
        
        // Include additional context if available
        if !context.metadata.isEmpty {
            logger.info("Error context: \(context.metadata)")
        }
    }
    
    private func determineStrategy(
        error: Error,
        context: ErrorContext,
        circuitBreaker: CircuitBreaker
    ) async -> ErrorHandlingStrategy {
        // Check circuit breaker state
        let state = await circuitBreaker.currentState
        if state == .open {
            return .failFast
        }
        
        // Determine strategy based on error type and context
        switch error {
        case is EmbeddingError:
            return await handleEmbeddingError(error as! EmbeddingError, context: context)
        case is NetworkError:
            return .retryWithBackoff
        case is ResourceError:
            return .degradeGracefully
        case is ValidationError:
            return .failFast
        default:
            return context.defaultStrategy
        }
    }
    
    private func handleEmbeddingError(_ error: EmbeddingError, context: ErrorContext) async -> ErrorHandlingStrategy {
        switch error {
        case .modelNotLoaded:
            return .retryAfterRecovery
        case .tokenizationFailed:
            return .useFallback
        case .invalidInput:
            return .retryWithBackoff
        default:
            return .retryWithBackoff
        }
    }
    
    private func shouldRetry(
        error: Error,
        context: ErrorContext,
        circuitBreaker: CircuitBreaker
    ) async -> Bool {
        let state = await circuitBreaker.currentState
        if state == .open {
            return false
        }
        
        let errorKey = "\(context.operation):\(type(of: error))"
        let currentCount = errorCounts[errorKey] ?? 0
        
        return currentCount < context.maxRetries && 
               !(error is ValidationError) &&
               !(error is CriticalError)
    }
    
    private func getCircuitBreakerStates() async -> [String: String] {
        var states: [String: String] = [:]
        for (operation, breaker) in circuitBreakers {
            states[operation] = await breaker.currentState.rawValue
        }
        return states
    }
}

/// Context information for error handling decisions
public struct ErrorContext {
    public let operation: String
    public let isCritical: Bool
    public let isExpected: Bool
    public let maxRetries: Int
    public let failureThreshold: Int
    public let recoveryTimeout: TimeInterval
    public let defaultStrategy: ErrorHandlingStrategy
    public let metadata: [String: String]
    public let fallbackProvider: (() -> Any?)?
    
    public init(
        operation: String,
        isCritical: Bool = false,
        isExpected: Bool = false,
        maxRetries: Int = 3,
        failureThreshold: Int = 5,
        recoveryTimeout: TimeInterval = 60.0,
        defaultStrategy: ErrorHandlingStrategy = .retryWithBackoff,
        metadata: [String: String] = [:],
        fallbackProvider: (() -> Any?)? = nil
    ) {
        self.operation = operation
        self.isCritical = isCritical
        self.isExpected = isExpected
        self.maxRetries = maxRetries
        self.failureThreshold = failureThreshold
        self.recoveryTimeout = recoveryTimeout
        self.defaultStrategy = defaultStrategy
        self.metadata = metadata
        self.fallbackProvider = fallbackProvider
    }
}

/// Result of error handling analysis
public struct ErrorHandlingResult {
    public let strategy: ErrorHandlingStrategy
    public let shouldRetry: Bool
    public let fallbackValue: Any?
    public let errorInfo: ErrorInfo
    
    public init(strategy: ErrorHandlingStrategy, shouldRetry: Bool, fallbackValue: Any?, errorInfo: ErrorInfo) {
        self.strategy = strategy
        self.shouldRetry = shouldRetry
        self.fallbackValue = fallbackValue
        self.errorInfo = errorInfo
    }
}

/// Information about a specific error occurrence
public struct ErrorInfo {
    public let error: Error
    public let operation: String
    public let timestamp: Date
    public let errorCount: Int
    
    public init(error: Error, operation: String, timestamp: Date, errorCount: Int) {
        self.error = error
        self.operation = operation
        self.timestamp = timestamp
        self.errorCount = errorCount
    }
}

/// Error handling strategies
public enum ErrorHandlingStrategy: String, CaseIterable {
    case retryWithBackoff = "retry_with_backoff"
    case retryAfterRecovery = "retry_after_recovery"
    case useFallback = "use_fallback"
    case degradeGracefully = "degrade_gracefully"
    case failFast = "fail_fast"
    case ignore = "ignore"
}

/// Circuit breaker for protecting against cascading failures
public actor CircuitBreaker {
    public enum State: String, Sendable {
        case closed = "closed"
        case open = "open"
        case halfOpen = "half_open"
    }
    
    private let failureThreshold: Int
    private let recoveryTimeout: TimeInterval
    private var failureCount: Int = 0
    private var lastFailureTime: Date?
    private var state: State = .closed
    
    public var currentState: State { state }
    
    public init(failureThreshold: Int = 5, recoveryTimeout: TimeInterval = 60.0) {
        self.failureThreshold = failureThreshold
        self.recoveryTimeout = recoveryTimeout
    }
    
    public func recordFailure() {
        failureCount += 1
        lastFailureTime = Date()
        
        if failureCount >= failureThreshold && state == .closed {
            state = .open
        }
    }
    
    public func recordSuccess() {
        failureCount = 0
        lastFailureTime = nil
        state = .closed
    }
    
    public func canExecute() -> Bool {
        switch state {
        case .closed:
            return true
        case .open:
            if let lastFailure = lastFailureTime,
               Date().timeIntervalSince(lastFailure) >= recoveryTimeout {
                state = .halfOpen
                return true
            }
            return false
        case .halfOpen:
            return true
        }
    }
}

/// Error statistics for monitoring
public struct ErrorStatistics: Sendable {
    public let totalErrorTypes: Int
    public let recentErrorCount: Int
    public let topErrors: [(String, Int)]
    public let circuitBreakerStates: [String: String]
    
    public init(totalErrorTypes: Int, recentErrorCount: Int, topErrors: [(String, Int)], circuitBreakerStates: [String: String]) {
        self.totalErrorTypes = totalErrorTypes
        self.recentErrorCount = recentErrorCount
        self.topErrors = topErrors
        self.circuitBreakerStates = circuitBreakerStates
    }
}

// MARK: - Error Types

/// Critical errors that require immediate attention
public protocol CriticalError: Error {}

/// Errors that can be recovered from
public protocol RecoverableError: Error {}

/// Network-related errors
public enum NetworkError: Error, RecoverableError {
    case connectionFailed
    case timeout
    case serverError(Int)
    case rateLimited
}

/// Resource-related errors
public enum ResourceError: Error, RecoverableError {
    case memoryPressure
    case diskSpaceLow
    case cpuOverload
    case gpuUnavailable
}

/// Validation errors (usually not retryable)
public enum ValidationError: Error {
    case invalidInput(String)
    case missingRequiredField(String)
    case formatError(String)
}

/// Graceful degradation system
public actor GracefulDegradationManager {
    private let logger = Logger(subsystem: "EmbedKit", category: "GracefulDegradation")
    
    private var degradationLevels: [String: DegradationLevel] = [:]
    private var resourceUsage: ResourceUsage = ResourceUsage()
    
    public enum DegradationLevel: Int, CaseIterable, Sendable, Comparable {
        case normal = 0
        case reduced = 1
        case minimal = 2
        case emergency = 3
        
        public var description: String {
            switch self {
            case .normal: return "Normal operation"
            case .reduced: return "Reduced functionality"
            case .minimal: return "Minimal functionality"
            case .emergency: return "Emergency mode"
            }
        }
        
        public static func < (lhs: DegradationLevel, rhs: DegradationLevel) -> Bool {
            lhs.rawValue < rhs.rawValue
        }
    }
    
    public init() {}
    
    /// Determine appropriate degradation level based on current conditions
    public func assessDegradationLevel(operation: String) async -> DegradationLevel {
        await updateResourceUsage()
        
        let memoryLevel = assessMemoryPressure()
        let errorLevel = assessErrorRate(operation: operation)
        let overallLevel = max(memoryLevel.rawValue, errorLevel.rawValue)
        
        let newLevel = DegradationLevel(rawValue: overallLevel) ?? .normal
        
        if degradationLevels[operation] != newLevel {
            degradationLevels[operation] = newLevel
            logger.info("Degradation level changed to \(newLevel.description) for \(operation)")
        }
        
        return newLevel
    }
    
    /// Apply degradation measures based on level
    public func applyDegradation(level: DegradationLevel, configuration: EmbeddingConfiguration) -> EmbeddingConfiguration {
        switch level {
        case .normal:
            return configuration // No changes
        case .reduced:
            return EmbeddingConfiguration(
                maxSequenceLength: min(configuration.maxSequenceLength, 256),
                normalizeEmbeddings: configuration.normalizeEmbeddings,
                poolingStrategy: configuration.poolingStrategy,
                batchSize: min(configuration.batchSize, 16)
            )
        case .minimal:
            return EmbeddingConfiguration(
                maxSequenceLength: min(configuration.maxSequenceLength, 128),
                normalizeEmbeddings: configuration.normalizeEmbeddings,
                poolingStrategy: configuration.poolingStrategy,
                batchSize: min(configuration.batchSize, 8)
            )
        case .emergency:
            return EmbeddingConfiguration(
                maxSequenceLength: min(configuration.maxSequenceLength, 64),
                normalizeEmbeddings: configuration.normalizeEmbeddings,
                poolingStrategy: .cls, // Fastest pooling
                batchSize: 1
            )
        }
    }
    
    /// Get current degradation status
    public func getDegradationStatus() async -> [String: DegradationLevel] {
        degradationLevels
    }
    
    // MARK: - Private Methods
    
    private func updateResourceUsage() async {
        // In a real implementation, this would query system resources
        // For now, we'll simulate based on some heuristics
        resourceUsage.memoryUsagePercent = getCurrentMemoryUsage()
    }
    
    private func assessMemoryPressure() -> DegradationLevel {
        switch resourceUsage.memoryUsagePercent {
        case 0..<0.7:
            return .normal
        case 0.7..<0.85:
            return .reduced
        case 0.85..<0.95:
            return .minimal
        default:
            return .emergency
        }
    }
    
    private func assessErrorRate(operation: String) -> DegradationLevel {
        // This would integrate with the error handling system
        // For now, return normal
        return .normal
    }
    
    private func getCurrentMemoryUsage() -> Double {
        // Simplified memory usage calculation
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        if kerr == KERN_SUCCESS {
            let usedMemory = Double(info.resident_size)
            let totalMemory = Double(ProcessInfo.processInfo.physicalMemory)
            return usedMemory / totalMemory
        }
        
        return 0.5 // Default assumption
    }
}

/// Resource usage tracking
public struct ResourceUsage: Sendable {
    public var memoryUsagePercent: Double = 0.0
    public var cpuUsagePercent: Double = 0.0
    public var diskUsagePercent: Double = 0.0
    
    public init() {}
}
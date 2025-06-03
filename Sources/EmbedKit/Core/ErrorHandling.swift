import Foundation
import OSLog



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


// MARK: - Error Types

/// Critical errors that require immediate attention
public protocol CriticalError: Error {}

/// Errors that can be recovered from
public protocol RecoverableError: Error {}


/// Graceful degradation system
public actor GracefulDegradationManager {
    private let logger = Logger(subsystem: "EmbedKit", category: "GracefulDegradation")
    
    private var degradationLevels: [String: DegradationLevel] = [:]
    private var resourceUsage: ResourceUsage = ResourceUsage()
    private var errorCounts: [String: Int] = [:]
    private var successCounts: [String: Int] = [:]
    
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
    public func applyDegradation(level: DegradationLevel, configuration: Configuration) -> Configuration {
        switch level {
        case .normal:
            return configuration // No changes
        case .reduced:
            return Configuration(
                model: ModelConfiguration(
                    identifier: configuration.model.identifier,
                    maxSequenceLength: min(configuration.model.maxSequenceLength, 256),
                    normalizeEmbeddings: configuration.model.normalizeEmbeddings,
                    poolingStrategy: configuration.model.poolingStrategy
                ),
                resources: ResourceConfiguration(
                    maxMemoryUsage: configuration.resources.maxMemoryUsage,
                    maxConcurrentOperations: configuration.resources.maxConcurrentOperations,
                    batchSize: min(configuration.resources.batchSize, 16)
                ),
                performance: configuration.performance,
                monitoring: configuration.monitoring,
                cache: configuration.cache,
                errorHandling: configuration.errorHandling
            )
        case .minimal:
            return Configuration(
                model: ModelConfiguration(
                    identifier: configuration.model.identifier,
                    maxSequenceLength: min(configuration.model.maxSequenceLength, 128),
                    normalizeEmbeddings: configuration.model.normalizeEmbeddings,
                    poolingStrategy: configuration.model.poolingStrategy
                ),
                resources: ResourceConfiguration(
                    maxMemoryUsage: configuration.resources.maxMemoryUsage,
                    maxConcurrentOperations: configuration.resources.maxConcurrentOperations,
                    batchSize: min(configuration.resources.batchSize, 8)
                ),
                performance: configuration.performance,
                monitoring: configuration.monitoring,
                cache: configuration.cache,
                errorHandling: configuration.errorHandling
            )
        case .emergency:
            return Configuration(
                model: ModelConfiguration(
                    identifier: configuration.model.identifier,
                    maxSequenceLength: min(configuration.model.maxSequenceLength, 64),
                    normalizeEmbeddings: configuration.model.normalizeEmbeddings,
                    poolingStrategy: .cls // Fastest pooling
                ),
                resources: ResourceConfiguration(
                    maxMemoryUsage: configuration.resources.maxMemoryUsage,
                    maxConcurrentOperations: configuration.resources.maxConcurrentOperations,
                    batchSize: 1
                ),
                performance: configuration.performance,
                monitoring: configuration.monitoring,
                cache: configuration.cache,
                errorHandling: configuration.errorHandling
            )
        }
    }
    
    /// Get current degradation status
    public func getDegradationStatus() async -> [String: DegradationLevel] {
        degradationLevels
    }
    
    // MARK: - Private Methods
    
    private func updateResourceUsage() async {
        // Update memory usage
        resourceUsage.memoryUsagePercent = getCurrentMemoryUsage()
        
        // Update CPU usage
        resourceUsage.cpuUsagePercent = getCurrentCPUUsage()
        
        // Update disk space (for model storage)
        resourceUsage.diskSpaceAvailable = getDiskSpaceAvailable()
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
        let recentErrors = errorCounts[operation] ?? 0
        let totalOps = max(1, (errorCounts[operation] ?? 0) + (successCounts[operation] ?? 0))
        let errorRate = Double(recentErrors) / Double(totalOps)
        
        switch errorRate {
        case 0..<0.05:
            return .normal
        case 0.05..<0.15:
            return .reduced
        case 0.15..<0.30:
            return .minimal
        default:
            return .emergency
        }
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
    
    private func getCurrentCPUUsage() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        if result == KERN_SUCCESS {
            let userTime = Double(info.user_time.seconds) + Double(info.user_time.microseconds) / 1_000_000
            let systemTime = Double(info.system_time.seconds) + Double(info.system_time.microseconds) / 1_000_000
            let totalTime = userTime + systemTime
            let currentTime = ProcessInfo.processInfo.systemUptime
            
            if currentTime > totalTime {
                return min((totalTime / currentTime) * 100.0, 100.0)
            }
        }
        
        return 0.0
    }
    
    private func getDiskSpaceAvailable() -> Int64 {
        do {
            let fileURL = URL(fileURLWithPath: NSHomeDirectory())
            let values = try fileURL.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
            if let capacity = values.volumeAvailableCapacityForImportantUsage {
                return capacity
            }
        } catch {
            logger.warning("Failed to get disk space: \(error)")
        }
        
        // Return a reasonable default (1GB)
        return 1_073_741_824
    }
    
    /// Record a successful operation
    public func recordSuccess(for operation: String) {
        successCounts[operation, default: 0] += 1
    }
    
    /// Record a failed operation
    public func recordError(for operation: String, error: Error) {
        errorCounts[operation, default: 0] += 1
        logger.error("Operation \(operation) failed: \(error)")
    }
}

/// Resource usage tracking
public struct ResourceUsage: Sendable {
    public var memoryUsagePercent: Double = 0.0
    public var cpuUsagePercent: Double = 0.0
    public var diskUsagePercent: Double = 0.0
    public var diskSpaceAvailable: Int64 = 0
    
    public init() {}
}
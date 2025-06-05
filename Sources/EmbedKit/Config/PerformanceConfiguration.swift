import Foundation

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
import Foundation

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
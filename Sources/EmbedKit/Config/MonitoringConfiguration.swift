import Foundation

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
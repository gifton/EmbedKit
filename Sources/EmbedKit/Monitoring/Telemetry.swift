import Foundation
import OSLog

/// Comprehensive telemetry system for monitoring EmbedKit operations
public actor TelemetrySystem {
    private let logger = Logger(subsystem: "EmbedKit", category: "Telemetry")
    
    private var metrics: [String: Metric] = [:]
    private var events: [TelemetryEvent] = []
    private var counters: [String: Int] = [:]
    private var timers: [String: [TimeInterval]] = [:]
    private var configuration: TelemetryConfiguration
    
    public init(configuration: TelemetryConfiguration = TelemetryConfiguration()) {
        self.configuration = configuration
    }
    
    /// Record a counter metric
    public func incrementCounter(_ name: String, by value: Int = 1, tags: [String: String] = [:]) {
        counters[name, default: 0] += value
        
        let metric = Metric(
            name: name,
            type: .counter,
            value: Double(value),
            tags: tags,
            timestamp: Date()
        )
        
        recordMetric(metric)
    }
    
    /// Record a timing metric
    public func recordTiming(_ name: String, duration: TimeInterval, tags: [String: String] = [:]) {
        timers[name, default: []].append(duration)
        
        let metric = Metric(
            name: name,
            type: .timing,
            value: duration,
            tags: tags,
            timestamp: Date()
        )
        
        recordMetric(metric)
    }
    
    /// Record a gauge metric (current value)
    public func recordGauge(_ name: String, value: Double, tags: [String: String] = [:]) {
        let metric = Metric(
            name: name,
            type: .gauge,
            value: value,
            tags: tags,
            timestamp: Date()
        )
        
        recordMetric(metric)
    }
    
    /// Record a histogram value
    public func recordHistogram(_ name: String, value: Double, tags: [String: String] = [:]) {
        let metric = Metric(
            name: name,
            type: .histogram,
            value: value,
            tags: tags,
            timestamp: Date()
        )
        
        recordMetric(metric)
    }
    
    /// Record a custom event
    public func recordEvent(_ event: TelemetryEvent) {
        events.append(event)
        
        // Keep only recent events to manage memory
        if events.count > configuration.maxEvents {
            events.removeFirst(events.count - configuration.maxEvents)
        }
        
        if configuration.logEvents {
            logger.info("Event: \(event.name) - \(event.description)")
        }
    }
    
    /// Start a timer for an operation
    public func startTimer(_ name: String) -> TimerToken {
        TimerToken(name: name, startTime: Date(), telemetry: self)
    }
    
    /// Get aggregated metrics for a specific metric name
    public func getMetricSummary(_ name: String) async -> MetricSummary? {
        let relatedMetrics = metrics.values.filter { $0.name == name }
        guard !relatedMetrics.isEmpty else { return nil }
        
        let values = relatedMetrics.map { $0.value }
        let timestamps = relatedMetrics.map { $0.timestamp }
        
        return MetricSummary(
            name: name,
            count: relatedMetrics.count,
            sum: values.reduce(0, +),
            min: values.min() ?? 0,
            max: values.max() ?? 0,
            average: values.reduce(0, +) / Double(values.count),
            firstTimestamp: timestamps.min() ?? Date(),
            lastTimestamp: timestamps.max() ?? Date()
        )
    }
    
    /// Get recent events
    public func getRecentEvents(limit: Int = 100) async -> [TelemetryEvent] {
        Array(events.suffix(limit))
    }
    
    /// Export all metrics as JSON
    public func exportMetrics() async -> Data? {
        let export = MetricsExport(
            timestamp: Date(),
            metrics: Array(metrics.values),
            events: events,
            counters: counters,
            timerSummaries: getTimerSummaries()
        )
        
        return try? JSONEncoder().encode(export)
    }
    
    /// Clear all collected data
    public func reset() {
        metrics.removeAll()
        events.removeAll()
        counters.removeAll()
        timers.removeAll()
        logger.info("Telemetry data reset")
    }
    
    /// Get current system metrics
    public func getSystemMetrics() async -> SystemMetrics {
        SystemMetrics(
            memoryUsage: getCurrentMemoryUsage(),
            cpuUsage: getCurrentCPUUsage(),
            timestamp: Date()
        )
    }
    
    // MARK: - Private Methods
    
    private func recordMetric(_ metric: Metric) {
        let key = "\(metric.name)_\(metric.timestamp.timeIntervalSince1970)"
        metrics[key] = metric
        
        // Clean up old metrics to prevent memory issues
        if metrics.count > configuration.maxMetrics {
            let sortedKeys = metrics.keys.sorted()
            let keysToRemove = sortedKeys.prefix(metrics.count - configuration.maxMetrics)
            for key in keysToRemove {
                metrics.removeValue(forKey: key)
            }
        }
        
        if configuration.logMetrics {
            logger.debug("Metric: \(metric.name) = \(metric.value)")
        }
    }
    
    private func getTimerSummaries() -> [String: TimerSummary] {
        var summaries: [String: TimerSummary] = [:]
        
        for (name, durations) in timers {
            guard !durations.isEmpty else { continue }
            
            summaries[name] = TimerSummary(
                name: name,
                count: durations.count,
                totalDuration: durations.reduce(0, +),
                averageDuration: durations.reduce(0, +) / Double(durations.count),
                minDuration: durations.min() ?? 0,
                maxDuration: durations.max() ?? 0
            )
        }
        
        return summaries
    }
    
    private func getCurrentMemoryUsage() -> Double {
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
            return Double(info.resident_size) / 1024 / 1024 // MB
        }
        
        return 0
    }
    
    private func getCurrentCPUUsage() -> Double {
        // Simplified CPU usage - in production would use more accurate methods
        return 0.0
    }
}

/// Configuration for telemetry system
public struct TelemetryConfiguration {
    public let maxMetrics: Int
    public let maxEvents: Int
    public let logMetrics: Bool
    public let logEvents: Bool
    public let exportInterval: TimeInterval?
    
    public init(
        maxMetrics: Int = 10000,
        maxEvents: Int = 1000,
        logMetrics: Bool = false,
        logEvents: Bool = true,
        exportInterval: TimeInterval? = nil
    ) {
        self.maxMetrics = maxMetrics
        self.maxEvents = maxEvents
        self.logMetrics = logMetrics
        self.logEvents = logEvents
        self.exportInterval = exportInterval
    }
}

/// Individual metric data point
public struct Metric: Codable {
    public let name: String
    public let type: MetricType
    public let value: Double
    public let tags: [String: String]
    public let timestamp: Date
    
    public init(name: String, type: MetricType, value: Double, tags: [String: String], timestamp: Date) {
        self.name = name
        self.type = type
        self.value = value
        self.tags = tags
        self.timestamp = timestamp
    }
}

/// Types of metrics
public enum MetricType: String, Codable, CaseIterable {
    case counter = "counter"
    case timing = "timing"
    case gauge = "gauge"
    case histogram = "histogram"
}

/// Telemetry event for tracking significant occurrences
public struct TelemetryEvent: Codable {
    public let name: String
    public let description: String
    public let severity: EventSeverity
    public let metadata: [String: String]
    public let timestamp: Date
    
    public init(
        name: String,
        description: String,
        severity: EventSeverity = .info,
        metadata: [String: String] = [:],
        timestamp: Date = Date()
    ) {
        self.name = name
        self.description = description
        self.severity = severity
        self.metadata = metadata
        self.timestamp = timestamp
    }
}

/// Event severity levels
public enum EventSeverity: String, Codable, CaseIterable {
    case debug = "debug"
    case info = "info"
    case warning = "warning"
    case error = "error"
    case critical = "critical"
}

/// Timer token for measuring operation duration
public struct TimerToken {
    private let name: String
    private let startTime: Date
    private let telemetry: TelemetrySystem
    
    init(name: String, startTime: Date, telemetry: TelemetrySystem) {
        self.name = name
        self.startTime = startTime
        self.telemetry = telemetry
    }
    
    /// Stop the timer and record the duration
    public func stop(tags: [String: String] = [:]) async {
        let duration = Date().timeIntervalSince(startTime)
        await telemetry.recordTiming(name, duration: duration, tags: tags)
    }
}

/// Summary statistics for a metric
public struct MetricSummary {
    public let name: String
    public let count: Int
    public let sum: Double
    public let min: Double
    public let max: Double
    public let average: Double
    public let firstTimestamp: Date
    public let lastTimestamp: Date
    
    public init(name: String, count: Int, sum: Double, min: Double, max: Double, average: Double, firstTimestamp: Date, lastTimestamp: Date) {
        self.name = name
        self.count = count
        self.sum = sum
        self.min = min
        self.max = max
        self.average = average
        self.firstTimestamp = firstTimestamp
        self.lastTimestamp = lastTimestamp
    }
}

/// Timer summary statistics
public struct TimerSummary: Codable {
    public let name: String
    public let count: Int
    public let totalDuration: TimeInterval
    public let averageDuration: TimeInterval
    public let minDuration: TimeInterval
    public let maxDuration: TimeInterval
    
    public init(name: String, count: Int, totalDuration: TimeInterval, averageDuration: TimeInterval, minDuration: TimeInterval, maxDuration: TimeInterval) {
        self.name = name
        self.count = count
        self.totalDuration = totalDuration
        self.averageDuration = averageDuration
        self.minDuration = minDuration
        self.maxDuration = maxDuration
    }
}

/// System metrics
public struct SystemMetrics: Sendable {
    public let memoryUsage: Double // MB
    public let cpuUsage: Double // Percentage
    public let timestamp: Date
    
    public init(memoryUsage: Double, cpuUsage: Double, timestamp: Date) {
        self.memoryUsage = memoryUsage
        self.cpuUsage = cpuUsage
        self.timestamp = timestamp
    }
}

/// Export format for metrics
public struct MetricsExport: Codable {
    public let timestamp: Date
    public let metrics: [Metric]
    public let events: [TelemetryEvent]
    public let counters: [String: Int]
    public let timerSummaries: [String: TimerSummary]
    
    public init(timestamp: Date, metrics: [Metric], events: [TelemetryEvent], counters: [String: Int], timerSummaries: [String: TimerSummary]) {
        self.timestamp = timestamp
        self.metrics = metrics
        self.events = events
        self.counters = counters
        self.timerSummaries = timerSummaries
    }
}

/// Embedding-specific telemetry metrics
public extension TelemetrySystem {
    /// Record embedding operation metrics
    func recordEmbeddingOperation(
        operation: String,
        duration: TimeInterval,
        inputLength: Int,
        outputDimensions: Int,
        batchSize: Int = 1,
        success: Bool = true
    ) async {
        let tags = [
            "operation": operation,
            "batch_size": String(batchSize),
            "success": String(success)
        ]
        
        recordTiming("embedding.duration", duration: duration, tags: tags)
        recordHistogram("embedding.input_length", value: Double(inputLength), tags: tags)
        recordGauge("embedding.output_dimensions", value: Double(outputDimensions), tags: tags)
        incrementCounter("embedding.operations", tags: tags)
        
        if !success {
            incrementCounter("embedding.errors", tags: tags)
        }
    }
    
    /// Record model loading metrics
    func recordModelLoad(
        modelId: String,
        loadDuration: TimeInterval,
        modelSize: Int,
        success: Bool = true
    ) async {
        let tags = [
            "model_id": modelId,
            "success": String(success)
        ]
        
        recordTiming("model.load_duration", duration: loadDuration, tags: tags)
        recordGauge("model.size_bytes", value: Double(modelSize), tags: tags)
        incrementCounter("model.loads", tags: tags)
        
        let event = TelemetryEvent(
            name: "model_loaded",
            description: "Model \(modelId) loaded in \(loadDuration)s",
            severity: success ? .info : .error,
            metadata: tags
        )
        recordEvent(event)
    }
    
    /// Record cache metrics
    func recordCacheOperation(
        operation: String,
        hit: Bool,
        keySize: Int = 0,
        valueSize: Int = 0
    ) {
        let tags = [
            "operation": operation,
            "result": hit ? "hit" : "miss"
        ]
        
        incrementCounter("cache.operations", tags: tags)
        
        if keySize > 0 {
            recordHistogram("cache.key_size", value: Double(keySize), tags: tags)
        }
        
        if valueSize > 0 {
            recordHistogram("cache.value_size", value: Double(valueSize), tags: tags)
        }
    }
}

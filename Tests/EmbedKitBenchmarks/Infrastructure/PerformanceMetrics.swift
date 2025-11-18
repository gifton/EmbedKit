import Foundation

/// Performance statistics calculated from benchmark timing samples
public struct PerformanceStatistics: Codable, Sendable {
    public let samples: [TimeInterval]

    // MARK: - Basic Statistics

    /// Arithmetic mean of all samples
    public var mean: TimeInterval {
        samples.reduce(0, +) / Double(samples.count)
    }

    /// Median value (50th percentile)
    public var median: TimeInterval {
        sorted[samples.count / 2]
    }

    /// Minimum observed time
    public var min: TimeInterval {
        samples.min() ?? 0
    }

    /// Maximum observed time
    public var max: TimeInterval {
        samples.max() ?? 0
    }

    // MARK: - Percentiles

    /// 50th percentile (same as median)
    public var p50: TimeInterval { median }

    /// 90th percentile
    public var p90: TimeInterval {
        sorted[Int(Double(samples.count) * 0.90)]
    }

    /// 95th percentile
    public var p95: TimeInterval {
        sorted[Int(Double(samples.count) * 0.95)]
    }

    /// 99th percentile - important for tail latency
    public var p99: TimeInterval {
        sorted[Int(Double(samples.count) * 0.99)]
    }

    // MARK: - Variability Metrics

    /// Standard deviation - measure of spread
    public var standardDeviation: TimeInterval {
        let avg = mean
        let variance = samples.map { pow($0 - avg, 2) }.reduce(0, +) / Double(samples.count)
        return sqrt(variance)
    }

    /// Coefficient of variation (std dev / mean)
    /// Values < 0.10 indicate stable measurements
    public var coefficientOfVariation: Double {
        standardDeviation / mean
    }

    // MARK: - Private

    private var sorted: [TimeInterval] {
        samples.sorted()
    }

    // MARK: - Formatting

    /// Format time in milliseconds
    public func formatMs(_ time: TimeInterval) -> String {
        String(format: "%.2f ms", time * 1000)
    }

    /// Format time in microseconds
    public func formatUs(_ time: TimeInterval) -> String {
        String(format: "%.2f μs", time * 1_000_000)
    }
}

/// Memory usage statistics from benchmark measurements
public struct MemoryStatistics: Codable, Sendable {
    public let deltas: [Int64]      // Per-iteration memory changes (bytes)
    public let total: Int64         // Total memory change (bytes)

    /// Average memory change per operation
    public var averageDelta: Int64 {
        guard !deltas.isEmpty else { return 0 }
        return deltas.reduce(0, +) / Int64(deltas.count)
    }

    /// Maximum memory change observed
    public var maxDelta: Int64 {
        deltas.max() ?? 0
    }

    /// Minimum memory change observed
    public var minDelta: Int64 {
        deltas.min() ?? 0
    }

    /// Format bytes as human-readable string
    public func formatBytes(_ bytes: Int64) -> String {
        let kb = Double(bytes) / 1024.0
        let mb = kb / 1024.0

        if mb >= 1.0 {
            return String(format: "%.2f MB", mb)
        } else if kb >= 1.0 {
            return String(format: "%.2f KB", kb)
        } else {
            return "\(bytes) bytes"
        }
    }
}

/// Complete result from a benchmark run
public struct BenchmarkResult: Codable, Sendable {
    public let name: String
    public let timing: PerformanceStatistics
    public let memory: MemoryStatistics?
    public let iterations: Int
    public let timestamp: Date
    public let hardware: HardwareInfo

    public init(
        name: String,
        timing: PerformanceStatistics,
        memory: MemoryStatistics? = nil,
        iterations: Int,
        timestamp: Date = Date(),
        hardware: HardwareInfo = .current
    ) {
        self.name = name
        self.timing = timing
        self.memory = memory
        self.iterations = iterations
        self.timestamp = timestamp
        self.hardware = hardware
    }

    /// Check if this benchmark meets specified performance targets
    /// - Parameters:
    ///   - p50: Target for 50th percentile (optional)
    ///   - p95: Target for 95th percentile (optional)
    ///   - p99: Target for 99th percentile (optional)
    /// - Returns: true if all specified targets are met
    public func meetsTarget(
        p50: TimeInterval? = nil,
        p95: TimeInterval? = nil,
        p99: TimeInterval? = nil
    ) -> Bool {
        if let p50Target = p50, timing.p50 > p50Target { return false }
        if let p95Target = p95, timing.p95 > p95Target { return false }
        if let p99Target = p99, timing.p99 > p99Target { return false }
        return true
    }
}

/// Hardware configuration for reproducibility
public struct HardwareInfo: Codable, Sendable {
    public let model: String
    public let processorCount: Int
    public let memoryGB: Int

    public init(model: String, processorCount: Int, memoryGB: Int) {
        self.model = model
        self.processorCount = processorCount
        self.memoryGB = memoryGB
    }

    /// Current hardware configuration
    public static var current: HardwareInfo {
        // Detect hardware model from system
        let model = detectHardwareModel()

        return HardwareInfo(
            model: model,
            processorCount: ProcessInfo.processInfo.processorCount,
            memoryGB: Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024))
        )
    }

    private static func detectHardwareModel() -> String {
        // Try to detect Apple Silicon vs Intel
        #if arch(arm64)
        // Apple Silicon - could parse from sysctl for M1/M2/M3/M4 detection
        return "Apple Silicon (ARM64)"
        #elseif arch(x86_64)
        return "Intel (x86_64)"
        #else
        return "Unknown"
        #endif
    }
}

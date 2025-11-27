// EmbedKit - Metrics

import Foundation
#if canImport(Darwin)
import Darwin.Mach
#endif

// MARK: - Histogram

/// A histogram for tracking latency distributions with configurable bucket boundaries.
///
/// Provides efficient percentile calculations and bucket-based distribution analysis.
///
/// Example:
/// ```swift
/// var histogram = LatencyHistogram()
/// histogram.record(0.015) // 15ms
/// histogram.record(0.023) // 23ms
///
/// let stats = histogram.statistics
/// print("p50: \(stats.p50)s, p99: \(stats.p99)s")
/// ```
public struct LatencyHistogram: Codable, Sendable {
    /// Bucket boundaries in seconds (e.g., [0.001, 0.005, 0.01, ...])
    public let bucketBoundaries: [TimeInterval]

    /// Count of values in each bucket (buckets.count == boundaries.count + 1)
    public private(set) var buckets: [Int]

    /// Raw values for percentile calculation (limited to windowSize)
    private var values: [TimeInterval]

    /// Total count of all recorded values
    public private(set) var count: Int

    /// Sum of all recorded values
    public private(set) var sum: TimeInterval

    /// Minimum recorded value
    public private(set) var min: TimeInterval

    /// Maximum recorded value
    public private(set) var max: TimeInterval

    /// Maximum number of raw values to retain for percentile calculation
    public let windowSize: Int

    /// Default bucket boundaries optimized for embedding latencies (1ms to 10s)
    public static let defaultBoundaries: [TimeInterval] = [
        0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0
    ]

    public init(
        bucketBoundaries: [TimeInterval] = LatencyHistogram.defaultBoundaries,
        windowSize: Int = 1024
    ) {
        self.bucketBoundaries = bucketBoundaries.sorted()
        self.buckets = [Int](repeating: 0, count: bucketBoundaries.count + 1)
        self.values = []
        self.count = 0
        self.sum = 0
        self.min = .greatestFiniteMagnitude
        self.max = 0
        self.windowSize = windowSize
    }

    /// Record a latency value
    public mutating func record(_ value: TimeInterval) {
        count += 1
        sum += value
        min = Swift.min(min, value)
        max = Swift.max(max, value)

        // Find bucket
        let bucketIndex = bucketBoundaries.firstIndex { value < $0 } ?? bucketBoundaries.count
        buckets[bucketIndex] += 1

        // Store for percentile calculation
        values.append(value)
        if values.count > windowSize {
            values.removeFirst(values.count - windowSize)
        }
    }

    /// Reset all histogram data
    public mutating func reset() {
        buckets = [Int](repeating: 0, count: bucketBoundaries.count + 1)
        values.removeAll(keepingCapacity: true)
        count = 0
        sum = 0
        min = .greatestFiniteMagnitude
        max = 0
    }

    /// Calculate percentile from recorded values
    public func percentile(_ p: Double) -> TimeInterval {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let rank = Swift.max(0, Swift.min(sorted.count - 1, Int(Double(sorted.count - 1) * p / 100.0)))
        return sorted[rank]
    }

    /// Get comprehensive statistics
    public var statistics: HistogramStatistics {
        HistogramStatistics(
            count: count,
            sum: sum,
            min: count > 0 ? min : 0,
            max: max,
            mean: count > 0 ? sum / Double(count) : 0,
            p50: percentile(50),
            p75: percentile(75),
            p90: percentile(90),
            p95: percentile(95),
            p99: percentile(99),
            p999: percentile(99.9),
            bucketBoundaries: bucketBoundaries,
            bucketCounts: buckets
        )
    }
}

/// Comprehensive statistics from a histogram
public struct HistogramStatistics: Codable, Sendable {
    public let count: Int
    public let sum: TimeInterval
    public let min: TimeInterval
    public let max: TimeInterval
    public let mean: TimeInterval
    public let p50: TimeInterval
    public let p75: TimeInterval
    public let p90: TimeInterval
    public let p95: TimeInterval
    public let p99: TimeInterval
    public let p999: TimeInterval
    public let bucketBoundaries: [TimeInterval]
    public let bucketCounts: [Int]

    /// Human-readable distribution summary
    public var distributionSummary: String {
        var lines: [String] = []
        lines.append("Latency Distribution (n=\(count)):")
        lines.append("  min: \(formatDuration(min))")
        lines.append("  p50: \(formatDuration(p50))")
        lines.append("  p95: \(formatDuration(p95))")
        lines.append("  p99: \(formatDuration(p99))")
        lines.append("  max: \(formatDuration(max))")
        return lines.joined(separator: "\n")
    }

    private func formatDuration(_ seconds: TimeInterval) -> String {
        if seconds < 0.001 {
            return String(format: "%.1fµs", seconds * 1_000_000)
        } else if seconds < 1.0 {
            return String(format: "%.1fms", seconds * 1000)
        } else {
            return String(format: "%.2fs", seconds)
        }
    }
}

// MARK: - Model Metrics

public struct ModelMetrics: Codable, Sendable {
    public let totalRequests: Int
    public let totalTokensProcessed: Int
    public let averageLatency: TimeInterval
    public let p50Latency: TimeInterval
    public let p95Latency: TimeInterval
    public let p99Latency: TimeInterval
    public let throughput: Double
    public let cacheHitRate: Double
    public let memoryUsage: Int64
    public let lastUsed: Date
    public let latencyHistogram: [TimeInterval]
    public let tokenHistogram: [Int]

    /// Detailed latency statistics including all percentiles
    public let latencyStats: HistogramStatistics?

    public init(
        totalRequests: Int,
        totalTokensProcessed: Int,
        averageLatency: TimeInterval,
        p50Latency: TimeInterval,
        p95Latency: TimeInterval,
        p99Latency: TimeInterval,
        throughput: Double,
        cacheHitRate: Double,
        memoryUsage: Int64,
        lastUsed: Date,
        latencyHistogram: [TimeInterval],
        tokenHistogram: [Int],
        latencyStats: HistogramStatistics? = nil
    ) {
        self.totalRequests = totalRequests
        self.totalTokensProcessed = totalTokensProcessed
        self.averageLatency = averageLatency
        self.p50Latency = p50Latency
        self.p95Latency = p95Latency
        self.p99Latency = p99Latency
        self.throughput = throughput
        self.cacheHitRate = cacheHitRate
        self.memoryUsage = memoryUsage
        self.lastUsed = lastUsed
        self.latencyHistogram = latencyHistogram
        self.tokenHistogram = tokenHistogram
        self.latencyStats = latencyStats
    }
}

struct MetricsData {
    private var requests: Int = 0
    private var tokens: Int = 0
    private var totalTime: TimeInterval = 0
    private var lastUsed: Date = .distantPast
    private var latencyHistogram = LatencyHistogram()
    private var tokenCounts: [Int] = []
    private let windowSize = 512

    mutating func record(tokenCount: Int, time: TimeInterval) {
        requests &+= 1
        tokens &+= tokenCount
        totalTime += time
        lastUsed = Date()

        latencyHistogram.record(time)

        tokenCounts.append(tokenCount)
        if tokenCounts.count > windowSize { tokenCounts.removeFirst(tokenCounts.count - windowSize) }
    }

    mutating func reset() {
        requests = 0
        tokens = 0
        totalTime = 0
        latencyHistogram.reset()
        tokenCounts.removeAll(keepingCapacity: true)
    }

    func snapshot(memoryUsage: Int64 = 0, cacheHitRate: Double = 0.0) -> ModelMetrics {
        let stats = latencyHistogram.statistics
        let throughput = totalTime > 0 ? Double(tokens) / totalTime : 0

        return ModelMetrics(
            totalRequests: requests,
            totalTokensProcessed: tokens,
            averageLatency: stats.mean,
            p50Latency: stats.p50,
            p95Latency: stats.p95,
            p99Latency: stats.p99,
            throughput: throughput,
            cacheHitRate: cacheHitRate,
            memoryUsage: memoryUsage,
            lastUsed: lastUsed,
            latencyHistogram: [],  // Deprecated: use latencyStats instead
            tokenHistogram: tokenCounts,
            latencyStats: stats
        )
    }
}

// MARK: - Memory

@inline(__always)
func currentMemoryUsage() -> Int64 {
    #if canImport(Darwin)
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout.size(ofValue: info) / MemoryLayout<integer_t>.size)
    let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) { ptr in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { iptr in
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), iptr, &count)
        }
    }
    if kerr == KERN_SUCCESS {
        return Int64(info.phys_footprint)
    } else {
        return 0
    }
    #else
    return 0
    #endif
}
// MARK: - Stage Metrics (tokenization, inference, pooling)

public struct StageMetrics: Codable, Sendable {
    public let tokenizationAverage: TimeInterval
    public let inferenceAverage: TimeInterval
    public let poolingAverage: TimeInterval
    public let samples: Int
    /// Average items per micro-batch across recordings since last reset.
    public let averageBatchSize: Double

    /// Detailed tokenization latency statistics (p50, p95, p99, etc.)
    public let tokenizationStats: HistogramStatistics?
    /// Detailed inference latency statistics
    public let inferenceStats: HistogramStatistics?
    /// Detailed pooling latency statistics
    public let poolingStats: HistogramStatistics?

    public init(
        tokenizationAverage: TimeInterval,
        inferenceAverage: TimeInterval,
        poolingAverage: TimeInterval,
        samples: Int,
        averageBatchSize: Double,
        tokenizationStats: HistogramStatistics? = nil,
        inferenceStats: HistogramStatistics? = nil,
        poolingStats: HistogramStatistics? = nil
    ) {
        self.tokenizationAverage = tokenizationAverage
        self.inferenceAverage = inferenceAverage
        self.poolingAverage = poolingAverage
        self.samples = samples
        self.averageBatchSize = averageBatchSize
        self.tokenizationStats = tokenizationStats
        self.inferenceStats = inferenceStats
        self.poolingStats = poolingStats
    }

    /// Human-readable summary of stage latencies
    public var summary: String {
        var lines: [String] = []
        lines.append("Stage Metrics (n=\(samples)):")
        lines.append("  Tokenization: avg=\(formatMs(tokenizationAverage))")
        if let stats = tokenizationStats {
            lines.append("    p50=\(formatMs(stats.p50)), p95=\(formatMs(stats.p95)), p99=\(formatMs(stats.p99))")
        }
        lines.append("  Inference: avg=\(formatMs(inferenceAverage))")
        if let stats = inferenceStats {
            lines.append("    p50=\(formatMs(stats.p50)), p95=\(formatMs(stats.p95)), p99=\(formatMs(stats.p99))")
        }
        lines.append("  Pooling: avg=\(formatMs(poolingAverage))")
        if let stats = poolingStats {
            lines.append("    p50=\(formatMs(stats.p50)), p95=\(formatMs(stats.p95)), p99=\(formatMs(stats.p99))")
        }
        lines.append("  Batch size: \(String(format: "%.1f", averageBatchSize))")
        return lines.joined(separator: "\n")
    }

    private func formatMs(_ seconds: TimeInterval) -> String {
        String(format: "%.2fms", seconds * 1000)
    }
}

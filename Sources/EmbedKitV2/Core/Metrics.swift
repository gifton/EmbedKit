// EmbedKitV2 - Metrics (Week 1)

import Foundation
#if canImport(Darwin)
import Darwin.Mach
#endif

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
        tokenHistogram: [Int]
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
    }
}

struct MetricsData {
    private var requests: Int = 0
    private var tokens: Int = 0
    private var totalTime: TimeInterval = 0
    private var lastUsed: Date = .distantPast
    private var latencies: [TimeInterval] = []
    private var tokenCounts: [Int] = []
    private let windowSize = 512

    mutating func record(tokenCount: Int, time: TimeInterval) {
        requests &+= 1
        tokens &+= tokenCount
        totalTime += time
        lastUsed = Date()

        latencies.append(time)
        if latencies.count > windowSize { latencies.removeFirst(latencies.count - windowSize) }

        tokenCounts.append(tokenCount)
        if tokenCounts.count > windowSize { tokenCounts.removeFirst(tokenCounts.count - windowSize) }
    }

    func snapshot(memoryUsage: Int64 = 0, cacheHitRate: Double = 0.0) -> ModelMetrics {
        let avg = requests > 0 ? totalTime / Double(requests) : 0
        let p50 = percentile(latencies, 50)
        let p95 = percentile(latencies, 95)
        let p99 = percentile(latencies, 99)
        let throughput = totalTime > 0 ? Double(tokens) / totalTime : 0

        return ModelMetrics(
            totalRequests: requests,
            totalTokensProcessed: tokens,
            averageLatency: avg,
            p50Latency: p50,
            p95Latency: p95,
            p99Latency: p99,
            throughput: throughput,
            cacheHitRate: cacheHitRate,
            memoryUsage: memoryUsage,
            lastUsed: lastUsed,
            latencyHistogram: latencies,
            tokenHistogram: tokenCounts
        )
    }

    private func percentile(_ values: [TimeInterval], _ p: Int) -> TimeInterval {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let rank = max(0, min(sorted.count - 1, (sorted.count - 1) * p / 100))
        return sorted[rank]
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

// EmbedKit - Histogram Metrics Tests

import Testing
import Foundation
@testable import EmbedKit

@Suite("Histogram Metrics")
struct HistogramMetricsTests {

    // MARK: - LatencyHistogram Tests

    @Test("Empty histogram has zero values")
    func emptyHistogram() {
        let histogram = LatencyHistogram()
        let stats = histogram.statistics

        #expect(stats.count == 0)
        #expect(stats.sum == 0)
        #expect(stats.mean == 0)
        #expect(stats.p50 == 0)
        #expect(stats.p99 == 0)
    }

    @Test("Single value histogram")
    func singleValueHistogram() {
        var histogram = LatencyHistogram()
        histogram.record(0.025)  // 25ms

        let stats = histogram.statistics
        #expect(stats.count == 1)
        #expect(abs(stats.sum - 0.025) < 0.0001)
        #expect(abs(stats.mean - 0.025) < 0.0001)
        #expect(abs(stats.min - 0.025) < 0.0001)
        #expect(abs(stats.max - 0.025) < 0.0001)
        #expect(abs(stats.p50 - 0.025) < 0.0001)
    }

    @Test("Histogram tracks min and max")
    func histogramMinMax() {
        var histogram = LatencyHistogram()
        histogram.record(0.010)  // 10ms
        histogram.record(0.050)  // 50ms
        histogram.record(0.025)  // 25ms

        let stats = histogram.statistics
        #expect(abs(stats.min - 0.010) < 0.0001)
        #expect(abs(stats.max - 0.050) < 0.0001)
    }

    @Test("Histogram calculates correct mean")
    func histogramMean() {
        var histogram = LatencyHistogram()
        histogram.record(0.010)
        histogram.record(0.020)
        histogram.record(0.030)

        let stats = histogram.statistics
        // Mean of 10, 20, 30 = 20ms
        #expect(abs(stats.mean - 0.020) < 0.0001)
    }

    @Test("Histogram calculates percentiles correctly")
    func histogramPercentiles() {
        var histogram = LatencyHistogram()

        // Record 100 values from 1ms to 100ms
        for i in 1...100 {
            histogram.record(Double(i) / 1000.0)
        }

        let stats = histogram.statistics
        #expect(stats.count == 100)

        // p50 should be around 50ms
        #expect(abs(stats.p50 - 0.050) < 0.002)

        // p95 should be around 95ms
        #expect(abs(stats.p95 - 0.095) < 0.002)

        // p99 should be around 99ms
        #expect(abs(stats.p99 - 0.099) < 0.002)
    }

    @Test("Histogram buckets count correctly")
    func histogramBuckets() {
        var histogram = LatencyHistogram()

        // Record values in different buckets
        histogram.record(0.0005)  // < 1ms bucket
        histogram.record(0.003)   // 2-5ms bucket
        histogram.record(0.015)   // 10-20ms bucket
        histogram.record(0.015)   // 10-20ms bucket
        histogram.record(15.0)    // > 10s bucket

        let stats = histogram.statistics
        #expect(stats.count == 5)

        // Verify bucket distribution
        let totalInBuckets = stats.bucketCounts.reduce(0, +)
        #expect(totalInBuckets == 5)
    }

    @Test("Histogram reset clears all data")
    func histogramReset() {
        var histogram = LatencyHistogram()
        histogram.record(0.010)
        histogram.record(0.020)

        #expect(histogram.count == 2)

        histogram.reset()

        let stats = histogram.statistics
        #expect(stats.count == 0)
        #expect(stats.sum == 0)
        #expect(stats.bucketCounts.allSatisfy { $0 == 0 })
    }

    @Test("Histogram window size limits stored values")
    func histogramWindowSize() {
        var histogram = LatencyHistogram(windowSize: 10)

        // Record 20 values
        for i in 1...20 {
            histogram.record(Double(i) / 1000.0)
        }

        // Count should be 20 (total recorded)
        #expect(histogram.count == 20)

        // But percentiles should be based on last 10 values (11-20ms)
        let stats = histogram.statistics
        // p50 of [11,12,13,14,15,16,17,18,19,20] = around 15ms
        #expect(stats.p50 >= 0.014 && stats.p50 <= 0.016)
    }

    @Test("Histogram custom boundaries")
    func histogramCustomBoundaries() {
        let boundaries: [TimeInterval] = [0.001, 0.01, 0.1, 1.0]
        var histogram = LatencyHistogram(bucketBoundaries: boundaries)

        histogram.record(0.0005)  // bucket 0: < 1ms
        histogram.record(0.005)   // bucket 1: 1-10ms
        histogram.record(0.05)    // bucket 2: 10-100ms
        histogram.record(0.5)     // bucket 3: 100ms-1s
        histogram.record(2.0)     // bucket 4: > 1s

        let stats = histogram.statistics
        #expect(stats.bucketCounts == [1, 1, 1, 1, 1])
    }

    // MARK: - HistogramStatistics Tests

    @Test("Distribution summary formatting")
    func distributionSummaryFormatting() {
        var histogram = LatencyHistogram()
        for i in 1...100 {
            histogram.record(Double(i) / 1000.0)
        }

        let stats = histogram.statistics
        let summary = stats.distributionSummary

        #expect(summary.contains("Latency Distribution"))
        #expect(summary.contains("min:"))
        #expect(summary.contains("p50:"))
        #expect(summary.contains("p95:"))
        #expect(summary.contains("p99:"))
        #expect(summary.contains("max:"))
    }

    // MARK: - ModelMetrics Integration Tests

    @Test("ModelMetrics includes latencyStats")
    func modelMetricsLatencyStats() {
        var metricsData = MetricsData()
        metricsData.record(tokenCount: 10, time: 0.015)
        metricsData.record(tokenCount: 20, time: 0.025)
        metricsData.record(tokenCount: 15, time: 0.020)

        let metrics = metricsData.snapshot()

        #expect(metrics.totalRequests == 3)
        #expect(metrics.latencyStats != nil)
        #expect(metrics.latencyStats?.count == 3)
        #expect(abs(metrics.averageLatency - 0.020) < 0.001)
    }

    // MARK: - StageMetrics Integration Tests

    @Test("StageMetrics includes histogram stats")
    func stageMetricsHistogramStats() {
        let metrics = StageMetrics(
            tokenizationAverage: 0.005,
            inferenceAverage: 0.020,
            poolingAverage: 0.002,
            samples: 10,
            averageBatchSize: 8.0,
            tokenizationStats: HistogramStatistics(
                count: 10, sum: 0.05, min: 0.003, max: 0.008,
                mean: 0.005, p50: 0.005, p75: 0.006, p90: 0.007,
                p95: 0.007, p99: 0.008, p999: 0.008,
                bucketBoundaries: [], bucketCounts: []
            ),
            inferenceStats: nil,
            poolingStats: nil
        )

        #expect(metrics.tokenizationStats != nil)
        #expect(metrics.tokenizationStats?.p50 == 0.005)
    }

    @Test("StageMetrics summary includes percentiles")
    func stageMetricsSummary() {
        let tokStats = HistogramStatistics(
            count: 10, sum: 0.05, min: 0.003, max: 0.008,
            mean: 0.005, p50: 0.005, p75: 0.006, p90: 0.007,
            p95: 0.007, p99: 0.008, p999: 0.008,
            bucketBoundaries: [], bucketCounts: []
        )

        let metrics = StageMetrics(
            tokenizationAverage: 0.005,
            inferenceAverage: 0.020,
            poolingAverage: 0.002,
            samples: 10,
            averageBatchSize: 8.0,
            tokenizationStats: tokStats,
            inferenceStats: nil,
            poolingStats: nil
        )

        let summary = metrics.summary
        #expect(summary.contains("Stage Metrics"))
        #expect(summary.contains("Tokenization:"))
        #expect(summary.contains("p50="))
        #expect(summary.contains("p95="))
        #expect(summary.contains("p99="))
    }

    // MARK: - Edge Cases

    @Test("Histogram handles very small values")
    func histogramSmallValues() {
        var histogram = LatencyHistogram()
        histogram.record(0.000001)  // 1 microsecond
        histogram.record(0.0000005) // 0.5 microseconds

        let stats = histogram.statistics
        #expect(stats.count == 2)
        #expect(stats.min > 0)
    }

    @Test("Histogram handles very large values")
    func histogramLargeValues() {
        var histogram = LatencyHistogram()
        histogram.record(100.0)  // 100 seconds
        histogram.record(200.0)  // 200 seconds

        let stats = histogram.statistics
        #expect(stats.count == 2)
        #expect(abs(stats.mean - 150.0) < 0.001)
    }

    @Test("Histogram handles negative values gracefully")
    func histogramNegativeValues() {
        var histogram = LatencyHistogram()
        histogram.record(-0.001)  // Shouldn't happen but handle gracefully
        histogram.record(0.010)

        let stats = histogram.statistics
        #expect(stats.count == 2)
    }
}

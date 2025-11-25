// Tests for FinalBenchmarks
import Testing
import Foundation
@testable import EmbedKit

@Suite("Final Benchmarks")
struct FinalBenchmarksTests {

    // MARK: - Metric Types Tests

    @Test
    func latencyMetrics_calculatesPercentiles() {
        let latencies: [TimeInterval] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        let metrics = LatencyMetrics(latencies: latencies)

        #expect(metrics.samples == 10)
        #expect(metrics.min == 0.1)
        #expect(metrics.max == 1.0)
        #expect(abs(metrics.mean - 0.55) < 0.01)
        #expect(metrics.p50 > 0)
        #expect(metrics.p95 > metrics.p50)
        #expect(metrics.p99 >= metrics.p95)
    }

    @Test
    func latencyMetrics_handlesEmptyArray() {
        let metrics = LatencyMetrics(latencies: [])

        #expect(metrics.samples == 0)
        #expect(metrics.min == 0)
        #expect(metrics.max == 0)
        #expect(metrics.mean == 0)
    }

    @Test
    func throughputMetrics_calculatesRates() {
        let metrics = ThroughputMetrics(
            documents: 100,
            tokens: 500,
            time: 2.0,
            batchSize: 10
        )

        #expect(metrics.documentsPerSecond == 50.0)
        #expect(metrics.tokensPerSecond == 250.0)
        #expect(metrics.totalDocuments == 100)
        #expect(metrics.batchSizeTested == 10)
    }

    @Test
    func throughputMetrics_handlesZeroTime() {
        let metrics = ThroughputMetrics(
            documents: 100,
            tokens: 500,
            time: 0,
            batchSize: 10
        )

        #expect(metrics.documentsPerSecond == 0)
        #expect(metrics.tokensPerSecond == 0)
    }

    @Test
    func memoryMetrics_calculatesPerDocument() {
        let metrics = MemoryMetrics(
            peak: 2_000_000,
            baseline: 1_000_000,
            documents: 100
        )

        #expect(metrics.memoryPerDocument == 10_000)
        #expect(metrics.peakMemory == 2_000_000)
        #expect(metrics.baselineMemory == 1_000_000)
    }

    @Test
    func concurrencyMetrics_calculatesSpeedup() {
        let metrics = ConcurrencyMetrics(
            sequential: 10.0,
            concurrent: 2.5,
            level: 4
        )

        #expect(metrics.speedup == 4.0)
        #expect(metrics.concurrencyLevel == 4)
    }

    // MARK: - Benchmark Report Tests

    @Test
    func benchmarkReport_initializesCorrectly() {
        let report = BenchmarkReport(modelID: "test/model@1.0")

        #expect(report.modelID == "test/model@1.0")
        #expect(report.modelLoading == nil)
        #expect(report.singleLatency == nil)
    }

    @Test
    func benchmarkReport_exportsJSON() throws {
        var report = BenchmarkReport(modelID: "test/model@1.0")
        report.singleLatency = LatencyMetrics(latencies: [0.1, 0.2, 0.3])

        let json = try report.exportJSON()
        #expect(!json.isEmpty)

        let string = try report.exportJSONString()
        // JSON encoder escapes slashes, so check for escaped version
        #expect(string.contains("test\\/model@1.0") || string.contains("test/model@1.0"))
        #expect(string.contains("singleLatency"))
    }

    @Test
    func systemInfo_capturesInfo() {
        let info = BenchmarkReport.SystemInfo()

        #expect(!info.platform.isEmpty)
        #expect(!info.osVersion.isEmpty)
    }

    // MARK: - Regression Report Tests

    @Test
    func regressionReport_detectsNoRegression() {
        var baseline = BenchmarkReport(modelID: "test")
        baseline.singleLatency = LatencyMetrics(latencies: [0.1, 0.1, 0.1])

        var current = BenchmarkReport(modelID: "test")
        current.singleLatency = LatencyMetrics(latencies: [0.1, 0.1, 0.1])

        let regression = RegressionReport(current: current, baseline: baseline, threshold: 0.1)

        #expect(!regression.hasRegressions)
    }

    @Test
    func regressionReport_detectsRegression() {
        var baseline = BenchmarkReport(modelID: "test")
        baseline.singleLatency = LatencyMetrics(latencies: [0.1, 0.1, 0.1])

        var current = BenchmarkReport(modelID: "test")
        // 50% slower
        current.singleLatency = LatencyMetrics(latencies: [0.15, 0.15, 0.15])

        let regression = RegressionReport(current: current, baseline: baseline, threshold: 0.1)

        #expect(regression.hasRegressions)
    }

    @Test
    func regressionReport_generatesSummary() {
        var baseline = BenchmarkReport(modelID: "test")
        baseline.singleLatency = LatencyMetrics(latencies: [0.1])

        var current = BenchmarkReport(modelID: "test")
        current.singleLatency = LatencyMetrics(latencies: [0.1])

        let regression = RegressionReport(current: current, baseline: baseline)

        let summary = regression.summary
        #expect(summary.contains("Benchmark Regression Report"))
    }

    // MARK: - Production Benchmarks Tests

    @Test
    func productionBenchmarks_runsLoadingBenchmark() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let benchmarks = ProductionBenchmarks(manager: manager, modelID: model.id)

        let metrics = try await benchmarks.benchmarkModelLoading()

        #expect(metrics.coldLoadTime >= 0)
        #expect(metrics.warmLoadTime >= 0)
    }

    @Test
    func productionBenchmarks_runsLatencyBenchmark() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let benchmarks = ProductionBenchmarks(manager: manager, modelID: model.id)

        let metrics = try await benchmarks.benchmarkSingleLatency(samples: 5)

        #expect(metrics.samples == 5)
        #expect(metrics.p50 >= 0)
        #expect(metrics.mean >= 0)
    }

    @Test
    func productionBenchmarks_runsThroughputBenchmark() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let benchmarks = ProductionBenchmarks(manager: manager, modelID: model.id)

        let metrics = try await benchmarks.benchmarkBatchThroughput(batchSize: 4, batches: 2)

        #expect(metrics.totalDocuments == 8)
        #expect(metrics.documentsPerSecond > 0)
    }

    @Test
    func productionBenchmarks_runsMemoryBenchmark() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let benchmarks = ProductionBenchmarks(manager: manager, modelID: model.id)

        let metrics = try await benchmarks.benchmarkMemory(documents: 5)

        #expect(metrics.documentsProcessed == 5)
        #expect(metrics.peakMemory >= metrics.baselineMemory)
    }

    @Test
    func productionBenchmarks_runsConcurrencyBenchmark() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let benchmarks = ProductionBenchmarks(manager: manager, modelID: model.id)

        let metrics = try await benchmarks.benchmarkConcurrency(level: 2, requests: 4)

        #expect(metrics.sequentialTime >= 0)
        #expect(metrics.concurrentTime >= 0)
        #expect(metrics.concurrencyLevel == 2)
    }

    @Test
    func productionBenchmarks_runsEdgeCaseBenchmark() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let benchmarks = ProductionBenchmarks(manager: manager, modelID: model.id)

        let metrics = try await benchmarks.benchmarkEdgeCases()

        #expect(metrics.emptyInputHandled)
        #expect(metrics.unicodeHandled)
        #expect(metrics.specialCharsHandled)
    }

    @Test
    func productionBenchmarks_runsCompleteSuite() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let benchmarks = ProductionBenchmarks(manager: manager, modelID: model.id)

        let report = try await benchmarks.runCompleteSuite(
            samples: 3,
            batchSize: 2,
            concurrencyLevel: 2
        )

        #expect(report.modelLoading != nil)
        #expect(report.singleLatency != nil)
        #expect(report.batchThroughput != nil)
        #expect(report.memoryEfficiency != nil)
        #expect(report.concurrency != nil)
        #expect(report.edgeCases != nil)
    }

    // MARK: - Quick Benchmark Test

    @Test
    func quickBenchmark_runsSuccessfully() async throws {
        let manager = ModelManager()
        let report = try await runQuickBenchmark(using: manager)

        #expect(!report.modelID.isEmpty)
        #expect(report.modelLoading != nil)
        #expect(report.singleLatency != nil)
    }
}

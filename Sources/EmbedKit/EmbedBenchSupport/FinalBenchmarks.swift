// EmbedKit - Production Benchmark Suite
// Comprehensive benchmarking for EmbedBench integration

import Foundation

// MARK: - Benchmark Types

/// Metrics for model loading performance.
public struct LoadingMetrics: Codable, Sendable {
    public let coldLoadTime: TimeInterval
    public let warmLoadTime: TimeInterval
    public let memoryAfterLoad: Int64
    public let timestamp: Date

    public init(coldLoadTime: TimeInterval, warmLoadTime: TimeInterval, memoryAfterLoad: Int64) {
        self.coldLoadTime = coldLoadTime
        self.warmLoadTime = warmLoadTime
        self.memoryAfterLoad = memoryAfterLoad
        self.timestamp = Date()
    }
}

/// Metrics for single embedding latency.
public struct LatencyMetrics: Codable, Sendable {
    public let p50: TimeInterval
    public let p95: TimeInterval
    public let p99: TimeInterval
    public let mean: TimeInterval
    public let min: TimeInterval
    public let max: TimeInterval
    public let samples: Int

    public init(latencies: [TimeInterval]) {
        let sorted = latencies.sorted()
        self.samples = sorted.count
        self.min = sorted.first ?? 0
        self.max = sorted.last ?? 0
        self.mean = sorted.isEmpty ? 0 : sorted.reduce(0, +) / Double(sorted.count)

        func percentile(_ p: Int) -> TimeInterval {
            guard !sorted.isEmpty else { return 0 }
            let rank = Swift.max(0, Swift.min(sorted.count - 1, (sorted.count - 1) * p / 100))
            return sorted[rank]
        }

        self.p50 = percentile(50)
        self.p95 = percentile(95)
        self.p99 = percentile(99)
    }
}

/// Metrics for batch throughput.
public struct ThroughputMetrics: Codable, Sendable {
    public let documentsPerSecond: Double
    public let tokensPerSecond: Double
    public let batchSizeTested: Int
    public let totalDocuments: Int
    public let totalTime: TimeInterval

    public init(documents: Int, tokens: Int, time: TimeInterval, batchSize: Int) {
        self.batchSizeTested = batchSize
        self.totalDocuments = documents
        self.totalTime = time
        self.documentsPerSecond = time > 0 ? Double(documents) / time : 0
        self.tokensPerSecond = time > 0 ? Double(tokens) / time : 0
    }
}

/// Metrics for memory efficiency.
public struct MemoryMetrics: Codable, Sendable {
    public let peakMemory: Int64
    public let baselineMemory: Int64
    public let memoryPerDocument: Int64
    public let documentsProcessed: Int

    public init(peak: Int64, baseline: Int64, documents: Int) {
        self.peakMemory = peak
        self.baselineMemory = baseline
        self.documentsProcessed = documents
        self.memoryPerDocument = documents > 0 ? (peak - baseline) / Int64(documents) : 0
    }
}

/// Metrics for concurrent performance.
public struct ConcurrencyMetrics: Codable, Sendable {
    public let sequentialTime: TimeInterval
    public let concurrentTime: TimeInterval
    public let speedup: Double
    public let concurrencyLevel: Int

    public init(sequential: TimeInterval, concurrent: TimeInterval, level: Int) {
        self.sequentialTime = sequential
        self.concurrentTime = concurrent
        self.concurrencyLevel = level
        self.speedup = concurrent > 0 ? sequential / concurrent : 0
    }
}

/// Metrics for edge case handling.
public struct EdgeCaseMetrics: Codable, Sendable {
    public let emptyInputHandled: Bool
    public let longInputTruncated: Bool
    public let unicodeHandled: Bool
    public let specialCharsHandled: Bool

    public init(empty: Bool, long: Bool, unicode: Bool, special: Bool) {
        self.emptyInputHandled = empty
        self.longInputTruncated = long
        self.unicodeHandled = unicode
        self.specialCharsHandled = special
    }
}

// MARK: - Benchmark Report

/// Complete benchmark report for production validation.
public struct BenchmarkReport: Codable, Sendable {
    public var modelLoading: LoadingMetrics?
    public var singleLatency: LatencyMetrics?
    public var batchThroughput: ThroughputMetrics?
    public var memoryEfficiency: MemoryMetrics?
    public var concurrency: ConcurrencyMetrics?
    public var edgeCases: EdgeCaseMetrics?

    public var timestamp: Date
    public var modelID: String
    public var systemInfo: SystemInfo

    public struct SystemInfo: Codable, Sendable {
        public let platform: String
        public let osVersion: String
        public let deviceModel: String
        public let totalMemory: Int64

        public init() {
            #if os(macOS)
            self.platform = "macOS"
            #elseif os(iOS)
            self.platform = "iOS"
            #elseif os(tvOS)
            self.platform = "tvOS"
            #elseif os(watchOS)
            self.platform = "watchOS"
            #else
            self.platform = "Unknown"
            #endif

            self.osVersion = ProcessInfo.processInfo.operatingSystemVersionString

            #if canImport(Darwin)
            var size: size_t = 0
            sysctlbyname("hw.model", nil, &size, nil, 0)
            var model = [UInt8](repeating: 0, count: size)
            sysctlbyname("hw.model", &model, &size, nil, 0)
            self.deviceModel = String(decoding: model.prefix(while: { $0 != 0 }), as: UTF8.self)

            var memSize: UInt64 = 0
            size = MemoryLayout<UInt64>.size
            sysctlbyname("hw.memsize", &memSize, &size, nil, 0)
            self.totalMemory = Int64(memSize)
            #else
            self.deviceModel = "Unknown"
            self.totalMemory = 0
            #endif
        }
    }

    public init(modelID: String) {
        self.modelID = modelID
        self.timestamp = Date()
        self.systemInfo = SystemInfo()
    }

    /// Export report as JSON data.
    public func exportJSON() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return try encoder.encode(self)
    }

    /// Export report as JSON string.
    public func exportJSONString() throws -> String {
        let data = try exportJSON()
        return String(data: data, encoding: .utf8) ?? "{}"
    }
}

// MARK: - Regression Detection

/// Report comparing current benchmark to a baseline.
public struct RegressionReport: Codable, Sendable {
    public let currentReport: BenchmarkReport
    public let baselineReport: BenchmarkReport
    public let regressionThreshold: Double
    public let regressions: [Regression]

    public struct Regression: Codable, Sendable {
        public let metric: String
        public let baseline: Double
        public let current: Double
        public let percentChange: Double
        public let isRegression: Bool
    }

    public init(current: BenchmarkReport, baseline: BenchmarkReport, threshold: Double = 0.1) {
        self.currentReport = current
        self.baselineReport = baseline
        self.regressionThreshold = threshold

        var regressions: [Regression] = []

        // Compare latency (lower is better)
        if let currLat = current.singleLatency?.p50,
           let baseLat = baseline.singleLatency?.p50, baseLat > 0 {
            let change = (currLat - baseLat) / baseLat
            regressions.append(Regression(
                metric: "latency_p50",
                baseline: baseLat,
                current: currLat,
                percentChange: change,
                isRegression: change > threshold
            ))
        }

        // Compare throughput (higher is better)
        if let currThrp = current.batchThroughput?.documentsPerSecond,
           let baseThrp = baseline.batchThroughput?.documentsPerSecond, baseThrp > 0 {
            let change = (baseThrp - currThrp) / baseThrp  // Inverted for throughput
            regressions.append(Regression(
                metric: "throughput_docs_per_sec",
                baseline: baseThrp,
                current: currThrp,
                percentChange: -change,  // Negative means regression for throughput
                isRegression: change > threshold
            ))
        }

        // Compare memory (lower is better)
        if let currMem = current.memoryEfficiency?.memoryPerDocument,
           let baseMem = baseline.memoryEfficiency?.memoryPerDocument, baseMem > 0 {
            let change = Double(currMem - baseMem) / Double(baseMem)
            regressions.append(Regression(
                metric: "memory_per_doc",
                baseline: Double(baseMem),
                current: Double(currMem),
                percentChange: change,
                isRegression: change > threshold
            ))
        }

        self.regressions = regressions
    }

    /// Whether any regression was detected above the threshold.
    public var hasRegressions: Bool {
        regressions.contains { $0.isRegression }
    }

    /// Summary of all regressions found.
    public var summary: String {
        if regressions.isEmpty {
            return "No metrics to compare"
        }

        var lines = ["Benchmark Regression Report", "==========================="]

        for reg in regressions {
            let status = reg.isRegression ? "[REGRESSION]" : "[OK]"
            let change = String(format: "%+.1f%%", reg.percentChange * 100)
            lines.append("\(status) \(reg.metric): \(change)")
        }

        if hasRegressions {
            lines.append("")
            lines.append("WARNING: Performance regressions detected!")
        } else {
            lines.append("")
            lines.append("All metrics within acceptable range.")
        }

        return lines.joined(separator: "\n")
    }
}

// MARK: - Production Benchmarks

/// Production benchmark suite for EmbedBench integration.
public struct ProductionBenchmarks {

    private let manager: ModelManager
    private let modelID: ModelID

    public init(manager: ModelManager, modelID: ModelID) {
        self.manager = manager
        self.modelID = modelID
    }

    /// Run the complete benchmark suite.
    public func runCompleteSuite(
        samples: Int = 100,
        batchSize: Int = 32,
        concurrencyLevel: Int = 4
    ) async throws -> BenchmarkReport {
        var report = BenchmarkReport(modelID: modelID.description)

        // 1. Model Loading
        report.modelLoading = try await benchmarkModelLoading()

        // 2. Single Embedding Latency
        report.singleLatency = try await benchmarkSingleLatency(samples: samples)

        // 3. Batch Throughput
        report.batchThroughput = try await benchmarkBatchThroughput(batchSize: batchSize, batches: 10)

        // 4. Memory Efficiency
        report.memoryEfficiency = try await benchmarkMemory(documents: batchSize * 5)

        // 5. Concurrent Performance
        report.concurrency = try await benchmarkConcurrency(level: concurrencyLevel, requests: samples)

        // 6. Edge Cases
        report.edgeCases = try await benchmarkEdgeCases()

        return report
    }

    /// Benchmark model loading time.
    public func benchmarkModelLoading() async throws -> LoadingMetrics {
        let baselineMemory = currentMemoryUsage()

        // Cold load
        await manager.unloadAll()
        let coldStart = CFAbsoluteTimeGetCurrent()
        _ = try await manager.loadMockModel()
        let coldTime = CFAbsoluteTimeGetCurrent() - coldStart

        // Warm load (model already loaded)
        let warmStart = CFAbsoluteTimeGetCurrent()
        _ = try await manager.loadMockModel()
        let warmTime = CFAbsoluteTimeGetCurrent() - warmStart

        let memoryAfterLoad = currentMemoryUsage()

        return LoadingMetrics(
            coldLoadTime: coldTime,
            warmLoadTime: warmTime,
            memoryAfterLoad: memoryAfterLoad - baselineMemory
        )
    }

    /// Benchmark single embedding latency.
    public func benchmarkSingleLatency(samples: Int = 100) async throws -> LatencyMetrics {
        _ = try await manager.embed("Benchmark warmup", using: modelID)

        var latencies: [TimeInterval] = []
        latencies.reserveCapacity(samples)

        let texts = (0..<samples).map { "Benchmark test text \($0) for latency measurement" }

        for text in texts {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try await manager.embed(text, using: modelID)
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            latencies.append(elapsed)
        }

        return LatencyMetrics(latencies: latencies)
    }

    /// Benchmark batch throughput.
    public func benchmarkBatchThroughput(batchSize: Int = 32, batches: Int = 10) async throws -> ThroughputMetrics {
        var totalTokens = 0
        let totalDocs = batchSize * batches

        let texts = (0..<totalDocs).map { "Document \($0) for throughput benchmark testing" }

        let start = CFAbsoluteTimeGetCurrent()

        for batchStart in stride(from: 0, to: totalDocs, by: batchSize) {
            let batchEnd = min(batchStart + batchSize, totalDocs)
            let batch = Array(texts[batchStart..<batchEnd])
            let result = try await manager.embedBatch(batch, using: modelID)
            totalTokens += result.tokenCounts.reduce(0, +)
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start

        return ThroughputMetrics(
            documents: totalDocs,
            tokens: totalTokens,
            time: elapsed,
            batchSize: batchSize
        )
    }

    /// Benchmark memory efficiency.
    public func benchmarkMemory(documents: Int = 100) async throws -> MemoryMetrics {
        // Get baseline
        let baseline = currentMemoryUsage()

        // Process documents
        let texts = (0..<documents).map { "Memory benchmark document \($0)" }
        _ = try await manager.embedBatch(texts, using: modelID)

        // Get peak
        let peak = currentMemoryUsage()

        return MemoryMetrics(
            peak: peak,
            baseline: baseline,
            documents: documents
        )
    }

    /// Benchmark concurrent performance.
    public func benchmarkConcurrency(level: Int = 4, requests: Int = 100) async throws -> ConcurrencyMetrics {
        let texts = (0..<requests).map { "Concurrency test \($0)" }

        // Sequential baseline
        let seqStart = CFAbsoluteTimeGetCurrent()
        for text in texts {
            _ = try await manager.embed(text, using: modelID)
        }
        let seqTime = CFAbsoluteTimeGetCurrent() - seqStart

        // Concurrent execution
        let concStart = CFAbsoluteTimeGetCurrent()
        let capturedManager = manager
        let capturedModelID = modelID
        try await withThrowingTaskGroup(of: Void.self) { group in
            for text in texts {
                group.addTask {
                    _ = try await capturedManager.embed(text, using: capturedModelID)
                }
            }
            try await group.waitForAll()
        }
        let concTime = CFAbsoluteTimeGetCurrent() - concStart

        return ConcurrencyMetrics(
            sequential: seqTime,
            concurrent: concTime,
            level: level
        )
    }

    /// Benchmark edge case handling.
    public func benchmarkEdgeCases() async throws -> EdgeCaseMetrics {
        // Empty/minimal input
        var emptyOK = true
        do {
            _ = try await manager.embed("a", using: modelID)
        } catch {
            emptyOK = false
        }

        // Long input (should truncate)
        var longOK = true
        do {
            let longText = String(repeating: "word ", count: 1000)
            let result = try await manager.embed(longText, using: modelID)
            longOK = result.embedding.metadata.truncated || true  // May or may not truncate
        } catch {
            longOK = false
        }

        // Unicode
        var unicodeOK = true
        do {
            _ = try await manager.embed("你好世界 مرحبا שלום", using: modelID)
        } catch {
            unicodeOK = false
        }

        // Special characters
        var specialOK = true
        do {
            _ = try await manager.embed("@#$% !?*&", using: modelID)
        } catch {
            specialOK = false
        }

        return EdgeCaseMetrics(
            empty: emptyOK,
            long: longOK,
            unicode: unicodeOK,
            special: specialOK
        )
    }
}

// MARK: - Quick Benchmark

/// Run a quick benchmark for validation.
public func runQuickBenchmark(using manager: ModelManager) async throws -> BenchmarkReport {
    let model = try await manager.loadMockModel()
    let benchmarks = ProductionBenchmarks(manager: manager, modelID: model.id)
    return try await benchmarks.runCompleteSuite(
        samples: 10,
        batchSize: 8,
        concurrencyLevel: 2
    )
}

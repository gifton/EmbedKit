import XCTest
import CoreML
@testable import EmbedKit

/// Performance benchmarks for CoreML inference operations
///
/// These benchmarks measure the critical path: model inference latency
/// Target: <50ms p50, <100ms p99 for single inference on Apple Silicon
final class CoreMLInferenceBenchmarks: XCTestCase {

    private var backend: CoreMLBackend!
    private var reporter: BenchmarkReporter!

    // Model path - using compiled .mlmodelc
    private let modelURL = URL(fileURLWithPath: "/Users/goftin/dev/gsuite/VSK/EmbedKit/MiniLM-L12-v2.mlmodelc")

    // MARK: - Setup & Teardown

    override func setUp() async throws {
        backend = CoreMLBackend()
        try await backend.loadModel(from: modelURL)
        reporter = BenchmarkReporter()
    }

    override func tearDown() async throws {
        // Print summary after all benchmarks complete
        await reporter.printSummary()

        // Export results for tracking
        let timestamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")

        // JSON for programmatic analysis
        let jsonURL = URL(fileURLWithPath: "benchmark_results_coreml_\(timestamp).json")
        try await reporter.exportJSON(to: jsonURL)

        // Markdown for documentation
        let markdown = await reporter.exportMarkdown()
        try markdown.write(
            to: URL(fileURLWithPath: "BENCHMARK_RESULTS_COREML.md"),
            atomically: true,
            encoding: .utf8
        )

        print("📄 Results exported to:")
        print("   - \(jsonURL.path)")
        print("   - BENCHMARK_RESULTS_COREML.md")
    }

    // MARK: - B-001a: Single Inference Latency

    func testSingleInferenceLatency() async throws {
        let localBackend = self.backend!
        let input = TestFixtures.createValidInput(sequenceLength: 512)
        let runner = BenchmarkRunner()

        let result = try await runner.measure(name: "CoreML Single Inference (512 tokens)") {
            try await localBackend.generateEmbeddings(for: input)
        }

        await reporter.record(result)

        // Verify meets target: <50ms p50, <100ms p99
        XCTAssertTrue(
            result.meetsTarget(p50: 0.050, p99: 0.100),
            """
            Single inference performance target not met:
            - p50: \(result.timing.formatMs(result.timing.p50)) (target: <50ms)
            - p99: \(result.timing.formatMs(result.timing.p99)) (target: <100ms)
            """
        )

        // Log actual performance
        print("✅ Single Inference:")
        print("   p50: \(result.timing.formatMs(result.timing.p50))")
        print("   p99: \(result.timing.formatMs(result.timing.p99))")
    }

    // MARK: - B-001b: Batch Inference (Small)

    func testBatchInference_5items() async throws {
        let localBackend = self.backend!
        let inputs = TestFixtures.createBatchInputs(count: 5, sequenceLength: 512)
        let runner = BenchmarkRunner()

        let result = try await runner.measure(name: "CoreML Batch Inference (5×512 tokens)") {
            try await localBackend.generateEmbeddings(for: inputs)
        }

        await reporter.record(result)

        // Target: <250ms p50 (50ms per item)
        XCTAssertLessThan(
            result.timing.p50,
            0.250,
            "Batch inference (5 items) p50: \(result.timing.formatMs(result.timing.p50))"
        )

        print("✅ Batch Inference (5 items):")
        print("   p50: \(result.timing.formatMs(result.timing.p50))")
        print("   Per-item: \(result.timing.formatMs(result.timing.p50 / 5.0))")
    }

    // MARK: - B-001c: Batch Inference (Medium)

    func testBatchInference_20items() async throws {
        let localBackend = self.backend!
        let inputs = TestFixtures.createBatchInputs(count: 20, sequenceLength: 512)

        // Use fewer iterations for longer-running benchmarks
        let runner = BenchmarkRunner(configuration: .init(
            warmupIterations: 5,
            measurementIterations: 20
        ))

        let result = try await runner.measure(name: "CoreML Batch Inference (20×512 tokens)") {
            try await localBackend.generateEmbeddings(for: inputs)
        }

        await reporter.record(result)

        // Target: <1000ms p50 (50ms per item)
        XCTAssertLessThan(
            result.timing.p50,
            1.000,
            "Batch inference (20 items) p50: \(result.timing.formatMs(result.timing.p50))"
        )

        print("✅ Batch Inference (20 items):")
        print("   p50: \(result.timing.formatMs(result.timing.p50))")
        print("   Per-item: \(result.timing.formatMs(result.timing.p50 / 20.0))")
    }

    // MARK: - B-001d: Batch Inference (Large)

    func testBatchInference_50items() async throws {
        let localBackend = self.backend!
        let inputs = TestFixtures.createBatchInputs(count: 50, sequenceLength: 512)

        // Minimal iterations for very long benchmarks
        let runner = BenchmarkRunner(configuration: .longRunning)

        let result = try await runner.measure(name: "CoreML Large Batch Inference (50×512 tokens)") {
            try await localBackend.generateEmbeddings(for: inputs)
        }

        await reporter.record(result)

        // Target: <2500ms p50 (50ms per item)
        XCTAssertLessThan(
            result.timing.p50,
            2.500,
            "Large batch inference (50 items) p50: \(result.timing.formatMs(result.timing.p50))"
        )

        let throughput = 50.0 / result.timing.median  // items per second
        print("✅ Large Batch Inference (50 items):")
        print("   p50: \(result.timing.formatMs(result.timing.p50))")
        print("   Per-item: \(result.timing.formatMs(result.timing.p50 / 50.0))")
        print("   Throughput: \(String(format: "%.1f", throughput)) inferences/sec")
    }

    // MARK: - B-001e: Different Sequence Lengths

    func testInferenceLatency_VariableLength() async throws {
        let localBackend = self.backend!
        // Test how sequence length affects latency
        let lengths = [128, 256, 512]
        var results: [BenchmarkResult] = []

        for length in lengths {
            let input = TestFixtures.createValidInput(sequenceLength: length)
            let runner = BenchmarkRunner(configuration: .init(
                warmupIterations: 5,
                measurementIterations: 50
            ))

            let result = try await runner.measure(name: "CoreML Inference (\(length) tokens)") {
                try await localBackend.generateEmbeddings(for: input)
            }

            results.append(result)
        }

        // Record all results
        for result in results {
            await reporter.record(result)
        }

        // Analyze: Should show relatively constant time (model processes full 512 internally)
        print("📊 Sequence Length Analysis:")
        for (i, length) in lengths.enumerated() {
            print("   \(length) tokens: \(results[i].timing.formatMs(results[i].timing.p50))")
        }
    }

    // MARK: - B-001f: Model Configuration Comparison

    func testInferenceLatency_CPUvsCPUAndNeuralEngine() async throws {
        let localBackend = self.backend!
        // Compare CPU-only vs ANE-enabled performance

        // CPU-only configuration
        let cpuConfig = CoreMLConfiguration(
            useNeuralEngine: false,
            allowCPUFallback: true,
            maxBatchSize: 32
        )

        let cpuBackend = CoreMLBackend(configuration: cpuConfig)
        try await cpuBackend.loadModel(from: modelURL)

        let input = TestFixtures.createValidInput(sequenceLength: 512)

        // Benchmark CPU-only
        let cpuRunner = BenchmarkRunner(configuration: .init(
            warmupIterations: 5,
            measurementIterations: 30
        ))

        let cpuResult = try await cpuRunner.measure(name: "CoreML Inference (CPU-only)") {
            try await cpuBackend.generateEmbeddings(for: input)
        }

        // Benchmark ANE-enabled (default backend)
        let aneRunner = BenchmarkRunner(configuration: .init(
            warmupIterations: 5,
            measurementIterations: 30
        ))

        let aneResult = try await aneRunner.measure(name: "CoreML Inference (ANE-enabled)") {
            try await localBackend.generateEmbeddings(for: input)
        }

        await reporter.record(cpuResult)
        await reporter.record(aneResult)

        // Compare
        let speedup = cpuResult.timing.median / aneResult.timing.median
        print("⚡ CPU vs ANE Comparison:")
        print("   CPU-only: \(cpuResult.timing.formatMs(cpuResult.timing.p50))")
        print("   ANE-enabled: \(aneResult.timing.formatMs(aneResult.timing.p50))")
        print("   Speedup: \(String(format: "%.1fx", speedup))")

        // ANE should be faster (or at least not significantly slower)
        XCTAssertLessThanOrEqual(
            aneResult.timing.median,
            cpuResult.timing.median * 1.2,  // Allow 20% variance
            "ANE should provide acceleration"
        )
    }
}

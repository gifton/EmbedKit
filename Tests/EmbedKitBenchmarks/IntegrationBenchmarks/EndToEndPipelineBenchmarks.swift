import XCTest
import CoreML
@testable import EmbedKit

/// End-to-end pipeline performance benchmarks
///
/// Measures complete text → embedding flow including:
/// - Tokenization
/// - Model inference
/// - Pooling
/// - Normalization
///
/// Target: <100ms p50, <300ms p99 for typical text
final class EndToEndPipelineBenchmarks: XCTestCase {

    private var pipeline: EmbeddingPipeline!
    private var reporter: BenchmarkReporter!

    private let modelURL = URL(fileURLWithPath: "/Users/goftin/dev/gsuite/VSK/EmbedKit/MiniLM-L12-v2.mlmodelc")

    // MARK: - Setup & Teardown

    override func setUp() async throws {
        // Create backend
        let backend = CoreMLBackend()

        // Create tokenizer
        let tokenizer = try await BERTTokenizer(maxSequenceLength: 512)

        // Create pipeline with GPU acceleration
        pipeline = EmbeddingPipeline(
            tokenizer: tokenizer,
            backend: backend,
            configuration: EmbeddingPipelineConfiguration(
                poolingStrategy: .mean,
                normalize: true,
                useGPUAcceleration: true
            )
        )

        // Load model through pipeline (this sets isModelLoaded flag)
        try await pipeline.loadModel(from: modelURL)

        reporter = BenchmarkReporter()
    }

    override func tearDown() async throws {
        await reporter.printSummary()

        // Export results
        let timestamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")

        let jsonURL = URL(fileURLWithPath: "benchmark_results_e2e_\(timestamp).json")
        try await reporter.exportJSON(to: jsonURL)

        let markdown = await reporter.exportMarkdown()
        try markdown.write(
            to: URL(fileURLWithPath: "BENCHMARK_RESULTS_E2E.md"),
            atomically: true,
            encoding: .utf8
        )

        print("📄 End-to-end results exported")
    }

    // MARK: - B-005a: Typical Text

    func testEndToEnd_TypicalText() async throws {
        let localPipeline = self.pipeline!
        let text = TestFixtures.typicalText30Words
        let runner = BenchmarkRunner()

        let result = try await runner.measure(name: "End-to-End - Typical Text (30 words)") {
            try await localPipeline.embed(text)
        }

        await reporter.record(result)

        // Target: <100ms p50, <300ms p99
        XCTAssertTrue(
            result.meetsTarget(p50: 0.100, p99: 0.300),
            """
            End-to-end performance target not met:
            - p50: \(result.timing.formatMs(result.timing.p50)) (target: <100ms)
            - p99: \(result.timing.formatMs(result.timing.p99)) (target: <300ms)
            """
        )

        print("✅ End-to-End Typical Text:")
        print("   p50: \(result.timing.formatMs(result.timing.p50))")
        print("   p99: \(result.timing.formatMs(result.timing.p99))")
    }

    // MARK: - B-005b: Short Text

    func testEndToEnd_ShortText() async throws {
        let localPipeline = self.pipeline!
        let text = TestFixtures.shortText
        let runner = BenchmarkRunner()

        let result = try await runner.measure(name: "End-to-End - Short Text (5 words)") {
            try await localPipeline.embed(text)
        }

        await reporter.record(result)

        // Should be similar to typical (model inference dominates)
        XCTAssertTrue(
            result.meetsTarget(p50: 0.100),
            "Short text p50: \(result.timing.formatMs(result.timing.p50))"
        )

        print("✅ End-to-End Short Text:")
        print("   p50: \(result.timing.formatMs(result.timing.p50))")
    }

    // MARK: - B-005c: Medium Text

    func testEndToEnd_MediumText() async throws {
        let localPipeline = self.pipeline!
        let text = TestFixtures.mediumText50Words
        let runner = BenchmarkRunner()

        let result = try await runner.measure(name: "End-to-End - Medium Text (50 words)") {
            try await localPipeline.embed(text)
        }

        await reporter.record(result)

        // Tokenization overhead increases slightly
        XCTAssertTrue(
            result.meetsTarget(p50: 0.110),
            "Medium text p50: \(result.timing.formatMs(result.timing.p50))"
        )

        print("✅ End-to-End Medium Text:")
        print("   p50: \(result.timing.formatMs(result.timing.p50))")
    }

    // MARK: - B-005d: Long Text

    func testEndToEnd_LongText() async throws {
        let localPipeline = self.pipeline!
        let text = TestFixtures.longText200Words
        let runner = BenchmarkRunner()

        let result = try await runner.measure(name: "End-to-End - Long Text (200 words)") {
            try await localPipeline.embed(text)
        }

        await reporter.record(result)

        // Tokenization overhead more noticeable but still dominated by inference
        XCTAssertTrue(
            result.meetsTarget(p50: 0.120),
            "Long text p50: \(result.timing.formatMs(result.timing.p50))"
        )

        print("✅ End-to-End Long Text:")
        print("   p50: \(result.timing.formatMs(result.timing.p50))")
    }

    // MARK: - B-005e: Batch Processing (Small)

    func testEndToEnd_BatchProcessing_10texts() async throws {
        let localPipeline = self.pipeline!
        let texts = TestFixtures.generateRandomTexts(count: 10, averageWords: 30)

        let runner = BenchmarkRunner(configuration: .init(
            warmupIterations: 3,
            measurementIterations: 20
        ))

        let result = try await runner.measure(name: "End-to-End - Batch 10 texts") {
            try await localPipeline.embed(batch: texts)
        }

        await reporter.record(result)

        let throughput = 10.0 / result.timing.median
        print("✅ End-to-End Batch (10 texts):")
        print("   p50: \(result.timing.formatMs(result.timing.p50))")
        print("   Throughput: \(String(format: "%.1f", throughput)) texts/sec")

        // Should batch efficiently
        XCTAssertLessThan(result.timing.p50, 1.0)  // <1s for 10 texts
    }

    // MARK: - B-005f: Batch Processing (Large)

    func testEndToEnd_BatchProcessing_100texts() async throws {
        let localPipeline = self.pipeline!
        let texts = TestFixtures.generateRandomTexts(count: 100, averageWords: 30)

        let runner = BenchmarkRunner(configuration: .longRunning)

        let result = try await runner.measure(name: "End-to-End - Batch 100 texts") {
            try await localPipeline.embed(batch: texts)
        }

        await reporter.record(result)

        let throughput = 100.0 / result.timing.median
        print("✅ End-to-End Batch (100 texts):")
        print("   p50: \(result.timing.formatMs(result.timing.p50))")
        print("   Throughput: \(String(format: "%.1f", throughput)) texts/sec")

        // Should complete in reasonable time
        XCTAssertLessThan(result.timing.p50, 10.0)  // <10s for 100 texts
    }

    // MARK: - B-005g: Different Pooling Strategies

    func testEndToEnd_PoolingStrategies() async throws {
        let text = TestFixtures.typicalText30Words
        let strategies: [PoolingStrategy] = [.mean, .cls, .max]

        var results: [BenchmarkResult] = []

        for strategy in strategies {
            // Create pipeline with specific pooling strategy
            let backend = CoreMLBackend()

            let tokenizer = try await BERTTokenizer(maxSequenceLength: 512)

            let testPipeline = EmbeddingPipeline(
                tokenizer: tokenizer,
                backend: backend,
                configuration: EmbeddingPipelineConfiguration(
                    poolingStrategy: strategy,
                    normalize: true,
                    useGPUAcceleration: true
                )
            )

            // Load model through pipeline
            try await testPipeline.loadModel(from: modelURL)

            let runner = BenchmarkRunner(configuration: .init(
                warmupIterations: 5,
                measurementIterations: 50
            ))

            let result = try await runner.measure(name: "End-to-End - Pooling Strategy: \(strategy)") {
                try await testPipeline.embed(text)
            }

            results.append(result)
        }

        // Record all results
        for result in results {
            await reporter.record(result)
        }

        // Compare strategies
        print("📊 Pooling Strategy Comparison:")
        for (i, strategy) in strategies.enumerated() {
            print("   \(strategy): \(results[i].timing.formatMs(results[i].timing.p50))")
        }

        // All strategies should complete in reasonable time
        for result in results {
            XCTAssertTrue(
                result.meetsTarget(p50: 0.120),
                "\(result.name) exceeded target"
            )
        }
    }

    // MARK: - B-005h: Memory Usage

    func testEndToEnd_MemoryUsage() async throws {
        let localPipeline = self.pipeline!
        let text = TestFixtures.typicalText30Words

        // Force memory collection
        let runner = BenchmarkRunner(configuration: .init(
            warmupIterations: 10,
            measurementIterations: 100,
            collectMemory: true
        ))

        let result = try await runner.measure(name: "End-to-End - Memory Profile") {
            try await localPipeline.embed(text)
        }

        await reporter.record(result)

        if let memory = result.memory {
            print("💾 Memory Usage:")
            print("   Average: \(memory.formatBytes(memory.averageDelta))")
            print("   Max: \(memory.formatBytes(memory.maxDelta))")
            print("   Total: \(memory.formatBytes(memory.total))")

            // Memory usage should be reasonable (<10MB per operation)
            XCTAssertLessThan(
                memory.averageDelta,
                10 * 1024 * 1024,  // 10MB
                "Memory usage too high"
            )
        }
    }

    // MARK: - B-005i: Concurrent Access

    func testEndToEnd_ConcurrentAccess() async throws {
        let localPipeline = self.pipeline!
        let texts = TestFixtures.generateRandomTexts(count: 10, averageWords: 30)

        let runner = BenchmarkRunner(configuration: .init(
            warmupIterations: 3,
            measurementIterations: 10
        ))

        let result = try await runner.measure(name: "End-to-End - Concurrent 10 texts") {
            // Process texts concurrently
            let embeddings = try await withThrowingTaskGroup(of: (Int, DynamicEmbedding).self) { group in
                for (index, text) in texts.enumerated() {
                    group.addTask {
                        let embedding = try await localPipeline.embed(text)
                        return (index, embedding)
                    }
                }

                var results: [(Int, DynamicEmbedding)] = []
                for try await result in group {
                    results.append(result)
                }
                return results.sorted { $0.0 < $1.0 }.map { $0.1 }
            }
            return embeddings
        }

        await reporter.record(result)

        print("⚡ Concurrent Access (10 texts):")
        print("   p50: \(result.timing.formatMs(result.timing.p50))")

        // Should handle concurrent access correctly
        XCTAssertLessThan(result.timing.p50, 5.0)  // <5s for 10 concurrent texts
    }
}

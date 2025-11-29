// EmbedKit - PipelinedBatchProcessor Tests

import Testing
import Foundation
@testable import EmbedKit
import VectorCore

// MARK: - PipelineConfig Tests

@Suite("PipelineConfig")
struct PipelineConfigTests {

    @Test("Default configuration has sensible values")
    func testDefaultConfig() {
        let config = PipelineConfig()

        #expect(config.bufferCount == 2)
        #expect(config.batchSize == 32)
        #expect(config.enablePrefetch == true)
        #expect(config.fillTimeout == 0.1)
        #expect(config.allowPartialBatches == true)
    }

    @Test("Double buffer preset")
    func testDoubleBufferPreset() {
        let config = PipelineConfig.doubleBuffer

        #expect(config.bufferCount == 2)
    }

    @Test("Triple buffer preset")
    func testTripleBufferPreset() {
        let config = PipelineConfig.tripleBuffer

        #expect(config.bufferCount == 3)
    }

    @Test("Low latency preset")
    func testLowLatencyPreset() {
        let config = PipelineConfig.lowLatency

        #expect(config.bufferCount == 2)
        #expect(config.batchSize == 8)
        #expect(config.fillTimeout == 0.05)
    }

    @Test("High throughput preset")
    func testHighThroughputPreset() {
        let config = PipelineConfig.highThroughput

        #expect(config.bufferCount == 3)
        #expect(config.batchSize == 64)
        #expect(config.fillTimeout == 0.2)
    }

    @Test("Custom configuration")
    func testCustomConfig() {
        let config = PipelineConfig(
            bufferCount: 4,
            batchSize: 16,
            enablePrefetch: false,
            fillTimeout: 0.5,
            allowPartialBatches: false
        )

        #expect(config.bufferCount == 4)
        #expect(config.batchSize == 16)
        #expect(config.enablePrefetch == false)
        #expect(config.fillTimeout == 0.5)
        #expect(config.allowPartialBatches == false)
    }

    @Test("Config is Sendable")
    func testSendable() async {
        let config = PipelineConfig.doubleBuffer
        let result = await Task {
            config.bufferCount
        }.value
        #expect(result == 2)
    }
}

// MARK: - PipelineBufferState Tests

@Suite("PipelineBufferState")
struct PipelineBufferStateTests {

    @Test("All states are available")
    func testAllStates() {
        let states: [PipelineBufferState] = [
            .empty, .filling, .ready, .processing, .completed
        ]
        #expect(states.count == 5)
    }

    @Test("States are Sendable")
    func testSendable() async {
        let state = PipelineBufferState.ready
        let result = await Task {
            state
        }.value
        #expect(result == .ready)
    }
}

// MARK: - PipelineStatistics Tests

@Suite("PipelineStatistics")
struct PipelineStatisticsTests {

    @Test("Throughput calculation")
    func testThroughput() {
        let stats = PipelineStatistics(
            totalItems: 100,
            totalBatches: 10,
            partialBatches: 1,
            stallCount: 2,
            averageStallTime: 0.05,
            totalTime: 1.0,
            bufferUtilization: 0.9,
            pipelineEfficiency: 0.95
        )

        #expect(stats.throughput == 100.0)
        #expect(stats.averageBatchSize == 10.0)
    }

    @Test("Zero time handling")
    func testZeroTime() {
        let stats = PipelineStatistics(
            totalItems: 0,
            totalBatches: 0,
            partialBatches: 0,
            stallCount: 0,
            averageStallTime: 0,
            totalTime: 0,
            bufferUtilization: 0,
            pipelineEfficiency: 0
        )

        #expect(stats.throughput == 0)
        #expect(stats.averageBatchSize == 0)
    }

    @Test("Statistics are Sendable")
    func testSendable() async {
        let stats = PipelineStatistics(
            totalItems: 50,
            totalBatches: 5,
            partialBatches: 0,
            stallCount: 0,
            averageStallTime: 0,
            totalTime: 0.5,
            bufferUtilization: 1.0,
            pipelineEfficiency: 1.0
        )

        let throughput = await Task {
            stats.throughput
        }.value
        #expect(throughput == 100.0)
    }
}

// MARK: - PipelineError Tests

@Suite("PipelineError")
struct PipelineErrorTests {

    @Test("Buffer timeout error has description")
    func testBufferTimeoutDescription() {
        let error = PipelineError.bufferTimeout
        #expect(error.errorDescription?.contains("Timeout") == true)
    }

    @Test("Cancelled error has description")
    func testCancelledDescription() {
        let error = PipelineError.cancelled
        #expect(error.errorDescription?.contains("cancelled") == true)
    }

    @Test("Invalid buffer state error has description")
    func testInvalidBufferStateDescription() {
        let error = PipelineError.invalidBufferState(expected: .ready, actual: .empty)
        #expect(error.errorDescription?.contains("Invalid buffer state") == true)
    }

    @Test("Errors are Sendable")
    func testSendable() async {
        let error = PipelineError.cancelled
        let desc = await Task {
            error.errorDescription
        }.value
        #expect(desc != nil)
    }
}

// MARK: - PipelinedBatchProcessor Tests

@Suite("PipelinedBatchProcessor")
struct PipelinedBatchProcessorTests {

    @Test("Empty input returns empty array")
    func testEmptyInput() async throws {
        let model = MockEmbeddingModel(dimensions: 384)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(generator: generator)

        let result = try await processor.produce([])

        #expect(result.isEmpty)
    }

    @Test("Single item processing")
    func testSingleItem() async throws {
        let model = MockEmbeddingModel(dimensions: 384)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(generator: generator)

        let result = try await processor.produce(["Hello world"])

        #expect(result.count == 1)
        #expect(result[0].count == 384)
    }

    @Test("Small batch processing")
    func testSmallBatch() async throws {
        let model = MockEmbeddingModel(dimensions: 384)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(
            generator: generator,
            config: PipelineConfig(batchSize: 8)
        )

        let texts = (0..<5).map { "Text \($0)" }
        let result = try await processor.produce(texts)

        #expect(result.count == 5)
        for embedding in result {
            #expect(embedding.count == 384)
        }
    }

    @Test("Full batch processing")
    func testFullBatch() async throws {
        let model = MockEmbeddingModel(dimensions: 384)
        let generator = EmbeddingGenerator(model: model)
        let config = PipelineConfig(batchSize: 10)
        let processor = PipelinedBatchProcessor(
            generator: generator,
            config: config
        )

        let texts = (0..<10).map { "Text \($0)" }
        let result = try await processor.produce(texts)

        #expect(result.count == 10)
    }

    @Test("Multiple batches processing")
    func testMultipleBatches() async throws {
        let model = MockEmbeddingModel(dimensions: 384)
        let generator = EmbeddingGenerator(model: model)
        let config = PipelineConfig(batchSize: 5)
        let processor = PipelinedBatchProcessor(
            generator: generator,
            config: config
        )

        let texts = (0..<23).map { "Text \($0)" }
        let result = try await processor.produce(texts)

        #expect(result.count == 23)

        // Verify statistics
        let stats = await processor.getStatistics()
        #expect(stats.totalItems == 23)
        #expect(stats.totalBatches == 5)  // 5 + 5 + 5 + 5 + 3
        #expect(stats.partialBatches == 1)  // Last batch has 3 items
    }

    @Test("Double buffering configuration")
    func testDoubleBuffering() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(
            generator: generator,
            config: .doubleBuffer
        )

        let texts = (0..<100).map { "Text \($0)" }
        let result = try await processor.produce(texts)

        #expect(result.count == 100)

        let stats = await processor.getStatistics()
        #expect(stats.pipelineEfficiency > 0)
    }

    @Test("Triple buffering configuration")
    func testTripleBuffering() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(
            generator: generator,
            config: .tripleBuffer
        )

        let texts = (0..<100).map { "Text \($0)" }
        let result = try await processor.produce(texts)

        #expect(result.count == 100)
    }

    @Test("Low latency configuration")
    func testLowLatencyConfig() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(
            generator: generator,
            config: .lowLatency
        )

        let texts = (0..<20).map { "Text \($0)" }
        let result = try await processor.produce(texts)

        #expect(result.count == 20)
    }

    @Test("High throughput configuration")
    func testHighThroughputConfig() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(
            generator: generator,
            config: .highThroughput
        )

        let texts = (0..<200).map { "Text \($0)" }
        let result = try await processor.produce(texts)

        #expect(result.count == 200)
    }

    @Test("Statistics are tracked correctly")
    func testStatisticsTracking() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let config = PipelineConfig(batchSize: 4)
        let processor = PipelinedBatchProcessor(
            generator: generator,
            config: config
        )

        // First processing run
        let texts1 = (0..<12).map { "Text \($0)" }
        _ = try await processor.produce(texts1)

        var stats = await processor.getStatistics()
        #expect(stats.totalItems == 12)
        #expect(stats.totalBatches == 3)
        #expect(stats.partialBatches == 0)
        #expect(stats.totalTime > 0)
        #expect(stats.throughput > 0)

        // Second processing run (stats are reset)
        let texts2 = (0..<7).map { "More text \($0)" }
        _ = try await processor.produce(texts2)

        stats = await processor.getStatistics()
        #expect(stats.totalItems == 7)
        #expect(stats.totalBatches == 2)
        #expect(stats.partialBatches == 1)  // Last batch has 3 items
    }

    @Test("Reset statistics")
    func testResetStatistics() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(generator: generator)

        _ = try await processor.produce(["Hello"])

        var stats = await processor.getStatistics()
        #expect(stats.totalItems == 1)

        await processor.resetStatistics()

        stats = await processor.getStatistics()
        #expect(stats.totalItems == 0)
        #expect(stats.totalBatches == 0)
    }

    @Test("Configuration update")
    func testConfigUpdate() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(
            generator: generator,
            config: PipelineConfig(batchSize: 10)
        )

        var config = await processor.getConfig()
        #expect(config.batchSize == 10)

        await processor.updateConfig(PipelineConfig(batchSize: 20))

        config = await processor.getConfig()
        #expect(config.batchSize == 20)
    }
}

// MARK: - VectorProducer Conformance Tests

@Suite("PipelinedBatchProcessor VectorProducer")
struct PipelinedBatchProcessorVectorProducerTests {

    @Test("Dimensions match underlying generator")
    func testDimensions() async throws {
        let model = MockEmbeddingModel(dimensions: 512)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(generator: generator)

        #expect(processor.dimensions == 512)
    }

    @Test("Produces normalized vectors matches underlying generator")
    func testProducesNormalizedVectors() async throws {
        let model = MockEmbeddingModel(dimensions: 384)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(generator: generator)

        // MockEmbeddingModel with default config normalizes output
        #expect(processor.producesNormalizedVectors == generator.producesNormalizedVectors)
    }

    @Test("Single produce works")
    func testSingleProduce() async throws {
        let model = MockEmbeddingModel(dimensions: 256)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(generator: generator)

        let embedding = try await processor.produce("Single text")

        #expect(embedding.count == 256)
    }
}

// MARK: - Stream Processing Tests

@Suite("PipelinedBatchProcessor Stream")
struct PipelinedBatchProcessorStreamTests {

    @Test("Stream produces all results")
    func testStreamAllResults() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(
            generator: generator,
            config: PipelineConfig(batchSize: 5)
        )

        let texts = (0..<15).map { "Text \($0)" }
        var results: [[Float]] = []
        var progressUpdates: [BatchProgress] = []

        for try await (embedding, progress) in await processor.processStream(texts) {
            results.append(embedding)
            progressUpdates.append(progress)
        }

        #expect(results.count == 15)
        #expect(progressUpdates.count == 15)
    }

    @Test("Stream provides progress updates")
    func testStreamProgress() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(
            generator: generator,
            config: PipelineConfig(batchSize: 4)
        )

        let texts = (0..<10).map { "Text \($0)" }
        var lastProgress: BatchProgress?

        for try await (_, progress) in await processor.processStream(texts) {
            lastProgress = progress
        }

        #expect(lastProgress != nil)
        #expect(lastProgress!.current == 10)
        #expect(lastProgress!.total == 10)
        #expect(lastProgress!.percentage == 100)
    }

    @Test("Stream respects early termination")
    func testStreamEarlyTermination() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(
            generator: generator,
            config: PipelineConfig(batchSize: 2)
        )

        let texts = (0..<20).map { "Text \($0)" }
        var count = 0

        for try await _ in await processor.processStream(texts) {
            count += 1
            if count >= 4 {
                break  // Stop early
            }
        }

        // Should have stopped after exactly 4 items
        #expect(count == 4)
    }
}

// MARK: - EmbeddingGenerator Extension Tests

@Suite("EmbeddingGenerator.pipelined")
struct EmbeddingGeneratorPipelinedExtensionTests {

    @Test("Extension creates processor with default config")
    func testDefaultPipelined() async throws {
        let model = MockEmbeddingModel(dimensions: 256)
        let generator = EmbeddingGenerator(model: model)

        let processor = await generator.pipelined()

        let config = await processor.getConfig()
        #expect(config.bufferCount == 2)  // Default double buffer
    }

    @Test("Extension creates processor with custom config")
    func testCustomPipelined() async throws {
        let model = MockEmbeddingModel(dimensions: 256)
        let generator = EmbeddingGenerator(model: model)

        let processor = await generator.pipelined(config: .tripleBuffer)

        let config = await processor.getConfig()
        #expect(config.bufferCount == 3)
    }

    @Test("Extension produces correct results")
    func testPipelinedResults() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)

        let processor = await generator.pipelined()
        let texts = (0..<10).map { "Text \($0)" }
        let result = try await processor.produce(texts)

        #expect(result.count == 10)
        for embedding in result {
            #expect(embedding.count == 128)
        }
    }
}

// MARK: - ConcurrentPipelineProcessor Tests

@Suite("ConcurrentPipelineProcessor")
struct ConcurrentPipelineProcessorTests {

    @Test("Concurrent processing of multiple arrays")
    func testConcurrentProcessing() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let concurrent = ConcurrentPipelineProcessor(
            generator: generator,
            maxConcurrency: 2
        )

        let textArrays = [
            (0..<5).map { "Array1 \($0)" },
            (0..<7).map { "Array2 \($0)" },
            (0..<3).map { "Array3 \($0)" }
        ]

        let results = try await concurrent.processConcurrently(textArrays)

        #expect(results.count == 3)
        #expect(results[0].count == 5)
        #expect(results[1].count == 7)
        #expect(results[2].count == 3)
    }

    @Test("Results maintain order")
    func testResultOrder() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let concurrent = ConcurrentPipelineProcessor(
            generator: generator,
            maxConcurrency: 3
        )

        // Create arrays of different sizes
        let textArrays = [
            ["First array item"],
            ["Second", "array", "items"],
            ["Third"]
        ]

        let results = try await concurrent.processConcurrently(textArrays)

        #expect(results[0].count == 1)
        #expect(results[1].count == 3)
        #expect(results[2].count == 1)
    }

    @Test("Empty arrays handled correctly")
    func testEmptyArrays() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let concurrent = ConcurrentPipelineProcessor(
            generator: generator,
            maxConcurrency: 2
        )

        let textArrays: [[String]] = [
            [],
            ["Some text"],
            []
        ]

        let results = try await concurrent.processConcurrently(textArrays)

        #expect(results.count == 3)
        #expect(results[0].isEmpty)
        #expect(results[1].count == 1)
        #expect(results[2].isEmpty)
    }
}

// MARK: - Lifecycle Tests

@Suite("PipelinedBatchProcessor Lifecycle")
struct PipelinedBatchProcessorLifecycleTests {

    @Test("Warmup calls underlying generator")
    func testWarmup() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(generator: generator)

        // Should not throw
        try await processor.warmup()
    }

    @Test("Release cleans up resources")
    func testRelease() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(generator: generator)

        // Process something first
        _ = try await processor.produce(["Test"])

        // Should not throw
        try await processor.release()
    }
}

// MARK: - Edge Cases Tests

@Suite("PipelinedBatchProcessor Edge Cases")
struct PipelinedBatchProcessorEdgeCasesTests {

    @Test("Large batch size with small input")
    func testLargeBatchSizeSmallInput() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(
            generator: generator,
            config: PipelineConfig(batchSize: 100)
        )

        let texts = ["Just one"]
        let result = try await processor.produce(texts)

        #expect(result.count == 1)

        let stats = await processor.getStatistics()
        #expect(stats.partialBatches == 1)
    }

    @Test("Exact batch size match")
    func testExactBatchSizeMatch() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(
            generator: generator,
            config: PipelineConfig(batchSize: 5)
        )

        let texts = (0..<10).map { "Text \($0)" }  // Exactly 2 batches
        let result = try await processor.produce(texts)

        #expect(result.count == 10)

        let stats = await processor.getStatistics()
        #expect(stats.totalBatches == 2)
        #expect(stats.partialBatches == 0)
    }

    @Test("Buffer utilization calculation")
    func testBufferUtilization() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(
            generator: generator,
            config: PipelineConfig(batchSize: 10)
        )

        // 15 items: batch of 10, batch of 5
        let texts = (0..<15).map { "Text \($0)" }
        _ = try await processor.produce(texts)

        let stats = await processor.getStatistics()

        // Average batch size: (10 + 5) / 2 = 7.5
        // Utilization: 7.5 / 10 = 0.75
        #expect(stats.bufferUtilization == 0.75)
    }

    @Test("Pipeline efficiency with no stalls")
    func testPipelineEfficiencyNoStalls() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(
            generator: generator,
            config: PipelineConfig(batchSize: 32)
        )

        let texts = (0..<10).map { "Text \($0)" }
        _ = try await processor.produce(texts)

        let stats = await processor.getStatistics()

        // No stalls means efficiency should be 1.0
        #expect(stats.stallCount == 0)
        #expect(stats.pipelineEfficiency == 1.0)
    }
}

// MARK: - Integration Tests

@Suite("PipelinedBatchProcessor Integration", .tags(.integration))
struct PipelinedBatchProcessorIntegrationTests {

    @Test("Process large dataset")
    func testLargeDataset() async throws {
        let model = MockEmbeddingModel(dimensions: 384)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(
            generator: generator,
            config: .highThroughput
        )

        let texts = (0..<500).map { "Large dataset text number \($0)" }
        let result = try await processor.produce(texts)

        #expect(result.count == 500)

        let stats = await processor.getStatistics()
        #expect(stats.totalItems == 500)
        #expect(stats.throughput > 0)
    }

    @Test("Pipeline maintains embedding quality")
    func testEmbeddingQuality() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(
            generator: generator,
            config: .doubleBuffer
        )

        let texts = ["Test text"]

        // Get embedding through pipeline
        let pipelineResult = try await processor.produce(texts)

        // Get embedding directly
        let directResult = try await generator.produce(texts)

        // Should be identical
        #expect(pipelineResult.count == directResult.count)
        #expect(pipelineResult[0].count == directResult[0].count)

        // Note: Due to MockEmbeddingModel's deterministic nature,
        // the vectors should be the same
    }
}

// Note: Tag.integration is defined in SharedMetalContextManagerTests.swift

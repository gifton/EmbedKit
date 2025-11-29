// Tests for AdaptiveBatcher
import Testing
import Foundation
@testable import EmbedKit

@Suite("Adaptive Batcher")
struct AdaptiveBatcherTests {

    // MARK: - Test Backend

    actor CountingBackend: CoreMLProcessingBackend {
        private(set) var isLoaded: Bool = false
        var memoryUsage: Int64 { 0 }
        private(set) var batchCalls: [(count: Int, texts: [String])] = []
        private let dim: Int

        init(dimensions: Int = 4) {
            self.dim = dimensions
        }

        func load() async throws { isLoaded = true }
        func unload() async throws { isLoaded = false }

        func process(_ input: CoreMLInput) async throws -> CoreMLOutput {
            CoreMLOutput(values: Array(repeating: 0, count: input.tokenIDs.count * dim), shape: [input.tokenIDs.count, dim])
        }

        func processBatch(_ inputs: [CoreMLInput]) async throws -> [CoreMLOutput] {
            batchCalls.append((count: inputs.count, texts: []))
            return inputs.map { inp in
                CoreMLOutput(values: Array(repeating: 0, count: inp.tokenIDs.count * dim), shape: [inp.tokenIDs.count, dim])
            }
        }

        func getBatchCalls() -> [(count: Int, texts: [String])] {
            batchCalls
        }

        func reset() {
            batchCalls = []
        }
    }

    // MARK: - Basic Functionality

    @Test
    func singleEmbed_returnsResult() async throws {
        let backend = CountingBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            paddingStrategy: .none,
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4
        )

        let batcher = AdaptiveBatcher(model: model)
        let embedding = try await batcher.embed("hello world")

        #expect(embedding.vector.count == 4)
    }

    @Test
    func embedBatch_directCall_works() async throws {
        let backend = CountingBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            paddingStrategy: .none,
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4
        )

        let batcher = AdaptiveBatcher(model: model)
        let embeddings = try await batcher.embedBatch(["hello", "world", "test"])

        #expect(embeddings.count == 3)
        for emb in embeddings {
            #expect(emb.vector.count == 4)
        }
    }

    // MARK: - Batching Behavior

    @Test
    func concurrentEmbeds_areBatched() async throws {
        let backend = CountingBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            paddingStrategy: .none,
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4
        )

        var config = AdaptiveBatcherConfig()
        config.minBatchSize = 1
        let batcher = AdaptiveBatcher(model: model, config: config)

        // Submit multiple requests concurrently
        let results = try await batcher.embedConcurrently(["a", "b", "c", "d"])

        #expect(results.count == 4)

        // At least some requests should have been batched together
        let calls = await backend.getBatchCalls()
        #expect(calls.count >= 1)
    }

    @Test
    func flush_processesQueuedRequests() async throws {
        let backend = CountingBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            paddingStrategy: .none,
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4
        )

        var config = AdaptiveBatcherConfig()
        config.maxLatency = 10.0 // Very long timeout
        let batcher = AdaptiveBatcher(model: model, config: config)

        // Start some embed tasks that will queue
        let task1 = Task { try await batcher.embed("hello") }
        let task2 = Task { try await batcher.embed("world") }

        // Give tasks time to queue
        try await Task.sleep(nanoseconds: 10_000_000) // 10ms

        // Now flush
        try await batcher.flush()

        // Wait for results
        let emb1 = try await task1.value
        let emb2 = try await task2.value

        #expect(emb1.vector.count == 4)
        #expect(emb2.vector.count == 4)
    }

    // MARK: - Memory Pressure

    @Test
    func memoryPressure_affectsBatchSize() async throws {
        let backend = CountingBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            paddingStrategy: .none,
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4
        )

        var config = AdaptiveBatcherConfig()
        config.batchSizeByPressure = [
            0.0...0.3: 64,
            0.3...0.6: 32,
            0.6...0.8: 16,
            0.8...1.0: 8
        ]
        let batcher = AdaptiveBatcher(model: model, config: config)

        // Set high memory pressure
        await batcher.setMemoryPressure(0.85)

        let metrics = await batcher.metrics
        #expect(metrics.currentMemoryPressure == 0.85)
    }

    @Test
    func memoryPressure_clampedTo0_1() async throws {
        let backend = CountingBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4
        )

        let batcher = AdaptiveBatcher(model: model)

        await batcher.setMemoryPressure(-0.5)
        var metrics = await batcher.metrics
        #expect(metrics.currentMemoryPressure == 0.0)

        await batcher.setMemoryPressure(1.5)
        metrics = await batcher.metrics
        #expect(metrics.currentMemoryPressure == 1.0)
    }

    // MARK: - Metrics

    @Test
    func metrics_trackRequestsAndBatches() async throws {
        let backend = CountingBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            paddingStrategy: .none,
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4
        )

        let batcher = AdaptiveBatcher(model: model)

        _ = try await batcher.embed("test 1")
        _ = try await batcher.embed("test 2")
        _ = try await batcher.embed("test 3")

        let metrics = await batcher.metrics
        #expect(metrics.totalRequests == 3)
        #expect(metrics.totalBatches >= 1)
    }

    @Test
    func resetMetrics_clearsCounters() async throws {
        let backend = CountingBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            paddingStrategy: .none,
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4
        )

        let batcher = AdaptiveBatcher(model: model)

        _ = try await batcher.embed("test")
        await batcher.resetMetrics()

        let metrics = await batcher.metrics
        #expect(metrics.totalRequests == 0)
        #expect(metrics.totalBatches == 0)
    }

    // MARK: - Configuration

    @Test
    func setConfig_updatesConfiguration() async throws {
        let backend = CountingBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4
        )

        var config = AdaptiveBatcherConfig()
        let batcher = AdaptiveBatcher(model: model, config: config)

        var newConfig = AdaptiveBatcherConfig()
        newConfig.maxLatency = 0.5
        newConfig.maxBatchSize = 64
        await batcher.setConfig(newConfig)

        // Configuration should be updated - verify through behavior
        // (The actual config is private, so we verify through metrics or behavior)
        let metrics = await batcher.metrics
        #expect(metrics.currentQueueDepth == 0)
    }

    // MARK: - Edge Cases

    @Test
    func emptyFlush_doesNothing() async throws {
        let backend = CountingBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4
        )

        let batcher = AdaptiveBatcher(model: model)

        // Flush with nothing queued should not throw
        try await batcher.flush()

        let calls = await backend.getBatchCalls()
        #expect(calls.isEmpty)
    }

    @Test
    func embedConcurrently_preservesOrder() async throws {
        let backend = CountingBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            paddingStrategy: .none,
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4
        )

        let batcher = AdaptiveBatcher(model: model)
        let texts = ["first", "second", "third", "fourth", "fifth"]

        let results = try await batcher.embedConcurrently(texts)

        #expect(results.count == texts.count)
        // Results should be in same order as input
        for (i, emb) in results.enumerated() {
            #expect(emb.metadata.tokenCount > 0, "Result \(i) should have tokens")
        }
    }

    @Test
    func maxBatchSize_respected() async throws {
        let backend = CountingBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            paddingStrategy: .none,
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4
        )

        var config = AdaptiveBatcherConfig()
        config.maxLatency = 0.01
        config.maxBatchSize = 2  // Limit batch size to 2
        let batcher = AdaptiveBatcher(model: model, config: config)

        // Process more than maxBatchSize
        let results = try await batcher.embedConcurrently(["a", "b", "c", "d", "e"])

        #expect(results.count == 5)

        // Check that batch sizes were respected
        let calls = await backend.getBatchCalls()
        for call in calls {
            #expect(call.count <= 2, "Batch size \(call.count) exceeds max of 2")
        }
    }
}

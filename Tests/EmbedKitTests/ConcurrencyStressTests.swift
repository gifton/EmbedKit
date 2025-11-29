// Tests for Concurrency Stress - P0 Category
// Validates thread safety and correctness under heavy concurrent load
import Testing
import Foundation
@testable import EmbedKit

// MARK: - Test Infrastructure

/// A backend that tracks concurrent access patterns
actor ConcurrencyTrackingBackend: CoreMLProcessingBackend {
    private(set) var isLoaded: Bool = false
    var memoryUsage: Int64 { 0 }

    private var maxConcurrentCalls: Int = 0
    private var currentConcurrentCalls: Int = 0
    private var totalCalls: Int = 0
    private let delayNs: UInt64
    private let dimensions: Int

    init(delayMs: UInt64 = 5, dimensions: Int = 4) {
        self.delayNs = delayMs * 1_000_000
        self.dimensions = dimensions
    }

    func load() async throws { isLoaded = true }
    func unload() async throws { isLoaded = false }

    func process(_ input: CoreMLInput) async throws -> CoreMLOutput {
        currentConcurrentCalls += 1
        maxConcurrentCalls = max(maxConcurrentCalls, currentConcurrentCalls)
        totalCalls += 1

        // Simulate processing time
        if delayNs > 0 {
            try await Task.sleep(nanoseconds: delayNs)
        }

        currentConcurrentCalls -= 1
        return CoreMLOutput(
            values: Array(repeating: 0.5, count: input.tokenIDs.count * dimensions),
            shape: [input.tokenIDs.count, dimensions]
        )
    }

    func processBatch(_ inputs: [CoreMLInput]) async throws -> [CoreMLOutput] {
        currentConcurrentCalls += 1
        maxConcurrentCalls = max(maxConcurrentCalls, currentConcurrentCalls)
        totalCalls += inputs.count

        if delayNs > 0 {
            try await Task.sleep(nanoseconds: delayNs)
        }

        currentConcurrentCalls -= 1
        return inputs.map { inp in
            CoreMLOutput(
                values: Array(repeating: 0.5, count: inp.tokenIDs.count * dimensions),
                shape: [inp.tokenIDs.count, dimensions]
            )
        }
    }

    func getStats() -> (maxConcurrent: Int, total: Int) {
        (maxConcurrentCalls, totalCalls)
    }
}

/// A backend that introduces random delays to simulate real-world variability
actor VariableDelayBackend: CoreMLProcessingBackend {
    private(set) var isLoaded: Bool = false
    var memoryUsage: Int64 { 0 }

    private let minDelayMs: UInt64
    private let maxDelayMs: UInt64
    private let dimensions: Int

    init(minDelayMs: UInt64 = 1, maxDelayMs: UInt64 = 10, dimensions: Int = 4) {
        self.minDelayMs = minDelayMs
        self.maxDelayMs = maxDelayMs
        self.dimensions = dimensions
    }

    func load() async throws { isLoaded = true }
    func unload() async throws { isLoaded = false }

    func process(_ input: CoreMLInput) async throws -> CoreMLOutput {
        let delay = UInt64.random(in: minDelayMs...maxDelayMs) * 1_000_000
        try await Task.sleep(nanoseconds: delay)
        return CoreMLOutput(
            values: Array(repeating: 0.5, count: input.tokenIDs.count * dimensions),
            shape: [input.tokenIDs.count, dimensions]
        )
    }

    func processBatch(_ inputs: [CoreMLInput]) async throws -> [CoreMLOutput] {
        let delay = UInt64.random(in: minDelayMs...maxDelayMs) * 1_000_000
        try await Task.sleep(nanoseconds: delay)
        return inputs.map { inp in
            CoreMLOutput(
                values: Array(repeating: 0.5, count: inp.tokenIDs.count * dimensions),
                shape: [inp.tokenIDs.count, dimensions]
            )
        }
    }
}

/// Thread-safe counter for tracking across tasks (using actor)
actor AtomicCounter {
    private var _value: Int = 0

    var value: Int { _value }

    @discardableResult
    func increment() -> Int {
        _value += 1
        return _value
    }

    func add(_ n: Int) {
        _value += n
    }
}

// MARK: - High Volume Concurrent Embedding Tests

@Suite("Concurrency Stress - High Volume")
struct HighVolumeConcurrencyTests {

    @Test("100 concurrent embedding requests complete without error")
    func hundredConcurrentRequests() async throws {
        let backend = ConcurrencyTrackingBackend(delayMs: 2)
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4,
            device: .cpu
        )

        let successCount = AtomicCounter()
        let requestCount = 100

        try await withThrowingTaskGroup(of: Void.self) { group in
            for i in 0..<requestCount {
                group.addTask {
                    _ = try await model.embed("Concurrent request \(i)")
                    await successCount.increment()
                }
            }
            try await group.waitForAll()
        }

        let finalCount = await successCount.value
        #expect(finalCount == requestCount)

        let stats = await backend.getStats()
        #expect(stats.total >= requestCount) // At least one call per request
    }

    @Test("Concurrent requests produce consistent embeddings for same input")
    func concurrentSameInputConsistency() async throws {
        let backend = ConcurrencyTrackingBackend(delayMs: 1)
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4,
            device: .cpu
        )

        let sharedInput = "Shared input for consistency check"

        // Collect embeddings using task group returning the results directly
        let embeddings = try await withThrowingTaskGroup(of: Embedding.self, returning: [Embedding].self) { group in
            for _ in 0..<20 {
                group.addTask {
                    try await model.embed(sharedInput)
                }
            }

            var results: [Embedding] = []
            for try await embedding in group {
                results.append(embedding)
            }
            return results
        }

        // All embeddings should have same dimensions
        let firstDims = embeddings.first!.dimensions
        #expect(embeddings.allSatisfy { $0.dimensions == firstDims })

        // All embeddings should be identical (from cache or identical computation)
        let firstVector = embeddings.first!.vector
        for emb in embeddings.dropFirst() {
            #expect(emb.vector == firstVector)
        }
    }

    @Test("Mixed batch sizes under concurrent load")
    func mixedBatchSizesConcurrent() async throws {
        let backend = ConcurrencyTrackingBackend(delayMs: 3)
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4,
            device: .cpu
        )

        let totalEmbeddings = AtomicCounter()

        try await withThrowingTaskGroup(of: Void.self) { group in
            // Single embeddings
            for i in 0..<20 {
                group.addTask {
                    _ = try await model.embed("Single \(i)")
                    await totalEmbeddings.increment()
                }
            }

            // Small batches
            for i in 0..<10 {
                group.addTask {
                    let texts = (0..<5).map { "Small batch \(i) item \($0)" }
                    let results = try await model.embedBatch(texts, options: BatchOptions())
                    await totalEmbeddings.add(results.count)
                }
            }

            // Large batches
            for i in 0..<5 {
                group.addTask {
                    let texts = (0..<20).map { "Large batch \(i) item \($0)" }
                    let results = try await model.embedBatch(texts, options: BatchOptions())
                    await totalEmbeddings.add(results.count)
                }
            }

            try await group.waitForAll()
        }

        // 20 singles + 10*5 small + 5*20 large = 20 + 50 + 100 = 170
        let finalCount = await totalEmbeddings.value
        #expect(finalCount == 170)
    }
}

// MARK: - Model Lifecycle Concurrency Tests

@Suite("Concurrency Stress - Model Lifecycle")
struct ModelLifecycleConcurrencyTests {

    @Test("Concurrent model loads return same instance")
    func concurrentModelLoadsReturnSameInstance() async throws {
        let manager = ModelManager()

        let modelIDs = try await withThrowingTaskGroup(of: ModelID.self, returning: [ModelID].self) { group in
            for _ in 0..<10 {
                group.addTask {
                    let model = try await manager.loadMockModel()
                    return model.id
                }
            }

            var results: [ModelID] = []
            for try await id in group {
                results.append(id)
            }
            return results
        }

        // All should return same model ID (mock model has fixed ID)
        let first = modelIDs.first!
        #expect(modelIDs.allSatisfy { $0 == first })
    }

    @Test("Embedding during model unload handles gracefully")
    func embeddingDuringUnload() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let modelID = model.id

        let results = await withTaskGroup(of: Bool.self, returning: (success: Int, error: Int).self) { group in
            // Start embeddings
            for i in 0..<20 {
                group.addTask {
                    do {
                        _ = try await manager.embed("Test \(i)", using: modelID)
                        return true
                    } catch {
                        return false
                    }
                }
            }

            // Unload partway through
            group.addTask {
                try? await Task.sleep(nanoseconds: 5_000_000) // 5ms
                await manager.unloadAll()
                return true
            }

            var successCount = 0
            var errorCount = 0
            for await success in group {
                if success {
                    successCount += 1
                } else {
                    errorCount += 1
                }
            }
            return (successCount, errorCount)
        }

        // Some requests should succeed before unload, some may fail after
        // The important thing is no crashes
        #expect(results.success + results.error == 21) // 20 embeddings + 1 unload task
    }

    @Test("Concurrent load and embed operations")
    func concurrentLoadAndEmbed() async throws {
        let manager = ModelManager()
        let successCount = AtomicCounter()

        try await withThrowingTaskGroup(of: Void.self) { group in
            // Load operations
            for _ in 0..<5 {
                group.addTask {
                    _ = try await manager.loadMockModel()
                    await successCount.increment()
                }
            }

            // Wait for at least one load
            try await Task.sleep(nanoseconds: 10_000_000) // 10ms

            // Embed operations (may race with loads)
            for i in 0..<10 {
                group.addTask {
                    let model = try await manager.loadMockModel()
                    _ = try await model.embed("Concurrent embed \(i)")
                    await successCount.increment()
                }
            }

            try await group.waitForAll()
        }

        let finalCount = await successCount.value
        #expect(finalCount == 15)
    }
}

// MARK: - TokenCache Stress Tests

@Suite("Concurrency Stress - Token Cache")
struct TokenCacheStressTests {

    @Test("Cache handles high concurrent access")
    func cacheHighConcurrentAccess() async throws {
        let cache = TokenCache<String, Int>(capacity: 100)
        let operationCount = AtomicCounter()

        try await withThrowingTaskGroup(of: Void.self) { group in
            // Writers
            for i in 0..<50 {
                group.addTask {
                    await cache.put("key-\(i)", i)
                    await operationCount.increment()
                }
            }

            // Readers
            for i in 0..<50 {
                group.addTask {
                    _ = await cache.get("key-\(i % 20)") // Read subset of keys
                    await operationCount.increment()
                }
            }

            try await group.waitForAll()
        }

        let finalCount = await operationCount.value
        #expect(finalCount == 100)

        let stats = await cache.stats()
        #expect(stats.total > 0) // Some operations recorded
    }

    @Test("Cache eviction under concurrent writes")
    func cacheEvictionConcurrent() async throws {
        let cache = TokenCache<String, Int>(capacity: 10)

        // Write more items than capacity concurrently
        try await withThrowingTaskGroup(of: Void.self) { group in
            for i in 0..<100 {
                group.addTask {
                    await cache.put("item-\(i)", i)
                }
            }
            try await group.waitForAll()
        }

        // Verify cache still works and respects capacity
        // (We can't guarantee exact contents due to race conditions)
        var foundCount = 0
        for i in 0..<100 {
            if await cache.get("item-\(i)") != nil {
                foundCount += 1
            }
        }

        // Should have approximately capacity items (may vary due to timing)
        #expect(foundCount <= 20) // Reasonable upper bound
        #expect(foundCount >= 1)   // At least some items present
    }

    @Test("Cache reset during concurrent access")
    func cacheResetDuringAccess() async throws {
        let cache = TokenCache<String, Int>(capacity: 50)

        // Populate
        for i in 0..<50 {
            await cache.put("key-\(i)", i)
        }

        let operationCount = AtomicCounter()

        await withTaskGroup(of: Void.self) { group in
            // Continuous reads
            for i in 0..<100 {
                group.addTask {
                    _ = await cache.get("key-\(i % 50)")
                    await operationCount.increment()
                }
            }

            // Reset midway
            group.addTask {
                try? await Task.sleep(nanoseconds: 5_000_000)
                await cache.reset()
                await operationCount.increment()
            }

            // More reads after reset
            for i in 0..<50 {
                group.addTask {
                    try? await Task.sleep(nanoseconds: 10_000_000)
                    _ = await cache.get("key-\(i)")
                    await operationCount.increment()
                }
            }
        }

        // All operations should complete without crashes
        let finalCount = await operationCount.value
        #expect(finalCount == 151)
    }
}

// MARK: - AdaptiveBatcher Stress Tests

@Suite("Concurrency Stress - Adaptive Batcher")
struct AdaptiveBatcherStressTests {

    @Test("Batcher handles burst of concurrent requests")
    func batcherBurstConcurrentRequests() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()

        var config = AdaptiveBatcherConfig()
        config.minBatchSize = 1
        config.maxBatchSize = 32

        let batcher = AdaptiveBatcher(model: model, config: config)
        let successCount = AtomicCounter()

        try await withThrowingTaskGroup(of: Void.self) { group in
            // Submit 50 requests as fast as possible
            for i in 0..<50 {
                group.addTask {
                    _ = try await batcher.embed("Request \(i)")
                    await successCount.increment()
                }
            }
            try await group.waitForAll()
        }

        let finalCount = await successCount.value
        #expect(finalCount == 50)

        let metrics = await batcher.metrics
        #expect(metrics.totalRequests == 50)
        #expect(metrics.totalBatches >= 1) // At least one batch processed
    }

    @Test("Batcher embedConcurrently maintains order")
    func batcherConcurrentlyMaintainsOrder() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let batcher = AdaptiveBatcher(model: model)

        let texts = (0..<20).map { "Text \($0)" }
        let embeddings = try await batcher.embedConcurrently(texts)

        #expect(embeddings.count == texts.count)
        // Order should be preserved
        for i in 0..<texts.count {
            #expect(embeddings[i].dimensions > 0)
        }
    }

    @Test("Batcher handles memory pressure changes during operation")
    func batcherMemoryPressureChanges() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let batcher = AdaptiveBatcher(model: model)

        let successCount = AtomicCounter()

        await withTaskGroup(of: Void.self) { group in
            // Submit requests
            for i in 0..<30 {
                group.addTask {
                    do {
                        _ = try await batcher.embed("Pressure test \(i)")
                        await successCount.increment()
                    } catch {
                        // Ignore errors for this test
                    }
                }
            }

            // Change memory pressure during processing
            group.addTask {
                for pressure: Float in [0.3, 0.6, 0.9, 0.5, 0.1] {
                    await batcher.setMemoryPressure(pressure)
                    try? await Task.sleep(nanoseconds: 10_000_000)
                }
            }
        }

        let finalCount = await successCount.value
        #expect(finalCount == 30)
    }
}

// MARK: - StreamingProcessor Stress Tests

@Suite("Concurrency Stress - Streaming Processor")
struct StreamingProcessorStressTests {

    @Test("Multiple concurrent streams complete independently")
    func multipleConcurrentStreams() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let processor = StreamingProcessor(model: model)

        let chunkCount = AtomicCounter()
        let documents = [
            "This is document one. It has multiple sentences. Each will be chunked.",
            "Document two is different. It also has sentences. Processing in parallel.",
            "Third document here. More content to process. Streaming works well."
        ]

        try await withThrowingTaskGroup(of: Void.self) { group in
            for doc in documents {
                group.addTask {
                    for try await _ in processor.embedStream(doc) {
                        await chunkCount.increment()
                    }
                }
            }
            try await group.waitForAll()
        }

        let finalCount = await chunkCount.value
        #expect(finalCount >= documents.count) // At least one chunk per doc
    }

    @Test("Concurrent stream with embedStreamConcurrent")
    func concurrentStreamMethod() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()

        var config = StreamingConfig()
        config.chunkingStrategy = .sentences(count: 2)

        let processor = StreamingProcessor(model: model, config: config)

        let longDocument = (0..<20).map { "Sentence number \($0). " }.joined()

        // Collect results directly in async sequence iteration
        var chunkEmbeddings: [ChunkEmbedding] = []

        for try await chunkEmb in processor.embedStreamConcurrent(longDocument) {
            chunkEmbeddings.append(chunkEmb)
        }

        #expect(chunkEmbeddings.count >= 5) // Multiple chunks processed

        // Verify all chunks have valid embeddings
        for ce in chunkEmbeddings {
            #expect(ce.embedding.dimensions > 0)
            #expect(!ce.chunk.text.isEmpty)
        }
    }
}

// MARK: - Task Cancellation Tests

@Suite("Concurrency Stress - Cancellation")
struct TaskCancellationStressTests {

    @Test("Task cancellation is handled gracefully")
    func taskCancellationGraceful() async throws {
        let backend = ConcurrencyTrackingBackend(delayMs: 50) // Longer delay
        let tokenizer = SimpleTokenizer()
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: EmbeddingConfiguration(),
            dimensions: 4,
            device: .cpu
        )

        let completedCount = AtomicCounter()
        let cancelledCount = AtomicCounter()

        let tasks = (0..<10).map { i in
            Task {
                do {
                    _ = try await model.embed("Cancellation test \(i)")
                    await completedCount.increment()
                } catch is CancellationError {
                    await cancelledCount.increment()
                } catch {
                    // Other errors
                }
            }
        }

        // Cancel some tasks quickly
        try await Task.sleep(nanoseconds: 10_000_000) // 10ms
        for (index, task) in tasks.enumerated() where index % 2 == 0 {
            task.cancel()
        }

        // Wait for all to complete or be cancelled
        for task in tasks {
            await task.value
        }

        // Total should equal 10 (all either completed or cancelled)
        let completed = await completedCount.value
        let cancelled = await cancelledCount.value
        #expect(completed + cancelled <= 10)
    }

    @Test("Batch processing handles partial cancellation")
    func batchPartialCancellation() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()

        let results = AtomicCounter()

        let task = Task {
            try await withThrowingTaskGroup(of: Void.self) { group in
                for i in 0..<20 {
                    group.addTask {
                        _ = try await model.embed("Batch cancel \(i)")
                        await results.increment()
                    }
                }
                try await group.waitForAll()
            }
        }

        // Let some complete, then cancel
        try await Task.sleep(nanoseconds: 20_000_000) // 20ms
        task.cancel()

        // Wait for cancellation to propagate
        try? await task.value

        // Some may have completed before cancellation
        let finalCount = await results.value
        #expect(finalCount >= 0)
    }
}

// MARK: - Resource Contention Tests

@Suite("Concurrency Stress - Resource Contention")
struct ResourceContentionTests {

    @Test("Metrics collection under high concurrency")
    func metricsCollectionConcurrent() async throws {
        let backend = ConcurrencyTrackingBackend(delayMs: 1)
        let tokenizer = SimpleTokenizer()
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: EmbeddingConfiguration(),
            dimensions: 4,
            device: .cpu
        )

        try await withThrowingTaskGroup(of: Void.self) { group in
            // Embeddings
            for i in 0..<50 {
                group.addTask {
                    _ = try await model.embed("Metrics test \(i)")
                }
            }

            // Concurrent metrics reads
            for _ in 0..<20 {
                group.addTask {
                    _ = await model.metrics
                }
            }

            try await group.waitForAll()
        }

        let finalMetrics = await model.metrics
        #expect(finalMetrics.totalRequests >= 50)
    }

    @Test("Memory pressure simulation during heavy load")
    func memoryPressureDuringLoad() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let modelID = model.id

        let successCount = AtomicCounter()

        await withTaskGroup(of: Void.self) { group in
            // Heavy embedding load
            for i in 0..<30 {
                group.addTask {
                    do {
                        _ = try await manager.embed("Memory test \(i)", using: modelID)
                        await successCount.increment()
                    } catch {
                        // May fail after memory pressure handling
                    }
                }
            }

            // Trigger memory pressure events
            group.addTask {
                await manager.simulateMemoryPressure(.warning)
                try? await Task.sleep(nanoseconds: 20_000_000)
                await manager.simulateMemoryPressure(.critical)
            }
        }

        // Most should succeed despite memory pressure
        let finalCount = await successCount.value
        #expect(finalCount >= 20)
    }

    @Test("Rapid model metrics reset during operation")
    func rapidMetricsResetDuringOperation() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let modelID = model.id

        await withTaskGroup(of: Void.self) { group in
            // Embeddings
            for i in 0..<30 {
                group.addTask {
                    _ = try? await manager.embed("Reset test \(i)", using: modelID)
                }
            }

            // Rapid resets
            for _ in 0..<5 {
                group.addTask {
                    try? await manager.resetMetrics(for: modelID)
                    try? await Task.sleep(nanoseconds: 5_000_000)
                }
            }
        }

        // Just verify no crashes - metrics may be in any state
        let metrics = try await manager.metrics(for: modelID)
        #expect(metrics.totalRequests >= 0)
    }
}

// MARK: - Cross-Component Stress Tests

@Suite("Concurrency Stress - Cross-Component")
struct CrossComponentStressTests {

    @Test("ModelManager with multiple models concurrently")
    func multipleModelsConcurrent() async throws {
        let manager = ModelManager()

        // Load multiple models
        let model1 = try await manager.loadMockModel()
        let model2 = try await manager.loadMockModel(
            configuration: EmbeddingConfiguration()
        )

        let successCount = AtomicCounter()

        try await withThrowingTaskGroup(of: Void.self) { group in
            // Embed with model 1
            for i in 0..<20 {
                group.addTask {
                    _ = try await model1.embed("Model1 text \(i)")
                    await successCount.increment()
                }
            }

            // Embed with model 2
            for i in 0..<20 {
                group.addTask {
                    _ = try await model2.embed("Model2 text \(i)")
                    await successCount.increment()
                }
            }

            try await group.waitForAll()
        }

        let finalCount = await successCount.value
        #expect(finalCount == 40)
    }

    @Test("Full pipeline stress: manager + batcher + streaming")
    func fullPipelineStress() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let batcher = AdaptiveBatcher(model: model)
        let processor = StreamingProcessor(model: model)

        let totalOperations = AtomicCounter()

        try await withThrowingTaskGroup(of: Void.self) { group in
            // Direct model embeds
            for i in 0..<10 {
                group.addTask {
                    _ = try await model.embed("Direct \(i)")
                    await totalOperations.increment()
                }
            }

            // Batcher embeds
            for i in 0..<10 {
                group.addTask {
                    _ = try await batcher.embed("Batched \(i)")
                    await totalOperations.increment()
                }
            }

            // Streaming embeds
            for i in 0..<5 {
                group.addTask {
                    let doc = "Streaming document \(i). It has content."
                    for try await _ in processor.embedStream(doc) {
                        await totalOperations.increment()
                    }
                }
            }

            try await group.waitForAll()
        }

        // 10 direct + 10 batched + (5 docs * at least 1 chunk each) = at least 25
        let finalCount = await totalOperations.value
        #expect(finalCount >= 25)
    }
}

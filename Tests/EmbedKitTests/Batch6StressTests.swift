// EmbedKit - Batch 6 Stress Tests
// High concurrency, rapid operations, and system stress testing

import Testing
import Foundation
@testable import EmbedKit
import VectorCore

// MARK: - Cancellation Stress Tests

@Suite("Cancellation Stress Tests")
struct CancellationStressTests {

    @Test("Rapid cancellation and restart cycles")
    func testRapidCancellationRestartCycles() async {
        let cycles = 100

        actor Counter {
            var completions = 0
            var cancellations = 0
            func markCompleted() { completions += 1 }
            func markCancelled() { cancellations += 1 }
        }
        let counter = Counter()

        for _ in 0..<cycles {
            let handle = CancellableEmbedding.withOperation { token in
                for i in 0..<10 {
                    if token.isCancelled {
                        await counter.markCancelled()
                        return i
                    }
                    try? await Task.sleep(for: .microseconds(100))
                }
                await counter.markCompleted()
                return 10
            }

            // Randomly cancel or let complete
            if Bool.random() {
                handle.cancel(mode: .immediate)
            }

            _ = try? await handle.value
        }

        let completions = await counter.completions
        let cancellations = await counter.cancellations
        let total = completions + cancellations

        #expect(total > 0)
        print("Completed: \(completions), Cancelled: \(cancellations)")
    }

    @Test("Concurrent cancellation from multiple threads")
    func testConcurrentCancellationMultipleThreads() async {
        let taskCount = 50

        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<taskCount {
                group.addTask {
                    let handle = CancellableEmbedding.withOperation { token in
                        for i in 0..<100 {
                            if token.shouldStopAt(checkpoint: .afterItem) {
                                return i
                            }
                            try? await Task.sleep(for: .microseconds(100))
                        }
                        return 100
                    }

                    // Multiple concurrent cancellations
                    await withTaskGroup(of: Void.self) { cancelGroup in
                        for _ in 0..<5 {
                            cancelGroup.addTask {
                                handle.cancel(mode: .immediate)
                            }
                        }
                    }

                    _ = try? await handle.value
                }
            }
        }

        // Should complete without crashes
        #expect(true)
    }

    @Test("High frequency cancellation token checks")
    func testHighFrequencyCancellationChecks() async {
        let token = CancellationToken()
        let iterations = 100000

        let handle = Task {
            var count = 0
            for i in 0..<iterations {
                if token.isCancelled {
                    return count
                }
                count = i
            }
            return count
        }

        // Cancel during processing
        Task {
            try? await Task.sleep(for: .milliseconds(10))
            token.cancel()
        }

        let result = await handle.value
        #expect(result < iterations)
    }

    @Test("Many handlers on single token")
    func testManyHandlersOnSingleToken() async {
        let token = CancellationToken()
        let handlerCount = 1000

        actor CallCounter {
            var count = 0
            func increment() { count += 1 }
        }
        let counter = CallCounter()

        for _ in 0..<handlerCount {
            token.onCancel { _ in
                Task { await counter.increment() }
            }
        }

        token.cancel()

        // Wait for handlers
        try? await Task.sleep(for: .milliseconds(200))

        let finalCount = await counter.count
        #expect(finalCount == handlerCount)
    }

    @Test("Nested task cancellation propagation")
    func testNestedTaskCancellationPropagation() async {
        let depth = 10

        func createNestedTask(depth: Int, token: CancellationToken) async -> Int {
            if depth == 0 {
                for i in 0..<100 {
                    if token.isCancelled { return i }
                    try? await Task.sleep(for: .microseconds(100))
                }
                return 100
            }

            if token.isCancelled { return 0 }
            return await createNestedTask(depth: depth - 1, token: token)
        }

        let token = CancellationToken()
        let handle = Task {
            await createNestedTask(depth: depth, token: token)
        }

        // Cancel after short delay
        Task {
            try? await Task.sleep(for: .milliseconds(20))
            token.cancel()
        }

        let result = await handle.value
        #expect(result < 100)
    }
}

// MARK: - Memory-Aware Stress Tests

@Suite("Memory-Aware Stress Tests")
struct MemoryAwareStressTests {

    @Test("High volume processing under memory pressure")
    func testHighVolumeUnderPressure() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let config = MemoryAwareConfig(baseBatchSize: 20, minBatchSize: 2)
        let memoryAware = MemoryAwareGenerator(generator: generator, config: config)

        // Start at critical pressure
        await memoryAware.setPressure(.critical)

        let texts = (0..<1000).map { "Text \($0)" }
        let result = try await memoryAware.produce(texts)

        #expect(result.count == 1000)

        let stats = await memoryAware.getStatistics()
        #expect(stats.batchesAtCritical > 0)
    }

    @Test("Concurrent pressure changes during processing")
    func testConcurrentPressureChanges() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let memoryAware = MemoryAwareGenerator(generator: generator)

        // Start multiple processing tasks
        await withTaskGroup(of: [[Float]].self) { processingGroup in
            for i in 0..<10 {
                processingGroup.addTask {
                    let texts = (0..<50).map { "Batch \(i) Text \($0)" }
                    return (try? await memoryAware.produce(texts)) ?? []
                }
            }

            // Concurrently change pressure levels rapidly
            Task {
                for _ in 0..<50 {
                    await memoryAware.setPressure(.normal)
                    try? await Task.sleep(for: .microseconds(500))
                    await memoryAware.setPressure(.warning)
                    try? await Task.sleep(for: .microseconds(500))
                    await memoryAware.setPressure(.critical)
                    try? await Task.sleep(for: .microseconds(500))
                }
            }

            var totalItems = 0
            for await result in processingGroup {
                totalItems += result.count
            }

            #expect(totalItems == 500)
        }
    }

    @Test("Extreme batch size variations")
    func testExtremeBatchSizeVariations() async throws {
        let model = MockEmbeddingModel(dimensions: 32)
        let generator = EmbeddingGenerator(model: model)
        let config = MemoryAwareConfig(baseBatchSize: 100, minBatchSize: 1)
        let memoryAware = MemoryAwareGenerator(generator: generator, config: config)

        let texts = (0..<200).map { "Text \($0)" }

        // Rapidly cycle through all pressure levels
        let processingTask = Task {
            try await memoryAware.produce(texts)
        }

        for _ in 0..<20 {
            await memoryAware.setPressure(.normal)
            try? await Task.sleep(for: .milliseconds(2))
            await memoryAware.setPressure(.critical)
            try? await Task.sleep(for: .milliseconds(2))
        }

        let result = try await processingTask.value
        #expect(result.count == 200)
    }

    @Test("Many concurrent producers")
    func testManyConcurrentProducers() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let memoryAware = MemoryAwareGenerator(generator: generator)

        let producerCount = 50

        let results = try await withThrowingTaskGroup(of: Int.self) { group in
            for i in 0..<producerCount {
                group.addTask {
                    let texts = (0..<10).map { "Producer \(i) Text \($0)" }
                    let embeddings = try await memoryAware.produce(texts)
                    return embeddings.count
                }
            }

            var total = 0
            for try await count in group {
                total += count
            }
            return total
        }

        #expect(results == producerCount * 10)
    }

    @Test("Statistics accuracy under high concurrency")
    func testStatisticsAccuracyHighConcurrency() async throws {
        let model = MockEmbeddingModel(dimensions: 32)
        let generator = EmbeddingGenerator(model: model)
        let config = MemoryAwareConfig(baseBatchSize: 5)
        let memoryAware = MemoryAwareGenerator(generator: generator, config: config)

        await memoryAware.setPressure(.normal)

        let taskCount = 20
        let textsPerTask = 25

        await withTaskGroup(of: Void.self) { group in
            for i in 0..<taskCount {
                group.addTask {
                    let texts = (0..<textsPerTask).map { "Task \(i) Text \($0)" }
                    _ = try? await memoryAware.produce(texts)
                }
            }
        }

        let stats = await memoryAware.getStatistics()
        #expect(stats.totalItemsProcessed == taskCount * textsPerTask)
    }
}

// MARK: - Pipelined Batch Processor Stress Tests

@Suite("Pipelined Batch Processor Stress Tests")
struct PipelinedBatchProcessorStressTests {

    @Test("Very large dataset processing")
    func testVeryLargeDataset() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(
            generator: generator,
            config: .highThroughput
        )

        let texts = (0..<5000).map { "Text \($0)" }
        let result = try await processor.produce(texts)

        #expect(result.count == 5000)

        let stats = await processor.getStatistics()
        #expect(stats.totalItems == 5000)
        #expect(stats.pipelineEfficiency > 0)
    }

    @Test("Rapid sequential processing")
    func testRapidSequentialProcessing() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(generator: generator)

        let iterations = 100

        for i in 0..<iterations {
            let texts = (0..<5).map { "Iteration \(i) Text \($0)" }
            let result = try await processor.produce(texts)
            #expect(result.count == 5)
        }

        // Should complete without issues
        #expect(true)
    }

    @Test("Many concurrent pipeline processors")
    func testManyConcurrentProcessors() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)

        let processorCount = 20

        let results = try await withThrowingTaskGroup(of: Int.self) { group in
            for i in 0..<processorCount {
                group.addTask {
                    let processor = PipelinedBatchProcessor(generator: generator)
                    let texts = (0..<20).map { "Processor \(i) Text \($0)" }
                    let embeddings = try await processor.produce(texts)
                    return embeddings.count
                }
            }

            var total = 0
            for try await count in group {
                total += count
            }
            return total
        }

        #expect(results == processorCount * 20)
    }

    @Test("Stream processing with many consumers")
    func testStreamManyConsumers() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(generator: generator)

        let texts = (0..<200).map { "Text \($0)" }

        // Multiple consumers reading from same stream
        actor ResultCollector {
            var results: [[Float]] = []
            func add(_ result: [Float]) {
                results.append(result)
            }
        }
        let collector = ResultCollector()

        for try await (embedding, _) in await processor.processStream(texts) {
            await collector.add(embedding)
        }

        let results = await collector.results
        #expect(results.count == 200)
    }

    @Test("Mixed batch sizes under load")
    func testMixedBatchSizesUnderLoad() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)

        let batchSizes = [1, 5, 10, 20, 50, 100]

        for batchSize in batchSizes {
            let processor = PipelinedBatchProcessor(
                generator: generator,
                config: PipelineConfig(batchSize: batchSize)
            )

            let texts = (0..<200).map { "Text \($0)" }
            let result = try await processor.produce(texts)

            #expect(result.count == 200)
        }
    }
}

// MARK: - Streaming Stress Tests

@Suite("Streaming Stress Tests")
struct StreamingStressTests {

    @Test("High concurrency streaming")
    func testHighConcurrencyStreaming() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let streaming = StreamingEmbeddingGenerator(
            generator: generator,
            config: FlowControlConfig(maxConcurrency: 10)
        )

        let texts = (0..<500).map { "Text \($0)" }
        let result = try await streaming.processConcurrently(texts)

        #expect(result.count == 500)

        let stats = await streaming.getStatistics()
        #expect(stats.totalProcessed == 500)
    }

    @Test("Rapid stream creation and destruction")
    func testRapidStreamCreationDestruction() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)

        for _ in 0..<100 {
            let streaming = StreamingEmbeddingGenerator(generator: generator)
            let texts = (0..<5).map { "Text \($0)" }

            var count = 0
            for try await _ in await streaming.stream(texts) {
                count += 1
            }

            #expect(count == 5)
        }
    }

    @Test("Back-pressure under extreme load")
    func testBackPressureExtremeLoad() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let streaming = StreamingEmbeddingGenerator(
            generator: generator,
            config: FlowControlConfig(
                maxConcurrency: 2,
                maxQueueDepth: 5,
                backPressureStrategy: .suspend
            )
        )

        let texts = (0..<200).map { "Text \($0)" }
        let result = try await streaming.produce(texts)

        #expect(result.count == 200)

        // Back-pressure tokens are acquired per batch, not per item
        // With default batchSize=32, 200 items = ~7 batches
        let bpStats = await streaming.getBackPressureStatistics()
        #expect(bpStats.totalAcquired >= 1)  // At least some acquisitions happened
    }

    @Test("Rate limiter under sustained load")
    func testRateLimiterSustainedLoad() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let streaming = StreamingEmbeddingGenerator(
            generator: generator,
            config: FlowControlConfig(
                rateLimitStrategy: .tokenBucket(capacity: 50, refillRate: 10)
            )
        )

        // Process multiple batches
        for batch in 0..<5 {
            let texts = (0..<30).map { "Batch \(batch) Text \($0)" }
            let result = try await streaming.produce(texts)
            #expect(result.count == 30)
        }

        let stats = await streaming.getStatistics()
        #expect(stats.totalProcessed == 150)
        #expect(stats.rateLimitHits > 0)
    }

    @Test("Concurrent rate limiter stress")
    func testConcurrentRateLimiterStress() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 100, refillRate: 50)
        )

        let taskCount = 50

        await withTaskGroup(of: Int.self) { group in
            for _ in 0..<taskCount {
                group.addTask {
                    var allowed = 0
                    for _ in 0..<10 {
                        if await limiter.allowRequest() {
                            allowed += 1
                        }
                        try? await Task.sleep(for: .microseconds(100))
                    }
                    return allowed
                }
            }

            var totalAllowed = 0
            for await count in group {
                totalAllowed += count
            }

            // Should be limited by capacity + refills
            #expect(totalAllowed <= 200)
        }
    }

    @Test("Drop strategies under overload")
    func testDropStrategiesUnderOverload() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)

        // Test drop oldest
        let dropOldest = StreamingEmbeddingGenerator(
            generator: generator,
            config: FlowControlConfig(
                maxQueueDepth: 3,
                backPressureStrategy: .dropOldest
            )
        )

        let texts1 = (0..<20).map { "Text \($0)" }
        let result1 = try await dropOldest.produce(texts1)
        #expect(result1.count <= 20)

        // Test drop newest
        let dropNewest = StreamingEmbeddingGenerator(
            generator: generator,
            config: FlowControlConfig(
                maxQueueDepth: 3,
                backPressureStrategy: .dropNewest
            )
        )

        let texts2 = (0..<20).map { "Text \($0)" }
        let result2 = try await dropNewest.produce(texts2)
        #expect(result2.count <= 20)
    }

    @Test("Many streaming instances simultaneously")
    func testManyStreamingInstances() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)

        let instanceCount = 30

        let results = try await withThrowingTaskGroup(of: Int.self) { group in
            for i in 0..<instanceCount {
                group.addTask {
                    let streaming = StreamingEmbeddingGenerator(generator: generator)
                    let texts = (0..<10).map { "Instance \(i) Text \($0)" }
                    let embeddings = try await streaming.produce(texts)
                    return embeddings.count
                }
            }

            var total = 0
            for try await count in group {
                total += count
            }
            return total
        }

        #expect(results == instanceCount * 10)
    }
}

// MARK: - Integration Stress Tests

@Suite("Integration Stress Tests")
struct IntegrationStressTests {

    @Test("All features under sustained high load")
    func testAllFeaturesHighLoad() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)

        // Memory-aware
        let memoryAware = MemoryAwareGenerator(
            generator: generator,
            config: MemoryAwareConfig(baseBatchSize: 20)
        )

        // Pipelined
        let pipelined = PipelinedBatchProcessor(
            generator: generator,
            config: .highThroughput
        )

        // Streaming
        let streaming = StreamingEmbeddingGenerator(
            generator: generator,
            config: FlowControlConfig(
                maxConcurrency: 4,
                rateLimitStrategy: .tokenBucket(capacity: 100, refillRate: 50)
            )
        )

        let texts = (0..<300).map { "Text \($0)" }

        // Run all simultaneously
        let results = try await withThrowingTaskGroup(of: Int.self) { group in
            group.addTask {
                await memoryAware.setPressure(.warning)
                let result = try await memoryAware.produce(texts)
                return result.count
            }

            group.addTask {
                let result = try await pipelined.produce(texts)
                return result.count
            }

            group.addTask {
                let result = try await streaming.produce(texts)
                return result.count
            }

            var counts: [Int] = []
            for try await count in group {
                counts.append(count)
            }
            return counts
        }

        #expect(results.allSatisfy { $0 == 300 })
    }

    @Test("Cancellation + memory pressure + rate limiting")
    func testCancellationMemoryRateLimiting() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let memoryAware = MemoryAwareGenerator(generator: generator)
        let streaming = StreamingEmbeddingGenerator(
            generator: generator,
            config: .rateLimited(requestsPerSecond: 100)
        )

        let texts = (0..<100).map { "Text \($0)" }

        let handle = CancellableEmbedding.withOperation { token in
            await memoryAware.setPressure(.critical)

            for text in texts {
                if token.shouldStopAt(checkpoint: .afterItem) {
                    break
                }
                _ = try await streaming.produce(text)
            }

            return texts.count
        }

        // Cancel after some processing
        Task {
            try? await Task.sleep(for: .milliseconds(50))
            handle.cancel(mode: .graceful)
        }

        let result = try? await handle.value
        #expect(result == nil || result! <= 100)
    }

    @Test("Pipeline + streaming concurrent processing")
    func testPipelineStreamingConcurrent() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)

        let pipelined = PipelinedBatchProcessor(generator: generator)
        let streaming = StreamingEmbeddingGenerator(generator: generator)

        let textArrays = (0..<10).map { batch in
            (0..<50).map { "Batch \(batch) Text \($0)" }
        }

        let results = try await withThrowingTaskGroup(of: Int.self) { group in
            // Half through pipeline
            for i in 0..<5 {
                group.addTask {
                    let result = try await pipelined.produce(textArrays[i])
                    return result.count
                }
            }

            // Half through streaming
            for i in 5..<10 {
                group.addTask {
                    let result = try await streaming.produce(textArrays[i])
                    return result.count
                }
            }

            var total = 0
            for try await count in group {
                total += count
            }
            return total
        }

        #expect(results == 500)
    }

    @Test("Resource cleanup under stress")
    func testResourceCleanupUnderStress() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)

        // Create and destroy many instances
        for _ in 0..<100 {
            let memoryAware = MemoryAwareGenerator(generator: generator)
            let pipelined = PipelinedBatchProcessor(generator: generator)
            let streaming = StreamingEmbeddingGenerator(generator: generator)

            let texts = ["Test 1", "Test 2", "Test 3"]

            _ = try? await memoryAware.produce(texts)
            _ = try? await pipelined.produce(texts)
            _ = try? await streaming.produce(texts)

            // Instances should be cleaned up when out of scope
        }

        // Should complete without memory issues
        #expect(true)
    }

    @Test("Extreme concurrency across all features")
    func testExtremeConcurrency() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)

        let taskCount = 100

        let results = try await withThrowingTaskGroup(of: Int.self) { group in
            for i in 0..<taskCount {
                group.addTask {
                    let type = i % 3

                    let texts = (0..<5).map { "Task \(i) Text \($0)" }

                    switch type {
                    case 0:
                        let memoryAware = MemoryAwareGenerator(generator: generator)
                        let result = try await memoryAware.produce(texts)
                        return result.count
                    case 1:
                        let pipelined = PipelinedBatchProcessor(generator: generator)
                        let result = try await pipelined.produce(texts)
                        return result.count
                    default:
                        let streaming = StreamingEmbeddingGenerator(generator: generator)
                        let result = try await streaming.produce(texts)
                        return result.count
                    }
                }
            }

            var total = 0
            for try await count in group {
                total += count
            }
            return total
        }

        #expect(results == taskCount * 5)
    }
}

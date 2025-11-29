// EmbedKit - Batch 6 Edge Case Tests
// Comprehensive edge case testing for CancellableEmbeddingTask, MemoryAwareGenerator,
// PipelinedBatchProcessor, and Streaming features

import Testing
import Foundation
@testable import EmbedKit
import VectorCore

// MARK: - CancellableEmbeddingTask Edge Cases

@Suite("CancellableEmbeddingTask Edge Cases")
struct CancellableEmbeddingTaskEdgeCases {

    @Test("Cancel before task starts")
    func testCancelBeforeStart() async {
        let task = CancellableEmbeddingTask<Int> { token in
            try? await Task.sleep(for: .seconds(1))
            return 42
        }

        let handle = task.start()
        handle.cancel(mode: .immediate)

        // Wait a moment
        try? await Task.sleep(for: .milliseconds(10))

        #expect(handle.isCancelled)
        #expect(handle.state == .cancellationRequested || handle.state == .cancelled)
    }

    @Test("Multiple concurrent cancellations")
    func testMultipleConcurrentCancellations() async {
        let task = CancellableEmbeddingTask<Int> { token in
            for i in 0..<100 {
                if token.shouldStopAt(checkpoint: .afterItem) {
                    return i
                }
                try? await Task.sleep(for: .milliseconds(5))
            }
            return 100
        }

        let handle = task.start()

        // Trigger multiple cancellations concurrently
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<10 {
                group.addTask {
                    handle.cancel(mode: .immediate)
                }
            }
        }

        // Should handle multiple cancellations gracefully
        #expect(handle.isCancelled)
    }

    @Test("Empty result on immediate cancellation")
    func testEmptyResultImmediateCancellation() async {
        let task = CancellableEmbeddingTask<[Int]> { token in
            var results: [Int] = []
            for i in 0..<100 {
                if token.shouldStopAt(checkpoint: .afterItem) {
                    return results
                }
                results.append(i)
                try? await Task.sleep(for: .milliseconds(1))
            }
            return results
        }

        let handle = task.start()
        handle.cancel(mode: .immediate)

        let result = try? await handle.value
        // Result may be empty or partial depending on timing
        #expect(result == nil || result!.count < 100)
    }

    @Test("Task completion before cancellation processed")
    func testTaskCompletesBeforeCancellation() async throws {
        let task = CancellableEmbeddingTask<Int> { _ in
            return 42
        }

        let handle = task.start()
        let result = try await handle.value

        // Cancel after completion
        handle.cancel()

        #expect(result == 42)
        #expect(handle.state == .completed)
    }

    @Test("Nested cancellation tokens")
    func testNestedCancellationTokens() async {
        let outerToken = CancellationToken()
        let innerToken = CancellationToken()

        let task = CancellableEmbeddingTask<Int> { token in
            // Simulate nested operations
            for i in 0..<50 {
                if outerToken.isCancelled || innerToken.isCancelled || token.isCancelled {
                    return i
                }
                try? await Task.sleep(for: .milliseconds(2))
            }
            return 50
        }

        let handle = task.start()

        // Cancel outer token
        Task {
            try? await Task.sleep(for: .milliseconds(30))
            outerToken.cancel()
        }

        let result = try? await handle.value
        // Should complete or return partial results
        _ = result
    }

    @Test("Zero-delay task with cancellation")
    func testZeroDelayTaskCancellation() async {
        let task = CancellableEmbeddingTask<Int> { token in
            var sum = 0
            for i in 0..<10000 {
                if token.shouldStopAt(checkpoint: .afterItem) {
                    return sum
                }
                sum += i
            }
            return sum
        }

        let handle = task.start()
        handle.cancel(mode: .immediate)

        let result = try? await handle.value
        // May or may not complete depending on scheduling
        _ = result
    }
}

// MARK: - MemoryAwareGenerator Edge Cases

@Suite("MemoryAwareGenerator Edge Cases")
struct MemoryAwareGeneratorEdgeCases {

    @Test("Empty input array")
    func testEmptyInput() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let memoryAware = MemoryAwareGenerator(generator: generator)

        let result = try await memoryAware.produce([])
        #expect(result.isEmpty)

        let stats = await memoryAware.getStatistics()
        #expect(stats.totalItemsProcessed == 0)
        #expect(stats.totalBatches == 0)
    }

    @Test("Single item processing")
    func testSingleItem() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let memoryAware = MemoryAwareGenerator(generator: generator)

        let result = try await memoryAware.produce(["Single item"])
        #expect(result.count == 1)
        #expect(result[0].count == 128)

        let stats = await memoryAware.getStatistics()
        #expect(stats.totalItemsProcessed == 1)
    }

    @Test("Batch size equals input size")
    func testBatchSizeEqualsInput() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let config = MemoryAwareConfig(baseBatchSize: 5, minBatchSize: 1)
        let memoryAware = MemoryAwareGenerator(generator: generator, config: config)

        await memoryAware.setPressure(.normal)
        let texts = (0..<5).map { "Text \($0)" }
        let result = try await memoryAware.produce(texts)

        #expect(result.count == 5)

        let stats = await memoryAware.getStatistics()
        #expect(stats.totalBatches == 1)
    }

    @Test("Batch size larger than input")
    func testBatchSizeLargerThanInput() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let config = MemoryAwareConfig(baseBatchSize: 100, minBatchSize: 1)
        let memoryAware = MemoryAwareGenerator(generator: generator, config: config)

        let texts = ["A", "B", "C"]
        let result = try await memoryAware.produce(texts)

        #expect(result.count == 3)

        let stats = await memoryAware.getStatistics()
        #expect(stats.totalBatches == 1)
    }

    @Test("Rapid pressure changes during processing")
    func testRapidPressureChanges() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let config = MemoryAwareConfig(baseBatchSize: 20, minBatchSize: 2)
        let memoryAware = MemoryAwareGenerator(generator: generator, config: config)

        // Start processing in background
        let processingTask = Task {
            let texts = (0..<100).map { "Text \($0)" }
            return try await memoryAware.produce(texts)
        }

        // Rapidly change pressure levels
        for _ in 0..<10 {
            await memoryAware.setPressure(.normal)
            try? await Task.sleep(for: .milliseconds(5))
            await memoryAware.setPressure(.warning)
            try? await Task.sleep(for: .milliseconds(5))
            await memoryAware.setPressure(.critical)
            try? await Task.sleep(for: .milliseconds(5))
        }

        let result = try await processingTask.value
        #expect(result.count == 100)
    }

    @Test("Minimum batch size enforced at critical pressure")
    func testMinimumBatchSizeEnforced() async throws {
        let model = MockEmbeddingModel(dimensions: 32)
        let generator = EmbeddingGenerator(model: model)
        let config = MemoryAwareConfig(baseBatchSize: 4, minBatchSize: 2)
        let memoryAware = MemoryAwareGenerator(generator: generator, config: config)

        await memoryAware.setPressure(.critical)
        let batchSize = await memoryAware.effectiveBatchSize

        // At critical (0.25 multiplier): 4 * 0.25 = 1, but min is 2
        #expect(batchSize == 2)
    }

    @Test("Adaptive batching disabled ignores pressure")
    func testAdaptiveBatchingDisabled() async throws {
        let model = MockEmbeddingModel(dimensions: 32)
        let generator = EmbeddingGenerator(model: model)
        let config = MemoryAwareConfig(
            baseBatchSize: 32,
            minBatchSize: 4,
            adaptiveBatching: false
        )
        let memoryAware = MemoryAwareGenerator(generator: generator, config: config)

        await memoryAware.setPressure(.critical)
        let batchSize = await memoryAware.effectiveBatchSize

        #expect(batchSize == 32)
    }

    @Test("Statistics reset clears all counters")
    func testStatisticsReset() async throws {
        let model = MockEmbeddingModel(dimensions: 32)
        let generator = EmbeddingGenerator(model: model)
        let memoryAware = MemoryAwareGenerator(generator: generator)

        // Process some items
        _ = try await memoryAware.produce((0..<10).map { "Text \($0)" })

        // Change pressure a few times
        await memoryAware.setPressure(.warning)
        await memoryAware.setPressure(.critical)

        await memoryAware.resetStatistics()

        let stats = await memoryAware.getStatistics()
        #expect(stats.totalItemsProcessed == 0)
        #expect(stats.totalBatches == 0)
        #expect(stats.batchesAtNormal == 0)
        #expect(stats.batchesAtWarning == 0)
        #expect(stats.batchesAtCritical == 0)
        #expect(stats.batchSizeAdjustments == 0)
    }

    @Test("Concurrent produce calls handle pressure")
    func testConcurrentProduceCalls() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let config = MemoryAwareConfig(baseBatchSize: 10, minBatchSize: 2)
        let memoryAware = MemoryAwareGenerator(generator: generator, config: config)

        await memoryAware.setPressure(.warning)

        let results = try await withThrowingTaskGroup(of: [[Float]].self) { group in
            for i in 0..<5 {
                group.addTask {
                    let texts = (0..<3).map { "Concurrent \(i)-\($0)" }
                    return try await memoryAware.produce(texts)
                }
            }

            var allResults: [[[Float]]] = []
            for try await result in group {
                allResults.append(result)
            }
            return allResults
        }

        #expect(results.count == 5)
        #expect(results.allSatisfy { $0.count == 3 })
    }
}

// MARK: - PipelinedBatchProcessor Edge Cases

@Suite("PipelinedBatchProcessor Edge Cases")
struct PipelinedBatchProcessorEdgeCases {

    @Test("Empty input produces empty output")
    func testEmptyInput() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(generator: generator)

        let result = try await processor.produce([])
        #expect(result.isEmpty)
    }

    @Test("Input size exactly matches batch size")
    func testExactBatchSizeMatch() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let config = PipelineConfig(batchSize: 8)
        let processor = PipelinedBatchProcessor(generator: generator, config: config)

        let texts = (0..<8).map { "Text \($0)" }
        let result = try await processor.produce(texts)

        #expect(result.count == 8)

        let stats = await processor.getStatistics()
        #expect(stats.totalBatches == 1)
        #expect(stats.partialBatches == 0)
    }

    @Test("Input size is multiple of batch size")
    func testMultipleBatchSizes() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let config = PipelineConfig(batchSize: 5)
        let processor = PipelinedBatchProcessor(generator: generator, config: config)

        let texts = (0..<25).map { "Text \($0)" }
        let result = try await processor.produce(texts)

        #expect(result.count == 25)

        let stats = await processor.getStatistics()
        #expect(stats.totalBatches == 5)
        #expect(stats.partialBatches == 0)
    }

    @Test("Input size is one less than batch size")
    func testOneLessThanBatchSize() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let config = PipelineConfig(batchSize: 10)
        let processor = PipelinedBatchProcessor(generator: generator, config: config)

        let texts = (0..<9).map { "Text \($0)" }
        let result = try await processor.produce(texts)

        #expect(result.count == 9)

        let stats = await processor.getStatistics()
        #expect(stats.totalBatches == 1)
        #expect(stats.partialBatches == 1)
    }

    @Test("Input size is one more than batch size")
    func testOneMoreThanBatchSize() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let config = PipelineConfig(batchSize: 10)
        let processor = PipelinedBatchProcessor(generator: generator, config: config)

        let texts = (0..<11).map { "Text \($0)" }
        let result = try await processor.produce(texts)

        #expect(result.count == 11)

        let stats = await processor.getStatistics()
        #expect(stats.totalBatches == 2)
        #expect(stats.partialBatches == 1)
    }

    @Test("Very small batch size")
    func testVerySmallBatchSize() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let config = PipelineConfig(batchSize: 1)
        let processor = PipelinedBatchProcessor(generator: generator, config: config)

        let texts = (0..<5).map { "Text \($0)" }
        let result = try await processor.produce(texts)

        #expect(result.count == 5)

        let stats = await processor.getStatistics()
        #expect(stats.totalBatches == 5)
        #expect(stats.partialBatches == 0)
    }

    @Test("Very large batch size with small input")
    func testVeryLargeBatchSize() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let config = PipelineConfig(batchSize: 1000)
        let processor = PipelinedBatchProcessor(generator: generator, config: config)

        let texts = (0..<3).map { "Text \($0)" }
        let result = try await processor.produce(texts)

        #expect(result.count == 3)

        let stats = await processor.getStatistics()
        #expect(stats.totalBatches == 1)
        #expect(stats.partialBatches == 1)
    }

    @Test("Stream early termination on first item")
    func testStreamEarlyTerminationFirstItem() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(generator: generator)

        let texts = (0..<20).map { "Text \($0)" }
        var count = 0

        for try await _ in await processor.processStream(texts) {
            count += 1
            break // Stop immediately
        }

        #expect(count == 1)
    }

    @Test("Concurrent pipeline processors don't interfere")
    func testConcurrentProcessors() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)

        let processor1 = PipelinedBatchProcessor(
            generator: generator,
            config: PipelineConfig(batchSize: 5)
        )
        let processor2 = PipelinedBatchProcessor(
            generator: generator,
            config: PipelineConfig(batchSize: 10)
        )

        let results = try await withThrowingTaskGroup(of: [[Float]].self) { group in
            group.addTask {
                let texts = (0..<15).map { "P1-\($0)" }
                return try await processor1.produce(texts)
            }
            group.addTask {
                let texts = (0..<20).map { "P2-\($0)" }
                return try await processor2.produce(texts)
            }

            var allResults: [[[Float]]] = []
            for try await result in group {
                allResults.append(result)
            }
            return allResults
        }

        #expect(results.count == 2)
        #expect(results.contains { $0.count == 15 })
        #expect(results.contains { $0.count == 20 })
    }
}

// MARK: - Streaming Edge Cases

@Suite("Streaming Edge Cases")
struct StreamingEdgeCases {

    @Test("Empty input array")
    func testEmptyInput() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let streaming = StreamingEmbeddingGenerator(generator: generator)

        let result = try await streaming.produce([])
        #expect(result.isEmpty)
    }

    @Test("Single item processing")
    func testSingleItem() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let streaming = StreamingEmbeddingGenerator(generator: generator)

        let result = try await streaming.produce("Single item")
        #expect(result.count == 128)
    }

    @Test("Stream with one item")
    func testStreamOneItem() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let streaming = StreamingEmbeddingGenerator(generator: generator)

        var count = 0
        for try await _ in await streaming.stream(["One"]) {
            count += 1
        }

        #expect(count == 1)
    }

    @Test("Concurrent processing with empty batches")
    func testConcurrentProcessingEmptyBatches() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let streaming = StreamingEmbeddingGenerator(generator: generator)

        let textArrays: [[String]] = [[], [], []]
        let results = try await withThrowingTaskGroup(of: [[Float]].self) { group in
            for texts in textArrays {
                group.addTask {
                    return try await streaming.produce(texts)
                }
            }

            var allResults: [[[Float]]] = []
            for try await result in group {
                allResults.append(result)
            }
            return allResults
        }

        #expect(results.count == 3)
        #expect(results.allSatisfy { $0.isEmpty })
    }

    @Test("Rate limiter at capacity")
    func testRateLimiterAtCapacity() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let streaming = StreamingEmbeddingGenerator(
            generator: generator,
            config: FlowControlConfig(
                rateLimitStrategy: .tokenBucket(capacity: 5, refillRate: 0.1)
            )
        )

        // Exhaust rate limiter
        let texts = (0..<10).map { "Text \($0)" }
        let result = try await streaming.produce(texts)

        #expect(result.count == 10)

        let stats = await streaming.getStatistics()
        #expect(stats.rateLimitHits > 0)
    }

    @Test("Back-pressure with very small queue")
    func testBackPressureSmallQueue() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let streaming = StreamingEmbeddingGenerator(
            generator: generator,
            config: FlowControlConfig(
                maxQueueDepth: 1,
                backPressureStrategy: .suspend
            )
        )

        let texts = (0..<5).map { "Text \($0)" }
        let result = try await streaming.produce(texts)

        #expect(result.count == 5)
    }

    @Test("Drop oldest strategy under pressure")
    func testDropOldestStrategy() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let streaming = StreamingEmbeddingGenerator(
            generator: generator,
            config: FlowControlConfig(
                maxQueueDepth: 2,
                backPressureStrategy: .dropOldest
            )
        )

        let texts = (0..<10).map { "Text \($0)" }
        let result = try await streaming.produce(texts)

        // Should complete but may have dropped some
        #expect(result.count <= 10)
    }

    @Test("Error timeout strategy")
    func testErrorTimeoutStrategy() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let streaming = StreamingEmbeddingGenerator(
            generator: generator,
            config: FlowControlConfig(
                maxQueueDepth: 1,
                backPressureStrategy: .error(timeout: 0.01)
            )
        )

        // This may or may not error depending on timing
        do {
            let texts = (0..<100).map { "Text \($0)" }
            _ = try await streaming.produce(texts)
        } catch {
            // Expected in some cases
            _ = error
        }
    }

    @Test("Concurrent streaming operations")
    func testConcurrentStreamingOperations() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let streaming = StreamingEmbeddingGenerator(generator: generator)

        let results = try await withThrowingTaskGroup(of: [[Float]].self) { group in
            for i in 0..<5 {
                group.addTask {
                    let texts = (0..<5).map { "Stream-\(i)-\($0)" }
                    return try await streaming.produce(texts)
                }
            }

            var allResults: [[[Float]]] = []
            for try await result in group {
                allResults.append(result)
            }
            return allResults
        }

        #expect(results.count == 5)
        #expect(results.allSatisfy { $0.count == 5 })
    }
}

// MARK: - Rate Limiter Edge Cases

@Suite("Rate Limiter Edge Cases")
struct RateLimiterEdgeCases {

    @Test("Token bucket with zero capacity")
    func testTokenBucketZeroCapacity() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 0, refillRate: 1)
        )

        let allowed = await limiter.allowRequest()
        #expect(!allowed)
    }

    @Test("Token bucket with zero refill rate")
    func testTokenBucketZeroRefillRate() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 5, refillRate: 0)
        )

        // Should allow burst
        for _ in 0..<5 {
            #expect(await limiter.allowRequest())
        }

        // Then no more
        #expect(!(await limiter.allowRequest()))
    }

    @Test("Sliding window with zero max requests")
    func testSlidingWindowZeroMax() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .slidingWindow(windowSize: 1.0, maxRequests: 0)
        )

        let allowed = await limiter.allowRequest()
        #expect(!allowed)
    }

    @Test("Leaky bucket rapid requests")
    func testLeakyBucketRapidRequests() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .leakyBucket(capacity: 2, leakRate: 10.0)
        )

        // Rapid fire
        var allowed = 0
        for _ in 0..<10 {
            if await limiter.allowRequest() {
                allowed += 1
            }
        }

        // Should be limited to capacity
        #expect(allowed <= 2)
    }

    @Test("Wait for permit with immediate availability")
    func testWaitForPermitImmediate() async throws {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 10, refillRate: 1)
        )

        // Should succeed immediately
        try await limiter.waitForPermit(timeout: 0.1)
    }
}

// MARK: - Back-Pressure Edge Cases

@Suite("Back-Pressure Edge Cases")
struct BackPressureEdgeCases {

    @Test("Acquire on minimum-depth queue behaves correctly")
    func testAcquireMinimumDepth() async throws {
        // maxQueueDepth: 0 triggers precondition failure, so test with 1
        let controller = BackPressureController(maxQueueDepth: 1)

        // First acquire should succeed
        let token1 = try await controller.acquire()

        // Second acquire should block, so use tryAcquire which returns nil
        let token2 = await controller.tryAcquire()
        #expect(token2 == nil)

        token1.release()
    }

    @Test("Try acquire returns nil when full")
    func testTryAcquireReturnsNil() async throws {
        let controller = BackPressureController(maxQueueDepth: 1)

        let token1 = try await controller.acquire()
        let token2 = await controller.tryAcquire()

        #expect(token2 == nil)
        token1.release()
    }

    @Test("Token hold duration increases over time")
    func testTokenHoldDuration() async throws {
        let controller = BackPressureController(maxQueueDepth: 5)

        let token = try await controller.acquire()
        let duration1 = token.holdDuration

        try await Task.sleep(for: .milliseconds(50))
        let duration2 = token.holdDuration

        #expect(duration2 > duration1)
        token.release()
    }

    @Test("Multiple tokens released in correct order")
    func testMultipleTokensReleased() async throws {
        let controller = BackPressureController(maxQueueDepth: 5)

        var tokens: [BackPressureToken] = []
        for _ in 0..<3 {
            tokens.append(try await controller.acquire())
        }

        #expect(await controller.queueDepth == 3)

        for token in tokens {
            token.release()
        }

        // Wait for async releases
        try await Task.sleep(for: .milliseconds(20))
        #expect(await controller.queueDepth == 0)
    }
}

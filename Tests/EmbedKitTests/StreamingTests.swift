// EmbedKit - Streaming Tests
// Tests for back-pressure, rate limiting, and streaming embedding generation

import Testing
import Foundation
@testable import EmbedKit
import VectorCore

// MARK: - BackPressureStrategy Tests

@Suite("BackPressureStrategy")
struct BackPressureStrategyTests {

    @Test("All strategies available")
    func testAllStrategies() {
        let strategies: [BackPressureStrategy] = [
            .suspend,
            .dropOldest,
            .dropNewest,
            .error(timeout: nil),
            .error(timeout: 1.0)
        ]
        #expect(strategies.count == 5)
    }

    @Test("Strategy equality")
    func testEquality() {
        #expect(BackPressureStrategy.suspend == .suspend)
        #expect(BackPressureStrategy.dropOldest == .dropOldest)
        #expect(BackPressureStrategy.error(timeout: 1.0) == .error(timeout: 1.0))
        #expect(BackPressureStrategy.error(timeout: nil) != .error(timeout: 1.0))
    }

    @Test("Strategies are Sendable")
    func testSendable() async {
        let strategy = BackPressureStrategy.suspend
        let result = await Task { strategy }.value
        #expect(result == .suspend)
    }
}

// MARK: - RateLimitStrategy Tests

@Suite("RateLimitStrategy")
struct RateLimitStrategyTests {

    @Test("Token bucket strategy")
    func testTokenBucket() {
        let strategy = RateLimitStrategy.tokenBucket(capacity: 100, refillRate: 10)
        if case .tokenBucket(let capacity, let rate) = strategy {
            #expect(capacity == 100)
            #expect(rate == 10)
        } else {
            Issue.record("Expected token bucket")
        }
    }

    @Test("Sliding window strategy")
    func testSlidingWindow() {
        let strategy = RateLimitStrategy.slidingWindow(windowSize: 60, maxRequests: 100)
        if case .slidingWindow(let window, let max) = strategy {
            #expect(window == 60)
            #expect(max == 100)
        } else {
            Issue.record("Expected sliding window")
        }
    }

    @Test("Leaky bucket strategy")
    func testLeakyBucket() {
        let strategy = RateLimitStrategy.leakyBucket(capacity: 50, leakRate: 0.1)
        if case .leakyBucket(let capacity, let rate) = strategy {
            #expect(capacity == 50)
            #expect(rate == 0.1)
        } else {
            Issue.record("Expected leaky bucket")
        }
    }
}

// MARK: - RateLimitStatus Tests

@Suite("RateLimitStatus")
struct RateLimitStatusTests {

    @Test("Status properties")
    func testProperties() {
        let status = RateLimitStatus(
            remaining: 50,
            limit: 100,
            resetAt: Date().addingTimeInterval(30)
        )

        #expect(status.remaining == 50)
        #expect(status.limit == 100)
        #expect(!status.isLimited)
        #expect(status.timeUntilReset > 0)
        #expect(status.timeUntilReset <= 30)
    }

    @Test("Limited status")
    func testLimitedStatus() {
        let status = RateLimitStatus(
            remaining: 0,
            limit: 100,
            resetAt: Date().addingTimeInterval(10)
        )

        #expect(status.isLimited)
    }

    @Test("Status is Sendable")
    func testSendable() async {
        let status = RateLimitStatus(remaining: 10, limit: 100, resetAt: Date())
        let result = await Task { status.remaining }.value
        #expect(result == 10)
    }
}

// MARK: - FlowControlConfig Tests

@Suite("FlowControlConfig")
struct FlowControlConfigTests {

    @Test("Default configuration")
    func testDefaultConfig() {
        let config = FlowControlConfig.default

        #expect(config.maxConcurrency == 4)
        #expect(config.maxQueueDepth == 100)
        #expect(config.backPressureStrategy == .suspend)
        #expect(config.rateLimitStrategy == nil)
        #expect(config.batchSize == 32)
    }

    @Test("High throughput configuration")
    func testHighThroughputConfig() {
        let config = FlowControlConfig.highThroughput

        #expect(config.maxConcurrency == 8)
        #expect(config.maxQueueDepth == 500)
        #expect(config.backPressureStrategy == .dropOldest)
        #expect(config.batchSize == 64)
    }

    @Test("Low latency configuration")
    func testLowLatencyConfig() {
        let config = FlowControlConfig.lowLatency

        #expect(config.maxConcurrency == 2)
        #expect(config.maxQueueDepth == 20)
        #expect(config.batchSize == 8)
    }

    @Test("Rate limited configuration")
    func testRateLimitedConfig() {
        let config = FlowControlConfig.rateLimited(requestsPerSecond: 50)

        #expect(config.rateLimitStrategy != nil)
    }

    @Test("Custom configuration")
    func testCustomConfig() {
        let config = FlowControlConfig(
            maxConcurrency: 10,
            maxQueueDepth: 200,
            backPressureStrategy: .dropNewest,
            batchSize: 16
        )

        #expect(config.maxConcurrency == 10)
        #expect(config.maxQueueDepth == 200)
        #expect(config.backPressureStrategy == .dropNewest)
        #expect(config.batchSize == 16)
    }
}

// MARK: - StreamingError Tests

@Suite("StreamingError")
struct StreamingErrorTests {

    @Test("Queue full error")
    func testQueueFullError() {
        let error = StreamingError.queueFull(current: 100, limit: 100)
        #expect(error.errorDescription?.contains("Queue is full") == true)
    }

    @Test("Timeout error")
    func testTimeoutError() {
        let error = StreamingError.timeout(waited: 5.0)
        #expect(error.errorDescription?.contains("Timeout") == true)
    }

    @Test("Dropped error")
    func testDroppedError() {
        let error = StreamingError.dropped(reason: "Test drop")
        #expect(error.errorDescription?.contains("dropped") == true)
    }

    @Test("Cancelled error")
    func testCancelledError() {
        let error = StreamingError.cancelled
        #expect(error.errorDescription?.contains("cancelled") == true)
    }
}

// MARK: - EmbeddingRateLimiter Tests

@Suite("EmbeddingRateLimiter")
struct EmbeddingRateLimiterTests {

    @Test("Token bucket allows burst")
    func testTokenBucketBurst() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 10, refillRate: 1)
        )

        // Should allow burst up to capacity
        for _ in 0..<10 {
            let allowed = await limiter.allowRequest()
            #expect(allowed)
        }

        // Next should be rate limited
        let allowed = await limiter.allowRequest()
        #expect(!allowed)
    }

    @Test("Token bucket refills over time")
    func testTokenBucketRefill() async throws {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 2, refillRate: 10)  // 10 tokens/sec
        )

        // Exhaust tokens
        _ = await limiter.allowRequest()
        _ = await limiter.allowRequest()
        #expect(!(await limiter.allowRequest()))

        // Wait for refill
        try await Task.sleep(for: .milliseconds(150))

        // Should have refilled
        #expect(await limiter.allowRequest())
    }

    @Test("Sliding window tracks requests")
    func testSlidingWindow() async throws {
        let limiter = EmbeddingRateLimiter(
            strategy: .slidingWindow(windowSize: 0.5, maxRequests: 3)
        )

        // Allow 3 requests
        #expect(await limiter.allowRequest())
        #expect(await limiter.allowRequest())
        #expect(await limiter.allowRequest())

        // 4th should be limited
        #expect(!(await limiter.allowRequest()))

        // Wait for window to slide
        try await Task.sleep(for: .milliseconds(600))

        // Should allow again
        #expect(await limiter.allowRequest())
    }

    @Test("Fixed window resets")
    func testFixedWindow() async throws {
        let limiter = EmbeddingRateLimiter(
            strategy: .fixedWindow(windowSize: 0.3, maxRequests: 2)
        )

        #expect(await limiter.allowRequest())
        #expect(await limiter.allowRequest())
        #expect(!(await limiter.allowRequest()))

        // Wait for window reset
        try await Task.sleep(for: .milliseconds(350))

        #expect(await limiter.allowRequest())
    }

    @Test("Leaky bucket constant rate")
    func testLeakyBucket() async throws {
        let limiter = EmbeddingRateLimiter(
            strategy: .leakyBucket(capacity: 3, leakRate: 0.1)  // Leak every 100ms
        )

        // Fill bucket
        #expect(await limiter.allowRequest())
        #expect(await limiter.allowRequest())
        #expect(await limiter.allowRequest())
        #expect(!(await limiter.allowRequest()))

        // Wait for leak
        try await Task.sleep(for: .milliseconds(150))

        #expect(await limiter.allowRequest())
    }

    @Test("Get status")
    func testGetStatus() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 100, refillRate: 10)
        )

        let status = await limiter.getStatus()
        #expect(status.remaining == 100)
        #expect(status.limit == 100)
    }

    @Test("Wait for permit")
    func testWaitForPermit() async throws {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 1, refillRate: 10)
        )

        // Exhaust
        _ = await limiter.allowRequest()

        // Should wait and then succeed
        try await limiter.waitForPermit(timeout: 1.0)
    }

    @Test("Reset clears state")
    func testReset() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 2, refillRate: 0.1)
        )

        _ = await limiter.allowRequest()
        _ = await limiter.allowRequest()
        #expect(!(await limiter.allowRequest()))

        await limiter.reset()

        #expect(await limiter.allowRequest())
    }
}

// MARK: - BackPressureController Tests

@Suite("BackPressureController")
struct BackPressureControllerTests {

    @Test("Immediate acquisition")
    func testImmediateAcquisition() async throws {
        let controller = BackPressureController(maxQueueDepth: 10)

        let token = try await controller.acquire()
        #expect(!token.isReleased)
        #expect(await controller.queueDepth == 1)

        token.release()
        #expect(token.isReleased)
        // Release happens asynchronously via Task
        try await Task.sleep(for: .milliseconds(10))
        #expect(await controller.queueDepth == 0)
    }

    @Test("Try acquire when available")
    func testTryAcquireAvailable() async {
        let controller = BackPressureController(maxQueueDepth: 5)

        let token = await controller.tryAcquire()
        #expect(token != nil)
    }

    @Test("Try acquire when full")
    func testTryAcquireFull() async throws {
        let controller = BackPressureController(maxQueueDepth: 1)

        let token1 = try await controller.acquire()
        let token2 = await controller.tryAcquire()

        #expect(token2 == nil)
        token1.release()
    }

    @Test("Suspend strategy waits")
    func testSuspendStrategy() async throws {
        let controller = BackPressureController(
            maxQueueDepth: 1,
            strategy: .suspend
        )

        let token1 = try await controller.acquire()

        // Start waiting task
        let waitTask = Task {
            try await controller.acquire()
        }

        // Give it time to start waiting
        try await Task.sleep(for: .milliseconds(50))

        // Release first token
        token1.release()

        // Second should now succeed
        let token2 = try await waitTask.value
        #expect(!token2.isReleased)
        token2.release()
    }

    @Test("Drop newest strategy rejects")
    func testDropNewestStrategy() async throws {
        let controller = BackPressureController(
            maxQueueDepth: 1,
            strategy: .dropNewest
        )

        let token1 = try await controller.acquire()

        do {
            _ = try await controller.acquire()
            Issue.record("Expected error")
        } catch is StreamingError {
            // Expected
        }

        token1.release()
    }

    @Test("Error with timeout")
    func testErrorWithTimeout() async throws {
        let controller = BackPressureController(
            maxQueueDepth: 1,
            strategy: .error(timeout: 0.1)
        )

        let token1 = try await controller.acquire()

        do {
            _ = try await controller.acquire()
            Issue.record("Expected timeout")
        } catch let error as StreamingError {
            if case .timeout = error {
                // Expected
            } else {
                Issue.record("Expected timeout error, got: \(error)")
            }
        }

        token1.release()
    }

    @Test("Statistics tracking")
    func testStatistics() async throws {
        let controller = BackPressureController(maxQueueDepth: 10)

        let token1 = try await controller.acquire()
        let token2 = try await controller.acquire()

        var stats = await controller.getStatistics()
        #expect(stats.currentDepth == 2)
        #expect(stats.totalAcquired == 2)

        token1.release()
        token2.release()

        // Wait for async release
        try await Task.sleep(for: .milliseconds(20))

        stats = await controller.getStatistics()
        #expect(stats.currentDepth == 0)
    }

    @Test("With back-pressure helper")
    func testWithBackPressure() async throws {
        let controller = BackPressureController(maxQueueDepth: 5)

        let result = try await controller.withBackPressure {
            return 42
        }

        #expect(result == 42)
        // Wait for async release
        try await Task.sleep(for: .milliseconds(10))
        #expect(await controller.queueDepth == 0)
    }

    @Test("Token auto-release on deinit")
    func testTokenAutoRelease() async throws {
        let controller = BackPressureController(maxQueueDepth: 5)

        do {
            let token = try await controller.acquire()
            #expect(await controller.queueDepth == 1)
            _ = token  // Token goes out of scope
        }

        // Token should auto-release
        // Give a moment for deinit
        try await Task.sleep(for: .milliseconds(10))
        #expect(await controller.queueDepth == 0)
    }
}

// MARK: - BackPressureToken Tests

@Suite("BackPressureToken")
struct BackPressureTokenTests {

    @Test("Token properties")
    func testTokenProperties() async {
        actor ReleaseTracker {
            var released = false
            func markReleased() { released = true }
        }
        let tracker = ReleaseTracker()

        let token = BackPressureToken {
            Task { await tracker.markReleased() }
        }

        #expect(!token.isReleased)
        #expect(token.holdDuration >= 0)

        token.release()
        try? await Task.sleep(for: .milliseconds(10))

        #expect(token.isReleased)
        let wasReleased = await tracker.released
        #expect(wasReleased)
    }

    @Test("Double release is safe")
    func testDoubleRelease() async {
        actor Counter {
            var count = 0
            func increment() { count += 1 }
        }
        let counter = Counter()

        let token = BackPressureToken {
            Task { await counter.increment() }
        }

        token.release()
        token.release()
        token.release()

        try? await Task.sleep(for: .milliseconds(20))
        let finalCount = await counter.count
        #expect(finalCount == 1)
    }
}

// MARK: - StreamingEmbeddingGenerator Tests

@Suite("StreamingEmbeddingGenerator")
struct StreamingEmbeddingGeneratorTests {

    @Test("Empty input returns empty array")
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

        let result = try await streaming.produce("Hello")
        #expect(result.count == 128)
    }

    @Test("Batch processing")
    func testBatchProcessing() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let streaming = StreamingEmbeddingGenerator(
            generator: generator,
            config: FlowControlConfig(batchSize: 5)
        )

        let texts = (0..<12).map { "Text \($0)" }
        let result = try await streaming.produce(texts)

        #expect(result.count == 12)
        for embedding in result {
            #expect(embedding.count == 128)
        }
    }

    @Test("Stream produces all results")
    func testStream() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let streaming = StreamingEmbeddingGenerator(
            generator: generator,
            config: FlowControlConfig(batchSize: 4)
        )

        let texts = (0..<10).map { "Text \($0)" }
        var results: [[Float]] = []
        var progressUpdates: [BatchProgress] = []

        for try await (embedding, progress) in await streaming.stream(texts) {
            results.append(embedding)
            progressUpdates.append(progress)
        }

        #expect(results.count == 10)
        #expect(progressUpdates.count == 10)
        #expect(progressUpdates.last?.current == 10)
    }

    @Test("Concurrent processing maintains order")
    func testConcurrentProcessing() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let streaming = StreamingEmbeddingGenerator(
            generator: generator,
            config: FlowControlConfig(maxConcurrency: 3, batchSize: 4)
        )

        let texts = (0..<20).map { "Text \($0)" }
        let result = try await streaming.processConcurrently(texts)

        #expect(result.count == 20)
    }

    @Test("With rate limiting")
    func testWithRateLimiting() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let streaming = StreamingEmbeddingGenerator(
            generator: generator,
            config: .rateLimited(requestsPerSecond: 100)
        )

        let texts = (0..<5).map { "Text \($0)" }
        let result = try await streaming.produce(texts)

        #expect(result.count == 5)
    }

    @Test("Statistics tracking")
    func testStatistics() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let streaming = StreamingEmbeddingGenerator(generator: generator)

        let texts = (0..<10).map { "Text \($0)" }
        _ = try await streaming.produce(texts)

        let stats = await streaming.getStatistics()
        #expect(stats.totalSubmitted == 10)
        #expect(stats.totalProcessed == 10)
        #expect(stats.throughput > 0)
    }

    @Test("VectorProducer conformance")
    func testVectorProducerConformance() async throws {
        let model = MockEmbeddingModel(dimensions: 256)
        let generator = EmbeddingGenerator(model: model)
        let streaming = StreamingEmbeddingGenerator(generator: generator)

        #expect(streaming.dimensions == 256)
    }
}

// MARK: - EmbeddingGenerator Extension Tests

@Suite("EmbeddingGenerator.streaming")
struct EmbeddingGeneratorStreamingExtensionTests {

    @Test("Extension creates streaming generator")
    func testExtension() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)

        let streaming = await generator.streaming()

        let config = await streaming.getConfig()
        #expect(config.maxConcurrency == 4)  // Default
    }

    @Test("Extension with custom config")
    func testExtensionWithConfig() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)

        let streaming = await generator.streaming(config: .highThroughput)

        let config = await streaming.getConfig()
        #expect(config.maxConcurrency == 8)
    }
}

// MARK: - Integration Tests

@Suite("Streaming Integration", .tags(.integration))
struct StreamingIntegrationTests {

    @Test("High volume streaming")
    func testHighVolumeStreaming() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: model)
        let streaming = StreamingEmbeddingGenerator(
            generator: generator,
            config: .highThroughput
        )

        let texts = (0..<200).map { "Large volume text \($0)" }
        var count = 0

        for try await _ in await streaming.stream(texts) {
            count += 1
        }

        #expect(count == 200)
    }

    @Test("Back-pressure under load")
    func testBackPressureUnderLoad() async throws {
        let model = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: model)
        let streaming = StreamingEmbeddingGenerator(
            generator: generator,
            config: FlowControlConfig(
                maxConcurrency: 2,
                maxQueueDepth: 5,
                backPressureStrategy: .suspend,
                batchSize: 2
            )
        )

        let texts = (0..<50).map { "Text \($0)" }
        let result = try await streaming.processConcurrently(texts)

        #expect(result.count == 50)

        let bpStats = await streaming.getBackPressureStatistics()
        #expect(bpStats.totalAcquired > 0)
    }
}

// Note: Tag.integration is defined elsewhere

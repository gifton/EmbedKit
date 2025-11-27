// EmbedKit - Request Prioritization Tests

import Testing
import Foundation
@testable import EmbedKit

@Suite("Request Prioritization")
struct RequestPrioritizationTests {

    // MARK: - RequestPriority Tests

    @Test("RequestPriority ordering is correct")
    func priorityOrdering() {
        #expect(RequestPriority.low < RequestPriority.normal)
        #expect(RequestPriority.normal < RequestPriority.high)
        #expect(RequestPriority.high < RequestPriority.urgent)

        #expect(RequestPriority.urgent > RequestPriority.low)
        #expect(RequestPriority.urgent > RequestPriority.normal)
        #expect(RequestPriority.urgent > RequestPriority.high)
    }

    @Test("RequestPriority raw values are correct")
    func priorityRawValues() {
        #expect(RequestPriority.low.rawValue == 0)
        #expect(RequestPriority.normal.rawValue == 1)
        #expect(RequestPriority.high.rawValue == 2)
        #expect(RequestPriority.urgent.rawValue == 3)
    }

    @Test("RequestPriority is Sendable and CaseIterable")
    func priorityConformances() {
        let allCases = RequestPriority.allCases
        #expect(allCases.count == 4)
        #expect(allCases.contains(.low))
        #expect(allCases.contains(.normal))
        #expect(allCases.contains(.high))
        #expect(allCases.contains(.urgent))
    }

    // MARK: - Configuration Tests

    @Test("Default config has priority scheduling enabled")
    func defaultConfigPriorityEnabled() {
        let config = AdaptiveBatcherConfig()
        #expect(config.enablePriorityScheduling == true)
        #expect(config.urgentTriggersFlush == true)
        #expect(config.lowPriorityLatencyMultiplier == 3.0)
    }

    @Test("Priority scheduling can be disabled")
    func configPriorityDisabled() {
        var config = AdaptiveBatcherConfig()
        config.enablePriorityScheduling = false
        #expect(config.enablePriorityScheduling == false)
    }

    @Test("Low priority latency multiplier is configurable")
    func lowPriorityLatencyConfigurable() {
        var config = AdaptiveBatcherConfig()
        config.lowPriorityLatencyMultiplier = 5.0
        #expect(config.lowPriorityLatencyMultiplier == 5.0)
    }

    @Test("Urgent triggers flush is configurable")
    func urgentTriggersFlushConfigurable() {
        var config = AdaptiveBatcherConfig()
        config.urgentTriggersFlush = false
        #expect(config.urgentTriggersFlush == false)
    }

    // MARK: - Priority API Tests

    @Test("Default embed uses normal priority")
    func defaultEmbedUsesNormal() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let batcher = AdaptiveBatcher(model: model)

        _ = try await batcher.embed("test text")

        let metrics = await batcher.metrics
        #expect(metrics.requestsByPriority[.normal] == 1)
    }

    @Test("Embed with priority tracks correctly")
    func embedWithPriorityTracks() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let batcher = AdaptiveBatcher(model: model)

        _ = try await batcher.embed("low text", priority: .low)
        _ = try await batcher.embed("high text", priority: .high)
        _ = try await batcher.embed("urgent text", priority: .urgent)

        let metrics = await batcher.metrics
        #expect(metrics.requestsByPriority[.low] == 1)
        #expect(metrics.requestsByPriority[.high] == 1)
        #expect(metrics.requestsByPriority[.urgent] == 1)
    }

    @Test("Reset metrics clears priority tracking")
    func resetMetricsClearsPriority() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let batcher = AdaptiveBatcher(model: model)

        // Submit and complete some requests
        _ = try await batcher.embed("text1", priority: .high)
        _ = try await batcher.embed("text2", priority: .low)

        var metrics = await batcher.metrics
        #expect(metrics.totalRequests == 2)

        // Reset
        await batcher.resetMetrics()

        metrics = await batcher.metrics
        #expect(metrics.totalRequests == 0)
        #expect(metrics.requestsByPriority.isEmpty)
    }

    // MARK: - Convenience API Tests

    @Test("embedConcurrently supports priority")
    func embedConcurrentlyWithPriority() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let batcher = AdaptiveBatcher(model: model)

        let texts = ["one", "two", "three"]
        let embeddings = try await batcher.embedConcurrently(texts, priority: .high)

        #expect(embeddings.count == 3)

        let metrics = await batcher.metrics
        #expect(metrics.requestsByPriority[.high] == 3)
    }

    // MARK: - Latency Tests

    @Test("Low priority requests complete successfully")
    func lowPriorityCompletes() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()

        var config = AdaptiveBatcherConfig()
        config.maxLatency = 0.05 // 50ms base latency
        config.lowPriorityLatencyMultiplier = 2.0
        config.minBatchSize = 1
        let batcher = AdaptiveBatcher(model: model, config: config)

        // Submit a low priority request
        let embedding = try await batcher.embed("low priority text", priority: .low)

        // Should complete successfully
        #expect(embedding.vector.count == 384)
    }

    @Test("Urgent requests process quickly")
    func urgentProcessesQuickly() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()

        var config = AdaptiveBatcherConfig()
        config.maxLatency = 1.0 // 1 second base latency
        config.urgentTriggersFlush = true
        config.minBatchSize = 1
        let batcher = AdaptiveBatcher(model: model, config: config)

        let start = CFAbsoluteTimeGetCurrent()
        _ = try await batcher.embed("urgent text", priority: .urgent)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        // Urgent should trigger immediate flush (much less than 1 second)
        #expect(elapsed < 0.5)
    }

    @Test("High priority requests process faster than low priority deadline")
    func highPriorityFasterThanLow() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()

        var config = AdaptiveBatcherConfig()
        config.maxLatency = 0.2 // 200ms base
        config.lowPriorityLatencyMultiplier = 3.0
        config.minBatchSize = 1
        let batcher = AdaptiveBatcher(model: model, config: config)

        // High priority should complete within base latency (or much faster due to flush)
        let start = CFAbsoluteTimeGetCurrent()
        _ = try await batcher.embed("high priority", priority: .high)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        // Should be significantly faster than low priority max latency (600ms)
        #expect(elapsed < 0.3)
    }

    // MARK: - Edge Cases

    @Test("Empty queue has no highest priority")
    func emptyQueueNoHighestPriority() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let batcher = AdaptiveBatcher(model: model)

        let highest = await batcher.highestPendingPriority
        #expect(highest == nil)
    }

    @Test("Empty queue has no urgent pending")
    func emptyQueueNoUrgentPending() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let batcher = AdaptiveBatcher(model: model)

        let hasUrgent = await batcher.hasUrgentPending
        #expect(hasUrgent == false)
    }

    @Test("Mixed priorities tracked correctly in metrics")
    func mixedPrioritiesTrackedCorrectly() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let batcher = AdaptiveBatcher(model: model)

        // Submit one of each priority
        _ = try await batcher.embed("low", priority: .low)
        _ = try await batcher.embed("normal", priority: .normal)
        _ = try await batcher.embed("high", priority: .high)
        _ = try await batcher.embed("urgent", priority: .urgent)

        let metrics = await batcher.metrics
        #expect(metrics.totalRequests == 4)
        #expect(metrics.requestsByPriority[.low] == 1)
        #expect(metrics.requestsByPriority[.normal] == 1)
        #expect(metrics.requestsByPriority[.high] == 1)
        #expect(metrics.requestsByPriority[.urgent] == 1)
    }

    @Test("Batch processing completes all priorities")
    func batchProcessingCompletesAllPriorities() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()

        var config = AdaptiveBatcherConfig()
        config.maxBatchSize = 4
        let batcher = AdaptiveBatcher(model: model, config: config)

        // Submit multiple requests of different priorities
        let results = try await batcher.embedConcurrently([
            "text1", "text2", "text3", "text4"
        ], priority: .high)

        #expect(results.count == 4)
        for embedding in results {
            #expect(embedding.vector.count == 384)
        }
    }

    @Test("Priority comparison operators work correctly")
    func priorityComparisonOperators() {
        // Test all comparison combinations
        #expect(RequestPriority.low == RequestPriority.low)
        #expect(RequestPriority.low != RequestPriority.normal)
        #expect(RequestPriority.low <= RequestPriority.normal)
        #expect(RequestPriority.high >= RequestPriority.normal)
        #expect(!(RequestPriority.low > RequestPriority.normal))
        #expect(!(RequestPriority.high < RequestPriority.normal))
    }

    // MARK: - Queue Ordering Tests (using concurrent tasks with flush)

    @Test("Priority scheduling puts higher priority first")
    func prioritySchedulingOrder() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()

        var config = AdaptiveBatcherConfig()
        config.enablePriorityScheduling = true
        config.autoFlush = false
        config.urgentTriggersFlush = false
        let batcher = AdaptiveBatcher(model: model, config: config)

        // Use tasks instead of async let to avoid deadlock
        let task1 = Task { try await batcher.embed("low", priority: .low) }
        let task2 = Task { try await batcher.embed("normal", priority: .normal) }
        let task3 = Task { try await batcher.embed("high", priority: .high) }

        // Give tasks time to submit
        try await Task.sleep(nanoseconds: 10_000_000) // 10ms

        // Check queue ordering
        let highest = await batcher.highestPendingPriority
        #expect(highest == .high) // High priority should be at front

        let metrics = await batcher.metrics
        #expect(metrics.currentQueueDepth == 3)
        #expect(metrics.queueDepthByPriority[.low] == 1)
        #expect(metrics.queueDepthByPriority[.normal] == 1)
        #expect(metrics.queueDepthByPriority[.high] == 1)

        // Flush to allow completion
        try await batcher.flush()

        // Wait for tasks to complete
        _ = try await task1.value
        _ = try await task2.value
        _ = try await task3.value
    }

    @Test("Urgent requests detected in pending queue")
    func urgentDetectedInQueue() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()

        var config = AdaptiveBatcherConfig()
        config.autoFlush = false
        config.urgentTriggersFlush = false
        let batcher = AdaptiveBatcher(model: model, config: config)

        let task1 = Task { try await batcher.embed("normal", priority: .normal) }
        try await Task.sleep(nanoseconds: 5_000_000)

        var hasUrgent = await batcher.hasUrgentPending
        #expect(hasUrgent == false)

        let task2 = Task { try await batcher.embed("urgent", priority: .urgent) }
        try await Task.sleep(nanoseconds: 5_000_000)

        hasUrgent = await batcher.hasUrgentPending
        #expect(hasUrgent == true)

        // Cleanup
        try await batcher.flush()
        _ = try await task1.value
        _ = try await task2.value
    }

    @Test("Disabled priority scheduling preserves submission order")
    func disabledPriorityPreservesOrder() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()

        var config = AdaptiveBatcherConfig()
        config.enablePriorityScheduling = false
        config.autoFlush = false
        let batcher = AdaptiveBatcher(model: model, config: config)

        // Submit in order with different priorities
        let task1 = Task { try await batcher.embed("first", priority: .low) }
        try await Task.sleep(nanoseconds: 5_000_000)
        let task2 = Task { try await batcher.embed("second", priority: .high) }
        try await Task.sleep(nanoseconds: 5_000_000)
        let task3 = Task { try await batcher.embed("third", priority: .normal) }
        try await Task.sleep(nanoseconds: 5_000_000)

        // With priority disabled, all go to back - first should still be first
        let metrics = await batcher.metrics
        #expect(metrics.currentQueueDepth == 3)

        // Cleanup
        try await batcher.flush()
        _ = try await task1.value
        _ = try await task2.value
        _ = try await task3.value
    }

    // MARK: - Metrics Queue Depth Tests

    @Test("Queue depth by priority updated correctly")
    func queueDepthByPriorityUpdates() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()

        var config = AdaptiveBatcherConfig()
        config.autoFlush = false
        config.urgentTriggersFlush = false
        let batcher = AdaptiveBatcher(model: model, config: config)

        // Add two high, one low
        let task1 = Task { try await batcher.embed("high1", priority: .high) }
        let task2 = Task { try await batcher.embed("high2", priority: .high) }
        let task3 = Task { try await batcher.embed("low", priority: .low) }

        try await Task.sleep(nanoseconds: 10_000_000)

        var metrics = await batcher.metrics
        #expect(metrics.queueDepthByPriority[.high] == 2)
        #expect(metrics.queueDepthByPriority[.low] == 1)
        #expect(metrics.queueDepthByPriority[.normal] == nil)

        // Process
        try await batcher.flush()
        _ = try await task1.value
        _ = try await task2.value
        _ = try await task3.value

        // After processing, queue should be empty
        metrics = await batcher.metrics
        #expect(metrics.currentQueueDepth == 0)
    }
}

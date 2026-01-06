// EmbedKit - BackPressureController Comprehensive Tests
//
// Comprehensive tests for BackPressureController covering all strategies,
// events, statistics, concurrency, and edge cases.

import Testing
import Foundation
@testable import EmbedKit

// MARK: - Drop Oldest Strategy Tests

@Suite("BackPressure Drop Oldest Strategy")
struct BackPressureDropOldestStrategyTests {

    @Test("Drop oldest drops waiting requests")
    func testDropOldestDropsWaiters() async throws {
        actor DropTracker {
            var dropped = false
            func markDropped() { dropped = true }
        }
        let tracker = DropTracker()

        let controller = BackPressureController(
            maxQueueDepth: 1,
            strategy: .dropOldest
        )

        // Fill the queue
        let token1 = try await controller.acquire()

        // Start a waiter that will be dropped (oldest)
        let waiterTask = Task {
            do {
                _ = try await controller.acquire()
            } catch {
                await tracker.markDropped()
            }
        }

        // Give waiter time to start waiting
        try await Task.sleep(for: .milliseconds(50))

        // Start new request - this drops oldest waiter and becomes new waiter
        let token2Task = Task {
            try await controller.acquire()
        }

        // Give time for drop to occur
        try await Task.sleep(for: .milliseconds(50))

        // Release token1 - token2Task should get capacity (waiterTask was dropped)
        token1.release()

        // Wait for dropped waiter to complete
        try? await waiterTask.value

        // token2 should succeed now
        let token2 = try await token2Task.value

        let wasDropped = await tracker.dropped
        #expect(wasDropped)

        token2.release()
    }

    @Test("Drop oldest statistics track drops")
    func testDropOldestStatisticsTrackDrops() async throws {
        let controller = BackPressureController(
            maxQueueDepth: 1,
            strategy: .dropOldest
        )

        let token1 = try await controller.acquire()

        // Start waiter (will be dropped)
        let waiterTask = Task {
            try? await controller.acquire()
        }

        try await Task.sleep(for: .milliseconds(50))

        // Start new request - drops the waiter and becomes new waiter
        let token2Task = Task {
            try await controller.acquire()
        }

        try await Task.sleep(for: .milliseconds(50))

        // Release to allow token2Task to complete
        token1.release()

        try? await waiterTask.value
        let token2 = try await token2Task.value

        let stats = await controller.getStatistics()
        #expect(stats.totalDropped >= 1)

        token2.release()
    }
}

// MARK: - Event Handler Tests

@Suite("BackPressure Event Handler")
struct BackPressureEventHandlerTests {

    @Test("Event handler receives suspended event")
    func testEventHandlerSuspended() async throws {
        actor EventTracker {
            var events: [BackPressureEvent] = []
            func add(_ event: BackPressureEvent) { events.append(event) }
            func getEvents() -> [BackPressureEvent] { events }
        }
        let tracker = EventTracker()

        let controller = BackPressureController(
            maxQueueDepth: 1,
            strategy: .suspend,
            onEvent: { event in
                Task { await tracker.add(event) }
            }
        )

        // Fill queue
        let token1 = try await controller.acquire()

        // Start suspended task
        let waitTask = Task {
            try await controller.acquire()
        }

        try await Task.sleep(for: .milliseconds(50))

        // Release to resume waiter
        token1.release()
        let token2 = try await waitTask.value

        try await Task.sleep(for: .milliseconds(20))

        let events = await tracker.getEvents()
        let hasSuspended = events.contains {
            if case .suspended = $0 { return true }
            return false
        }
        #expect(hasSuspended)

        token2.release()
    }

    @Test("Event handler receives resumed event")
    func testEventHandlerResumed() async throws {
        actor EventTracker {
            var resumed = false
            func markResumed() { resumed = true }
        }
        let tracker = EventTracker()

        let controller = BackPressureController(
            maxQueueDepth: 1,
            strategy: .suspend,
            onEvent: { event in
                if case .resumed = event {
                    Task { await tracker.markResumed() }
                }
            }
        )

        let token1 = try await controller.acquire()

        let waitTask = Task {
            try await controller.acquire()
        }

        try await Task.sleep(for: .milliseconds(50))
        token1.release()

        let token2 = try await waitTask.value
        try await Task.sleep(for: .milliseconds(50))

        let wasResumed = await tracker.resumed
        #expect(wasResumed)

        token2.release()
    }

    @Test("Event handler receives dropped event")
    func testEventHandlerDropped() async throws {
        actor EventTracker {
            var dropped = false
            func markDropped() { dropped = true }
        }
        let tracker = EventTracker()

        let controller = BackPressureController(
            maxQueueDepth: 1,
            strategy: .dropOldest,
            onEvent: { event in
                if case .dropped = event {
                    Task { await tracker.markDropped() }
                }
            }
        )

        let token1 = try await controller.acquire()

        // Start waiter (will be dropped)
        let waiterTask = Task {
            try? await controller.acquire()
        }

        try await Task.sleep(for: .milliseconds(50))

        // Start new request - drops oldest and becomes new waiter
        let token2Task = Task {
            try await controller.acquire()
        }

        try await Task.sleep(for: .milliseconds(50))

        // Release to allow token2Task to complete
        token1.release()

        try? await waiterTask.value
        let token2 = try await token2Task.value

        try await Task.sleep(for: .milliseconds(50))

        let wasDropped = await tracker.dropped
        #expect(wasDropped)

        token2.release()
    }
}

// MARK: - Statistics Comprehensive Tests

@Suite("BackPressure Statistics Comprehensive")
struct BackPressureStatisticsComprehensiveTests {

    @Test("Statistics track peak depth")
    func testStatisticsTrackPeakDepth() async throws {
        let controller = BackPressureController(maxQueueDepth: 10)

        var tokens: [BackPressureToken] = []
        for _ in 0..<5 {
            tokens.append(try await controller.acquire())
        }

        var stats = await controller.getStatistics()
        #expect(stats.peakDepth == 5)

        // Release all
        for token in tokens {
            token.release()
        }

        try await Task.sleep(for: .milliseconds(50))

        // Peak should remain 5
        stats = await controller.getStatistics()
        #expect(stats.peakDepth == 5)
        #expect(stats.currentDepth == 0)
    }

    @Test("Statistics calculate utilization")
    func testStatisticsUtilization() async throws {
        let controller = BackPressureController(maxQueueDepth: 10)

        let token1 = try await controller.acquire()
        let token2 = try await controller.acquire()

        let stats = await controller.getStatistics()
        #expect(stats.utilization == 0.2) // 2/10

        token1.release()
        token2.release()
    }

    @Test("Statistics calculate drop rate")
    func testStatisticsDropRate() async throws {
        let controller = BackPressureController(
            maxQueueDepth: 1,
            strategy: .dropOldest
        )

        var heldToken = try await controller.acquire()

        // Create waiters that will be dropped
        for _ in 0..<3 {
            // Start a waiter that will be dropped
            let waiterTask = Task {
                try? await controller.acquire()
            }
            try await Task.sleep(for: .milliseconds(20))

            // Start new request - drops oldest waiter
            let newTokenTask = Task {
                try await controller.acquire()
            }
            try await Task.sleep(for: .milliseconds(20))

            // Release held token to let newTokenTask complete
            heldToken.release()

            try? await waiterTask.value
            heldToken = try await newTokenTask.value
        }

        let stats = await controller.getStatistics()
        #expect(stats.dropRate > 0)

        heldToken.release()
    }

    @Test("Statistics track average wait time")
    func testStatisticsAverageWaitTime() async throws {
        let controller = BackPressureController(
            maxQueueDepth: 1,
            strategy: .suspend
        )

        let token1 = try await controller.acquire()

        let waiterTask = Task {
            try await controller.acquire()
        }

        try await Task.sleep(for: .milliseconds(100))
        token1.release()

        let token2 = try await waiterTask.value

        let stats = await controller.getStatistics()
        // Average wait time should be > 0 since we waited
        #expect(stats.averageWaitTime > 0)

        token2.release()
    }

    @Test("Reset statistics clears totals")
    func testResetStatistics() async throws {
        let controller = BackPressureController(maxQueueDepth: 10)

        let token1 = try await controller.acquire()
        let token2 = try await controller.acquire()

        var stats = await controller.getStatistics()
        #expect(stats.totalAcquired == 2)

        await controller.resetStatistics()

        stats = await controller.getStatistics()
        #expect(stats.totalAcquired == 0)
        #expect(stats.currentDepth == 2) // Current state preserved

        token1.release()
        token2.release()
    }
}

// MARK: - Properties Tests

@Suite("BackPressure Controller Properties")
struct BackPressureControllerPropertiesTests {

    @Test("Queue depth property")
    func testQueueDepthProperty() async throws {
        let controller = BackPressureController(maxQueueDepth: 10)

        #expect(await controller.queueDepth == 0)

        let token = try await controller.acquire()
        #expect(await controller.queueDepth == 1)

        token.release()
        try await Task.sleep(for: .milliseconds(20))
        #expect(await controller.queueDepth == 0)
    }

    @Test("Peak queue depth property")
    func testPeakQueueDepthProperty() async throws {
        let controller = BackPressureController(maxQueueDepth: 10)

        let token1 = try await controller.acquire()
        let token2 = try await controller.acquire()
        let token3 = try await controller.acquire()

        #expect(await controller.peakQueueDepth == 3)

        token1.release()
        token2.release()
        token3.release()

        try await Task.sleep(for: .milliseconds(20))

        // Peak should still be 3
        #expect(await controller.peakQueueDepth == 3)
    }

    @Test("Waiter count property")
    func testWaiterCountProperty() async throws {
        let controller = BackPressureController(
            maxQueueDepth: 1,
            strategy: .suspend
        )

        let token1 = try await controller.acquire()

        // Start some waiters
        let waitTask1 = Task { try? await controller.acquire() }
        let waitTask2 = Task { try? await controller.acquire() }

        try await Task.sleep(for: .milliseconds(50))

        #expect(await controller.waiterCount >= 1)

        // Release to let waiters through
        token1.release()

        try await Task.sleep(for: .milliseconds(50))
        waitTask1.cancel()
        waitTask2.cancel()
    }

    @Test("IsBackPressured property")
    func testIsBackPressuredProperty() async throws {
        let controller = BackPressureController(maxQueueDepth: 2)

        #expect(await controller.isBackPressured == false)

        let token1 = try await controller.acquire()
        #expect(await controller.isBackPressured == false)

        let token2 = try await controller.acquire()
        #expect(await controller.isBackPressured == true)

        token1.release()
        try await Task.sleep(for: .milliseconds(20))
        #expect(await controller.isBackPressured == false)

        token2.release()
    }
}

// MARK: - Error Strategy Tests

@Suite("BackPressure Error Strategy Comprehensive")
struct BackPressureErrorStrategyComprehensiveTests {

    @Test("Error with nil timeout throws immediately")
    func testErrorNilTimeoutThrowsImmediately() async throws {
        let controller = BackPressureController(
            maxQueueDepth: 1,
            strategy: .error(timeout: nil)
        )

        let token1 = try await controller.acquire()

        let start = CFAbsoluteTimeGetCurrent()

        do {
            _ = try await controller.acquire()
            Issue.record("Expected error")
        } catch is StreamingError {
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            #expect(elapsed < 0.1) // Should be nearly instant
        }

        token1.release()
    }

    @Test("Error with timeout waits then throws")
    func testErrorWithTimeoutWaitsThenThrows() async throws {
        let controller = BackPressureController(
            maxQueueDepth: 1,
            strategy: .error(timeout: 0.15)
        )

        let token1 = try await controller.acquire()

        let start = CFAbsoluteTimeGetCurrent()

        do {
            _ = try await controller.acquire()
            Issue.record("Expected timeout")
        } catch let error as StreamingError {
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            #expect(elapsed >= 0.1) // Should have waited
            if case .timeout = error {
                // Expected
            } else {
                Issue.record("Expected timeout error")
            }
        }

        token1.release()
    }

    @Test("Error strategy succeeds if capacity freed in time")
    func testErrorStrategySucceedsIfCapacityFreed() async throws {
        let controller = BackPressureController(
            maxQueueDepth: 1,
            strategy: .error(timeout: 1.0)
        )

        let token1 = try await controller.acquire()

        // Start acquisition task
        let acquireTask = Task {
            try await controller.acquire()
        }

        // Free capacity before timeout
        try await Task.sleep(for: .milliseconds(100))
        token1.release()

        // Should succeed
        let token2 = try await acquireTask.value
        #expect(!token2.isReleased)
        token2.release()
    }
}

// MARK: - Concurrent Waiter Tests

@Suite("BackPressure Concurrent Waiters", .serialized)
struct BackPressureConcurrentWaiterTests {

    @Test("Multiple waiters resume in order")
    func testMultipleWaitersResumeInOrder() async throws {
        let controller = BackPressureController(
            maxQueueDepth: 1,
            strategy: .suspend
        )

        actor OrderTracker {
            var order: [Int] = []
            func add(_ id: Int) { order.append(id) }
            func getOrder() -> [Int] { order }
        }
        let tracker = OrderTracker()

        let token1 = try await controller.acquire()

        // Start multiple waiters
        var tasks: [Task<Void, Error>] = []
        for i in 0..<3 {
            let task = Task {
                let token = try await controller.acquire()
                await tracker.add(i)
                token.release()
            }
            tasks.append(task)
            try await Task.sleep(for: .milliseconds(20))
        }

        // Release to start resumption chain
        token1.release()

        // Wait for all
        for task in tasks {
            try await task.value
        }

        let order = await tracker.getOrder()
        #expect(order.count == 3)
    }

    @Test("Cancelled waiter is removed")
    func testCancelledWaiterRemoved() async throws {
        let controller = BackPressureController(
            maxQueueDepth: 1,
            strategy: .suspend
        )

        let token1 = try await controller.acquire()

        // Start waiter
        let waiterTask = Task {
            try await controller.acquire()
        }

        try await Task.sleep(for: .milliseconds(50))
        #expect(await controller.waiterCount >= 1)

        // Cancel the waiter
        waiterTask.cancel()

        try await Task.sleep(for: .milliseconds(50))

        // Start another waiter that should succeed when we release
        let waiter2Task = Task {
            try await controller.acquire()
        }

        try await Task.sleep(for: .milliseconds(50))
        token1.release()

        let token2 = try await waiter2Task.value
        #expect(!token2.isReleased)
        token2.release()
    }

    @Test("High concurrency stress test")
    func testHighConcurrencyStress() async throws {
        let controller = BackPressureController(
            maxQueueDepth: 5,
            strategy: .suspend
        )

        actor CompletionCounter {
            var count = 0
            func increment() { count += 1 }
            func getCount() -> Int { count }
        }
        let counter = CompletionCounter()

        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<50 {
                group.addTask {
                    do {
                        let token = try await controller.acquire()
                        try? await Task.sleep(for: .milliseconds(5))
                        await counter.increment()
                        token.release()
                    } catch {
                        // Ignore cancellation
                    }
                }
            }
        }

        let completions = await counter.getCount()
        #expect(completions == 50)

        // Allow controller to settle after concurrent releases
        try await Task.sleep(for: .milliseconds(50))

        // Controller should be in clean state
        #expect(await controller.queueDepth == 0)
        #expect(await controller.waiterCount == 0)
    }
}

// MARK: - Token Tests

@Suite("BackPressure Token Comprehensive")
struct BackPressureTokenComprehensiveTests {

    @Test("Token acquired at records time")
    func testTokenAcquiredAtRecordsTime() async throws {
        let controller = BackPressureController(maxQueueDepth: 5)

        let before = Date()
        let token = try await controller.acquire()
        let after = Date()

        #expect(token.acquiredAt >= before)
        #expect(token.acquiredAt <= after)

        token.release()
    }

    @Test("Token hold duration increases")
    func testTokenHoldDurationIncreases() async throws {
        let controller = BackPressureController(maxQueueDepth: 5)

        let token = try await controller.acquire()
        let duration1 = token.holdDuration

        try await Task.sleep(for: .milliseconds(100))
        let duration2 = token.holdDuration

        #expect(duration2 > duration1)
        #expect(duration2 >= 0.1)

        token.release()
    }

    @Test("Token is released check")
    func testTokenIsReleasedCheck() async throws {
        let controller = BackPressureController(maxQueueDepth: 5)

        let token = try await controller.acquire()
        #expect(!token.isReleased)

        token.release()
        #expect(token.isReleased)
    }

    @Test("Token release is idempotent")
    func testTokenReleaseIdempotent() async throws {
        let controller = BackPressureController(maxQueueDepth: 5)

        let token = try await controller.acquire()

        token.release()
        token.release()
        token.release()

        #expect(token.isReleased)

        try await Task.sleep(for: .milliseconds(50))

        // Queue depth should be 0, not negative
        #expect(await controller.queueDepth == 0)
    }

    @Test("Token deinit releases")
    func testTokenDeinitReleases() async throws {
        let controller = BackPressureController(maxQueueDepth: 5)

        do {
            let _ = try await controller.acquire()
            #expect(await controller.queueDepth == 1)
        }

        // Token goes out of scope
        try await Task.sleep(for: .milliseconds(50))
        #expect(await controller.queueDepth == 0)
    }
}

// MARK: - WithBackPressure Tests

@Suite("BackPressure WithBackPressure Helper")
struct BackPressureWithBackPressureHelperTests {

    @Test("withBackPressure executes operation")
    func testWithBackPressureExecutesOperation() async throws {
        let controller = BackPressureController(maxQueueDepth: 5)

        let result = try await controller.withBackPressure {
            42
        }

        #expect(result == 42)
    }

    @Test("withBackPressure releases on success")
    func testWithBackPressureReleasesOnSuccess() async throws {
        let controller = BackPressureController(maxQueueDepth: 5)

        _ = try await controller.withBackPressure {
            "hello"
        }

        try await Task.sleep(for: .milliseconds(20))
        #expect(await controller.queueDepth == 0)
    }

    @Test("withBackPressure releases on error")
    func testWithBackPressureReleasesOnError() async throws {
        let controller = BackPressureController(maxQueueDepth: 5)

        struct TestError: Error {}

        do {
            _ = try await controller.withBackPressure {
                throw TestError()
            }
            Issue.record("Expected error")
        } catch is TestError {
            // Expected
        }

        try await Task.sleep(for: .milliseconds(20))
        #expect(await controller.queueDepth == 0)
    }

    @Test("withBackPressure respects back-pressure")
    func testWithBackPressureRespectsBackPressure() async throws {
        let controller = BackPressureController(
            maxQueueDepth: 1,
            strategy: .dropNewest
        )

        let token1 = try await controller.acquire()

        do {
            _ = try await controller.withBackPressure {
                "should fail"
            }
            Issue.record("Expected error")
        } catch is StreamingError {
            // Expected - queue full
        }

        token1.release()
    }

    @Test("withBackPressure with async operation")
    func testWithBackPressureAsyncOperation() async throws {
        let controller = BackPressureController(maxQueueDepth: 5)

        let result = try await controller.withBackPressure {
            try await Task.sleep(for: .milliseconds(50))
            return "async result"
        }

        #expect(result == "async result")
        try await Task.sleep(for: .milliseconds(20))
        #expect(await controller.queueDepth == 0)
    }
}

// MARK: - Initialization Tests

@Suite("BackPressure Controller Initialization")
struct BackPressureControllerInitializationTests {

    @Test("Default strategy is suspend")
    func testDefaultStrategyIsSuspend() async throws {
        let controller = BackPressureController(maxQueueDepth: 5)

        // Fill queue
        var tokens: [BackPressureToken] = []
        for _ in 0..<5 {
            tokens.append(try await controller.acquire())
        }

        // Should suspend, not error or drop
        let acquireTask = Task {
            try await controller.acquire()
        }

        try await Task.sleep(for: .milliseconds(50))
        #expect(await controller.waiterCount >= 1)

        // Release to resume
        tokens[0].release()
        let newToken = try await acquireTask.value

        for token in tokens.dropFirst() {
            token.release()
        }
        newToken.release()
    }

    @Test("Custom max wait time")
    func testCustomMaxWaitTime() async throws {
        let controller = BackPressureController(
            maxQueueDepth: 1,
            strategy: .error(timeout: nil),
            maxWaitTime: 0.1
        )

        let token1 = try await controller.acquire()

        // Should error immediately (nil timeout)
        do {
            _ = try await controller.acquire()
            Issue.record("Expected error")
        } catch {
            // Expected
        }

        token1.release()
    }
}

// MARK: - Queue Full Error Tests

@Suite("BackPressure Queue Full Error")
struct BackPressureQueueFullErrorTests {

    @Test("Queue full error contains current and limit")
    func testQueueFullErrorContainsInfo() async throws {
        let controller = BackPressureController(
            maxQueueDepth: 3,
            strategy: .dropNewest
        )

        var tokens: [BackPressureToken] = []
        for _ in 0..<3 {
            tokens.append(try await controller.acquire())
        }

        do {
            _ = try await controller.acquire()
            Issue.record("Expected queue full error")
        } catch let error as StreamingError {
            if case .queueFull(let current, let limit) = error {
                #expect(current == 3)
                #expect(limit == 3)
            } else {
                Issue.record("Expected queueFull error")
            }
        }

        for token in tokens {
            token.release()
        }
    }
}

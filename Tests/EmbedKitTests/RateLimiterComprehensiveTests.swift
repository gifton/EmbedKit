// EmbedKit - RateLimiter Comprehensive Tests
//
// Comprehensive tests for EmbeddingRateLimiter covering all strategies,
// edge cases, concurrency, and internal implementations.

import Testing
import Foundation
@testable import EmbedKit

// MARK: - Load Factor Actor for Testing

actor LoadFactorTracker {
    var value: Double = 0.5
    func getValue() -> Double { value }
    func setValue(_ newValue: Double) { value = newValue }
}

// MARK: - Adaptive Strategy Tests

@Suite("Adaptive Rate Limiting")
struct AdaptiveRateLimitingTests {

    @Test("Adaptive strategy uses load factor")
    func testAdaptiveWithLoadFactor() async {
        let loadTracker = LoadFactorTracker()

        let limiter = EmbeddingRateLimiter(
            strategy: .adaptive(baseRate: 10, loadFactor: { await loadTracker.getValue() })
        )

        // At 50% load, effective capacity = baseRate * (2.0 - 0.5) = 15
        let status = await limiter.getStatus()
        #expect(status.limit == 10) // Reports base rate

        // Should allow requests based on adjusted capacity
        var allowedCount = 0
        for _ in 0..<20 {
            if await limiter.allowRequest() {
                allowedCount += 1
            }
        }
        #expect(allowedCount > 0)
    }

    @Test("Adaptive strategy responds to load changes")
    func testAdaptiveRespondsToLoadChanges() async {
        let loadTracker = LoadFactorTracker()
        await loadTracker.setValue(0.0)

        let limiter = EmbeddingRateLimiter(
            strategy: .adaptive(baseRate: 5, loadFactor: { await loadTracker.getValue() })
        )

        // At 0% load, capacity = 5 * 2.0 = 10
        var count1 = 0
        for _ in 0..<15 {
            if await limiter.allowRequest() {
                count1 += 1
            }
        }

        // Reset and change load
        await limiter.reset()
        await loadTracker.setValue(0.9) // At 90% load, capacity = 5 * 1.1 = 5.5

        var count2 = 0
        for _ in 0..<15 {
            if await limiter.allowRequest() {
                count2 += 1
            }
        }

        // Higher load should result in fewer allowed requests
        #expect(count1 > count2)
    }

    @Test("Adaptive strategy with full load")
    func testAdaptiveFullLoad() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .adaptive(baseRate: 10, loadFactor: { 1.0 })
        )

        // At 100% load, capacity = 10 * (2.0 - 1.0) = 10
        var count = 0
        for _ in 0..<15 {
            if await limiter.allowRequest() {
                count += 1
            }
        }
        #expect(count <= 10)
    }

    @Test("Adaptive strategy with zero load")
    func testAdaptiveZeroLoad() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .adaptive(baseRate: 5, loadFactor: { 0.0 })
        )

        // At 0% load, capacity = 5 * (2.0 - 0.0) = 10
        var count = 0
        for _ in 0..<12 {
            if await limiter.allowRequest() {
                count += 1
            }
        }
        #expect(count <= 10)
    }
}

// MARK: - Multiple Identifier Tests

@Suite("Rate Limiter Multiple Identifiers")
struct RateLimiterMultipleIdentifiersTests {

    @Test("Separate buckets per identifier")
    func testSeparateBucketsPerIdentifier() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 3, refillRate: 0)
        )

        // Exhaust bucket for user1
        #expect(await limiter.allowRequest(identifier: "user1"))
        #expect(await limiter.allowRequest(identifier: "user1"))
        #expect(await limiter.allowRequest(identifier: "user1"))
        #expect(!(await limiter.allowRequest(identifier: "user1")))

        // user2 should still have full capacity
        #expect(await limiter.allowRequest(identifier: "user2"))
        #expect(await limiter.allowRequest(identifier: "user2"))
        #expect(await limiter.allowRequest(identifier: "user2"))
        #expect(!(await limiter.allowRequest(identifier: "user2")))
    }

    @Test("Reset specific identifier")
    func testResetSpecificIdentifier() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 2, refillRate: 0)
        )

        // Exhaust both
        _ = await limiter.allowRequest(identifier: "a")
        _ = await limiter.allowRequest(identifier: "a")
        _ = await limiter.allowRequest(identifier: "b")
        _ = await limiter.allowRequest(identifier: "b")

        #expect(!(await limiter.allowRequest(identifier: "a")))
        #expect(!(await limiter.allowRequest(identifier: "b")))

        // Reset only "a"
        await limiter.reset(identifier: "a")

        #expect(await limiter.allowRequest(identifier: "a"))
        #expect(!(await limiter.allowRequest(identifier: "b")))
    }

    @Test("Reset all identifiers")
    func testResetAllIdentifiers() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 1, refillRate: 0)
        )

        _ = await limiter.allowRequest(identifier: "x")
        _ = await limiter.allowRequest(identifier: "y")
        _ = await limiter.allowRequest(identifier: "z")

        #expect(!(await limiter.allowRequest(identifier: "x")))
        #expect(!(await limiter.allowRequest(identifier: "y")))
        #expect(!(await limiter.allowRequest(identifier: "z")))

        await limiter.reset()

        #expect(await limiter.allowRequest(identifier: "x"))
        #expect(await limiter.allowRequest(identifier: "y"))
        #expect(await limiter.allowRequest(identifier: "z"))
    }

    @Test("Status per identifier")
    func testStatusPerIdentifier() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 10, refillRate: 1)
        )

        // Use some from user1
        _ = await limiter.allowRequest(identifier: "user1")
        _ = await limiter.allowRequest(identifier: "user1")
        _ = await limiter.allowRequest(identifier: "user1")

        let status1 = await limiter.getStatus(identifier: "user1")
        let status2 = await limiter.getStatus(identifier: "user2")

        // user1 has used 3, user2 hasn't used any (new bucket)
        #expect(status1.remaining < status2.remaining)
    }
}

// MARK: - Cost Parameter Tests

@Suite("Rate Limiter Cost Parameter")
struct RateLimiterCostParameterTests {

    @Test("Token bucket respects cost parameter")
    func testTokenBucketCost() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 10, refillRate: 0)
        )

        // Cost of 5 should consume 5 tokens
        #expect(await limiter.allowRequest(cost: 5))
        #expect(await limiter.allowRequest(cost: 5))
        #expect(!(await limiter.allowRequest(cost: 1))) // Only 0 remaining
    }

    @Test("Cost exceeding capacity")
    func testCostExceedingCapacity() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 5, refillRate: 0)
        )

        // Cost of 10 exceeds capacity of 5
        #expect(!(await limiter.allowRequest(cost: 10)))
        // But should still allow smaller costs
        #expect(await limiter.allowRequest(cost: 5))
    }

    @Test("Fractional cost")
    func testFractionalCost() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 1, refillRate: 0)
        )

        #expect(await limiter.allowRequest(cost: 0.3))
        #expect(await limiter.allowRequest(cost: 0.3))
        #expect(await limiter.allowRequest(cost: 0.3))
        #expect(!(await limiter.allowRequest(cost: 0.3))) // 0.9 used, only 0.1 remaining
    }

    @Test("Zero cost always allowed")
    func testZeroCost() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 0, refillRate: 0)
        )

        // Zero cost should always be allowed even with zero capacity
        #expect(await limiter.allowRequest(cost: 0))
        #expect(await limiter.allowRequest(cost: 0))
    }
}

// MARK: - Wait For Permit Tests

@Suite("Wait For Permit Comprehensive")
struct WaitForPermitComprehensiveTests {

    @Test("Wait for permit times out")
    func testWaitForPermitTimeout() async throws {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 0, refillRate: 0)
        )

        do {
            try await limiter.waitForPermit(timeout: 0.1)
            Issue.record("Expected timeout")
        } catch let error as StreamingError {
            if case .timeout(let waited) = error {
                #expect(waited >= 0.1)
            } else {
                Issue.record("Expected timeout error, got: \(error)")
            }
        }
    }

    @Test("Wait for permit succeeds after refill")
    func testWaitForPermitSucceedsAfterRefill() async throws {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 1, refillRate: 20) // 20 tokens/sec
        )

        // Exhaust
        _ = await limiter.allowRequest()
        #expect(!(await limiter.allowRequest()))

        // Wait for permit should succeed within timeout
        try await limiter.waitForPermit(timeout: 0.5)
    }

    @Test("Wait for permit with custom identifier")
    func testWaitForPermitCustomIdentifier() async throws {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 1, refillRate: 10)
        )

        // Exhaust for specific identifier
        _ = await limiter.allowRequest(identifier: "limited")
        #expect(!(await limiter.allowRequest(identifier: "limited")))

        // Different identifier should succeed immediately
        try await limiter.waitForPermit(identifier: "other", timeout: 0.1)
    }

    @Test("Wait for permit with cost")
    func testWaitForPermitWithCost() async throws {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 5, refillRate: 20)
        )

        // Use 4 tokens
        _ = await limiter.allowRequest(cost: 4)

        // Should wait for 2 more tokens to refill for cost of 3
        try await limiter.waitForPermit(cost: 3, timeout: 0.5)
    }
}

// MARK: - Fixed Window Comprehensive Tests

@Suite("Fixed Window Comprehensive")
struct FixedWindowComprehensiveTests {

    @Test("Fixed window counts correctly")
    func testFixedWindowCounting() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .fixedWindow(windowSize: 1.0, maxRequests: 5)
        )

        for i in 0..<5 {
            let allowed = await limiter.allowRequest()
            #expect(allowed, "Request \(i) should be allowed")
        }

        #expect(!(await limiter.allowRequest()))
    }

    @Test("Fixed window status shows remaining")
    func testFixedWindowStatus() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .fixedWindow(windowSize: 1.0, maxRequests: 10)
        )

        _ = await limiter.allowRequest()
        _ = await limiter.allowRequest()
        _ = await limiter.allowRequest()

        let status = await limiter.getStatus()
        #expect(status.remaining == 7)
        #expect(status.limit == 10)
    }

    @Test("Fixed window resets after window expires")
    func testFixedWindowResetsAfterExpiry() async throws {
        let limiter = EmbeddingRateLimiter(
            strategy: .fixedWindow(windowSize: 0.2, maxRequests: 2)
        )

        #expect(await limiter.allowRequest())
        #expect(await limiter.allowRequest())
        #expect(!(await limiter.allowRequest()))

        // Wait for window to expire
        try await Task.sleep(for: .milliseconds(250))

        // Should allow requests again
        #expect(await limiter.allowRequest())
    }
}

// MARK: - Sliding Window Comprehensive Tests

@Suite("Sliding Window Comprehensive")
struct SlidingWindowComprehensiveTests {

    @Test("Sliding window tracks individual requests")
    func testSlidingWindowTracksRequests() async throws {
        let limiter = EmbeddingRateLimiter(
            strategy: .slidingWindow(windowSize: 0.5, maxRequests: 3)
        )

        // Make requests at different times
        #expect(await limiter.allowRequest())
        try await Task.sleep(for: .milliseconds(100))
        #expect(await limiter.allowRequest())
        try await Task.sleep(for: .milliseconds(100))
        #expect(await limiter.allowRequest())

        // Should be rate limited
        #expect(!(await limiter.allowRequest()))

        // Wait for first request to slide out
        try await Task.sleep(for: .milliseconds(350))

        // Should allow one more
        #expect(await limiter.allowRequest())
    }

    @Test("Sliding window status accuracy")
    func testSlidingWindowStatusAccuracy() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .slidingWindow(windowSize: 1.0, maxRequests: 5)
        )

        _ = await limiter.allowRequest()
        _ = await limiter.allowRequest()

        let status = await limiter.getStatus()
        #expect(status.remaining == 3)
        #expect(status.limit == 5)
    }
}

// MARK: - Leaky Bucket Comprehensive Tests

@Suite("Leaky Bucket Comprehensive")
struct LeakyBucketComprehensiveTests {

    @Test("Leaky bucket fills then drains")
    func testLeakyBucketFillsAndDrains() async throws {
        let limiter = EmbeddingRateLimiter(
            strategy: .leakyBucket(capacity: 3, leakRate: 0.05) // Leak every 50ms
        )

        // Fill bucket
        #expect(await limiter.allowRequest())
        #expect(await limiter.allowRequest())
        #expect(await limiter.allowRequest())
        #expect(!(await limiter.allowRequest()))

        // Wait for some leaks
        try await Task.sleep(for: .milliseconds(120)) // Should leak ~2 items

        // Should allow more
        #expect(await limiter.allowRequest())
    }

    @Test("Leaky bucket status shows current level")
    func testLeakyBucketStatus() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .leakyBucket(capacity: 10, leakRate: 1.0)
        )

        _ = await limiter.allowRequest()
        _ = await limiter.allowRequest()
        _ = await limiter.allowRequest()

        let status = await limiter.getStatus()
        #expect(status.remaining == 7) // 10 - 3
        #expect(status.limit == 10)
    }

    @Test("Leaky bucket constant output rate")
    func testLeakyBucketConstantRate() async throws {
        let limiter = EmbeddingRateLimiter(
            strategy: .leakyBucket(capacity: 100, leakRate: 0.02) // 50 per second
        )

        // Fill quickly
        for _ in 0..<100 {
            _ = await limiter.allowRequest()
        }

        // Measure how many we can process over time
        var count = 0
        let start = CFAbsoluteTimeGetCurrent()

        while CFAbsoluteTimeGetCurrent() - start < 0.2 {
            if await limiter.allowRequest() {
                count += 1
            }
            try await Task.sleep(for: .milliseconds(5))
        }

        // Should have leaked approximately 10 items (50/sec * 0.2s)
        #expect(count >= 5 && count <= 15)
    }
}

// MARK: - Concurrency Stress Tests

@Suite("Rate Limiter Concurrency Stress", .serialized)
struct RateLimiterConcurrencyStressTests {

    @Test("High concurrency token bucket")
    func testHighConcurrencyTokenBucket() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 100, refillRate: 100)
        )

        actor Counter {
            var count = 0
            func increment() { count += 1 }
            func getCount() -> Int { count }
        }
        let counter = Counter()

        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<50 {
                group.addTask {
                    for _ in 0..<10 {
                        if await limiter.allowRequest() {
                            await counter.increment()
                        }
                    }
                }
            }
        }

        let total = await counter.getCount()
        // Should be around capacity due to refill
        #expect(total > 50)
    }

    @Test("Concurrent identifier access")
    func testConcurrentIdentifierAccess() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 5, refillRate: 0)
        )

        await withTaskGroup(of: Void.self) { group in
            for i in 0..<20 {
                let identifier = "user\(i % 5)"
                group.addTask {
                    for _ in 0..<10 {
                        _ = await limiter.allowRequest(identifier: identifier)
                    }
                }
            }
        }

        // Each identifier should have 5 capacity
        // Verify state consistency
        for i in 0..<5 {
            let status = await limiter.getStatus(identifier: "user\(i)")
            #expect(status.limit == 5)
        }
    }

    @Test("Concurrent reset and request")
    func testConcurrentResetAndRequest() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 10, refillRate: 0)
        )

        await withTaskGroup(of: Void.self) { group in
            // Request tasks
            for _ in 0..<10 {
                group.addTask {
                    for _ in 0..<20 {
                        _ = await limiter.allowRequest()
                    }
                }
            }

            // Reset tasks
            for _ in 0..<5 {
                group.addTask {
                    try? await Task.sleep(for: .milliseconds(10))
                    await limiter.reset()
                }
            }
        }

        // Should complete without crashes
        let status = await limiter.getStatus()
        #expect(status.limit == 10)
    }
}

// MARK: - Time-Based Behavior Tests

@Suite("Rate Limiter Time Behavior")
struct RateLimiterTimeBehaviorTests {

    @Test("Token bucket refill is continuous")
    func testTokenBucketContinuousRefill() async throws {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 10, refillRate: 100) // 100 tokens/sec
        )

        // Exhaust
        for _ in 0..<10 {
            _ = await limiter.allowRequest()
        }
        #expect(!(await limiter.allowRequest()))

        // Wait 50ms, should have ~5 tokens
        try await Task.sleep(for: .milliseconds(50))

        var count = 0
        for _ in 0..<10 {
            if await limiter.allowRequest() {
                count += 1
            }
        }

        // Should have refilled approximately 5 tokens
        #expect(count >= 3 && count <= 7)
    }

    @Test("Time until reset is accurate")
    func testTimeUntilResetAccurate() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .fixedWindow(windowSize: 1.0, maxRequests: 5)
        )

        _ = await limiter.allowRequest()

        let status = await limiter.getStatus()
        #expect(status.timeUntilReset > 0)
        #expect(status.timeUntilReset <= 1.0)
    }
}

// MARK: - Default Identifier Tests

@Suite("Rate Limiter Default Identifier")
struct RateLimiterDefaultIdentifierTests {

    @Test("Default identifier is 'default'")
    func testDefaultIdentifier() async {
        let limiter = EmbeddingRateLimiter(
            strategy: .tokenBucket(capacity: 5, refillRate: 0)
        )

        // These should share the same bucket
        _ = await limiter.allowRequest() // Uses "default"
        _ = await limiter.allowRequest(identifier: "default")

        let status1 = await limiter.getStatus()
        let status2 = await limiter.getStatus(identifier: "default")

        #expect(status1.remaining == status2.remaining)
        #expect(status1.remaining == 3)
    }
}

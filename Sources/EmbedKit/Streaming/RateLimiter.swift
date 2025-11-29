// EmbedKit - RateLimiter
// Rate limiting for streaming embedding generation
// Adapted from PipelineKit's rate limiting algorithms

import Foundation

// MARK: - Rate Limiter

/// Actor-based rate limiter with multiple strategies for controlling embedding throughput.
///
/// `EmbeddingRateLimiter` provides sophisticated rate limiting to prevent overwhelming
/// the embedding model or downstream systems. It supports multiple algorithms:
/// - Token bucket for burst control
/// - Sliding window for accurate limiting
/// - Fixed window for simple counting
/// - Leaky bucket for constant output rate
/// - Adaptive limiting based on system load
///
/// ## Example Usage
/// ```swift
/// let limiter = EmbeddingRateLimiter(
///     strategy: .tokenBucket(capacity: 100, refillRate: 10)
/// )
///
/// // Check before processing
/// if await limiter.allowRequest(identifier: "batch-1") {
///     let embeddings = try await generator.produce(texts)
/// } else {
///     // Handle rate limit
/// }
///
/// // Or wait for permit
/// try await limiter.waitForPermit(identifier: "batch-1")
/// let embeddings = try await generator.produce(texts)
/// ```
public actor EmbeddingRateLimiter {
    private let strategy: RateLimitStrategy
    private var buckets: [String: TokenBucketImpl] = [:]
    private var slidingWindows: [String: SlidingWindowImpl] = [:]
    private var fixedWindows: [String: FixedWindowImpl] = [:]
    private var leakyBuckets: [String: LeakyBucketImpl] = [:]

    private let cleanupInterval: TimeInterval = 300  // 5 minutes
    private var lastCleanup = Date()

    /// Creates a rate limiter with the specified strategy.
    ///
    /// - Parameter strategy: The rate limiting algorithm to use.
    public init(strategy: RateLimitStrategy) {
        self.strategy = strategy
    }

    // MARK: - Core Operations

    /// Checks if a request is allowed under the current rate limit.
    ///
    /// - Parameters:
    ///   - identifier: Identifier for the rate limit bucket (e.g., user ID, batch ID).
    ///   - cost: Cost of this request (default: 1.0).
    /// - Returns: `true` if the request is allowed, `false` if rate limited.
    public func allowRequest(identifier: String = "default", cost: Double = 1.0) async -> Bool {
        await cleanupIfNeeded()

        switch strategy {
        case let .tokenBucket(capacity, refillRate):
            return checkTokenBucket(
                identifier: identifier,
                capacity: capacity,
                refillRate: refillRate,
                cost: cost
            )

        case let .slidingWindow(windowSize, maxRequests):
            return checkSlidingWindow(
                identifier: identifier,
                windowSize: windowSize,
                maxRequests: maxRequests
            )

        case let .fixedWindow(windowSize, maxRequests):
            return checkFixedWindow(
                identifier: identifier,
                windowSize: windowSize,
                maxRequests: maxRequests
            )

        case let .leakyBucket(capacity, leakRate):
            return checkLeakyBucket(
                identifier: identifier,
                capacity: capacity,
                leakRate: leakRate
            )

        case let .adaptive(baseRate, loadFactor):
            let factor = await loadFactor()
            let adjustedCapacity = Double(baseRate) * (2.0 - factor)
            return checkTokenBucket(
                identifier: identifier,
                capacity: adjustedCapacity,
                refillRate: adjustedCapacity / 10.0,
                cost: cost
            )
        }
    }

    /// Waits until a request is allowed, or throws on timeout.
    ///
    /// - Parameters:
    ///   - identifier: Identifier for the rate limit bucket.
    ///   - cost: Cost of this request (default: 1.0).
    ///   - timeout: Maximum time to wait (default: 30 seconds).
    /// - Throws: `StreamingError.timeout` if waiting exceeds timeout.
    public func waitForPermit(
        identifier: String = "default",
        cost: Double = 1.0,
        timeout: TimeInterval = 30.0
    ) async throws {
        let startTime = CFAbsoluteTimeGetCurrent()

        while true {
            if await allowRequest(identifier: identifier, cost: cost) {
                return
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            if elapsed >= timeout {
                throw StreamingError.timeout(waited: elapsed)
            }

            // Wait a small interval before retrying
            let status = await getStatus(identifier: identifier)
            let waitTime = min(status.timeUntilReset, timeout - elapsed, 0.1)
            if waitTime > 0 {
                try await Task.sleep(nanoseconds: UInt64(waitTime * 1_000_000_000))
            }
        }
    }

    /// Gets the current rate limit status for an identifier.
    ///
    /// - Parameter identifier: The identifier to check.
    /// - Returns: Current rate limit status.
    public func getStatus(identifier: String = "default") -> RateLimitStatus {
        switch strategy {
        case let .tokenBucket(capacity, refillRate):
            let bucket = buckets[identifier] ?? TokenBucketImpl(capacity: capacity, refillRate: refillRate)
            return RateLimitStatus(
                remaining: Int(bucket.tokens),
                limit: Int(capacity),
                resetAt: Date().addingTimeInterval(bucket.timeToNextToken())
            )

        case let .slidingWindow(windowSize, maxRequests):
            let window = slidingWindows[identifier] ?? SlidingWindowImpl(windowSize: windowSize)
            let count = window.requestCount(since: Date().addingTimeInterval(-windowSize))
            return RateLimitStatus(
                remaining: max(0, maxRequests - count),
                limit: maxRequests,
                resetAt: Date().addingTimeInterval(windowSize)
            )

        case let .fixedWindow(windowSize, maxRequests):
            let window = fixedWindows[identifier] ?? FixedWindowImpl(windowSize: windowSize)
            return RateLimitStatus(
                remaining: max(0, maxRequests - window.count),
                limit: maxRequests,
                resetAt: Date().addingTimeInterval(window.timeUntilReset())
            )

        case let .leakyBucket(capacity, _):
            let bucket = leakyBuckets[identifier] ?? LeakyBucketImpl(capacity: capacity, leakRate: 1.0)
            return RateLimitStatus(
                remaining: max(0, capacity - bucket.currentLevel),
                limit: capacity,
                resetAt: Date().addingTimeInterval(bucket.timeUntilNextLeak())
            )

        case let .adaptive(baseRate, _):
            let bucket = buckets[identifier] ?? TokenBucketImpl(
                capacity: Double(baseRate),
                refillRate: Double(baseRate) / 10.0
            )
            return RateLimitStatus(
                remaining: Int(bucket.tokens),
                limit: baseRate,
                resetAt: Date().addingTimeInterval(bucket.timeToNextToken())
            )
        }
    }

    /// Resets rate limits for a specific identifier or all identifiers.
    ///
    /// - Parameter identifier: The identifier to reset, or nil to reset all.
    public func reset(identifier: String? = nil) {
        if let identifier = identifier {
            buckets.removeValue(forKey: identifier)
            slidingWindows.removeValue(forKey: identifier)
            fixedWindows.removeValue(forKey: identifier)
            leakyBuckets.removeValue(forKey: identifier)
        } else {
            buckets.removeAll()
            slidingWindows.removeAll()
            fixedWindows.removeAll()
            leakyBuckets.removeAll()
        }
    }

    // MARK: - Private Implementation

    private func checkTokenBucket(
        identifier: String,
        capacity: Double,
        refillRate: Double,
        cost: Double
    ) -> Bool {
        let bucket = buckets[identifier] ?? TokenBucketImpl(capacity: capacity, refillRate: refillRate)
        buckets[identifier] = bucket

        bucket.refill()
        return bucket.consume(tokens: cost)
    }

    private func checkSlidingWindow(
        identifier: String,
        windowSize: TimeInterval,
        maxRequests: Int
    ) -> Bool {
        let window = slidingWindows[identifier] ?? SlidingWindowImpl(windowSize: windowSize)
        slidingWindows[identifier] = window

        window.recordRequest()
        let count = window.requestCount(since: Date().addingTimeInterval(-windowSize))
        return count <= maxRequests
    }

    private func checkFixedWindow(
        identifier: String,
        windowSize: TimeInterval,
        maxRequests: Int
    ) -> Bool {
        let window = fixedWindows[identifier] ?? FixedWindowImpl(windowSize: windowSize)
        fixedWindows[identifier] = window

        let count = window.recordRequest()
        return count <= maxRequests
    }

    private func checkLeakyBucket(
        identifier: String,
        capacity: Int,
        leakRate: TimeInterval
    ) -> Bool {
        let bucket = leakyBuckets[identifier] ?? LeakyBucketImpl(capacity: capacity, leakRate: leakRate)
        leakyBuckets[identifier] = bucket

        return bucket.tryAdd()
    }

    private func cleanupIfNeeded() async {
        let now = Date()
        guard now.timeIntervalSince(lastCleanup) >= cleanupInterval else { return }

        // Clean up unused buckets - remove those not accessed within cleanup interval
        buckets = buckets.filter { now.timeIntervalSince($0.value.lastAccess) < cleanupInterval }
        slidingWindows = slidingWindows.filter { $0.value.hasRecentRequests(within: cleanupInterval) }
        // Fixed windows and leaky buckets clean themselves

        lastCleanup = now
    }
}

// MARK: - Token Bucket Implementation

/// Token bucket algorithm implementation for rate limiting.
fileprivate final class TokenBucketImpl: @unchecked Sendable {
    private let lock = NSLock()
    private let capacity: Double
    private let refillRate: Double
    private var _tokens: Double
    private var _lastRefill: Date
    private var _lastAccess: Date

    var tokens: Double {
        lock.lock()
        defer { lock.unlock() }
        return _tokens
    }

    var lastAccess: Date {
        lock.lock()
        defer { lock.unlock() }
        return _lastAccess
    }

    init(capacity: Double, refillRate: Double) {
        self.capacity = capacity
        self.refillRate = refillRate
        self._tokens = capacity
        self._lastRefill = Date()
        self._lastAccess = Date()
    }

    func refill() {
        lock.lock()
        defer { lock.unlock() }

        let now = Date()
        let elapsed = now.timeIntervalSince(_lastRefill)
        let tokensToAdd = elapsed * refillRate

        _tokens = min(capacity, _tokens + tokensToAdd)
        _lastRefill = now
        _lastAccess = now
    }

    func consume(tokens: Double) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        guard _tokens >= tokens else { return false }
        _tokens -= tokens
        _lastAccess = Date()
        return true
    }

    func timeToNextToken() -> TimeInterval {
        lock.lock()
        defer { lock.unlock() }

        guard _tokens < capacity, refillRate > 0 else { return 0 }
        return 1.0 / refillRate
    }
}

// MARK: - Sliding Window Implementation

/// Sliding window algorithm implementation for rate limiting.
fileprivate final class SlidingWindowImpl: @unchecked Sendable {
    private let lock = NSLock()
    private let windowSize: TimeInterval
    private var requests: [Date] = []

    init(windowSize: TimeInterval) {
        self.windowSize = windowSize
    }

    func recordRequest() {
        lock.lock()
        defer { lock.unlock() }

        let now = Date()
        requests.append(now)

        // Clean up old requests
        let cutoff = now.addingTimeInterval(-windowSize * 2)
        requests.removeAll { $0 < cutoff }
    }

    func requestCount(since date: Date) -> Int {
        lock.lock()
        defer { lock.unlock() }

        return requests.filter { $0 >= date }.count
    }

    func hasRecentRequests(within interval: TimeInterval) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        let cutoff = Date().addingTimeInterval(-interval)
        return requests.contains { $0 >= cutoff }
    }
}

// MARK: - Fixed Window Implementation

/// Fixed window counter implementation for rate limiting.
fileprivate final class FixedWindowImpl: @unchecked Sendable {
    private let lock = NSLock()
    private let windowSize: TimeInterval
    private var windowStart: Date
    private var _count: Int = 0

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        resetIfNeeded()
        return _count
    }

    init(windowSize: TimeInterval) {
        self.windowSize = windowSize
        self.windowStart = Date()
    }

    func recordRequest() -> Int {
        lock.lock()
        defer { lock.unlock() }

        resetIfNeeded()
        _count += 1
        return _count
    }

    func timeUntilReset() -> TimeInterval {
        lock.lock()
        defer { lock.unlock() }

        let windowEnd = windowStart.addingTimeInterval(windowSize)
        return max(0, windowEnd.timeIntervalSinceNow)
    }

    private func resetIfNeeded() {
        let now = Date()
        if now.timeIntervalSince(windowStart) >= windowSize {
            windowStart = now
            _count = 0
        }
    }
}

// MARK: - Leaky Bucket Implementation

/// Leaky bucket algorithm implementation for rate limiting.
fileprivate final class LeakyBucketImpl: @unchecked Sendable {
    private let lock = NSLock()
    private let capacity: Int
    private let leakRate: TimeInterval
    private var _currentLevel: Int = 0
    private var lastLeakTime: Date

    var currentLevel: Int {
        lock.lock()
        defer { lock.unlock() }
        leak()
        return _currentLevel
    }

    init(capacity: Int, leakRate: TimeInterval) {
        self.capacity = capacity
        self.leakRate = leakRate
        self.lastLeakTime = Date()
    }

    func tryAdd() -> Bool {
        lock.lock()
        defer { lock.unlock() }

        leak()
        guard _currentLevel < capacity else { return false }
        _currentLevel += 1
        return true
    }

    func timeUntilNextLeak() -> TimeInterval {
        lock.lock()
        defer { lock.unlock() }

        guard _currentLevel > 0 else { return 0 }
        let nextLeakTime = lastLeakTime.addingTimeInterval(leakRate)
        return max(0, nextLeakTime.timeIntervalSinceNow)
    }

    private func leak() {
        let now = Date()
        let timeSinceLastLeak = now.timeIntervalSince(lastLeakTime)
        let leakedItems = Int(timeSinceLastLeak / leakRate)

        if leakedItems > 0 {
            _currentLevel = max(0, _currentLevel - leakedItems)
            lastLeakTime = lastLeakTime.addingTimeInterval(Double(leakedItems) * leakRate)
        }
    }
}

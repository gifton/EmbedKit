// EmbedKit - Streaming Types
// Core types for back-pressure and rate limiting in streaming embedding generation

import Foundation

// MARK: - Back-Pressure Strategy

/// Strategy for handling back-pressure when the consumer can't keep up with the producer.
public enum BackPressureStrategy: Sendable, Equatable {
    /// Suspend the producer until capacity becomes available.
    /// This provides natural flow control by slowing down producers.
    case suspend

    /// Drop the oldest queued item to make room for new ones.
    /// Useful when recent embeddings are more valuable than older ones.
    case dropOldest

    /// Drop the newest item if the queue is full.
    /// Preserves existing work at the cost of rejecting new requests.
    case dropNewest

    /// Throw an error after the specified timeout.
    /// - Parameter timeout: Maximum time to wait before throwing. If nil, throws immediately.
    case error(timeout: TimeInterval?)
}

// MARK: - Rate Limit Strategy

/// Strategy for rate limiting embedding requests.
public enum RateLimitStrategy: Sendable {
    /// Token bucket algorithm with burst control.
    /// - Parameters:
    ///   - capacity: Maximum tokens in the bucket (burst size)
    ///   - refillRate: Tokens added per second
    case tokenBucket(capacity: Double, refillRate: Double)

    /// Sliding window for accurate rate limiting.
    /// - Parameters:
    ///   - windowSize: Window duration in seconds
    ///   - maxRequests: Maximum requests allowed in the window
    case slidingWindow(windowSize: TimeInterval, maxRequests: Int)

    /// Fixed window counter (simpler but less accurate at boundaries).
    /// - Parameters:
    ///   - windowSize: Window duration in seconds
    ///   - maxRequests: Maximum requests allowed in the window
    case fixedWindow(windowSize: TimeInterval, maxRequests: Int)

    /// Leaky bucket for constant output rate.
    /// - Parameters:
    ///   - capacity: Maximum queue size
    ///   - leakRate: Seconds between processing each item
    case leakyBucket(capacity: Int, leakRate: TimeInterval)

    /// Adaptive rate limiting based on system load.
    /// - Parameters:
    ///   - baseRate: Base requests per second
    ///   - loadFactor: Closure returning current load (0.0-1.0)
    case adaptive(baseRate: Int, loadFactor: @Sendable () async -> Double)
}

// MARK: - Rate Limit Status

/// Current status of rate limiting for a given identifier.
public struct RateLimitStatus: Sendable {
    /// Remaining requests allowed in the current window.
    public let remaining: Int

    /// Total limit for the window.
    public let limit: Int

    /// When the rate limit will reset.
    public let resetAt: Date

    /// Whether requests are currently being rate limited.
    public var isLimited: Bool { remaining <= 0 }

    /// Time until the rate limit resets.
    public var timeUntilReset: TimeInterval {
        max(0, resetAt.timeIntervalSinceNow)
    }
}

// MARK: - Streaming Configuration

/// Configuration for streaming embedding generation with back-pressure and rate limiting.
public struct FlowControlConfig: Sendable {
    /// Maximum concurrent embedding operations.
    public var maxConcurrency: Int

    /// Maximum items that can be queued for processing.
    public var maxQueueDepth: Int

    /// Back-pressure strategy when queue is full.
    public var backPressureStrategy: BackPressureStrategy

    /// Optional rate limiting strategy.
    public var rateLimitStrategy: RateLimitStrategy?

    /// Batch size for processing.
    public var batchSize: Int

    /// Maximum time an item can wait in queue before timing out.
    public var maxQueueWait: TimeInterval

    /// Whether to enable adaptive batching based on system load.
    public var adaptiveBatching: Bool

    /// Callback when back-pressure is applied.
    public var onBackPressure: (@Sendable (BackPressureEvent) -> Void)?

    /// Creates a streaming configuration.
    ///
    /// - Parameters:
    ///   - maxConcurrency: Maximum concurrent embedding operations (must be > 0).
    ///   - maxQueueDepth: Maximum items that can be queued (must be > 0).
    ///   - backPressureStrategy: Strategy when queue is full.
    ///   - rateLimitStrategy: Optional rate limiting strategy.
    ///   - batchSize: Batch size for processing (must be > 0).
    ///   - maxQueueWait: Maximum time an item can wait in queue.
    ///   - adaptiveBatching: Enable adaptive batching based on system load.
    ///   - onBackPressure: Callback when back-pressure is applied.
    public init(
        maxConcurrency: Int = 4,
        maxQueueDepth: Int = 100,
        backPressureStrategy: BackPressureStrategy = .suspend,
        rateLimitStrategy: RateLimitStrategy? = nil,
        batchSize: Int = 32,
        maxQueueWait: TimeInterval = 30.0,
        adaptiveBatching: Bool = true,
        onBackPressure: (@Sendable (BackPressureEvent) -> Void)? = nil
    ) {
        precondition(maxConcurrency > 0, "maxConcurrency must be > 0")
        precondition(maxQueueDepth > 0, "maxQueueDepth must be > 0")
        precondition(batchSize > 0, "batchSize must be > 0")
        precondition(maxQueueWait > 0, "maxQueueWait must be > 0")

        self.maxConcurrency = maxConcurrency
        self.maxQueueDepth = maxQueueDepth
        self.backPressureStrategy = backPressureStrategy
        self.rateLimitStrategy = rateLimitStrategy
        self.batchSize = batchSize
        self.maxQueueWait = maxQueueWait
        self.adaptiveBatching = adaptiveBatching
        self.onBackPressure = onBackPressure
    }

    /// Default configuration for most use cases.
    public static let `default` = FlowControlConfig()

    /// Configuration optimized for high throughput.
    public static let highThroughput = FlowControlConfig(
        maxConcurrency: 8,
        maxQueueDepth: 500,
        backPressureStrategy: .dropOldest,
        batchSize: 64,
        maxQueueWait: 60.0
    )

    /// Configuration optimized for low latency.
    public static let lowLatency = FlowControlConfig(
        maxConcurrency: 2,
        maxQueueDepth: 20,
        backPressureStrategy: .error(timeout: 1.0),
        batchSize: 8,
        maxQueueWait: 5.0
    )

    /// Configuration with strict rate limiting.
    public static func rateLimited(requestsPerSecond: Double) -> FlowControlConfig {
        FlowControlConfig(
            maxConcurrency: 4,
            maxQueueDepth: 100,
            backPressureStrategy: .suspend,
            rateLimitStrategy: .tokenBucket(
                capacity: requestsPerSecond * 2,  // Allow burst of 2 seconds
                refillRate: requestsPerSecond
            ),
            batchSize: 32
        )
    }
}

// MARK: - Back-Pressure Event

/// Events related to back-pressure handling.
public enum BackPressureEvent: Sendable {
    /// Producer was suspended due to full queue.
    case suspended(queueDepth: Int)

    /// Producer resumed after queue drained.
    case resumed(queueDepth: Int)

    /// Item was dropped due to back-pressure.
    case dropped(reason: String)

    /// Queue depth changed significantly.
    case queueDepthChanged(old: Int, new: Int)

    /// Rate limit was hit.
    case rateLimited(remaining: Int, resetAt: Date)
}

// MARK: - Streaming Errors

/// Errors that can occur during streaming embedding generation.
public enum StreamingError: Error, LocalizedError, Sendable {
    /// Queue is full and back-pressure strategy doesn't allow waiting.
    case queueFull(current: Int, limit: Int)

    /// Operation timed out waiting for queue capacity.
    case timeout(waited: TimeInterval)

    /// Rate limit exceeded.
    case rateLimited(status: RateLimitStatus)

    /// Item was dropped due to back-pressure.
    case dropped(reason: String)

    /// Stream was cancelled.
    case cancelled

    /// Producer finished before consumer could process all items.
    case producerFinished

    public var errorDescription: String? {
        switch self {
        case .queueFull(let current, let limit):
            return "Queue is full: \(current)/\(limit) items"
        case .timeout(let waited):
            return "Timeout after waiting \(String(format: "%.2f", waited))s for queue capacity"
        case .rateLimited(let status):
            return "Rate limited: \(status.remaining)/\(status.limit) remaining, resets in \(String(format: "%.1f", status.timeUntilReset))s"
        case .dropped(let reason):
            return "Item dropped: \(reason)"
        case .cancelled:
            return "Stream was cancelled"
        case .producerFinished:
            return "Producer finished unexpectedly"
        }
    }
}

// MARK: - Streaming Statistics

/// Statistics for streaming embedding operations.
public struct StreamingStatistics: Sendable {
    /// Total items submitted to the stream.
    public let totalSubmitted: Int

    /// Total items successfully processed.
    public let totalProcessed: Int

    /// Total items dropped due to back-pressure.
    public let totalDropped: Int

    /// Total times rate limiting was applied.
    public let rateLimitHits: Int

    /// Total time spent waiting due to back-pressure.
    public let totalWaitTime: TimeInterval

    /// Current queue depth.
    public let currentQueueDepth: Int

    /// Peak queue depth observed.
    public let peakQueueDepth: Int

    /// Average processing time per item.
    public let averageProcessingTime: TimeInterval

    /// Current throughput (items per second).
    public let throughput: Double

    /// Creates streaming statistics.
    public init(
        totalSubmitted: Int,
        totalProcessed: Int,
        totalDropped: Int,
        rateLimitHits: Int,
        totalWaitTime: TimeInterval,
        currentQueueDepth: Int,
        peakQueueDepth: Int,
        averageProcessingTime: TimeInterval,
        throughput: Double
    ) {
        self.totalSubmitted = totalSubmitted
        self.totalProcessed = totalProcessed
        self.totalDropped = totalDropped
        self.rateLimitHits = rateLimitHits
        self.totalWaitTime = totalWaitTime
        self.currentQueueDepth = currentQueueDepth
        self.peakQueueDepth = peakQueueDepth
        self.averageProcessingTime = averageProcessingTime
        self.throughput = throughput
    }

    /// Queue utilization (current/max).
    public var queueUtilization: Double {
        guard peakQueueDepth > 0 else { return 0 }
        return Double(currentQueueDepth) / Double(peakQueueDepth)
    }

    /// Drop rate (dropped/submitted).
    public var dropRate: Double {
        guard totalSubmitted > 0 else { return 0 }
        return Double(totalDropped) / Double(totalSubmitted)
    }
}

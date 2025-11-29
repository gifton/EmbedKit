// EmbedKit - BackPressureController
// Back-pressure management for streaming embedding generation
// Adapted from PipelineKit's back-pressure semaphore

import Foundation

// MARK: - Back-Pressure Controller

/// Actor-based back-pressure controller for managing producer-consumer flow.
///
/// `BackPressureController` prevents overwhelming the embedding system by controlling
/// how many items can be queued for processing. When the queue fills up, it applies
/// the configured back-pressure strategy.
///
/// ## Strategies
/// - `.suspend`: Block producers until capacity is available
/// - `.dropOldest`: Remove oldest queued items to make room
/// - `.dropNewest`: Reject new items when full
/// - `.error(timeout:)`: Throw an error after optional timeout
///
/// ## Example Usage
/// ```swift
/// let controller = BackPressureController(
///     maxQueueDepth: 100,
///     strategy: .suspend
/// )
///
/// // Acquire a slot before submitting work
/// let token = try await controller.acquire()
/// defer { token.release() }
///
/// let embedding = try await generator.produce(text)
/// ```
public actor BackPressureController {
    // MARK: - Configuration

    private let maxQueueDepth: Int
    private let strategy: BackPressureStrategy
    private let maxWaitTime: TimeInterval

    // MARK: - State

    private var currentDepth: Int = 0
    private var peakDepth: Int = 0
    private var waiters: [Waiter] = []
    private var totalAcquired: Int = 0
    private var totalDropped: Int = 0
    private var totalWaitTime: TimeInterval = 0

    private var eventHandler: (@Sendable (BackPressureEvent) -> Void)?

    private struct Waiter: Identifiable {
        let id = UUID()
        let continuation: CheckedContinuation<BackPressureToken, any Error>
        let enqueuedAt: Date
        var isCancelled: Bool = false
    }

    // MARK: - Initialization

    /// Creates a back-pressure controller.
    ///
    /// - Parameters:
    ///   - maxQueueDepth: Maximum items that can be queued.
    ///   - strategy: Strategy for handling full queue.
    ///   - maxWaitTime: Maximum time to wait for capacity (default: 30s).
    ///   - onEvent: Optional handler for back-pressure events.
    public init(
        maxQueueDepth: Int,
        strategy: BackPressureStrategy = .suspend,
        maxWaitTime: TimeInterval = 30.0,
        onEvent: (@Sendable (BackPressureEvent) -> Void)? = nil
    ) {
        precondition(maxQueueDepth > 0, "maxQueueDepth must be > 0")

        self.maxQueueDepth = maxQueueDepth
        self.strategy = strategy
        self.maxWaitTime = maxWaitTime
        self.eventHandler = onEvent
    }

    // MARK: - Acquisition

    /// Acquires a slot in the queue, waiting if necessary.
    ///
    /// - Returns: A token that must be released when done.
    /// - Throws: `StreamingError` if acquisition fails.
    public func acquire() async throws -> BackPressureToken {
        // Fast path: immediate acquisition
        if currentDepth < maxQueueDepth && waiters.isEmpty {
            return acquireImmediate()
        }

        // Handle based on strategy
        switch strategy {
        case .suspend:
            return try await acquireWithSuspend()

        case .dropOldest:
            return try await acquireWithDropOldest()

        case .dropNewest:
            throw StreamingError.queueFull(current: currentDepth, limit: maxQueueDepth)

        case .error(let timeout):
            if let timeout = timeout {
                return try await acquireWithTimeout(timeout)
            } else {
                throw StreamingError.queueFull(current: currentDepth, limit: maxQueueDepth)
            }
        }
    }

    /// Attempts to acquire a slot without waiting.
    ///
    /// - Returns: A token if acquired, nil if queue is full.
    public func tryAcquire() -> BackPressureToken? {
        guard currentDepth < maxQueueDepth && waiters.isEmpty else {
            return nil
        }
        return acquireImmediate()
    }

    /// Releases a slot back to the pool.
    fileprivate func release() {
        let oldDepth = currentDepth
        currentDepth -= 1

        // Notify of depth change
        if abs(oldDepth - currentDepth) >= maxQueueDepth / 10 {
            eventHandler?(.queueDepthChanged(old: oldDepth, new: currentDepth))
        }

        // Resume a waiting acquirer if any
        if let waiter = extractNextWaiter() {
            let waitTime = Date().timeIntervalSince(waiter.enqueuedAt)
            totalWaitTime += waitTime

            currentDepth += 1
            totalAcquired += 1

            let token = BackPressureToken { [weak self] in
                Task { await self?.release() }
            }
            waiter.continuation.resume(returning: token)

            eventHandler?(.resumed(queueDepth: currentDepth))
        }
    }

    // MARK: - Statistics

    /// Current queue depth.
    public var queueDepth: Int { currentDepth }

    /// Peak queue depth observed.
    public var peakQueueDepth: Int { peakDepth }

    /// Number of waiters currently blocked.
    public var waiterCount: Int { waiters.count }

    /// Whether back-pressure is currently being applied.
    public var isBackPressured: Bool { currentDepth >= maxQueueDepth }

    /// Get comprehensive statistics.
    public func getStatistics() -> BackPressureStatistics {
        BackPressureStatistics(
            maxQueueDepth: maxQueueDepth,
            currentDepth: currentDepth,
            peakDepth: peakDepth,
            waitingCount: waiters.count,
            totalAcquired: totalAcquired,
            totalDropped: totalDropped,
            totalWaitTime: totalWaitTime,
            averageWaitTime: totalAcquired > 0 ? totalWaitTime / Double(totalAcquired) : 0
        )
    }

    /// Reset statistics (not the current state).
    public func resetStatistics() {
        peakDepth = currentDepth
        totalAcquired = 0
        totalDropped = 0
        totalWaitTime = 0
    }

    // MARK: - Private Implementation

    private func acquireImmediate() -> BackPressureToken {
        currentDepth += 1
        totalAcquired += 1
        peakDepth = max(peakDepth, currentDepth)

        return BackPressureToken { [weak self] in
            Task { await self?.release() }
        }
    }

    private func acquireWithSuspend() async throws -> BackPressureToken {
        eventHandler?(.suspended(queueDepth: currentDepth))

        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let waiter = Waiter(continuation: continuation, enqueuedAt: Date())
                waiters.append(waiter)
            }
        } onCancel: {
            Task { [weak self] in
                await self?.cancelWaiter()
            }
        }
    }

    private func acquireWithDropOldest() async throws -> BackPressureToken {
        // If we have waiters, drop the oldest
        if let oldest = waiters.first {
            waiters.removeFirst()
            totalDropped += 1
            oldest.continuation.resume(throwing: StreamingError.dropped(reason: "Dropped to make room for newer request"))
            eventHandler?(.dropped(reason: "Oldest waiter dropped"))
        }

        // Now try to acquire
        if currentDepth < maxQueueDepth {
            return acquireImmediate()
        }

        // If still full, wait
        return try await acquireWithSuspend()
    }

    private func acquireWithTimeout(_ timeout: TimeInterval) async throws -> BackPressureToken {
        let deadline = Date().addingTimeInterval(timeout)

        return try await withThrowingTaskGroup(of: BackPressureToken.self) { group in
            group.addTask {
                try await self.acquireWithSuspend()
            }

            group.addTask {
                let waitTime = deadline.timeIntervalSinceNow
                if waitTime > 0 {
                    try await Task.sleep(nanoseconds: UInt64(waitTime * 1_000_000_000))
                }
                throw StreamingError.timeout(waited: timeout)
            }

            // Return first successful result
            let result = try await group.next()!
            group.cancelAll()
            return result
        }
    }

    private func extractNextWaiter() -> Waiter? {
        while !waiters.isEmpty {
            let waiter = waiters.removeFirst()
            if !waiter.isCancelled {
                return waiter
            }
        }
        return nil
    }

    private func cancelWaiter() {
        // Find and remove the most recent non-cancelled waiter
        if let index = waiters.lastIndex(where: { !$0.isCancelled }) {
            let waiter = waiters.remove(at: index)
            waiter.continuation.resume(throwing: CancellationError())
        }
    }
}

// MARK: - Back-Pressure Token

/// A token representing an acquired slot in the queue.
///
/// The slot is automatically released when the token is deallocated,
/// preventing resource leaks even if tasks are cancelled or errors occur.
public final class BackPressureToken: @unchecked Sendable {
    private let releaseHandler: @Sendable () -> Void
    private var released = false
    private let lock = NSLock()

    /// When this token was acquired.
    public let acquiredAt: Date

    init(releaseHandler: @Sendable @escaping () -> Void) {
        self.releaseHandler = releaseHandler
        self.acquiredAt = Date()
    }

    /// How long this token has been held.
    public var holdDuration: TimeInterval {
        Date().timeIntervalSince(acquiredAt)
    }

    /// Whether this token has been released.
    public var isReleased: Bool {
        lock.lock()
        defer { lock.unlock() }
        return released
    }

    /// Explicitly releases the slot.
    ///
    /// This method is idempotent - calling it multiple times is safe.
    public func release() {
        lock.lock()
        guard !released else {
            lock.unlock()
            return
        }
        released = true
        lock.unlock()

        releaseHandler()
    }

    deinit {
        release()
    }
}

// MARK: - Back-Pressure Statistics

/// Statistics about back-pressure operations.
public struct BackPressureStatistics: Sendable {
    /// Maximum queue depth configured.
    public let maxQueueDepth: Int

    /// Current queue depth.
    public let currentDepth: Int

    /// Peak queue depth observed.
    public let peakDepth: Int

    /// Number of waiters currently blocked.
    public let waitingCount: Int

    /// Total acquisitions.
    public let totalAcquired: Int

    /// Total items dropped due to back-pressure.
    public let totalDropped: Int

    /// Total time spent waiting for capacity.
    public let totalWaitTime: TimeInterval

    /// Average wait time per acquisition.
    public let averageWaitTime: TimeInterval

    /// Queue utilization (current/max).
    public var utilization: Double {
        guard maxQueueDepth > 0 else { return 0 }
        return Double(currentDepth) / Double(maxQueueDepth)
    }

    /// Drop rate (dropped/acquired).
    public var dropRate: Double {
        let total = totalAcquired + totalDropped
        guard total > 0 else { return 0 }
        return Double(totalDropped) / Double(total)
    }
}

// MARK: - Convenience Extensions

extension BackPressureController {
    /// Execute a block with automatic back-pressure management.
    ///
    /// - Parameter operation: The operation to execute.
    /// - Returns: The result of the operation.
    /// - Throws: Any error from the operation or back-pressure.
    public func withBackPressure<T: Sendable>(
        _ operation: @Sendable () async throws -> T
    ) async throws -> T {
        let token = try await acquire()
        defer { token.release() }
        return try await operation()
    }
}

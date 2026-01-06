// EmbedKit - Adaptive Batcher
// Intelligent request batching with memory-aware dynamic sizing

import Foundation
import Dispatch
import VectorCore

// MARK: - Request Priority

/// Priority level for embedding requests.
///
/// Higher priority requests are processed before lower priority ones,
/// regardless of submission order. This allows latency-sensitive requests
/// (e.g., user-facing search) to be processed ahead of background work.
public enum RequestPriority: Int, Comparable, Sendable, CaseIterable {
    /// Background work that can tolerate high latency.
    case low = 0
    /// Standard priority for most requests.
    case normal = 1
    /// Elevated priority for time-sensitive operations.
    case high = 2
    /// Highest priority, processed immediately if possible.
    case urgent = 3

    public static func < (lhs: RequestPriority, rhs: RequestPriority) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

// MARK: - Configuration

/// Configuration for adaptive batching behavior.
public struct AdaptiveBatcherConfig: Sendable {
    /// Maximum time (in seconds) a request can wait in queue before forcing a flush.
    public var maxLatency: TimeInterval = 0.1

    /// Target batch sizes at different memory pressure levels (0.0 - 1.0).
    /// Lower pressure allows larger batches for throughput; higher pressure reduces batch size.
    public var batchSizeByPressure: [ClosedRange<Float>: Int] = [
        0.0...0.3: 128,
        0.3...0.6: 64,
        0.6...0.8: 32,
        0.8...1.0: 16
    ]

    /// Minimum batch size (will wait for more requests unless maxLatency exceeded).
    public var minBatchSize: Int = 1

    /// Maximum batch size (hard cap regardless of pressure).
    public var maxBatchSize: Int = 128

    /// Enable automatic flushing when optimal batch size is reached.
    public var autoFlush: Bool = true

    /// Number of recent batch timings to track for performance adaptation.
    public var performanceWindowSize: Int = 10

    /// Options passed to the underlying model's embedBatch call.
    public var batchOptions: BatchOptions = BatchOptions()

    /// Enable priority-based request ordering within batches.
    /// When enabled, higher priority requests are processed first.
    public var enablePriorityScheduling: Bool = true

    /// Maximum latency multiplier for low-priority requests.
    /// Low-priority requests may wait up to `maxLatency * lowPriorityLatencyMultiplier`.
    public var lowPriorityLatencyMultiplier: Double = 3.0

    /// Whether urgent requests should trigger immediate processing.
    public var urgentTriggersFlush: Bool = true

    public init() {}
}

// MARK: - Pending Request

/// Internal representation of a queued embedding request.
fileprivate final class PendingRequest: @unchecked Sendable {
    let text: String
    let submittedAt: CFAbsoluteTime
    let priority: RequestPriority
    private let continuation: UnsafeContinuation<Embedding, Error>

    init(text: String, priority: RequestPriority = .normal, continuation: UnsafeContinuation<Embedding, Error>) {
        self.text = text
        self.submittedAt = CFAbsoluteTimeGetCurrent()
        self.priority = priority
        self.continuation = continuation
    }

    var age: TimeInterval {
        CFAbsoluteTimeGetCurrent() - submittedAt
    }

    /// Effective max latency based on priority (lower priority = more patient).
    func effectiveMaxLatency(baseLatency: TimeInterval, lowPriorityMultiplier: Double) -> TimeInterval {
        switch priority {
        case .urgent:
            return baseLatency * 0.25  // Urgent gets 25% of normal latency
        case .high:
            return baseLatency * 0.5   // High gets 50% of normal latency
        case .normal:
            return baseLatency
        case .low:
            return baseLatency * lowPriorityMultiplier
        }
    }

    func complete(with embedding: Embedding) {
        continuation.resume(returning: embedding)
    }

    func fail(with error: Error) {
        continuation.resume(throwing: error)
    }
}

// MARK: - Adaptive Batcher

/// An actor that intelligently batches embedding requests for optimal throughput.
///
/// `AdaptiveBatcher` provides a simple `embed(_:)` interface that transparently batches
/// concurrent requests. It monitors system memory pressure and adjusts batch sizes
/// dynamically to balance throughput and resource usage.
///
/// ## Example Usage
/// ```swift
/// let model = try await ModelManager.shared.model(for: modelID)
/// let batcher = AdaptiveBatcher(model: model)
///
/// // These concurrent calls will be batched automatically
/// async let emb1 = batcher.embed("Hello world")
/// async let emb2 = batcher.embed("How are you?")
/// async let emb3 = batcher.embed("Swift is great")
///
/// let embeddings = try await [emb1, emb2, emb3]
/// ```
public actor AdaptiveBatcher {
    // MARK: - Properties

    private let model: any EmbeddingModel
    private var config: AdaptiveBatcherConfig
    private var pendingRequests: [PendingRequest] = []
    private var currentMemoryPressure: Float = 0.0
    private var recentBatchTimes: [Double] = []
    private var flushTask: Task<Void, Never>?
    private var isProcessing: Bool = false

    // MARK: - Metrics

    /// Statistics about batching performance.
    public struct BatcherMetrics: Sendable {
        public let totalRequests: Int
        public let totalBatches: Int
        public let averageBatchSize: Double
        public let averageBatchLatency: TimeInterval
        public let currentQueueDepth: Int
        public let currentMemoryPressure: Float
        /// Breakdown of requests by priority level.
        public let requestsByPriority: [RequestPriority: Int]
        /// Current queue depth by priority level.
        public let queueDepthByPriority: [RequestPriority: Int]
    }

    private var totalRequests: Int = 0
    private var totalBatches: Int = 0
    private var requestsByPriority: [RequestPriority: Int] = [:]

    // MARK: - Initialization

    /// Creates a new adaptive batcher wrapping the given embedding model.
    ///
    /// - Parameters:
    ///   - model: The embedding model to use for processing batches.
    ///   - config: Configuration for batching behavior. Defaults to sensible values.
    public init(model: any EmbeddingModel, config: AdaptiveBatcherConfig = AdaptiveBatcherConfig()) {
        self.model = model
        self.config = config
    }

    // MARK: - Public API

    /// Embed a single text, automatically batching with other concurrent requests.
    ///
    /// This method queues the request and waits for the result. Multiple concurrent
    /// calls to this method will be batched together for efficient processing.
    ///
    /// - Parameter text: The text to embed.
    /// - Returns: The computed embedding.
    /// - Throws: Any error from the underlying model.
    public func embed(_ text: String) async throws -> Embedding {
        try await embed(text, priority: .normal)
    }

    /// Embed a single text with specified priority.
    ///
    /// Higher priority requests are processed before lower priority ones.
    /// Urgent requests may trigger immediate batch processing.
    ///
    /// - Parameters:
    ///   - text: The text to embed.
    ///   - priority: The priority level for this request.
    /// - Returns: The computed embedding.
    /// - Throws: Any error from the underlying model.
    public func embed(_ text: String, priority: RequestPriority) async throws -> Embedding {
        try await withUnsafeThrowingContinuation { continuation in
            let request = PendingRequest(text: text, priority: priority, continuation: continuation)
            insertByPriority(request)
            totalRequests += 1
            requestsByPriority[priority, default: 0] += 1
            scheduleFlushIfNeeded(urgentAdded: priority == .urgent)
        }
    }

    /// Embed multiple texts directly, bypassing the queue.
    ///
    /// Use this when you already have a batch of texts to process together.
    /// This does not affect or interact with pending queued requests.
    ///
    /// - Parameter texts: The texts to embed.
    /// - Returns: The computed embeddings in the same order as input.
    /// - Throws: Any error from the underlying model.
    public func embedBatch(_ texts: [String]) async throws -> [Embedding] {
        try await model.embedBatch(texts, options: config.batchOptions)
    }

    /// Force flush all pending requests immediately.
    ///
    /// Call this to ensure all queued requests are processed without waiting
    /// for the batch to fill or the timeout to expire.
    public func flush() async throws {
        guard !pendingRequests.isEmpty else { return }
        try await processPendingBatch()
    }

    /// Update the batcher configuration.
    ///
    /// - Parameter config: The new configuration to use.
    public func setConfig(_ config: AdaptiveBatcherConfig) {
        self.config = config
    }

    /// Get current batching metrics.
    public var metrics: BatcherMetrics {
        let avgBatchSize = totalBatches > 0 ? Double(totalRequests) / Double(totalBatches) : 0
        let avgLatency = recentBatchTimes.isEmpty ? 0 : recentBatchTimes.reduce(0, +) / Double(recentBatchTimes.count)

        // Compute current queue depth by priority
        var queueByPriority: [RequestPriority: Int] = [:]
        for request in pendingRequests {
            queueByPriority[request.priority, default: 0] += 1
        }

        return BatcherMetrics(
            totalRequests: totalRequests,
            totalBatches: totalBatches,
            averageBatchSize: avgBatchSize,
            averageBatchLatency: avgLatency,
            currentQueueDepth: pendingRequests.count,
            currentMemoryPressure: currentMemoryPressure,
            requestsByPriority: requestsByPriority,
            queueDepthByPriority: queueByPriority
        )
    }

    /// Reset metrics counters.
    public func resetMetrics() {
        totalRequests = 0
        totalBatches = 0
        recentBatchTimes.removeAll()
        requestsByPriority.removeAll()
    }

    // MARK: - Private Implementation

    private func scheduleFlushIfNeeded(urgentAdded: Bool = false) {
        // When autoFlush is disabled, process immediately to avoid hanging
        guard config.autoFlush else {
            flushTask = Task {
                try? await self.processPendingBatch()
            }
            return
        }

        // Cancel existing flush task if we're scheduling a new one
        flushTask?.cancel()

        // Urgent requests trigger immediate processing if configured
        let shouldFlushImmediately = shouldProcessBatch() ||
            (urgentAdded && config.urgentTriggersFlush && pendingRequests.count >= config.minBatchSize)

        if shouldFlushImmediately {
            flushTask = Task {
                try? await self.processPendingBatch()
            }
        } else {
            // Schedule a flush after maxLatency
            // Use the shortest effective latency from pending requests
            let shortestLatency = computeShortestEffectiveLatency()
            flushTask = Task {
                try? await Task.sleep(nanoseconds: UInt64(shortestLatency * 1_000_000_000))
                if !Task.isCancelled {
                    try? await self.processPendingBatch()
                }
            }
        }
    }

    /// Computes the shortest effective latency across all pending requests.
    private func computeShortestEffectiveLatency() -> TimeInterval {
        guard !pendingRequests.isEmpty else { return config.maxLatency }

        var shortest = config.maxLatency
        for request in pendingRequests {
            let effective = request.effectiveMaxLatency(
                baseLatency: config.maxLatency,
                lowPriorityMultiplier: config.lowPriorityLatencyMultiplier
            )
            let remaining = max(0, effective - request.age)
            shortest = min(shortest, remaining)
        }
        return shortest
    }

    /// Inserts a request in priority order (higher priority = earlier in array).
    private func insertByPriority(_ request: PendingRequest) {
        guard config.enablePriorityScheduling else {
            pendingRequests.append(request)
            return
        }

        // Find insertion point: insert after last request with same or higher priority
        var insertIndex = pendingRequests.count
        for i in (0..<pendingRequests.count).reversed() {
            if pendingRequests[i].priority >= request.priority {
                insertIndex = i + 1
                break
            }
            if i == 0 {
                insertIndex = 0
            }
        }
        pendingRequests.insert(request, at: insertIndex)
    }

    private func shouldProcessBatch() -> Bool {
        guard !pendingRequests.isEmpty else { return false }

        let queueSize = pendingRequests.count
        let optimalSize = optimalBatchSize()

        // Check if any request has exceeded its effective max latency
        let anyExpired = pendingRequests.contains { request in
            let effectiveLatency = request.effectiveMaxLatency(
                baseLatency: config.maxLatency,
                lowPriorityMultiplier: config.lowPriorityLatencyMultiplier
            )
            return request.age >= effectiveLatency
        }

        // Check for high-priority requests that are aging
        let hasAgingHighPriority = pendingRequests.contains { request in
            (request.priority >= .high) && (request.age >= config.maxLatency * 0.5)
        }

        // Flush conditions:
        // 1. Queue reached optimal size
        // 2. Any request exceeded its effective max latency
        // 3. Memory pressure is high (flush smaller batches more frequently)
        // 4. Queue reached max batch size
        // 5. High-priority requests are aging
        return queueSize >= optimalSize ||
               anyExpired ||
               (currentMemoryPressure > 0.8 && queueSize >= config.minBatchSize) ||
               queueSize >= config.maxBatchSize ||
               (hasAgingHighPriority && queueSize >= config.minBatchSize)
    }

    private func optimalBatchSize() -> Int {
        // Find the batch size for current memory pressure level
        for (range, size) in config.batchSizeByPressure {
            if range.contains(currentMemoryPressure) {
                return min(size, config.maxBatchSize)
            }
        }
        // Default fallback
        return config.maxBatchSize
    }

    private func processPendingBatch() async throws {
        guard !pendingRequests.isEmpty && !isProcessing else { return }

        isProcessing = true
        defer { isProcessing = false }

        // Take current batch
        let batchSize = min(pendingRequests.count, config.maxBatchSize)
        let batch = Array(pendingRequests.prefix(batchSize))
        pendingRequests.removeFirst(batchSize)

        let texts = batch.map { $0.text }
        let startTime = CFAbsoluteTimeGetCurrent()

        do {
            let embeddings = try await model.embedBatch(texts, options: config.batchOptions)

            // Record timing for performance tracking
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            recordBatchTime(elapsed)

            // Complete all requests
            for (i, request) in batch.enumerated() {
                if i < embeddings.count {
                    request.complete(with: embeddings[i])
                } else {
                    request.fail(with: EmbedKitError.invalidConfiguration("Batch result count mismatch"))
                }
            }

            totalBatches += 1

        } catch {
            // Fail all requests in this batch
            for request in batch {
                request.fail(with: error)
            }
        }

        // If there are more pending requests, schedule another flush
        if !pendingRequests.isEmpty {
            scheduleFlushIfNeeded()
        }
    }

    private func recordBatchTime(_ time: Double) {
        recentBatchTimes.append(time)
        if recentBatchTimes.count > config.performanceWindowSize {
            recentBatchTimes.removeFirst()
        }
    }

    // MARK: - Memory Pressure

    /// Manually set memory pressure level for testing or external monitoring.
    ///
    /// - Parameter pressure: A value from 0.0 (no pressure) to 1.0 (critical).
    public func setMemoryPressure(_ pressure: Float) {
        currentMemoryPressure = max(0, min(1, pressure))
    }
}

// MARK: - Progress Reporting

extension AdaptiveBatcher {
    /// Progress callback type for batch operations
    public typealias ProgressCallback = @Sendable (BatchProgress) -> Void

    /// Embed multiple texts with progress reporting via callback.
    ///
    /// This method processes all texts in batches and reports progress after each batch.
    /// Use this for progress bars or status indicators.
    ///
    /// - Parameters:
    ///   - texts: The texts to embed.
    ///   - priority: Priority level for all requests.
    ///   - onProgress: Callback invoked after each batch completes.
    /// - Returns: All embeddings in the same order as input.
    /// - Throws: Any error from the underlying model.
    ///
    /// ## Example
    /// ```swift
    /// let embeddings = try await batcher.embedWithProgress(texts) { progress in
    ///     print("[\(progress.percentage)%] \(progress.itemsPerSecond ?? 0) items/sec")
    /// }
    /// ```
    public func embedWithProgress(
        _ texts: [String],
        priority: RequestPriority = .normal,
        onProgress: ProgressCallback? = nil
    ) async throws -> [Embedding] {
        guard !texts.isEmpty else { return [] }

        let batchSize = optimalBatchSize()
        let batches = stride(from: 0, to: texts.count, by: batchSize).map {
            Array(texts[$0..<min($0 + batchSize, texts.count)])
        }
        let totalBatches = batches.count
        let startTime = CFAbsoluteTimeGetCurrent()

        var results: [Embedding] = []
        results.reserveCapacity(texts.count)

        var totalTokens = 0

        // Report started
        onProgress?(BatchProgress.started(total: texts.count, totalBatches: totalBatches))

        for (batchIndex, batch) in batches.enumerated() {
            let batchEmbeddings = try await model.embedBatch(batch, options: config.batchOptions)
            results.append(contentsOf: batchEmbeddings)

            // Estimate tokens (rough approximation: 1 token per 4 characters)
            let tokensInBatch = batch.reduce(0) { $0 + max(1, $1.count / 4) }
            totalTokens += tokensInBatch

            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let progress = BatchProgress.batchCompleted(
                itemsCompleted: results.count,
                totalItems: texts.count,
                batchIndex: batchIndex,
                totalBatches: totalBatches,
                batchSize: batch.count,
                tokensInBatch: tokensInBatch,
                totalTokens: totalTokens,
                elapsedTime: elapsed
            )
            onProgress?(progress)
        }

        return results
    }

    /// Embed multiple texts and yield results with progress as an AsyncSequence.
    ///
    /// This method returns a stream that yields (embedding, progress) tuples.
    /// Results are yielded as batches complete, not waiting for all to finish.
    ///
    /// - Parameters:
    ///   - texts: The texts to embed.
    ///   - priority: Priority level for all requests.
    /// - Returns: An async stream of (embeddings, progress) tuples.
    ///
    /// ## Example
    /// ```swift
    /// for try await (embeddings, progress) in batcher.embedBatchStream(texts) {
    ///     print("Batch done: \(embeddings.count) items, \(progress.percentage)% complete")
    ///     allResults.append(contentsOf: embeddings)
    /// }
    /// ```
    public func embedBatchStream(
        _ texts: [String],
        priority: RequestPriority = .normal
    ) -> AsyncThrowingStream<([Embedding], BatchProgress), Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    guard !texts.isEmpty else {
                        continuation.finish()
                        return
                    }

                    let batchSize = optimalBatchSize()
                    let batches = stride(from: 0, to: texts.count, by: batchSize).map {
                        Array(texts[$0..<min($0 + batchSize, texts.count)])
                    }
                    let totalBatches = batches.count
                    let startTime = CFAbsoluteTimeGetCurrent()

                    var processedCount = 0
                    var totalTokens = 0

                    for (batchIndex, batch) in batches.enumerated() {
                        let batchEmbeddings = try await model.embedBatch(batch, options: config.batchOptions)
                        processedCount += batchEmbeddings.count

                        // Estimate tokens
                        let tokensInBatch = batch.reduce(0) { $0 + max(1, $1.count / 4) }
                        totalTokens += tokensInBatch

                        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                        let progress = BatchProgress.batchCompleted(
                            itemsCompleted: processedCount,
                            totalItems: texts.count,
                            batchIndex: batchIndex,
                            totalBatches: totalBatches,
                            batchSize: batch.count,
                            tokensInBatch: tokensInBatch,
                            totalTokens: totalTokens,
                            elapsedTime: elapsed
                        )

                        continuation.yield((batchEmbeddings, progress))
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Embed texts and yield individual embeddings with progress.
    ///
    /// Unlike `embedBatchStream`, this yields one embedding at a time with progress,
    /// making it suitable for progress bars that update per-item.
    ///
    /// - Parameters:
    ///   - texts: The texts to embed.
    /// - Returns: An EmbeddingProgressStream (each element is ([Float], BatchProgress))
    public func embedWithProgressStream(
        _ texts: [String]
    ) -> EmbeddingProgressStream {
        let stream = AsyncThrowingStream<([Float], OperationProgress), Error> { continuation in
            Task {
                do {
                    guard !texts.isEmpty else {
                        continuation.finish()
                        return
                    }

                    let batchSize = optimalBatchSize()
                    let batches = stride(from: 0, to: texts.count, by: batchSize).map {
                        Array(texts[$0..<min($0 + batchSize, texts.count)])
                    }
                    let totalBatches = batches.count
                    let startTime = CFAbsoluteTimeGetCurrent()

                    var processedCount = 0
                    var totalTokens = 0

                    for (batchIndex, batch) in batches.enumerated() {
                        let batchEmbeddings = try await model.embedBatch(batch, options: config.batchOptions)

                        // Yield each embedding individually with updated progress
                        for (itemIndex, embedding) in batchEmbeddings.enumerated() {
                            processedCount += 1
                            let tokensInItem = max(1, batch[itemIndex].count / 4)
                            totalTokens += tokensInItem

                            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                            let itemsPerSec = elapsed > 0 ? Double(processedCount) / elapsed : nil
                            let remaining = texts.count - processedCount
                            let eta = itemsPerSec.map { remaining > 0 && $0 > 0 ? Double(remaining) / $0 : 0 }

                            let progress = OperationProgress(
                                current: processedCount,
                                total: texts.count,
                                phase: batchIndex < totalBatches - 1 ? "Processing" : "Finalizing",
                                message: "Item \(processedCount)/\(texts.count)",
                                estimatedTimeRemaining: eta
                            )

                            continuation.yield((embedding.vector, progress))
                        }
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }

        return EmbeddingProgressStream(stream)
    }
}

// MARK: - Convenience Extensions

extension AdaptiveBatcher {
    /// Embed multiple texts concurrently through the batcher queue.
    ///
    /// Unlike `embedBatch(_:)`, this method submits each text individually
    /// through the queue, allowing them to be batched with other concurrent requests.
    ///
    /// - Parameter texts: The texts to embed.
    /// - Returns: The computed embeddings in the same order as input.
    /// - Throws: Any error from the underlying model.
    public func embedConcurrently(_ texts: [String]) async throws -> [Embedding] {
        try await embedConcurrently(texts, priority: .normal)
    }

    /// Embed multiple texts concurrently through the batcher queue with specified priority.
    ///
    /// - Parameters:
    ///   - texts: The texts to embed.
    ///   - priority: The priority level for all requests.
    /// - Returns: The computed embeddings in the same order as input.
    /// - Throws: Any error from the underlying model.
    public func embedConcurrently(_ texts: [String], priority: RequestPriority) async throws -> [Embedding] {
        try await withThrowingTaskGroup(of: (Int, Embedding).self) { group in
            for (i, text) in texts.enumerated() {
                group.addTask {
                    let embedding = try await self.embed(text, priority: priority)
                    return (i, embedding)
                }
            }

            var results: [(Int, Embedding)] = []
            for try await result in group {
                results.append(result)
            }

            return results.sorted { $0.0 < $1.0 }.map { $0.1 }
        }
    }

    /// Get the highest priority in the current queue.
    public var highestPendingPriority: RequestPriority? {
        pendingRequests.first?.priority
    }

    /// Check if there are any urgent requests pending.
    public var hasUrgentPending: Bool {
        pendingRequests.contains { $0.priority == .urgent }
    }
}

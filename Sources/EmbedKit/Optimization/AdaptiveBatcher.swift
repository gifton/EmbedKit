// EmbedKit - Adaptive Batcher
// Intelligent request batching with memory-aware dynamic sizing

import Foundation
import Dispatch

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

    public init() {}
}

// MARK: - Pending Request

/// Internal representation of a queued embedding request.
fileprivate final class PendingRequest: @unchecked Sendable {
    let text: String
    let submittedAt: CFAbsoluteTime
    private let continuation: UnsafeContinuation<Embedding, Error>

    init(text: String, continuation: UnsafeContinuation<Embedding, Error>) {
        self.text = text
        self.submittedAt = CFAbsoluteTimeGetCurrent()
        self.continuation = continuation
    }

    var age: TimeInterval {
        CFAbsoluteTimeGetCurrent() - submittedAt
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
    }

    private var totalRequests: Int = 0
    private var totalBatches: Int = 0

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
        try await withUnsafeThrowingContinuation { continuation in
            let request = PendingRequest(text: text, continuation: continuation)
            pendingRequests.append(request)
            totalRequests += 1
            scheduleFlushIfNeeded()
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
        return BatcherMetrics(
            totalRequests: totalRequests,
            totalBatches: totalBatches,
            averageBatchSize: avgBatchSize,
            averageBatchLatency: avgLatency,
            currentQueueDepth: pendingRequests.count,
            currentMemoryPressure: currentMemoryPressure
        )
    }

    /// Reset metrics counters.
    public func resetMetrics() {
        totalRequests = 0
        totalBatches = 0
        recentBatchTimes.removeAll()
    }

    // MARK: - Private Implementation

    private func scheduleFlushIfNeeded() {
        guard config.autoFlush else { return }

        // Cancel existing flush task if we're scheduling a new one
        flushTask?.cancel()

        // Check if we should flush immediately
        if shouldProcessBatch() {
            flushTask = Task {
                try? await self.processPendingBatch()
            }
        } else {
            // Schedule a flush after maxLatency
            flushTask = Task {
                try? await Task.sleep(nanoseconds: UInt64(config.maxLatency * 1_000_000_000))
                if !Task.isCancelled {
                    try? await self.processPendingBatch()
                }
            }
        }
    }

    private func shouldProcessBatch() -> Bool {
        guard !pendingRequests.isEmpty else { return false }

        let queueSize = pendingRequests.count
        let oldestAge = pendingRequests.first?.age ?? 0
        let optimalSize = optimalBatchSize()

        // Flush conditions:
        // 1. Queue reached optimal size
        // 2. Oldest request exceeded max latency
        // 3. Memory pressure is high (flush smaller batches more frequently)
        // 4. Queue reached max batch size
        return queueSize >= optimalSize ||
               oldestAge >= config.maxLatency ||
               (currentMemoryPressure > 0.8 && queueSize >= config.minBatchSize) ||
               queueSize >= config.maxBatchSize
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
        try await withThrowingTaskGroup(of: (Int, Embedding).self) { group in
            for (i, text) in texts.enumerated() {
                group.addTask {
                    let embedding = try await self.embed(text)
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
}

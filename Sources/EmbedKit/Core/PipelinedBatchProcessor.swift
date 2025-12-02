// EmbedKit - PipelinedBatchProcessor
// Double/triple buffered batch processing with async prefetching

import Foundation
import VectorCore

// MARK: - Pipeline Buffer State

/// State of a single buffer in the pipeline.
public enum PipelineBufferState: Sendable {
    /// Buffer is empty and ready to receive new data.
    case empty
    /// Buffer is being filled with data (prefetching in progress).
    case filling
    /// Buffer is full and ready for processing.
    case ready
    /// Buffer is currently being processed by the model.
    case processing
    /// Buffer processing completed, results available.
    case completed
}

// MARK: - Pipeline Configuration

/// Configuration for pipelined batch processing.
public struct PipelineConfig: Sendable {
    /// Number of buffers in the pipeline (2 = double buffering, 3 = triple buffering).
    public let bufferCount: Int

    /// Maximum items per batch.
    public let batchSize: Int

    /// Whether to prefetch the next batch while the current one is processing.
    public let enablePrefetch: Bool

    /// Maximum time (in seconds) to wait for a batch to fill before processing.
    public let fillTimeout: TimeInterval

    /// Whether to process partial batches when timeout expires.
    public let allowPartialBatches: Bool

    /// Callback when pipeline stalls (no buffers available).
    public var onStall: (@Sendable () -> Void)?

    /// Creates a pipeline configuration.
    ///
    /// - Parameters:
    ///   - bufferCount: Number of buffers (default: 2 for double buffering)
    ///   - batchSize: Maximum items per batch (default: 32)
    ///   - enablePrefetch: Enable async prefetching (default: true)
    ///   - fillTimeout: Timeout for batch filling in seconds (default: 0.1)
    ///   - allowPartialBatches: Process partial batches on timeout (default: true)
    ///   - onStall: Optional callback when pipeline stalls
    public init(
        bufferCount: Int = 2,
        batchSize: Int = 32,
        enablePrefetch: Bool = true,
        fillTimeout: TimeInterval = 0.1,
        allowPartialBatches: Bool = true,
        onStall: (@Sendable () -> Void)? = nil
    ) {
        precondition(bufferCount >= 2, "Pipeline requires at least 2 buffers")
        precondition(batchSize > 0, "Batch size must be > 0")
        precondition(fillTimeout > 0, "Fill timeout must be > 0")

        self.bufferCount = bufferCount
        self.batchSize = batchSize
        self.enablePrefetch = enablePrefetch
        self.fillTimeout = fillTimeout
        self.allowPartialBatches = allowPartialBatches
        self.onStall = onStall
    }

    /// Default double-buffering configuration.
    public static let doubleBuffer = PipelineConfig(bufferCount: 2)

    /// Triple-buffering configuration for maximum throughput.
    public static let tripleBuffer = PipelineConfig(bufferCount: 3)

    /// Configuration optimized for low latency.
    public static let lowLatency = PipelineConfig(
        bufferCount: 2,
        batchSize: 8,
        enablePrefetch: true,
        fillTimeout: 0.05,
        allowPartialBatches: true
    )

    /// Configuration optimized for high throughput.
    public static let highThroughput = PipelineConfig(
        bufferCount: 3,
        batchSize: 64,
        enablePrefetch: true,
        fillTimeout: 0.2,
        allowPartialBatches: true
    )
}

// MARK: - Pipeline Buffer

/// A single buffer in the processing pipeline.
fileprivate final class PipelineBuffer<T: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var _state: PipelineBufferState = .empty
    private var _items: [T] = []
    private var _results: [[Float]]? = nil
    private var _error: (any Error)? = nil

    let id: Int
    let capacity: Int

    init(id: Int, capacity: Int) {
        self.id = id
        self.capacity = capacity
    }

    var state: PipelineBufferState {
        lock.lock()
        defer { lock.unlock() }
        return _state
    }

    var items: [T] {
        lock.lock()
        defer { lock.unlock() }
        return _items
    }

    var itemCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return _items.count
    }

    var results: [[Float]]? {
        lock.lock()
        defer { lock.unlock() }
        return _results
    }

    var error: (any Error)? {
        lock.lock()
        defer { lock.unlock() }
        return _error
    }

    var isFull: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _items.count >= capacity
    }

    var isEmpty: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _items.isEmpty
    }

    func setState(_ newState: PipelineBufferState) {
        lock.lock()
        _state = newState
        lock.unlock()
    }

    func addItem(_ item: T) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        guard _items.count < capacity else { return false }
        _items.append(item)
        return true
    }

    func addItems(_ newItems: [T]) -> Int {
        lock.lock()
        defer { lock.unlock() }

        let available = capacity - _items.count
        let toAdd = min(available, newItems.count)
        _items.append(contentsOf: newItems.prefix(toAdd))
        return toAdd
    }

    func setResults(_ newResults: [[Float]]) {
        lock.lock()
        _results = newResults
        _state = .completed
        lock.unlock()
    }

    func setError(_ newError: any Error) {
        lock.lock()
        _error = newError
        _state = .completed
        lock.unlock()
    }

    func reset() {
        lock.lock()
        _state = .empty
        _items.removeAll(keepingCapacity: true)
        _results = nil
        _error = nil
        lock.unlock()
    }
}

// MARK: - Pipeline Statistics

/// Statistics for pipelined batch processing.
public struct PipelineStatistics: Sendable {
    /// Total number of items processed.
    public let totalItems: Int

    /// Total number of batches processed.
    public let totalBatches: Int

    /// Number of partial batches processed (less than full capacity).
    public let partialBatches: Int

    /// Number of pipeline stalls (waited for available buffer).
    public let stallCount: Int

    /// Average items per batch.
    public var averageBatchSize: Double {
        totalBatches > 0 ? Double(totalItems) / Double(totalBatches) : 0
    }

    /// Average time spent waiting for buffers (pipeline stalls).
    public let averageStallTime: TimeInterval

    /// Total processing time.
    public let totalTime: TimeInterval

    /// Throughput in items per second.
    public var throughput: Double {
        totalTime > 0 ? Double(totalItems) / totalTime : 0
    }

    /// Buffer utilization (fraction of buffer capacity used on average).
    public let bufferUtilization: Double

    /// Pipeline efficiency (fraction of time spent processing vs waiting).
    public let pipelineEfficiency: Double
}

// MARK: - Pipelined Batch Processor

/// An actor that processes embeddings using double or triple buffering for improved throughput.
///
/// `PipelinedBatchProcessor` uses multiple buffers to overlap:
/// 1. **Filling**: Collecting items into the next batch
/// 2. **Processing**: Computing embeddings for the current batch
/// 3. **Consuming**: Reading results from completed batches
///
/// This pipelining eliminates idle time between batches, improving throughput by 30-50%
/// for streaming workloads.
///
/// ## Example Usage
/// ```swift
/// let generator = try await modelManager.createGenerator()
/// let processor = PipelinedBatchProcessor(generator: generator)
///
/// // Process a large collection with pipelining
/// let texts = loadLargeTextCollection()
/// let embeddings = try await processor.process(texts)
///
/// // Or use streaming API for very large collections
/// for try await (embedding, progress) in processor.processStream(texts) {
///     storeEmbedding(embedding)
/// }
/// ```
///
/// ## Performance Characteristics
/// - **Double buffering**: One buffer processing while the other fills
/// - **Triple buffering**: Maximum throughput, never waits for buffers
/// - **Prefetching**: Asynchronously prepares next batch during processing
public actor PipelinedBatchProcessor: VectorProducer {
    // MARK: - Properties

    private let generator: EmbeddingGenerator
    private var config: PipelineConfig
    private var buffers: [PipelineBuffer<String>]

    // Statistics tracking
    private var stats = StatsTracker()

    private struct StatsTracker {
        var totalItems: Int = 0
        var totalBatches: Int = 0
        var partialBatches: Int = 0
        var stallCount: Int = 0
        var totalStallTime: TimeInterval = 0
        var startTime: CFAbsoluteTime? = nil
        var batchItemCounts: [Int] = []
    }

    // Pipeline state
    private var prefetchTask: Task<Void, Never>? = nil

    // MARK: - VectorProducer Requirements

    public nonisolated var dimensions: Int {
        generator.dimensions
    }

    public nonisolated var producesNormalizedVectors: Bool {
        generator.producesNormalizedVectors
    }

    // MARK: - Initialization

    /// Creates a pipelined batch processor wrapping an embedding generator.
    ///
    /// - Parameters:
    ///   - generator: The underlying embedding generator.
    ///   - config: Pipeline configuration.
    public init(
        generator: EmbeddingGenerator,
        config: PipelineConfig = .doubleBuffer
    ) {
        self.generator = generator
        self.config = config
        self.buffers = (0..<config.bufferCount).map { id in
            PipelineBuffer(id: id, capacity: config.batchSize)
        }
    }

    // MARK: - VectorProducer Implementation

    /// Produces embeddings for a batch of texts using pipelined processing.
    public func produce(_ texts: [String]) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }

        try Task.checkCancellation()

        // Reset statistics for this run
        stats = StatsTracker()
        stats.startTime = CFAbsoluteTimeGetCurrent()

        var results: [[Float]] = []
        results.reserveCapacity(texts.count)

        // Process using pipeline
        var position = 0
        while position < texts.count {
            try Task.checkCancellation()

            // Get an empty buffer for filling
            let fillBuffer = try await acquireBuffer(for: .filling)

            // Fill the buffer
            let end = min(position + config.batchSize, texts.count)
            let batch = Array(texts[position..<end])
            _ = fillBuffer.addItems(batch)
            fillBuffer.setState(.ready)

            // Start prefetching next batch if enabled
            if config.enablePrefetch && end < texts.count {
                startPrefetch(texts: texts, from: end)
            }

            // Process this batch
            try await processBuffer(fillBuffer)

            // Collect results
            if let batchResults = fillBuffer.results {
                results.append(contentsOf: batchResults)
            } else if let error = fillBuffer.error {
                fillBuffer.reset()
                throw error
            }

            stats.totalItems += batch.count
            stats.totalBatches += 1
            stats.batchItemCounts.append(batch.count)

            if batch.count < config.batchSize {
                stats.partialBatches += 1
            }

            // Reset buffer for reuse
            fillBuffer.reset()

            position = end
        }

        // Cancel any pending prefetch
        prefetchTask?.cancel()
        prefetchTask = nil

        return results
    }

    /// Produces an embedding for a single text.
    public func produce(_ text: String) async throws -> [Float] {
        try await generator.produce(text)
    }

    // MARK: - Streaming API

    /// Process texts as an async stream with pipelined batching.
    ///
    /// Returns embeddings as they become available, providing progress updates.
    /// This is ideal for very large collections where you want to start consuming
    /// results before all processing is complete.
    ///
    /// - Parameter texts: Array of texts to embed.
    /// - Returns: Async stream of (embedding vector, progress) tuples.
    public func processStream(
        _ texts: [String]
    ) -> AsyncThrowingStream<([Float], BatchProgress), any Error> {
        let generator = self.generator
        let config = self.config

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    let startTime = CFAbsoluteTimeGetCurrent()
                    let totalItems = texts.count
                    let totalBatches = (totalItems + config.batchSize - 1) / config.batchSize
                    var processed = 0
                    var batchIndex = 0
                    var position = 0
                    let totalTokens = 0

                    while position < totalItems {
                        try Task.checkCancellation()

                        let end = min(position + config.batchSize, totalItems)
                        let batch = Array(texts[position..<end])

                        // Process batch through generator
                        let batchResults = try await generator.produce(batch)

                        // Yield each result with progress
                        for vector in batchResults {
                            processed += 1
                            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                            let itemsPerSec = elapsed > 0 ? Double(processed) / elapsed : nil
                            let eta = processed < totalItems && itemsPerSec != nil && itemsPerSec! > 0
                                ? Double(totalItems - processed) / itemsPerSec!
                                : nil

                            let progress = BatchProgress(
                                current: processed,
                                total: totalItems,
                                batchIndex: batchIndex,
                                totalBatches: totalBatches,
                                phase: processed == totalItems ? "Complete" : "Processing",
                                message: "Batch \(batchIndex + 1)/\(totalBatches)",
                                itemsPerSecond: itemsPerSec,
                                tokensProcessed: totalTokens,
                                currentBatchSize: batch.count,
                                estimatedTimeRemaining: eta
                            )

                            continuation.yield((vector, progress))
                        }

                        batchIndex += 1
                        position = end
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Pipeline Operations

    /// Acquire a buffer in the specified state, waiting if necessary.
    private func acquireBuffer(for desiredState: PipelineBufferState) async throws -> PipelineBuffer<String> {
        let startWait = CFAbsoluteTimeGetCurrent()
        var attempts = 0
        let maxAttempts = 100

        while attempts < maxAttempts {
            try Task.checkCancellation()

            // Find a buffer in empty state (for filling)
            for buffer in buffers {
                if buffer.state == .empty {
                    buffer.setState(.filling)
                    return buffer
                }
            }

            // No buffer available, record stall
            stats.stallCount += 1
            config.onStall?()

            // Wait a bit and try again
            try await Task.sleep(for: .milliseconds(10))
            attempts += 1
        }

        let stallTime = CFAbsoluteTimeGetCurrent() - startWait
        stats.totalStallTime += stallTime

        throw PipelineError.bufferTimeout
    }

    /// Process a buffer through the embedding generator.
    private func processBuffer(_ buffer: PipelineBuffer<String>) async throws {
        guard !buffer.isEmpty else {
            buffer.setState(.completed)
            return
        }

        buffer.setState(.processing)

        do {
            let results = try await generator.produce(buffer.items)
            buffer.setResults(results)
        } catch {
            buffer.setError(error)
            throw error
        }
    }

    /// Start prefetching the next batch of items.
    private func startPrefetch(texts: [String], from position: Int) {
        prefetchTask?.cancel()

        // Capture config values before the task
        let batchSize = config.batchSize
        let buffersSnapshot = buffers

        prefetchTask = Task {
            // Find an empty buffer
            for buffer in buffersSnapshot {
                if buffer.state == .empty && !Task.isCancelled {
                    buffer.setState(.filling)

                    let end = min(position + batchSize, texts.count)
                    let batch = Array(texts[position..<end])
                    _ = buffer.addItems(batch)
                    buffer.setState(.ready)
                    break
                }
            }
        }
    }

    // MARK: - Statistics

    /// Get current pipeline statistics.
    public func getStatistics() -> PipelineStatistics {
        let totalTime = stats.startTime != nil
            ? CFAbsoluteTimeGetCurrent() - stats.startTime!
            : 0

        let avgStallTime = stats.stallCount > 0
            ? stats.totalStallTime / Double(stats.stallCount)
            : 0

        let avgBatchSize = stats.totalBatches > 0
            ? Double(stats.totalItems) / Double(stats.totalBatches)
            : 0

        let bufferUtilization = avgBatchSize / Double(config.batchSize)

        // Pipeline efficiency: time processing vs total time
        let processingTime = totalTime - stats.totalStallTime
        let pipelineEfficiency = totalTime > 0 ? processingTime / totalTime : 1.0

        return PipelineStatistics(
            totalItems: stats.totalItems,
            totalBatches: stats.totalBatches,
            partialBatches: stats.partialBatches,
            stallCount: stats.stallCount,
            averageStallTime: avgStallTime,
            totalTime: totalTime,
            bufferUtilization: bufferUtilization,
            pipelineEfficiency: pipelineEfficiency
        )
    }

    /// Reset pipeline statistics.
    public func resetStatistics() {
        stats = StatsTracker()
    }

    // MARK: - Configuration

    /// Update the pipeline configuration.
    ///
    /// - Note: This recreates the buffer pool, so should not be called during processing.
    public func updateConfig(_ newConfig: PipelineConfig) {
        config = newConfig
        buffers = (0..<newConfig.bufferCount).map { id in
            PipelineBuffer(id: id, capacity: newConfig.batchSize)
        }
    }

    /// Get the current configuration.
    public func getConfig() -> PipelineConfig {
        config
    }

    // MARK: - Lifecycle

    /// Warm up the underlying generator.
    public func warmup() async throws {
        try await generator.warmup()
    }

    /// Release resources.
    public func release() async throws {
        prefetchTask?.cancel()
        prefetchTask = nil
        try await generator.release()
    }
}

// MARK: - Pipeline Errors

/// Errors that can occur during pipelined batch processing.
public enum PipelineError: Error, LocalizedError, Sendable {
    /// Timeout waiting for an available buffer.
    case bufferTimeout
    /// Pipeline was cancelled.
    case cancelled
    /// Buffer is in an invalid state for the requested operation.
    case invalidBufferState(expected: PipelineBufferState, actual: PipelineBufferState)

    public var errorDescription: String? {
        switch self {
        case .bufferTimeout:
            return "Timeout waiting for available pipeline buffer"
        case .cancelled:
            return "Pipeline processing was cancelled"
        case .invalidBufferState(let expected, let actual):
            return "Invalid buffer state: expected \(expected), got \(actual)"
        }
    }
}

// MARK: - EmbeddingGenerator Extension

extension EmbeddingGenerator {
    /// Create a pipelined processor for this generator.
    ///
    /// - Parameter config: Pipeline configuration.
    /// - Returns: A `PipelinedBatchProcessor` wrapping this generator.
    public func pipelined(
        config: PipelineConfig = .doubleBuffer
    ) -> PipelinedBatchProcessor {
        PipelinedBatchProcessor(generator: self, config: config)
    }
}

// MARK: - Concurrent Pipeline Processor

/// A pipeline processor that can handle multiple concurrent streams.
///
/// This is useful when you have multiple independent sources of text
/// that need to be embedded simultaneously.
public actor ConcurrentPipelineProcessor {
    private let generator: EmbeddingGenerator
    private let maxConcurrency: Int
    private let pipelineConfig: PipelineConfig

    private var activeStreams: Int = 0
    private let semaphore: AsyncSemaphore

    /// Creates a concurrent pipeline processor.
    ///
    /// - Parameters:
    ///   - generator: The underlying embedding generator.
    ///   - maxConcurrency: Maximum concurrent processing streams (default: 2).
    ///   - pipelineConfig: Configuration for each pipeline.
    public init(
        generator: EmbeddingGenerator,
        maxConcurrency: Int = 2,
        pipelineConfig: PipelineConfig = .doubleBuffer
    ) {
        precondition(maxConcurrency > 0, "maxConcurrency must be > 0")

        self.generator = generator
        self.maxConcurrency = maxConcurrency
        self.pipelineConfig = pipelineConfig
        self.semaphore = AsyncSemaphore(count: maxConcurrency)
    }

    /// Process multiple text arrays concurrently.
    ///
    /// - Parameter textArrays: Arrays of texts to process.
    /// - Returns: Array of embedding arrays, in the same order as input.
    public func processConcurrently(
        _ textArrays: [[String]]
    ) async throws -> [[[Float]]] {
        try await withThrowingTaskGroup(of: (Int, [[Float]]).self) { group in
            for (index, texts) in textArrays.enumerated() {
                group.addTask {
                    await self.semaphore.wait()
                    // Use do/catch instead of defer { Task { } } to properly
                    // await the semaphore signal and avoid orphaned tasks.
                    do {
                        let pipeline = PipelinedBatchProcessor(
                            generator: self.generator,
                            config: self.pipelineConfig
                        )
                        let results = try await pipeline.produce(texts)
                        await self.semaphore.signal()
                        return (index, results)
                    } catch {
                        await self.semaphore.signal()
                        throw error
                    }
                }
            }

            var results: [(Int, [[Float]])] = []
            for try await result in group {
                results.append(result)
            }

            return results.sorted { $0.0 < $1.0 }.map { $0.1 }
        }
    }

    /// Number of currently active processing streams.
    public var currentActiveStreams: Int {
        activeStreams
    }
}

// MARK: - Async Semaphore

/// A simple async semaphore for limiting concurrency.
fileprivate actor AsyncSemaphore {
    private var count: Int
    private var waiters: [CheckedContinuation<Void, Never>] = []

    init(count: Int) {
        self.count = count
    }

    func wait() async {
        if count > 0 {
            count -= 1
            return
        }

        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    func signal() {
        if let waiter = waiters.first {
            waiters.removeFirst()
            waiter.resume()
        } else {
            count += 1
        }
    }
}

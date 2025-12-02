// EmbedKit - StreamingEmbeddingGenerator
// Advanced streaming API with back-pressure and rate limiting

import Foundation
import VectorCore

// MARK: - Streaming Embedding Generator

/// An embedding generator with built-in back-pressure and rate limiting for streaming workloads.
///
/// `StreamingEmbeddingGenerator` provides production-ready streaming of embeddings with:
/// - **Back-pressure**: Prevents overwhelming the system when producers are faster than consumers
/// - **Rate limiting**: Controls throughput to respect API limits or system capacity
/// - **Adaptive batching**: Dynamically adjusts batch sizes based on load
/// - **Flow control**: Multiple strategies for handling overload conditions
///
/// ## Example Usage
/// ```swift
/// let generator = try await modelManager.createGenerator()
/// let streaming = StreamingEmbeddingGenerator(
///     generator: generator,
///     config: .rateLimited(requestsPerSecond: 100)
/// )
///
/// // Stream embeddings with automatic flow control
/// for try await (embedding, progress) in streaming.stream(largeTextArray) {
///     await store.add(embedding)
/// }
///
/// // Or use with async sequences
/// let embeddings = await streaming.processStream(textStream)
/// ```
public actor StreamingEmbeddingGenerator: VectorProducer {
    // MARK: - Properties

    private let generator: EmbeddingGenerator
    private var config: FlowControlConfig
    private let backPressure: BackPressureController
    private let rateLimiter: EmbeddingRateLimiter?

    // Statistics
    private var stats = StatsTracker()

    private struct StatsTracker {
        var totalSubmitted: Int = 0
        var totalProcessed: Int = 0
        var totalDropped: Int = 0
        var rateLimitHits: Int = 0
        var totalWaitTime: TimeInterval = 0
        var peakQueueDepth: Int = 0
        var processingTimes: [TimeInterval] = []
        var startTime: CFAbsoluteTime? = nil
    }

    // MARK: - Statistics Helpers

    /// Increment rate limit hits counter (callable from async contexts).
    private func recordRateLimitHit() {
        stats.rateLimitHits += 1
    }

    /// Record processed items count.
    private func recordProcessed(count: Int) {
        stats.totalProcessed += count
    }

    /// Record submitted items count.
    private func recordSubmitted(count: Int) {
        stats.totalSubmitted += count
    }

    /// Record wait time.
    private func recordWaitTime(_ time: TimeInterval) {
        stats.totalWaitTime += time
    }

    /// Update peak queue depth.
    private func updatePeakQueueDepth(_ depth: Int) {
        stats.peakQueueDepth = max(stats.peakQueueDepth, depth)
    }

    // MARK: - VectorProducer Requirements

    public nonisolated var dimensions: Int {
        generator.dimensions
    }

    public nonisolated var producesNormalizedVectors: Bool {
        generator.producesNormalizedVectors
    }

    // MARK: - Initialization

    /// Creates a streaming embedding generator.
    ///
    /// - Parameters:
    ///   - generator: The underlying embedding generator.
    ///   - config: Streaming configuration.
    public init(
        generator: EmbeddingGenerator,
        config: FlowControlConfig = .default
    ) {
        self.generator = generator
        self.config = config

        self.backPressure = BackPressureController(
            maxQueueDepth: config.maxQueueDepth,
            strategy: config.backPressureStrategy,
            maxWaitTime: config.maxQueueWait,
            onEvent: config.onBackPressure
        )

        self.rateLimiter = config.rateLimitStrategy.map {
            EmbeddingRateLimiter(strategy: $0)
        }
    }

    // MARK: - VectorProducer Implementation

    /// Produces embeddings for a batch of texts with streaming flow control.
    public func produce(_ texts: [String]) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }

        try Task.checkCancellation()
        stats.startTime = CFAbsoluteTimeGetCurrent()

        var results: [[Float]] = []
        results.reserveCapacity(texts.count)

        // Process in batches with back-pressure
        var position = 0
        while position < texts.count {
            try Task.checkCancellation()

            // Calculate batch size first so we can use it for rate limiting cost
            let end = min(position + config.batchSize, texts.count)
            let batch = Array(texts[position..<end])
            let batchCost = Double(batch.count)

            // Apply rate limiting if configured (cost = batch size)
            if let limiter = rateLimiter {
                let startWait = CFAbsoluteTimeGetCurrent()
                let allowed = await limiter.allowRequest(cost: batchCost)
                if !allowed {
                    stats.rateLimitHits += 1
                    try await limiter.waitForPermit(cost: batchCost)
                }
                stats.totalWaitTime += CFAbsoluteTimeGetCurrent() - startWait
            }

            // Acquire back-pressure token
            let token = try await backPressure.acquire()
            defer { token.release() }

            stats.totalSubmitted += batch.count

            // Track peak queue depth
            let currentDepth = await backPressure.queueDepth
            stats.peakQueueDepth = max(stats.peakQueueDepth, currentDepth)

            let batchStart = CFAbsoluteTimeGetCurrent()
            let batchResults = try await generator.produce(batch)
            let batchTime = CFAbsoluteTimeGetCurrent() - batchStart

            results.append(contentsOf: batchResults)
            stats.totalProcessed += batchResults.count
            stats.processingTimes.append(batchTime)

            position = end
        }

        return results
    }

    /// Produces an embedding for a single text.
    public func produce(_ text: String) async throws -> [Float] {
        // Apply rate limiting
        if let limiter = rateLimiter {
            if !(await limiter.allowRequest()) {
                stats.rateLimitHits += 1
                try await limiter.waitForPermit()
            }
        }

        // Acquire back-pressure token
        let token = try await backPressure.acquire()
        defer { token.release() }

        stats.totalSubmitted += 1
        let result = try await generator.produce(text)
        stats.totalProcessed += 1

        return result
    }

    // MARK: - Streaming API

    /// Streams embeddings for an array of texts with progress updates.
    ///
    /// - Parameter texts: Array of texts to embed.
    /// - Returns: Async stream of (embedding, progress) tuples.
    public func stream(
        _ texts: [String]
    ) -> AsyncThrowingStream<([Float], BatchProgress), any Error> {
        let generator = self.generator
        let config = self.config
        let rateLimiter = self.rateLimiter
        let backPressure = self.backPressure
        let streamingSelf = self  // Capture self for stats recording

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    let startTime = CFAbsoluteTimeGetCurrent()
                    let totalItems = texts.count
                    let totalBatches = (totalItems + config.batchSize - 1) / config.batchSize
                    var processed = 0
                    var batchIndex = 0
                    var position = 0

                    while position < totalItems {
                        try Task.checkCancellation()

                        // Calculate batch first for rate limiting cost
                        let end = min(position + config.batchSize, totalItems)
                        let batch = Array(texts[position..<end])
                        let batchCost = Double(batch.count)

                        // Rate limiting (cost = batch size)
                        if let limiter = rateLimiter {
                            let waitStart = CFAbsoluteTimeGetCurrent()
                            if !(await limiter.allowRequest(cost: batchCost)) {
                                await streamingSelf.recordRateLimitHit()
                                try await limiter.waitForPermit(cost: batchCost)
                            }
                            await streamingSelf.recordWaitTime(CFAbsoluteTimeGetCurrent() - waitStart)
                        }

                        // Back-pressure
                        let token = try await backPressure.acquire()
                        defer { token.release() }
                        await streamingSelf.recordSubmitted(count: batch.count)
                        let batchResults = try await generator.produce(batch)
                        await streamingSelf.recordProcessed(count: batchResults.count)

                        // Yield each result with progress
                        for vector in batchResults {
                            processed += 1
                            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                            let itemsPerSec = elapsed > 0 ? Double(processed) / elapsed : nil
                            let eta = processed < totalItems && itemsPerSec != nil && itemsPerSec! > 0
                                ? Double(totalItems - processed) / itemsPerSec!
                                : nil

                            let bpStats = await backPressure.getStatistics()

                            let progress = BatchProgress(
                                current: processed,
                                total: totalItems,
                                batchIndex: batchIndex,
                                totalBatches: totalBatches,
                                phase: processed == totalItems ? "Complete" : "Processing",
                                message: "Batch \(batchIndex + 1)/\(totalBatches), Queue: \(bpStats.currentDepth)/\(bpStats.maxQueueDepth)",
                                itemsPerSecond: itemsPerSec,
                                tokensProcessed: 0,
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

    /// Processes an async stream of texts, returning embeddings as they complete.
    ///
    /// - Parameter textStream: Async stream of texts to embed.
    /// - Returns: Async stream of embedding vectors.
    public func processStream<S: AsyncSequence>(
        _ textStream: S
    ) -> AsyncThrowingStream<[Float], any Error> where S.Element == String, S: Sendable {
        let generator = self.generator
        let config = self.config
        let rateLimiter = self.rateLimiter
        let backPressure = self.backPressure
        let streamingSelf = self  // Capture self for stats recording

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    var batch: [String] = []
                    batch.reserveCapacity(config.batchSize)

                    for try await text in textStream {
                        try Task.checkCancellation()

                        batch.append(text)

                        // Process when batch is full
                        if batch.count >= config.batchSize {
                            // Rate limiting (cost = batch size)
                            let batchCost = Double(batch.count)
                            if let limiter = rateLimiter {
                                let waitStart = CFAbsoluteTimeGetCurrent()
                                if !(await limiter.allowRequest(cost: batchCost)) {
                                    await streamingSelf.recordRateLimitHit()
                                    try await limiter.waitForPermit(cost: batchCost)
                                }
                                await streamingSelf.recordWaitTime(CFAbsoluteTimeGetCurrent() - waitStart)
                            }

                            // Back-pressure
                            let token = try await backPressure.acquire()

                            await streamingSelf.recordSubmitted(count: batch.count)
                            let results = try await generator.produce(batch)
                            await streamingSelf.recordProcessed(count: results.count)
                            token.release()

                            for vector in results {
                                continuation.yield(vector)
                            }

                            batch.removeAll(keepingCapacity: true)
                        }
                    }

                    // Process remaining items
                    if !batch.isEmpty {
                        // Rate limiting (cost = remaining batch size)
                        let batchCost = Double(batch.count)
                        if let limiter = rateLimiter {
                            let waitStart = CFAbsoluteTimeGetCurrent()
                            if !(await limiter.allowRequest(cost: batchCost)) {
                                await streamingSelf.recordRateLimitHit()
                                try await limiter.waitForPermit(cost: batchCost)
                            }
                            await streamingSelf.recordWaitTime(CFAbsoluteTimeGetCurrent() - waitStart)
                        }

                        let token = try await backPressure.acquire()
                        await streamingSelf.recordSubmitted(count: batch.count)
                        let results = try await generator.produce(batch)
                        await streamingSelf.recordProcessed(count: results.count)
                        token.release()

                        for vector in results {
                            continuation.yield(vector)
                        }
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Processes texts concurrently with controlled parallelism.
    ///
    /// - Parameters:
    ///   - texts: Array of texts to embed.
    ///   - maxConcurrency: Maximum concurrent batches (overrides config if specified).
    /// - Returns: Array of embedding vectors in order.
    public func processConcurrently(
        _ texts: [String],
        maxConcurrency: Int? = nil
    ) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }

        let concurrency = maxConcurrency ?? config.maxConcurrency
        let batchSize = config.batchSize

        // Split into batches
        var batches: [[String]] = []
        var position = 0
        while position < texts.count {
            let end = min(position + batchSize, texts.count)
            batches.append(Array(texts[position..<end]))
            position = end
        }

        // Process batches with controlled concurrency
        return try await withThrowingTaskGroup(of: (Int, [[Float]]).self) { group in
            var activeCount = 0
            var batchIndex = 0
            var results: [(Int, [[Float]])] = []

            // Submit initial batches
            while activeCount < concurrency && batchIndex < batches.count {
                let batch = batches[batchIndex]
                let index = batchIndex

                group.addTask {
                    // Rate limiting (cost = batch size)
                    let batchCost = Double(batch.count)
                    if let limiter = self.rateLimiter {
                        let startWait = CFAbsoluteTimeGetCurrent()
                        if !(await limiter.allowRequest(cost: batchCost)) {
                            await self.recordRateLimitHit()
                            try await limiter.waitForPermit(cost: batchCost)
                        }
                        await self.recordWaitTime(CFAbsoluteTimeGetCurrent() - startWait)
                    }

                    // Back-pressure
                    let token = try await self.backPressure.acquire()
                    defer { token.release() }

                    await self.recordSubmitted(count: batch.count)
                    let embeddings = try await self.generator.produce(batch)
                    await self.recordProcessed(count: embeddings.count)
                    return (index, embeddings)
                }

                activeCount += 1
                batchIndex += 1
            }

            // Process results and submit more batches
            for try await result in group {
                results.append(result)
                activeCount -= 1

                // Submit another batch if available
                if batchIndex < batches.count {
                    let batch = batches[batchIndex]
                    let index = batchIndex

                    group.addTask {
                        // Rate limiting (cost = batch size)
                        let batchCost = Double(batch.count)
                        if let limiter = self.rateLimiter {
                            let startWait = CFAbsoluteTimeGetCurrent()
                            if !(await limiter.allowRequest(cost: batchCost)) {
                                await self.recordRateLimitHit()
                                try await limiter.waitForPermit(cost: batchCost)
                            }
                            await self.recordWaitTime(CFAbsoluteTimeGetCurrent() - startWait)
                        }

                        let token = try await self.backPressure.acquire()
                        defer { token.release() }

                        await self.recordSubmitted(count: batch.count)
                        let embeddings = try await self.generator.produce(batch)
                        await self.recordProcessed(count: embeddings.count)
                        return (index, embeddings)
                    }

                    activeCount += 1
                    batchIndex += 1
                }
            }

            // Sort by index and flatten
            return results.sorted { $0.0 < $1.0 }.flatMap { $0.1 }
        }
    }

    // MARK: - Statistics

    /// Get current streaming statistics.
    public func getStatistics() async -> StreamingStatistics {
        let totalTime = stats.startTime != nil
            ? CFAbsoluteTimeGetCurrent() - stats.startTime!
            : 0

        let avgProcessingTime = stats.processingTimes.isEmpty
            ? 0
            : stats.processingTimes.reduce(0, +) / Double(stats.processingTimes.count)

        let throughput = totalTime > 0
            ? Double(stats.totalProcessed) / totalTime
            : 0

        let bpStats = await backPressure.getStatistics()

        return StreamingStatistics(
            totalSubmitted: stats.totalSubmitted,
            totalProcessed: stats.totalProcessed,
            totalDropped: stats.totalDropped,
            rateLimitHits: stats.rateLimitHits,
            totalWaitTime: stats.totalWaitTime,
            currentQueueDepth: bpStats.currentDepth,
            peakQueueDepth: stats.peakQueueDepth,
            averageProcessingTime: avgProcessingTime,
            throughput: throughput
        )
    }

    /// Reset statistics.
    public func resetStatistics() {
        stats = StatsTracker()
    }

    /// Get rate limit status.
    public func getRateLimitStatus() async -> RateLimitStatus? {
        await rateLimiter?.getStatus()
    }

    /// Get back-pressure statistics.
    public func getBackPressureStatistics() async -> BackPressureStatistics {
        await backPressure.getStatistics()
    }

    // MARK: - Configuration

    /// Update configuration.
    public func updateConfig(_ newConfig: FlowControlConfig) {
        config = newConfig
    }

    /// Get current configuration.
    public func getConfig() -> FlowControlConfig {
        config
    }

    // MARK: - Lifecycle

    /// Warm up the underlying generator.
    public func warmup() async throws {
        try await generator.warmup()
    }

    /// Release resources.
    public func release() async throws {
        try await generator.release()
    }
}

// MARK: - EmbeddingGenerator Extension

extension EmbeddingGenerator {
    /// Create a streaming generator with back-pressure and rate limiting.
    ///
    /// - Parameter config: Streaming configuration.
    /// - Returns: A `StreamingEmbeddingGenerator` wrapping this generator.
    public func streaming(
        config: FlowControlConfig = .default
    ) -> StreamingEmbeddingGenerator {
        StreamingEmbeddingGenerator(generator: self, config: config)
    }
}

// EmbedKit - EmbeddingGenerator
// High-level actor for generating embeddings with progress tracking

import Foundation
import VectorCore
import Logging

/// Actor for generating text embeddings with progress tracking and cancellation support.
///
/// `EmbeddingGenerator` provides a high-level API for embedding generation that conforms
/// to the `VectorProducer` protocol from VectorCore, enabling seamless integration with
/// VectorIndex and other VSK components.
///
/// ## Features
/// - **VectorProducer Conformance**: Works with VectorIndex for semantic search
/// - **Progress Streaming**: Track embedding progress with `generateWithProgress`
/// - **Cancellation**: Respects Swift Task cancellation
/// - **Batch Optimization**: Efficient batching with configurable options
///
/// ## Example Usage
/// ```swift
/// let generator = try await modelManager.createGenerator()
///
/// // Simple batch embedding (VectorProducer)
/// let vectors = try await generator.produce(["Hello", "World"])
///
/// // With progress tracking
/// for try await (embedding, progress) in generator.generateWithProgress(texts) {
///     updateUI(progress.percentage)
///     results.append(embedding)
/// }
/// ```
public actor EmbeddingGenerator: VectorProducer {
    // MARK: - Properties

    private let model: any EmbeddingModel
    private let configuration: EmbeddingConfiguration
    private let batchOptions: BatchOptions
    private let logger = Logger(label: "EmbedKit.EmbeddingGenerator")

    // MARK: - Internal Accessors (for extensions in other files)

    /// Internal access to the underlying model.
    internal var internalModel: any EmbeddingModel { model }

    /// Internal access to batch options.
    internal var internalBatchOptions: BatchOptions { batchOptions }

    /// Internal access to configuration.
    internal var internalConfiguration: EmbeddingConfiguration { configuration }

    // MARK: - VectorProducer Requirements

    /// The dimensionality of produced embedding vectors.
    public nonisolated var dimensions: Int {
        model.dimensions
    }

    /// Whether output vectors are L2-normalized.
    public nonisolated var producesNormalizedVectors: Bool {
        configuration.normalizeOutput
    }

    // MARK: - Initialization

    /// Creates an embedding generator wrapping the given model.
    ///
    /// - Parameters:
    ///   - model: The underlying embedding model to use
    ///   - configuration: Embedding configuration (defaults to model's config or `.default`)
    ///   - batchOptions: Batch processing options (defaults to `.default`)
    public init(
        model: any EmbeddingModel,
        configuration: EmbeddingConfiguration = .default,
        batchOptions: BatchOptions = .default
    ) {
        self.model = model
        self.configuration = configuration
        self.batchOptions = batchOptions
    }

    // MARK: - VectorProducer Implementation

    /// Produces embeddings for a batch of texts.
    ///
    /// This is the primary `VectorProducer` method for batch embedding.
    /// Supports Task cancellation - throws `CancellationError` if cancelled.
    ///
    /// - Parameter texts: Array of texts to embed
    /// - Returns: Array of embedding vectors (same order as input)
    /// - Throws: `EmbedKitError` on failure, `CancellationError` if cancelled
    public func produce(_ texts: [String]) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }

        // Check for cancellation before starting
        try Task.checkCancellation()

        let embeddings = try await model.embedBatch(texts, options: batchOptions)

        // Check for cancellation after completion
        try Task.checkCancellation()

        return embeddings.map { $0.vector }
    }

    /// Produces an embedding for a single text.
    ///
    /// - Parameter text: The text to embed
    /// - Returns: The embedding vector
    /// - Throws: `EmbedKitError` on failure
    public func produce(_ text: String) async throws -> [Float] {
        try Task.checkCancellation()
        let embedding = try await model.embed(text)
        return embedding.vector
    }

    // MARK: - Extended API with Progress

    /// Generates embeddings with detailed progress tracking.
    ///
    /// Returns a `ProgressStream` that yields each embedding along with
    /// progress information. Supports cancellation by breaking the loop.
    ///
    /// - Parameter texts: Array of texts to embed
    /// - Returns: Stream of (embedding vector, progress) tuples
    ///
    /// ## Example
    /// ```swift
    /// var embeddings: [[Float]] = []
    /// for try await (vector, progress) in await generator.generateWithProgress(texts) {
    ///     print("Progress: \(progress.percentage)%")
    ///     embeddings.append(vector)
    /// }
    /// ```
    public func generateWithProgress(
        _ texts: [String]
    ) -> ProgressStream<[Float]> {
        let model = self.model
        let batchOptions = self.batchOptions

        let (stream, continuation) = AsyncThrowingStream.makeStream(
            of: ([Float], OperationProgress).self
        )

        Task {
            do {
                try await Self.processWithProgressStatic(
                    texts: texts,
                    model: model,
                    batchOptions: batchOptions,
                    continuation: continuation
                )
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }

        return ProgressStream(stream)
    }

    /// Generates full `Embedding` objects with progress tracking.
    ///
    /// Unlike `generateWithProgress`, this returns complete `Embedding` objects
    /// with metadata (token count, processing time, etc.).
    ///
    /// - Parameter texts: Array of texts to embed
    /// - Returns: Stream of (Embedding, progress) tuples
    public func generateEmbeddingsWithProgress(
        _ texts: [String]
    ) -> AsyncThrowingStream<(Embedding, BatchProgress), any Error> {
        let model = self.model
        let batchOptions = self.batchOptions

        let (stream, continuation) = AsyncThrowingStream.makeStream(
            of: (Embedding, BatchProgress).self
        )

        Task {
            do {
                try await Self.processEmbeddingsWithProgressStatic(
                    texts: texts,
                    model: model,
                    batchOptions: batchOptions,
                    continuation: continuation
                )
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }

        return stream
    }

    // MARK: - Batch Processing with Progress

    /// Generates embeddings in batches with progress, returning all at once.
    ///
    /// Use this when you want batch efficiency but need all results together.
    /// Progress updates are provided via the optional callback.
    ///
    /// - Parameters:
    ///   - texts: Array of texts to embed
    ///   - onProgress: Optional callback for progress updates
    /// - Returns: Array of embedding vectors
    /// - Throws: `EmbedKitError` on failure
    public func generateBatch(
        _ texts: [String],
        onProgress: ((BatchProgress) -> Void)? = nil
    ) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }

        let startTime = CFAbsoluteTimeGetCurrent()
        let totalItems = texts.count
        let batchSize = batchOptions.maxBatchSize
        let totalBatches = (totalItems + batchSize - 1) / batchSize

        // Report start
        onProgress?(.started(total: totalItems, totalBatches: totalBatches))

        var results: [[Float]] = []
        results.reserveCapacity(totalItems)
        var totalTokens = 0

        for batchIndex in 0..<totalBatches {
            try Task.checkCancellation()

            let start = batchIndex * batchSize
            let end = min(start + batchSize, totalItems)
            let batch = Array(texts[start..<end])

            let embeddings = try await model.embedBatch(batch, options: batchOptions)
            let batchTokens = embeddings.reduce(0) { $0 + $1.metadata.tokenCount }
            totalTokens += batchTokens

            results.append(contentsOf: embeddings.map { $0.vector })

            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let progress = BatchProgress.batchCompleted(
                itemsCompleted: results.count,
                totalItems: totalItems,
                batchIndex: batchIndex,
                totalBatches: totalBatches,
                batchSize: batch.count,
                tokensInBatch: batchTokens,
                totalTokens: totalTokens,
                elapsedTime: elapsed
            )
            onProgress?(progress)
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        let throughput = elapsed > 0 ? Double(totalItems) / elapsed : nil
        onProgress?(.completed(
            total: totalItems,
            totalBatches: totalBatches,
            tokensProcessed: totalTokens,
            itemsPerSecond: throughput
        ))

        return results
    }

    // MARK: - Convenience Methods

    /// The underlying model's identifier.
    public var modelID: ModelID {
        get async { model.id }
    }

    /// Current model metrics.
    public var metrics: ModelMetrics {
        get async { await model.metrics }
    }

    /// Hints for consumers about this producer's characteristics.
    public var hints: VectorProducerHints {
        VectorProducerHints(
            dimensions: dimensions,
            isNormalized: producesNormalizedVectors,
            optimalBatchSize: batchOptions.maxBatchSize,
            maxBatchSize: batchOptions.maxBatchSize
        )
    }

    /// Warms up the model for faster first inference.
    public func warmup() async throws {
        try await model.warmup()
    }

    /// Releases model resources.
    public func release() async throws {
        try await model.release()
    }

    // MARK: - ID-Preserving Batch Methods

    /// Produces embeddings for items with associated identifiers.
    ///
    /// This method preserves the association between each input item's ID and its
    /// resulting embedding vector, eliminating the need for manual `zip` operations.
    /// The output order is guaranteed to match the input order.
    ///
    /// - Parameter items: Array of (id, text) tuples to embed
    /// - Returns: Array of (id, vector) tuples in the same order as input
    /// - Throws: `EmbedKitError` on failure, `CancellationError` if cancelled
    ///
    /// ## Example
    /// ```swift
    /// // Embed fragments with their UUIDs
    /// let fragments = entry.fragments.map { ($0.id, $0.content) }
    /// let results = try await generator.produceWithIDs(fragments)
    ///
    /// // results[i].id == fragments[i].id guaranteed
    /// for (id, vector) in results {
    ///     fragmentVectors[id] = vector
    /// }
    /// ```
    public func produceWithIDs<ID: Hashable & Sendable>(
        _ items: [(id: ID, text: String)]
    ) async throws -> [(id: ID, vector: [Float])] {
        guard !items.isEmpty else { return [] }

        try Task.checkCancellation()

        let texts = items.map(\.text)
        let vectors = try await produce(texts)

        try Task.checkCancellation()

        // zip maintains order - vectors[i] corresponds to items[i]
        return zip(items, vectors).map { item, vector in
            (id: item.id, vector: vector)
        }
    }

    /// Produces embeddings with IDs and progress tracking via streaming.
    ///
    /// Returns an async stream that yields each embedding paired with its ID
    /// and progress information. Useful for large batches where you want to
    /// process results incrementally.
    ///
    /// - Parameter items: Array of (id, text) tuples to embed
    /// - Returns: Async stream of (id, vector, progress) tuples
    ///
    /// ## Example
    /// ```swift
    /// var embeddings: [UUID: [Float]] = [:]
    /// for try await (id, vector, progress) in await generator.produceWithIDsAndProgress(fragments) {
    ///     embeddings[id] = vector
    ///     updateUI(progress.percentage)
    /// }
    /// ```
    public func produceWithIDsAndProgress<ID: Hashable & Sendable>(
        _ items: [(id: ID, text: String)]
    ) -> AsyncThrowingStream<(id: ID, vector: [Float], progress: OperationProgress), Error> {
        let items = items
        let model = self.model
        let batchOptions = self.batchOptions

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    try await Self.processWithIDsAndProgressStatic(
                        items: items,
                        model: model,
                        batchOptions: batchOptions,
                        continuation: continuation
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Generates embeddings with IDs in batches, with progress callback.
    ///
    /// Like `generateBatch`, but preserves ID associations. Use this when you
    /// want batch efficiency with all results returned at once, plus progress updates.
    ///
    /// - Parameters:
    ///   - items: Array of (id, text) tuples to embed
    ///   - onProgress: Optional callback for progress updates
    /// - Returns: Array of (id, vector) tuples in the same order as input
    /// - Throws: `EmbedKitError` on failure
    ///
    /// ## Example
    /// ```swift
    /// let results = try await generator.generateBatchWithIDs(fragments) { progress in
    ///     print("Progress: \(progress.percentage)%")
    /// }
    /// ```
    public func generateBatchWithIDs<ID: Hashable & Sendable>(
        _ items: [(id: ID, text: String)],
        onProgress: ((BatchProgress) -> Void)? = nil
    ) async throws -> [(id: ID, vector: [Float])] {
        guard !items.isEmpty else { return [] }

        let startTime = CFAbsoluteTimeGetCurrent()
        let totalItems = items.count
        let batchSize = batchOptions.maxBatchSize
        let totalBatches = (totalItems + batchSize - 1) / batchSize

        // Report start
        onProgress?(.started(total: totalItems, totalBatches: totalBatches))

        var results: [(id: ID, vector: [Float])] = []
        results.reserveCapacity(totalItems)
        var totalTokens = 0

        for batchIndex in 0..<totalBatches {
            try Task.checkCancellation()

            let start = batchIndex * batchSize
            let end = min(start + batchSize, totalItems)
            let batchItems = Array(items[start..<end])
            let batchTexts = batchItems.map(\.text)

            let embeddings = try await model.embedBatch(batchTexts, options: batchOptions)
            let batchTokens = embeddings.reduce(0) { $0 + $1.metadata.tokenCount }
            totalTokens += batchTokens

            // Pair each embedding with its ID, maintaining order
            for (item, embedding) in zip(batchItems, embeddings) {
                results.append((id: item.id, vector: embedding.vector))
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let progress = BatchProgress.batchCompleted(
                itemsCompleted: results.count,
                totalItems: totalItems,
                batchIndex: batchIndex,
                totalBatches: totalBatches,
                batchSize: batchItems.count,
                tokensInBatch: batchTokens,
                totalTokens: totalTokens,
                elapsedTime: elapsed
            )
            onProgress?(progress)
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        let throughput = elapsed > 0 ? Double(totalItems) / elapsed : nil
        onProgress?(.completed(
            total: totalItems,
            totalBatches: totalBatches,
            tokensProcessed: totalTokens,
            itemsPerSecond: throughput
        ))

        return results
    }

    // MARK: - Private Static Implementation

    /// Static method to process embeddings with progress (avoids actor isolation issues)
    private static func processWithProgressStatic(
        texts: [String],
        model: any EmbeddingModel,
        batchOptions: BatchOptions,
        continuation: AsyncThrowingStream<([Float], OperationProgress), any Error>.Continuation
    ) async throws {
        guard !texts.isEmpty else { return }

        let startTime = CFAbsoluteTimeGetCurrent()
        let totalItems = texts.count
        let batchSize = batchOptions.maxBatchSize
        var processed = 0

        // Process in batches
        var position = 0
        while position < totalItems {
            try Task.checkCancellation()

            let end = min(position + batchSize, totalItems)
            let batch = Array(texts[position..<end])

            let embeddings = try await model.embedBatch(batch, options: batchOptions)

            // Yield each embedding with progress
            for embedding in embeddings {
                processed += 1
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let eta = processed < totalItems && elapsed > 0
                    ? elapsed / Double(processed) * Double(totalItems - processed)
                    : nil

                let progress = OperationProgress(
                    current: processed,
                    total: totalItems,
                    phase: processed == totalItems ? "Complete" : "Processing",
                    message: "Embedded \(processed)/\(totalItems)",
                    estimatedTimeRemaining: eta
                )

                continuation.yield((embedding.vector, progress))
            }

            position = end
        }
    }

    /// Static method to process embeddings with batch progress (avoids actor isolation issues)
    private static func processEmbeddingsWithProgressStatic(
        texts: [String],
        model: any EmbeddingModel,
        batchOptions: BatchOptions,
        continuation: AsyncThrowingStream<(Embedding, BatchProgress), any Error>.Continuation
    ) async throws {
        guard !texts.isEmpty else { return }

        let startTime = CFAbsoluteTimeGetCurrent()
        let totalItems = texts.count
        let batchSize = batchOptions.maxBatchSize
        let totalBatches = (totalItems + batchSize - 1) / batchSize
        var processed = 0
        var totalTokens = 0
        var batchIndex = 0

        // Process in batches
        var position = 0
        while position < totalItems {
            try Task.checkCancellation()

            let end = min(position + batchSize, totalItems)
            let batch = Array(texts[position..<end])

            let embeddings = try await model.embedBatch(batch, options: batchOptions)
            let batchTokens = embeddings.reduce(0) { $0 + $1.metadata.tokenCount }
            totalTokens += batchTokens

            // Yield each embedding with progress
            for embedding in embeddings {
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

                continuation.yield((embedding, progress))
            }

            batchIndex += 1
            position = end
        }
    }

    /// Static method to process embeddings with IDs and progress (avoids actor isolation issues)
    private static func processWithIDsAndProgressStatic<ID: Hashable & Sendable>(
        items: [(id: ID, text: String)],
        model: any EmbeddingModel,
        batchOptions: BatchOptions,
        continuation: AsyncThrowingStream<(id: ID, vector: [Float], progress: OperationProgress), Error>.Continuation
    ) async throws {
        guard !items.isEmpty else { return }

        let startTime = CFAbsoluteTimeGetCurrent()
        let totalItems = items.count
        let batchSize = batchOptions.maxBatchSize
        var processed = 0

        // Process in batches
        var position = 0
        while position < totalItems {
            try Task.checkCancellation()

            let end = min(position + batchSize, totalItems)
            let batchItems = Array(items[position..<end])
            let batchTexts = batchItems.map(\.text)

            let embeddings = try await model.embedBatch(batchTexts, options: batchOptions)

            // Yield each embedding paired with its ID
            for (item, embedding) in zip(batchItems, embeddings) {
                processed += 1
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let eta = processed < totalItems && elapsed > 0
                    ? elapsed / Double(processed) * Double(totalItems - processed)
                    : nil

                let progress = OperationProgress(
                    current: processed,
                    total: totalItems,
                    phase: processed == totalItems ? "Complete" : "Processing",
                    message: "Embedded \(processed)/\(totalItems)",
                    estimatedTimeRemaining: eta
                )

                continuation.yield((id: item.id, vector: embedding.vector, progress: progress))
            }

            position = end
        }
    }
}

// MARK: - Generator Configuration

/// Configuration for creating an EmbeddingGenerator.
public struct GeneratorConfiguration: Sendable {
    /// Embedding configuration for the underlying model
    public let embedding: EmbeddingConfiguration

    /// Batch processing options
    public let batch: BatchOptions

    /// Creates a generator configuration.
    ///
    /// - Parameters:
    ///   - embedding: Embedding configuration (default: `.default`)
    ///   - batch: Batch options (default: `.default`)
    public init(
        embedding: EmbeddingConfiguration = .default,
        batch: BatchOptions = .default
    ) {
        self.embedding = embedding
        self.batch = batch
    }

    /// Default configuration for most use cases.
    public static let `default` = GeneratorConfiguration()

    /// Configuration optimized for high throughput.
    public static let highThroughput = GeneratorConfiguration(
        embedding: .performant,
        batch: .highThroughput
    )

    /// Configuration optimized for low latency.
    public static let lowLatency = GeneratorConfiguration(
        embedding: .default,
        batch: .lowLatency
    )

    /// Configuration for semantic search applications.
    public static func forSemanticSearch(maxLength: Int = 512) -> GeneratorConfiguration {
        GeneratorConfiguration(
            embedding: .forSemanticSearch(maxLength: maxLength),
            batch: .default
        )
    }

    /// Configuration for RAG applications.
    public static func forRAG(chunkSize: Int = 256) -> GeneratorConfiguration {
        GeneratorConfiguration(
            embedding: .forRAG(chunkSize: chunkSize),
            batch: .highThroughput
        )
    }
}

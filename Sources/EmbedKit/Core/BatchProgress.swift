// EmbedKit - BatchProgress
// Embedding-specific progress tracking with throughput metrics

import Foundation
import VectorCore

/// Progress information for batch embedding operations.
///
/// Extends `OperationProgress` with embedding-specific metrics like throughput,
/// batch indexing, and token counts. Use this for detailed progress tracking
/// in long-running embedding operations.
///
/// ## Example
/// ```swift
/// for try await (embedding, progress) in generator.generateWithProgress(texts) {
///     print("[\(progress.percentage)%] \(progress.itemsPerSecond ?? 0) items/sec")
///     results.append(embedding)
/// }
/// ```
public struct BatchProgress: Sendable, Equatable {
    // MARK: - Core Progress (delegates to OperationProgress)

    /// The underlying operation progress
    public let base: OperationProgress

    // MARK: - Batch-Specific Fields

    /// Current batch index (0-based)
    public let batchIndex: Int

    /// Total number of batches
    public let totalBatches: Int

    /// Items processed per second (nil if not yet calculable)
    public let itemsPerSecond: Double?

    /// Total tokens processed so far
    public let tokensProcessed: Int

    /// Current batch size
    public let currentBatchSize: Int

    // MARK: - Convenience Accessors (delegate to base)

    /// Current item being processed
    public var current: Int { base.current }

    /// Total items to process
    public var total: Int { base.total }

    /// Progress as a fraction [0, 1]
    public var fraction: Double { base.fraction }

    /// Progress as a percentage [0, 100]
    public var percentage: Int { base.percentage }

    /// Whether processing is complete
    public var isComplete: Bool { base.isComplete }

    /// Current processing phase
    public var phase: String { base.phase }

    /// Optional message about current operation
    public var message: String? { base.message }

    /// Timestamp when this progress was created
    public var timestamp: Date { base.timestamp }

    /// Estimated time remaining in seconds
    public var estimatedTimeRemaining: TimeInterval? { base.estimatedTimeRemaining }

    // MARK: - Initialization

    /// Creates a batch progress instance with full control over all fields.
    ///
    /// - Parameters:
    ///   - current: Current item index
    ///   - total: Total items to process
    ///   - batchIndex: Current batch index (0-based)
    ///   - totalBatches: Total number of batches
    ///   - phase: Current processing phase
    ///   - message: Optional status message
    ///   - itemsPerSecond: Throughput metric
    ///   - tokensProcessed: Total tokens processed
    ///   - currentBatchSize: Size of current batch
    ///   - estimatedTimeRemaining: Estimated completion time
    public init(
        current: Int,
        total: Int,
        batchIndex: Int = 0,
        totalBatches: Int = 1,
        phase: String = "Processing",
        message: String? = nil,
        itemsPerSecond: Double? = nil,
        tokensProcessed: Int = 0,
        currentBatchSize: Int = 1,
        estimatedTimeRemaining: TimeInterval? = nil
    ) {
        self.base = OperationProgress(
            current: current,
            total: total,
            phase: phase,
            message: message,
            estimatedTimeRemaining: estimatedTimeRemaining
        )
        self.batchIndex = max(0, batchIndex)
        self.totalBatches = max(1, totalBatches)
        self.itemsPerSecond = itemsPerSecond
        self.tokensProcessed = max(0, tokensProcessed)
        self.currentBatchSize = max(0, currentBatchSize)
    }

    /// Creates a batch progress from an existing OperationProgress.
    ///
    /// - Parameters:
    ///   - base: The underlying operation progress
    ///   - batchIndex: Current batch index
    ///   - totalBatches: Total number of batches
    ///   - itemsPerSecond: Throughput metric
    ///   - tokensProcessed: Total tokens processed
    ///   - currentBatchSize: Size of current batch
    public init(
        base: OperationProgress,
        batchIndex: Int = 0,
        totalBatches: Int = 1,
        itemsPerSecond: Double? = nil,
        tokensProcessed: Int = 0,
        currentBatchSize: Int = 1
    ) {
        self.base = base
        self.batchIndex = max(0, batchIndex)
        self.totalBatches = max(1, totalBatches)
        self.itemsPerSecond = itemsPerSecond
        self.tokensProcessed = max(0, tokensProcessed)
        self.currentBatchSize = max(0, currentBatchSize)
    }

    // MARK: - Factory Methods

    /// Creates a progress indicating the start of batch processing.
    ///
    /// - Parameters:
    ///   - total: Total items to process
    ///   - totalBatches: Number of batches planned
    /// - Returns: A progress instance at 0%
    public static func started(total: Int, totalBatches: Int = 1) -> BatchProgress {
        BatchProgress(
            current: 0,
            total: total,
            batchIndex: 0,
            totalBatches: totalBatches,
            phase: "Starting"
        )
    }

    /// Creates a progress indicating completion.
    ///
    /// - Parameters:
    ///   - total: Total items processed
    ///   - totalBatches: Number of batches completed
    ///   - tokensProcessed: Total tokens processed
    ///   - itemsPerSecond: Final throughput
    /// - Returns: A progress instance at 100%
    public static func completed(
        total: Int,
        totalBatches: Int = 1,
        tokensProcessed: Int = 0,
        itemsPerSecond: Double? = nil
    ) -> BatchProgress {
        BatchProgress(
            current: total,
            total: total,
            batchIndex: totalBatches - 1,
            totalBatches: totalBatches,
            phase: "Complete",
            itemsPerSecond: itemsPerSecond,
            tokensProcessed: tokensProcessed
        )
    }

    /// Creates a progress for a specific batch completion.
    ///
    /// - Parameters:
    ///   - itemsCompleted: Total items completed so far
    ///   - totalItems: Total items to process
    ///   - batchIndex: Just-completed batch index
    ///   - totalBatches: Total number of batches
    ///   - batchSize: Size of the completed batch
    ///   - tokensInBatch: Tokens processed in this batch
    ///   - elapsedTime: Time elapsed since start
    /// - Returns: A progress instance reflecting batch completion
    public static func batchCompleted(
        itemsCompleted: Int,
        totalItems: Int,
        batchIndex: Int,
        totalBatches: Int,
        batchSize: Int,
        tokensInBatch: Int,
        totalTokens: Int,
        elapsedTime: TimeInterval
    ) -> BatchProgress {
        let itemsPerSec = elapsedTime > 0 ? Double(itemsCompleted) / elapsedTime : nil
        let remaining = totalItems - itemsCompleted
        let eta = itemsPerSec.map { remaining > 0 && $0 > 0 ? Double(remaining) / $0 : 0 }

        return BatchProgress(
            current: itemsCompleted,
            total: totalItems,
            batchIndex: batchIndex,
            totalBatches: totalBatches,
            phase: batchIndex < totalBatches - 1 ? "Processing" : "Finalizing",
            message: "Batch \(batchIndex + 1)/\(totalBatches) complete",
            itemsPerSecond: itemsPerSec,
            tokensProcessed: totalTokens,
            currentBatchSize: batchSize,
            estimatedTimeRemaining: eta
        )
    }

    // MARK: - Progress Updates

    /// Creates a new progress with updated current count.
    ///
    /// - Parameter newCurrent: The new current item count
    /// - Returns: Updated progress instance
    public func withCurrent(_ newCurrent: Int) -> BatchProgress {
        BatchProgress(
            current: newCurrent,
            total: total,
            batchIndex: batchIndex,
            totalBatches: totalBatches,
            phase: phase,
            message: message,
            itemsPerSecond: itemsPerSecond,
            tokensProcessed: tokensProcessed,
            currentBatchSize: currentBatchSize,
            estimatedTimeRemaining: estimatedTimeRemaining
        )
    }

    /// Creates a new progress with updated throughput.
    ///
    /// - Parameter throughput: Items per second
    /// - Returns: Updated progress instance
    public func withThroughput(_ throughput: Double) -> BatchProgress {
        BatchProgress(
            current: current,
            total: total,
            batchIndex: batchIndex,
            totalBatches: totalBatches,
            phase: phase,
            message: message,
            itemsPerSecond: throughput,
            tokensProcessed: tokensProcessed,
            currentBatchSize: currentBatchSize,
            estimatedTimeRemaining: estimatedTimeRemaining
        )
    }

    /// Creates a new progress for the next batch.
    ///
    /// - Parameter batchSize: Size of the next batch
    /// - Returns: Updated progress instance
    public func nextBatch(size batchSize: Int) -> BatchProgress {
        BatchProgress(
            current: current,
            total: total,
            batchIndex: batchIndex + 1,
            totalBatches: totalBatches,
            phase: "Processing",
            message: "Starting batch \(batchIndex + 2)/\(totalBatches)",
            itemsPerSecond: itemsPerSecond,
            tokensProcessed: tokensProcessed,
            currentBatchSize: batchSize,
            estimatedTimeRemaining: estimatedTimeRemaining
        )
    }
}

// MARK: - CustomStringConvertible

extension BatchProgress: CustomStringConvertible {
    public var description: String {
        var parts = ["\(percentage)%"]
        parts.append("(\(current)/\(total))")
        parts.append("batch \(batchIndex + 1)/\(totalBatches)")
        if let ips = itemsPerSecond {
            parts.append(String(format: "%.1f items/s", ips))
        }
        if tokensProcessed > 0 {
            parts.append("\(tokensProcessed) tokens")
        }
        return parts.joined(separator: " | ")
    }
}

// MARK: - EmbeddingProgressStream

/// Type alias for embedding progress streams.
/// Each element is a tuple of (embedding vector, progress).
public typealias EmbeddingProgressStream = ProgressStream<[Float]>

/// Type alias for batch embedding progress streams.
/// Each element is a tuple of (batch of embeddings, progress).
public typealias BatchEmbeddingProgressStream = ProgressStream<[[Float]]>

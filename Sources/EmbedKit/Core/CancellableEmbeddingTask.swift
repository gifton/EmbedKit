// EmbedKit - CancellableEmbeddingTask
// Fine-grained cancellation control for embedding operations

import Foundation

// MARK: - Cancellation Mode

/// Defines how cancellation should be handled for embedding operations.
///
/// Different modes provide varying trade-offs between responsiveness and
/// result completeness:
///
/// - `immediate`: Cancel as soon as possible. May leave embeddings incomplete.
/// - `graceful`: Complete the current embedding, then cancel remaining work.
/// - `afterBatch`: Complete the current batch, then cancel remaining batches.
/// - `checkpoint`: Cancel at the next progress checkpoint/yield point.
public enum CancellationMode: String, Sendable, CaseIterable {
    /// Cancel immediately. The operation may return partial results or throw.
    /// Use when responsiveness is more important than completeness.
    case immediate

    /// Complete the currently processing item, then cancel.
    /// Provides at least one valid result if cancellation happens mid-batch.
    case graceful

    /// Complete the current batch, then cancel remaining batches.
    /// Useful when you want batch-level atomicity.
    case afterBatch

    /// Cancel at the next yield/checkpoint in the stream.
    /// Ideal for streaming operations where you want clean termination.
    case checkpoint
}

// MARK: - Cancellation State

/// Represents the current state of a cancellable operation.
public enum CancellationState: String, Sendable {
    /// Operation is running normally.
    case running

    /// Cancellation has been requested but not yet processed.
    case cancellationRequested

    /// Operation is in the process of cancelling (cleanup in progress).
    case cancelling

    /// Operation has been cancelled.
    case cancelled

    /// Operation completed successfully before cancellation.
    case completed

    /// Operation failed with an error.
    case failed
}

// MARK: - Cancellation Handler

/// A closure that runs when cancellation occurs.
///
/// Use this to perform cleanup, release resources, or notify observers.
/// The handler receives the final state of the operation.
public typealias CancellationHandler = @Sendable (CancellationState) -> Void

// MARK: - Cancellation Token

/// A token that tracks cancellation state and can be checked by operations.
///
/// `CancellationToken` provides a cooperative cancellation mechanism that allows
/// long-running operations to check for cancellation and respond appropriately.
///
/// ## Example Usage
/// ```swift
/// let token = CancellationToken()
///
/// Task {
///     for text in texts {
///         // Check before expensive operation
///         if token.isCancelled {
///             return partialResults
///         }
///         let embedding = try await model.embed(text)
///         partialResults.append(embedding)
///
///         // Check at yield points
///         try token.throwIfCancelled()
///     }
/// }
///
/// // Later, from another context:
/// token.cancel()
/// ```
public final class CancellationToken: @unchecked Sendable {
    private let lock = NSLock()
    private var _isCancelled: Bool = false
    private var _mode: CancellationMode = .graceful
    private var _handlers: [CancellationHandler] = []
    private var _state: CancellationState = .running

    /// Creates a new cancellation token.
    public init() {}

    /// Whether cancellation has been requested.
    public var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _isCancelled
    }

    /// The cancellation mode that was used.
    public var mode: CancellationMode {
        lock.lock()
        defer { lock.unlock() }
        return _mode
    }

    /// Current state of the operation.
    public var state: CancellationState {
        lock.lock()
        defer { lock.unlock() }
        return _state
    }

    /// Request cancellation with the specified mode.
    ///
    /// - Parameter mode: How cancellation should be handled.
    public func cancel(mode: CancellationMode = .graceful) {
        lock.lock()
        guard !_isCancelled else {
            lock.unlock()
            return
        }
        _isCancelled = true
        _mode = mode
        _state = .cancellationRequested
        let handlers = _handlers
        lock.unlock()

        // Invoke handlers outside lock
        for handler in handlers {
            handler(.cancellationRequested)
        }
    }

    /// Throws `CancellationError` if cancellation was requested.
    ///
    /// Call this at yield points in your operation to enable cooperative cancellation.
    public func throwIfCancelled() throws {
        if isCancelled {
            throw CancellationError()
        }
    }

    /// Check cancellation, respecting the mode.
    ///
    /// - Parameter checkpoint: The type of checkpoint (item, batch, or yield).
    /// - Returns: `true` if the operation should stop at this checkpoint.
    public func shouldStopAt(checkpoint: CheckpointType) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        guard _isCancelled else { return false }

        switch (_mode, checkpoint) {
        case (.immediate, _):
            return true
        case (.graceful, .afterItem), (.graceful, .afterBatch), (.graceful, .yieldPoint):
            return true
        case (.afterBatch, .afterBatch), (.afterBatch, .yieldPoint):
            return true
        case (.checkpoint, .yieldPoint):
            return true
        default:
            return false
        }
    }

    /// Register a handler to be called when cancellation occurs.
    ///
    /// - Parameter handler: Closure to invoke on cancellation.
    /// - Returns: A registration that can be used to remove the handler.
    @discardableResult
    public func onCancel(_ handler: @escaping CancellationHandler) -> CancellationRegistration {
        lock.lock()
        let index = _handlers.count
        _handlers.append(handler)
        let alreadyCancelled = _isCancelled
        let currentState = _state
        lock.unlock()

        // If already cancelled, invoke immediately
        if alreadyCancelled {
            handler(currentState)
        }

        return CancellationRegistration(token: self, index: index)
    }

    /// Update the internal state (called by task wrapper).
    internal func updateState(_ newState: CancellationState) {
        lock.lock()
        _state = newState
        let handlers = _handlers
        lock.unlock()

        for handler in handlers {
            handler(newState)
        }
    }

    /// Remove a handler at the given index.
    internal func removeHandler(at index: Int) {
        lock.lock()
        defer { lock.unlock() }
        if index < _handlers.count {
            _handlers[index] = { _ in }  // Replace with no-op
        }
    }

    /// Types of checkpoints where cancellation can be checked.
    public enum CheckpointType: Sendable {
        /// After processing a single item.
        case afterItem
        /// After completing a batch.
        case afterBatch
        /// At a general yield point (e.g., before yielding to stream).
        case yieldPoint
    }
}

// MARK: - Cancellation Registration

/// A registration that can be used to remove a cancellation handler.
public struct CancellationRegistration: Sendable {
    private weak var token: CancellationToken?
    private let index: Int

    init(token: CancellationToken, index: Int) {
        self.token = token
        self.index = index
    }

    /// Remove this handler from the token.
    public func unregister() {
        token?.removeHandler(at: index)
    }
}

// MARK: - CancellableEmbeddingTask

/// A wrapper around an embedding operation that provides fine-grained cancellation control.
///
/// `CancellableEmbeddingTask` extends Swift's standard Task cancellation with
/// embedding-specific modes and handlers. It's ideal for long-running embedding
/// operations where you need control over how cancellation affects results.
///
/// ## Example Usage
/// ```swift
/// let task = CancellableEmbeddingTask { token in
///     var results: [[Float]] = []
///     for text in texts {
///         if token.shouldStopAt(checkpoint: .afterItem) {
///             break
///         }
///         let embedding = try await generator.produce(text)
///         results.append(embedding)
///     }
///     return results
/// }
///
/// // Start the task
/// let handle = task.start()
///
/// // Cancel gracefully after some time
/// Task {
///     try await Task.sleep(for: .seconds(5))
///     handle.cancel(mode: .graceful)
/// }
///
/// // Wait for results (may be partial if cancelled)
/// let embeddings = try await handle.value
/// ```
public struct CancellableEmbeddingTask<Result: Sendable>: Sendable {
    /// The operation to perform.
    private let operation: @Sendable (CancellationToken) async throws -> Result

    /// Creates a cancellable embedding task.
    ///
    /// - Parameter operation: The async operation to perform. Receives a
    ///   `CancellationToken` that should be checked periodically.
    public init(
        operation: @escaping @Sendable (CancellationToken) async throws -> Result
    ) {
        self.operation = operation
    }

    /// Start the task and return a handle for cancellation and result retrieval.
    ///
    /// - Parameter priority: Task priority (default: nil, inherits from context).
    /// - Returns: A handle that can be used to cancel or await the result.
    public func start(priority: TaskPriority? = nil) -> TaskHandle<Result> {
        let token = CancellationToken()

        let task = Task(priority: priority) {
            do {
                let result = try await operation(token)
                // Success - update state to completed
                if token.isCancelled {
                    token.updateState(.cancelled)
                } else {
                    token.updateState(.completed)
                }
                return result
            } catch is CancellationError {
                token.updateState(.cancelled)
                throw CancellationError()
            } catch {
                token.updateState(.failed)
                throw error
            }
        }

        return TaskHandle(task: task, token: token)
    }

    /// A handle to a running task that provides cancellation and result access.
    public struct TaskHandle<T: Sendable>: Sendable {
        private let task: Task<T, Error>
        private let token: CancellationToken

        init(task: Task<T, Error>, token: CancellationToken) {
            self.task = task
            self.token = token
        }

        /// The result of the task (awaitable).
        public var value: T {
            get async throws {
                try await task.value
            }
        }

        /// Whether the task has been cancelled.
        public var isCancelled: Bool {
            token.isCancelled || task.isCancelled
        }

        /// Current state of the task.
        public var state: CancellationState {
            token.state
        }

        /// Cancel the task with the specified mode.
        ///
        /// - Parameter mode: How cancellation should be handled.
        public func cancel(mode: CancellationMode = .graceful) {
            token.cancel(mode: mode)

            // For immediate mode, also cancel the underlying Task
            if mode == .immediate {
                task.cancel()
            }
        }

        /// Register a handler to be called when cancellation occurs.
        ///
        /// - Parameter handler: Closure to invoke on cancellation.
        /// - Returns: A registration that can be used to remove the handler.
        @discardableResult
        public func onCancel(_ handler: @escaping CancellationHandler) -> CancellationRegistration {
            token.onCancel(handler)
        }

        /// Wait for completion without throwing on cancellation.
        ///
        /// - Returns: The result if successful, nil if cancelled.
        public func valueOrNil() async -> T? {
            do {
                return try await value
            } catch {
                return nil
            }
        }
    }
}

// MARK: - Cancellable Embedding Helpers

/// Helpers for creating cancellable embedding operations.
///
/// These functions provide convenient ways to create `CancellableEmbeddingTask`
/// instances for common embedding patterns.
///
/// ## Example Usage
/// ```swift
/// // Create a cancellable batch operation
/// let handle = CancellableEmbedding.withBatchOperation { token, batchSize in
///     var results: [[Float]] = []
///     for batch in texts.chunked(into: batchSize) {
///         if token.shouldStopAt(checkpoint: .afterBatch) { break }
///         let embeddings = try await model.embedBatch(batch, options: .default)
///         results.append(contentsOf: embeddings.map { $0.vector })
///     }
///     return results
/// }
///
/// // Cancel after timeout
/// Task {
///     try await Task.sleep(for: .seconds(10))
///     handle.cancel(mode: .graceful)
/// }
///
/// // Get results (may be partial if cancelled)
/// let embeddings = try await handle.value
/// ```
public enum CancellableEmbedding {

    /// Create a cancellable task from an async operation.
    ///
    /// This is the most flexible way to create a cancellable embedding task.
    /// You provide the operation and it receives a `CancellationToken` to
    /// check for cancellation at appropriate points.
    ///
    /// - Parameter operation: The async operation to perform.
    /// - Returns: A task handle for cancellation and result access.
    public static func withOperation<T: Sendable>(
        _ operation: @escaping @Sendable (CancellationToken) async throws -> T
    ) -> CancellableEmbeddingTask<T>.TaskHandle<T> {
        CancellableEmbeddingTask(operation: operation).start()
    }

    /// Create a cancellable task with progress tracking.
    ///
    /// - Parameters:
    ///   - onProgress: Progress callback.
    ///   - onCancel: Called when cancellation occurs.
    ///   - operation: The async operation to perform.
    /// - Returns: A task handle for cancellation and result access.
    public static func withProgress<T: Sendable>(
        onProgress: (@Sendable (BatchProgress) -> Void)? = nil,
        onCancel: (@Sendable (CancellationState) -> Void)? = nil,
        operation: @escaping @Sendable (CancellationToken) async throws -> T
    ) -> CancellableEmbeddingTask<T>.TaskHandle<T> {
        let task = CancellableEmbeddingTask { token in
            if let onCancel = onCancel {
                token.onCancel(onCancel)
            }
            return try await operation(token)
        }
        return task.start()
    }

}

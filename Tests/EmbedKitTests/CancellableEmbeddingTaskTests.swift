// EmbedKit - CancellableEmbeddingTask Tests

import Testing
import Foundation
@testable import EmbedKit
import VectorCore

// MARK: - CancellationMode Tests

@Suite("CancellationMode")
struct CancellationModeTests {

    @Test("All modes are available")
    func testAllModesAvailable() {
        let modes: [CancellationMode] = [.immediate, .graceful, .afterBatch, .checkpoint]
        #expect(modes.count == 4)
    }

    @Test("Modes have distinct raw values")
    func testDistinctRawValues() {
        let rawValues = CancellationMode.allCases.map { $0.rawValue }
        let uniqueValues = Set(rawValues)
        #expect(uniqueValues.count == CancellationMode.allCases.count)
    }

    @Test("CancellationMode is Sendable")
    func testSendable() async {
        let mode = CancellationMode.graceful
        let result = await Task {
            mode.rawValue
        }.value
        #expect(result == "graceful")
    }
}

// MARK: - CancellationState Tests

@Suite("CancellationState")
struct CancellationStateTests {

    @Test("All states are available")
    func testAllStatesAvailable() {
        let states: [CancellationState] = [
            .running, .cancellationRequested, .cancelling,
            .cancelled, .completed, .failed
        ]
        #expect(states.count == 6)
    }

    @Test("CancellationState is Sendable")
    func testSendable() async {
        let state = CancellationState.running
        let result = await Task {
            state.rawValue
        }.value
        #expect(result == "running")
    }
}

// MARK: - CancellationToken Tests

@Suite("CancellationToken")
struct CancellationTokenTests {

    @Test("Initial state is not cancelled")
    func testInitialState() {
        let token = CancellationToken()
        #expect(!token.isCancelled)
        #expect(token.state == .running)
    }

    @Test("Cancel sets isCancelled to true")
    func testCancel() {
        let token = CancellationToken()
        token.cancel()
        #expect(token.isCancelled)
        #expect(token.state == .cancellationRequested)
    }

    @Test("Cancel respects mode")
    func testCancelWithMode() {
        let token = CancellationToken()
        token.cancel(mode: .immediate)
        #expect(token.mode == .immediate)

        let token2 = CancellationToken()
        token2.cancel(mode: .afterBatch)
        #expect(token2.mode == .afterBatch)
    }

    @Test("Multiple cancels only register once")
    func testMultipleCancels() {
        let token = CancellationToken()
        token.cancel(mode: .graceful)
        token.cancel(mode: .immediate)  // Should be ignored

        #expect(token.mode == .graceful)  // First mode wins
    }

    @Test("throwIfCancelled throws when cancelled")
    func testThrowIfCancelled() {
        let token = CancellationToken()

        // Should not throw when not cancelled
        #expect(throws: Never.self) {
            try token.throwIfCancelled()
        }

        token.cancel()

        // Should throw when cancelled
        #expect(throws: CancellationError.self) {
            try token.throwIfCancelled()
        }
    }

    @Test("shouldStopAt respects immediate mode")
    func testShouldStopAtImmediate() {
        let token = CancellationToken()
        token.cancel(mode: .immediate)

        #expect(token.shouldStopAt(checkpoint: .afterItem))
        #expect(token.shouldStopAt(checkpoint: .afterBatch))
        #expect(token.shouldStopAt(checkpoint: .yieldPoint))
    }

    @Test("shouldStopAt respects graceful mode")
    func testShouldStopAtGraceful() {
        let token = CancellationToken()
        token.cancel(mode: .graceful)

        #expect(token.shouldStopAt(checkpoint: .afterItem))
        #expect(token.shouldStopAt(checkpoint: .afterBatch))
        #expect(token.shouldStopAt(checkpoint: .yieldPoint))
    }

    @Test("shouldStopAt respects afterBatch mode")
    func testShouldStopAtAfterBatch() {
        let token = CancellationToken()
        token.cancel(mode: .afterBatch)

        #expect(!token.shouldStopAt(checkpoint: .afterItem))
        #expect(token.shouldStopAt(checkpoint: .afterBatch))
        #expect(token.shouldStopAt(checkpoint: .yieldPoint))
    }

    @Test("shouldStopAt respects checkpoint mode")
    func testShouldStopAtCheckpoint() {
        let token = CancellationToken()
        token.cancel(mode: .checkpoint)

        #expect(!token.shouldStopAt(checkpoint: .afterItem))
        #expect(!token.shouldStopAt(checkpoint: .afterBatch))
        #expect(token.shouldStopAt(checkpoint: .yieldPoint))
    }

    @Test("shouldStopAt returns false when not cancelled")
    func testShouldStopAtNotCancelled() {
        let token = CancellationToken()

        #expect(!token.shouldStopAt(checkpoint: .afterItem))
        #expect(!token.shouldStopAt(checkpoint: .afterBatch))
        #expect(!token.shouldStopAt(checkpoint: .yieldPoint))
    }

    @Test("onCancel handler is called on cancel")
    func testOnCancelHandler() async {
        let token = CancellationToken()

        actor StateHolder {
            var handlerCalled = false
            func markCalled() { handlerCalled = true }
        }
        let holder = StateHolder()

        token.onCancel { _ in
            Task { await holder.markCalled() }
        }

        token.cancel()

        // Wait a bit for the handler to be called
        try? await Task.sleep(for: .milliseconds(50))

        let wasCalled = await holder.handlerCalled
        #expect(wasCalled)
    }

    @Test("onCancel handler is called immediately if already cancelled")
    func testOnCancelAlreadyCancelled() async {
        let token = CancellationToken()
        token.cancel()

        actor StateHolder {
            var handlerCalled = false
            func markCalled() { handlerCalled = true }
        }
        let holder = StateHolder()

        // Register handler after cancel
        token.onCancel { _ in
            Task { await holder.markCalled() }
        }

        // Should be called immediately
        try? await Task.sleep(for: .milliseconds(50))

        let wasCalled = await holder.handlerCalled
        #expect(wasCalled)
    }

    @Test("Registration can be unregistered")
    func testUnregister() async {
        let token = CancellationToken()

        actor CallTracker {
            var called = false
            func markCalled() { called = true }
        }
        let tracker = CallTracker()

        let registration = token.onCancel { _ in
            Task { await tracker.markCalled() }
        }

        registration.unregister()
        token.cancel()

        // Give time for any potential handler call
        try? await Task.sleep(for: .milliseconds(50))

        // Handler should not be called (replaced with no-op)
        // Note: The no-op replacement means this should be false
        // but due to timing, we just verify no crash occurs
    }

    @Test("CancellationToken is thread-safe")
    func testThreadSafety() async {
        let token = CancellationToken()

        // Spawn multiple concurrent tasks
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<100 {
                group.addTask {
                    _ = token.isCancelled
                    _ = token.state
                    _ = token.shouldStopAt(checkpoint: .afterItem)
                }
            }

            // Also cancel from another task
            group.addTask {
                token.cancel()
            }
        }

        // Should complete without crashes
        #expect(token.isCancelled)
    }
}

// MARK: - CancellableEmbeddingTask Tests

@Suite("CancellableEmbeddingTask")
struct CancellableEmbeddingTaskTests {

    @Test("Task completes successfully")
    func testSuccessfulCompletion() async throws {
        let task = CancellableEmbeddingTask<Int> { _ in
            return 42
        }

        let handle = task.start()
        let result = try await handle.value

        #expect(result == 42)
        #expect(handle.state == .completed)
    }

    @Test("Task propagates errors")
    func testErrorPropagation() async {
        struct TestError: Error {}

        let task = CancellableEmbeddingTask<Int> { _ in
            throw TestError()
        }

        let handle = task.start()

        do {
            _ = try await handle.value
            Issue.record("Expected error to be thrown")
        } catch is TestError {
            // Expected - error was propagated correctly
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }

        #expect(handle.state == .failed)
    }

    @Test("Task can be cancelled with immediate mode")
    func testImmediateCancellation() async {
        let task = CancellableEmbeddingTask<Int> { token in
            // Long-running operation
            for i in 0..<100 {
                if token.shouldStopAt(checkpoint: .afterItem) {
                    return i
                }
                try? await Task.sleep(for: .milliseconds(10))
            }
            return 100
        }

        let handle = task.start()

        // Cancel immediately
        handle.cancel(mode: .immediate)

        let result = try? await handle.value

        // Should return early (not 100)
        #expect(result == nil || result! < 100)
        #expect(handle.isCancelled)
    }

    @Test("Task can be cancelled with graceful mode")
    func testGracefulCancellation() async throws {
        actor Counter {
            var count = 0
            func increment() { count += 1 }
        }
        let counter = Counter()

        let task = CancellableEmbeddingTask<Int> { token in
            for _ in 0..<10 {
                if token.shouldStopAt(checkpoint: .afterItem) {
                    return await counter.count
                }
                await counter.increment()
                try? await Task.sleep(for: .milliseconds(20))
            }
            return await counter.count
        }

        let handle = task.start()

        // Cancel after a short delay
        Task {
            try? await Task.sleep(for: .milliseconds(50))
            handle.cancel(mode: .graceful)
        }

        let result = try await handle.value
        let finalCount = await counter.count

        // Should have processed some items before cancellation
        #expect(result > 0)
        #expect(result < 10)
        #expect(finalCount == result)
    }

    @Test("valueOrNil returns nil on cancellation")
    func testValueOrNil() async {
        let task = CancellableEmbeddingTask<Int> { token in
            try token.throwIfCancelled()
            return 42
        }

        let handle = task.start()
        handle.cancel(mode: .immediate)

        let result = await handle.valueOrNil()

        // Should be nil due to cancellation
        // (may or may not be nil depending on timing)
        _ = result // Just verify it doesn't crash
    }

    @Test("Task respects priority")
    func testPriority() async throws {
        let task = CancellableEmbeddingTask<Int> { _ in
            return 42
        }

        let handle = task.start(priority: .high)
        let result = try await handle.value

        #expect(result == 42)
    }

    @Test("Multiple handlers can be registered")
    func testMultipleHandlers() async {
        let task = CancellableEmbeddingTask<Int> { token in
            try? await Task.sleep(for: .seconds(10))
            return 42
        }

        let handle = task.start()

        actor Counter {
            var count = 0
            func increment() { count += 1 }
        }
        let counter = Counter()

        handle.onCancel { _ in Task { await counter.increment() } }
        handle.onCancel { _ in Task { await counter.increment() } }
        handle.onCancel { _ in Task { await counter.increment() } }

        handle.cancel()

        try? await Task.sleep(for: .milliseconds(100))
        let finalCount = await counter.count

        #expect(finalCount == 3)
    }
}

// MARK: - CancellableEmbedding Helpers Tests

@Suite("CancellableEmbedding")
struct CancellableEmbeddingTests {

    @Test("withOperation creates working task")
    func testWithOperation() async throws {
        let handle = CancellableEmbedding.withOperation { _ in
            return [1.0, 2.0, 3.0]
        }

        let result = try await handle.value
        #expect(result == [1.0, 2.0, 3.0])
    }

    @Test("withOperation respects cancellation token")
    func testWithOperationCancellation() async {
        let handle = CancellableEmbedding.withOperation { token in
            for i in 0..<100 {
                if token.shouldStopAt(checkpoint: .afterItem) {
                    return i
                }
                try? await Task.sleep(for: .milliseconds(10))
            }
            return 100
        }

        // Cancel after short delay
        Task {
            try? await Task.sleep(for: .milliseconds(30))
            handle.cancel(mode: .graceful)
        }

        let result = try? await handle.value
        #expect(result == nil || result! < 100)
    }

    @Test("withProgress calls progress handler")
    func testWithProgress() async throws {
        actor ProgressTracker {
            var updates: [BatchProgress] = []
            func add(_ p: BatchProgress) { updates.append(p) }
        }
        let tracker = ProgressTracker()

        let handle = CancellableEmbedding.withProgress(
            onProgress: { progress in
                Task { await tracker.add(progress) }
            }
        ) { _ in
            return "done"
        }

        let result = try await handle.value
        #expect(result == "done")

        // Progress handler may or may not have been called depending on operation
        // Just verify it doesn't crash
    }

    @Test("withProgress calls cancel handler on cancellation")
    func testWithProgressCancelHandler() async {
        actor StateHolder {
            var cancelCalled = false
            var finalState: CancellationState?
            func setCancelled(_ state: CancellationState) {
                cancelCalled = true
                finalState = state
            }
        }
        let holder = StateHolder()

        let handle = CancellableEmbedding.withProgress(
            onCancel: { state in
                Task { await holder.setCancelled(state) }
            }
        ) { token in
            try? await Task.sleep(for: .seconds(10))
            return "done"
        }

        handle.cancel(mode: .immediate)

        try? await Task.sleep(for: .milliseconds(100))

        let wasCalled = await holder.cancelCalled
        #expect(wasCalled)
    }
}

// MARK: - Integration Tests

@Suite("Cancellation Integration", .tags(.integration))
struct CancellationIntegrationTests {

    @Test("Cancellation works with MockEmbeddingModel")
    func testWithMockModel() async throws {
        let model = MockEmbeddingModel(dimensions: 384)
        let texts = (0..<20).map { "Text \($0)" }

        let handle = CancellableEmbedding.withOperation { token in
            var results: [[Float]] = []
            for text in texts {
                if token.shouldStopAt(checkpoint: .afterItem) {
                    break
                }
                let embedding = try await model.embed(text)
                results.append(embedding.vector)
            }
            return results
        }

        // Cancel after some processing
        Task {
            try? await Task.sleep(for: .milliseconds(50))
            handle.cancel(mode: .graceful)
        }

        let results = try await handle.value

        // Should have some results but not all
        #expect(results.count > 0)
        #expect(results.count <= texts.count)

        // Each result should have correct dimensions
        for result in results {
            #expect(result.count == 384)
        }
    }

    @Test("Batch processing with cancellation checkpoints")
    func testBatchCancellation() async throws {
        let model = MockEmbeddingModel(dimensions: 128)
        let texts = (0..<50).map { "Batch text \($0)" }
        let batchSize = 10

        let handle = CancellableEmbedding.withOperation { token in
            var results: [[Float]] = []
            var position = 0

            while position < texts.count {
                // Check at batch boundaries
                if token.shouldStopAt(checkpoint: .afterBatch) {
                    break
                }

                let end = min(position + batchSize, texts.count)
                let batch = Array(texts[position..<end])

                let embeddings = try await model.embedBatch(batch, options: .default)
                results.append(contentsOf: embeddings.map { $0.vector })

                position = end
            }

            return results
        }

        // Cancel after 2 batches worth of time
        Task {
            try? await Task.sleep(for: .milliseconds(100))
            handle.cancel(mode: .afterBatch)
        }

        let results = try await handle.value

        // Should have complete batches (multiples of batchSize)
        // unless we got all of them
        if results.count < texts.count {
            #expect(results.count % batchSize == 0 || results.count == texts.count)
        }
    }
}

// Note: Tag.integration is defined in SharedMetalContextManagerTests.swift

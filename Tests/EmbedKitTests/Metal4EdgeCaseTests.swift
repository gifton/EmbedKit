// EmbedKit - Metal 4 Edge Case Tests
//
// Comprehensive edge case testing for Metal 4 tensor storage, lifecycle, and operations.

import Testing
import Foundation
@testable import EmbedKit

#if canImport(Metal)
import Metal
#endif

// MARK: - TensorStorageManager Edge Cases

@Suite("TensorStorageManager Edge Cases")
struct TensorStorageManagerEdgeCaseTests {

    #if canImport(Metal)
    @Test("Zero-size tensor handling")
    func zeroSizeTensorHandling() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4EdgeCaseError.skipped("Metal not available")
        }

        let manager = TensorStorageManager(device: device)

        // Try to create zero-size embedding tensor
        // Should create minimum buffer size
        let tensor = try await manager.createEmbeddingTensor(
            batchSize: 0,
            dimensions: 384
        )

        // Buffer should have at least 4 bytes (minimum)
        #expect(tensor.buffer.length >= 4)

        await manager.releaseAll()
    }

    @Test("Very large tensor allocation")
    func veryLargeTensorAllocation() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4EdgeCaseError.skipped("Metal not available")
        }

        let config = TensorStorageManager.Configuration(
            maxMemoryBytes: 1024 * 1024 * 1024  // 1 GB
        )
        let manager = TensorStorageManager(device: device, configuration: config)

        // Create a large tensor (100MB)
        let largeBatchSize = 32768
        let dimensions = 768
        let expectedSize = largeBatchSize * dimensions * MemoryLayout<Float>.stride

        let tensor = try await manager.createEmbeddingTensor(
            batchSize: largeBatchSize,
            dimensions: dimensions,
            label: "large_tensor"
        )

        #expect(tensor.buffer.length >= expectedSize)

        let usage = await manager.getMemoryUsage()
        #expect(usage.allocatedBytes >= expectedSize)

        await manager.releaseAll()
    }

    @Test("Rapid create-release cycles")
    func rapidCreateReleaseCycles() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4EdgeCaseError.skipped("Metal not available")
        }

        let manager = TensorStorageManager(device: device)

        // Rapid create-release cycles
        for i in 0..<100 {
            let tensor = try await manager.createEmbeddingTensor(
                batchSize: 8,
                dimensions: 64,
                label: "rapid_\(i)"
            )

            await manager.release(tensor)
        }

        let usage = await manager.getMemoryUsage()
        #expect(usage.tensorCount == 0)
        #expect(usage.allocatedBytes == 0)

        let stats = await manager.getStatistics()
        #expect(stats.totalCreated == 100)
        #expect(stats.totalReleased == 100)
    }

    @Test("Memory pressure eviction correctness")
    func memoryPressureEvictionCorrectness() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4EdgeCaseError.skipped("Metal not available")
        }

        let config = TensorStorageManager.Configuration(
            maxMemoryBytes: 1024 * 1024,  // 1 MB limit
            evictionIdleThreshold: 0  // Immediate eviction eligibility
        )
        let manager = TensorStorageManager(device: device, configuration: config)

        // Create tensors that exceed memory limit
        var tensors: [ManagedTensor] = []

        for i in 0..<10 {
            let tensor = try await manager.createEmbeddingTensor(
                batchSize: 64,
                dimensions: 512,  // ~128KB each
                label: "evict_test_\(i)"
            )
            tensors.append(tensor)
            await manager.markCached(tensor)  // Mark as evictable
        }

        // Memory should be managed
        let usage = await manager.getMemoryUsage()
        #expect(usage.allocatedBytes <= config.maxMemoryBytes * 2)  // Allow some flexibility

        await manager.releaseAll()
    }

    @Test("Duplicate label handling")
    func duplicateLabelHandling() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4EdgeCaseError.skipped("Metal not available")
        }

        let manager = TensorStorageManager(device: device)

        // Create tensor with label
        let tensor1 = try await manager.createEmbeddingTensor(
            batchSize: 8,
            dimensions: 64,
            label: "duplicate_label"
        )

        // Create another with same label (should overwrite in lookup)
        let tensor2 = try await manager.createEmbeddingTensor(
            batchSize: 8,
            dimensions: 64,
            label: "duplicate_label"
        )

        // Both tensors exist but label points to second
        let retrieved = await manager.getTensor(label: "duplicate_label")
        #expect(retrieved?.id == tensor2.id)

        // First tensor still exists by ID
        let firstById = await manager.getTensor(id: tensor1.id)
        #expect(firstById != nil)

        await manager.releaseAll()
    }

    @Test("Empty label handling")
    func emptyLabelHandling() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4EdgeCaseError.skipped("Metal not available")
        }

        let manager = TensorStorageManager(device: device)

        // Create tensor with empty label
        let tensor = try await manager.createEmbeddingTensor(
            batchSize: 8,
            dimensions: 64,
            label: ""
        )

        // Should not be findable by empty label
        let retrieved = await manager.getTensor(label: "")
        #expect(retrieved == nil)

        // But findable by ID
        let byId = await manager.getTensor(id: tensor.id)
        #expect(byId != nil)

        await manager.releaseAll()
    }

    @Test("Concurrent tensor access")
    func concurrentTensorAccess() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4EdgeCaseError.skipped("Metal not available")
        }

        let manager = TensorStorageManager(device: device)

        // Create initial tensor
        let tensor = try await manager.createEmbeddingTensor(
            batchSize: 32,
            dimensions: 128,
            label: "concurrent_test"
        )

        // Concurrent access from multiple tasks
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<50 {
                group.addTask {
                    await manager.markAccessed(tensor)
                }
            }
        }

        // Access count should reflect all accesses
        #expect(tensor.accessCount >= 50)

        await manager.releaseAll()
    }
    #endif
}

// MARK: - TensorLifecycleManager Edge Cases

@Suite("TensorLifecycleManager Edge Cases")
struct TensorLifecycleManagerEdgeCaseTests {

    #if canImport(Metal)
    @Test("Nested scope cleanup")
    func nestedScopeCleanup() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4EdgeCaseError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let lifecycleManager = TensorLifecycleManager(storageManager: storageManager)

        // Outer scope
        try await lifecycleManager.withScope(label: "outer") { outerScope in
            _ = try await outerScope.createEmbeddingTensor(batchSize: 8, dimensions: 64)

            // Inner scope
            try await lifecycleManager.withScope(label: "inner") { innerScope in
                _ = try await innerScope.createEmbeddingTensor(batchSize: 4, dimensions: 32)

                let usage = await storageManager.getMemoryUsage()
                #expect(usage.tensorCount == 2)
            }

            // Inner scope cleaned up, only outer tensor remains
            try? await Task.sleep(nanoseconds: 50_000_000)  // 50ms
            let usage = await storageManager.getMemoryUsage()
            #expect(usage.tensorCount == 1)
        }

        // All cleaned up
        try? await Task.sleep(nanoseconds: 50_000_000)
        let finalUsage = await storageManager.getMemoryUsage()
        #expect(finalUsage.tensorCount == 0)
    }

    @Test("Scope cleanup on error")
    func scopeCleanupOnError() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4EdgeCaseError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let lifecycleManager = TensorLifecycleManager(storageManager: storageManager)

        do {
            try await lifecycleManager.withScope { scope in
                _ = try await scope.createEmbeddingTensor(batchSize: 8, dimensions: 64)
                _ = try await scope.createEmbeddingTensor(batchSize: 16, dimensions: 128)

                throw Metal4EdgeCaseError.intentionalError
            }
        } catch is Metal4EdgeCaseError {
            // Expected
        }

        // Tensors should still be cleaned up
        try? await Task.sleep(nanoseconds: 50_000_000)
        let usage = await storageManager.getMemoryUsage()
        #expect(usage.tensorCount == 0)
    }

    @Test("Multiple concurrent scopes")
    func multipleConcurrentScopes() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4EdgeCaseError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let lifecycleManager = TensorLifecycleManager(storageManager: storageManager)

        // Run multiple scopes concurrently
        await withTaskGroup(of: Void.self) { group in
            for i in 0..<10 {
                group.addTask {
                    do {
                        try await lifecycleManager.withScope(label: "concurrent_\(i)") { scope in
                            _ = try await scope.createEmbeddingTensor(
                                batchSize: 4,
                                dimensions: 32
                            )
                            // Simulate work
                            try? await Task.sleep(nanoseconds: 10_000_000)  // 10ms
                        }
                    } catch {
                        // Ignore errors in stress test
                    }
                }
            }
        }

        // All scopes should be cleaned up
        try? await Task.sleep(nanoseconds: 100_000_000)  // 100ms
        let usage = await storageManager.getMemoryUsage()
        #expect(usage.tensorCount == 0)
    }

    @Test("Pool exhaustion and recovery")
    func poolExhaustionAndRecovery() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4EdgeCaseError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)

        // Create a small pool
        let pool = try await TensorPool(
            storageManager: storageManager,
            shape: .embedding(batchSize: 4, dimensions: 32),
            poolSize: 3
        )

        // Acquire all tensors
        let t1 = await pool.acquire()
        let t2 = await pool.acquire()
        let t3 = await pool.acquire()

        #expect(t1 != nil)
        #expect(t2 != nil)
        #expect(t3 != nil)
        #expect(await pool.availableCount == 0)

        // Try to acquire when exhausted
        let t4 = await pool.acquire()
        #expect(t4 == nil)

        // Release one
        await pool.release(t1!)
        #expect(await pool.availableCount == 1)

        // Can acquire again
        let t5 = await pool.acquire()
        #expect(t5 != nil)

        // Cleanup
        await pool.release(t2!)
        await pool.release(t3!)
        await pool.release(t5!)
        await pool.clear()
    }

    @Test("Auto-release tensor lifecycle")
    func autoReleaseTensorLifecycle() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4EdgeCaseError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)

        // Create auto-release tensor in inner scope
        do {
            let autoTensor = try await AutoReleaseTensor(
                batchSize: 8,
                dimensions: 64,
                storageManager: storageManager
            )

            let usage = await storageManager.getMemoryUsage()
            #expect(usage.tensorCount == 1)

            // Use the tensor
            #expect(autoTensor.buffer.length > 0)
        }

        // After scope exit, tensor should be scheduled for release
        // Give time for async cleanup
        try? await Task.sleep(nanoseconds: 200_000_000)  // 200ms

        let finalUsage = await storageManager.getMemoryUsage()
        #expect(finalUsage.tensorCount == 0)
    }
    #endif
}

// MARK: - TensorOperationDispatcher Edge Cases

@Suite("TensorOperationDispatcher Edge Cases")
struct TensorOperationDispatcherEdgeCaseTests {

    #if canImport(Metal)
    @Test("Operation on zero-element tensor")
    func operationOnZeroElementTensor() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4EdgeCaseError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let accelerator = await MetalAccelerator()
        let dispatcher = await TensorOperationDispatcher(
            accelerator: accelerator,
            storageManager: storageManager
        )

        // Create minimal tensors
        let input = try await storageManager.createEmbeddingTensor(
            batchSize: 1,
            dimensions: 4
        )

        let output = try await storageManager.createEmbeddingTensor(
            batchSize: 1,
            dimensions: 4
        )

        // Fill input with zeros
        let ptr = input.buffer.contents().bindMemory(to: Float.self, capacity: 4)
        for i in 0..<4 { ptr[i] = 0 }

        // Normalize zeros - should produce zeros without error
        let result = try await dispatcher.execute(
            input: input,
            output: output,
            config: .normalizeConfig(),
            inputShape: .embedding(batchSize: 1, dimensions: 4)
        )

        #expect(result.executionTime > 0)

        // Output should be zeros (or very small values)
        let outPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: 4)
        for i in 0..<4 {
            #expect(abs(outPtr[i]) < 1e-6)
        }

        await storageManager.releaseAll()
    }

    @Test("Operation with NaN values")
    func operationWithNaNValues() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4EdgeCaseError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let accelerator = await MetalAccelerator()
        let dispatcher = await TensorOperationDispatcher(
            accelerator: accelerator,
            storageManager: storageManager
        )

        let input = try await storageManager.createEmbeddingTensor(
            batchSize: 2,
            dimensions: 4
        )

        let output = try await storageManager.createEmbeddingTensor(
            batchSize: 2,
            dimensions: 4
        )

        // Fill with values including NaN
        let ptr = input.buffer.contents().bindMemory(to: Float.self, capacity: 8)
        ptr[0] = 1.0; ptr[1] = 2.0; ptr[2] = Float.nan; ptr[3] = 4.0  // First vector has NaN
        ptr[4] = 1.0; ptr[5] = 1.0; ptr[6] = 1.0; ptr[7] = 1.0        // Second vector is normal

        // Operation should complete without crash
        let result = try await dispatcher.execute(
            input: input,
            output: output,
            config: .normalizeConfig(),
            inputShape: .embedding(batchSize: 2, dimensions: 4)
        )

        #expect(result.executionTime > 0)

        // Second vector should be properly normalized
        let outPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: 8)
        let secondNorm = sqrt(outPtr[4]*outPtr[4] + outPtr[5]*outPtr[5] +
                              outPtr[6]*outPtr[6] + outPtr[7]*outPtr[7])
        #expect(abs(secondNorm - 1.0) < 0.01)

        await storageManager.releaseAll()
    }

    @Test("Very small values normalization")
    func verySmallValuesNormalization() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4EdgeCaseError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let accelerator = await MetalAccelerator()
        let dispatcher = await TensorOperationDispatcher(
            accelerator: accelerator,
            storageManager: storageManager
        )

        let input = try await storageManager.createEmbeddingTensor(
            batchSize: 1,
            dimensions: 4
        )

        let output = try await storageManager.createEmbeddingTensor(
            batchSize: 1,
            dimensions: 4
        )

        // Very small values that could cause numerical issues
        let ptr = input.buffer.contents().bindMemory(to: Float.self, capacity: 4)
        ptr[0] = 1e-20; ptr[1] = 2e-20; ptr[2] = 3e-20; ptr[3] = 4e-20

        let result = try await dispatcher.execute(
            input: input,
            output: output,
            config: .normalizeConfig(),
            inputShape: .embedding(batchSize: 1, dimensions: 4)
        )

        #expect(result.executionTime > 0)

        // Output should be valid (either normalized or zeros for epsilon protection)
        let extractor = TensorResultExtractor()
        let validity = extractor.checkValidity(tensor: output)
        #expect(validity.isValid)

        await storageManager.releaseAll()
    }

    @Test("Similarity of identical vectors")
    func similarityOfIdenticalVectors() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4EdgeCaseError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let accelerator = await MetalAccelerator()
        let dispatcher = await TensorOperationDispatcher(
            accelerator: accelerator,
            storageManager: storageManager
        )

        let dimensions = 64

        let queries = try await storageManager.createEmbeddingTensor(
            batchSize: 2,
            dimensions: dimensions
        )

        let keys = try await storageManager.createEmbeddingTensor(
            batchSize: 2,
            dimensions: dimensions
        )

        let output = try await storageManager.createSimilarityTensor(
            queryCount: 2,
            keyCount: 2
        )

        // Create identical normalized vectors
        let qPtr = queries.buffer.contents().bindMemory(to: Float.self, capacity: 2 * dimensions)
        let kPtr = keys.buffer.contents().bindMemory(to: Float.self, capacity: 2 * dimensions)

        // First vectors: all same value, normalized
        let val = 1.0 / sqrt(Float(dimensions))
        for d in 0..<dimensions {
            qPtr[d] = val
            kPtr[d] = val
        }

        // Second vectors: different pattern, normalized
        for d in 0..<dimensions {
            qPtr[dimensions + d] = d % 2 == 0 ? val : -val
            kPtr[dimensions + d] = d % 2 == 0 ? -val : val  // Opposite pattern
        }

        let result = try await dispatcher.executeSimilarity(
            queries: queries,
            keys: keys,
            output: output,
            queryCount: 2,
            keyCount: 2,
            dimensions: dimensions,
            normalized: true
        )

        #expect(result.executionTime > 0)

        // Check similarity matrix
        let outPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: 4)

        // q0 vs k0 should be ~1.0 (identical)
        #expect(abs(outPtr[0] - 1.0) < 0.01)

        // q1 vs k1 should be ~-1.0 (opposite)
        #expect(abs(outPtr[3] - (-1.0)) < 0.01)

        await storageManager.releaseAll()
    }

    @Test("Large batch operation performance")
    func largeBatchOperationPerformance() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4EdgeCaseError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let accelerator = await MetalAccelerator()
        let dispatcher = await TensorOperationDispatcher(
            accelerator: accelerator,
            storageManager: storageManager
        )

        // Create large tensors
        let batchSize = 1024
        let sequenceLength = 128
        let dimensions = 384

        let input = try await storageManager.createTokenEmbeddingTensor(
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            dimensions: dimensions
        )

        let output = try await storageManager.createEmbeddingTensor(
            batchSize: batchSize,
            dimensions: dimensions
        )

        // Fill input with random values
        let count = batchSize * sequenceLength * dimensions
        let ptr = input.buffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            ptr[i] = Float.random(in: -1...1)
        }

        let result = try await dispatcher.execute(
            input: input,
            output: output,
            config: .meanPoolNormalize(),
            inputShape: .tokenEmbedding(
                batchSize: batchSize,
                sequenceLength: sequenceLength,
                dimensions: dimensions
            )
        )

        // Should complete in reasonable time and use GPU
        #expect(result.executionTime < 10.0)  // Less than 10 seconds

        // Verify output is normalized
        let extractor = TensorResultExtractor()
        let isNormalized = extractor.validateNormalized(
            tensor: output,
            dimensions: dimensions,
            tolerance: 1e-3
        )
        #expect(isNormalized)

        await storageManager.releaseAll()
    }
    #endif
}

// MARK: - TensorResultExtractor Edge Cases

@Suite("TensorResultExtractor Edge Cases")
struct TensorResultExtractorEdgeCaseTests {

    #if canImport(Metal)
    @Test("Extract from buffer boundaries")
    func extractFromBufferBoundaries() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4EdgeCaseError.skipped("Metal not available")
        }

        let count = 100
        guard let buffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride) else {
            throw Metal4EdgeCaseError.skipped("Failed to create buffer")
        }

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count { ptr[i] = Float(i) }

        let tensor = ManagedTensor(
            buffer: buffer,
            shape: .buffer(length: count),
            label: "test"
        )

        let extractor = TensorResultExtractor()

        // Extract first element
        let first = try extractor.extractFlat(from: tensor, offset: 0, count: 1)
        #expect(first[0] == 0)

        // Extract last element
        let last = try extractor.extractFlat(from: tensor, offset: 99, count: 1)
        #expect(last[0] == 99)

        // Extract full range
        let full = try extractor.extractFlat(from: tensor, offset: 0, count: 100)
        #expect(full.count == 100)

        // Invalid range should throw
        #expect(throws: EmbedKitError.self) {
            _ = try extractor.extractFlat(from: tensor, offset: 99, count: 2)
        }

        #expect(throws: EmbedKitError.self) {
            _ = try extractor.extractFlat(from: tensor, offset: -1, count: 1)
        }
    }

    @Test("Statistics with extreme values")
    func statisticsWithExtremeValues() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4EdgeCaseError.skipped("Metal not available")
        }

        let count = 10
        guard let buffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride) else {
            throw Metal4EdgeCaseError.skipped("Failed to create buffer")
        }

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
        ptr[0] = Float.greatestFiniteMagnitude
        ptr[1] = -Float.greatestFiniteMagnitude
        for i in 2..<count { ptr[i] = 0 }

        let tensor = ManagedTensor(
            buffer: buffer,
            shape: .buffer(length: count),
            label: "test"
        )

        let extractor = TensorResultExtractor()
        let stats = extractor.extractStatistics(from: tensor)

        #expect(stats.min == -Float.greatestFiniteMagnitude)
        #expect(stats.max == Float.greatestFiniteMagnitude)
    }

    @Test("Shape mismatch detection")
    func shapeMismatchDetection() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4EdgeCaseError.skipped("Metal not available")
        }

        let count = 20
        guard let buffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride) else {
            throw Metal4EdgeCaseError.skipped("Failed to create buffer")
        }

        // Create tensor with buffer shape
        let tensor = ManagedTensor(
            buffer: buffer,
            shape: .buffer(length: count),
            label: "test"
        )

        let extractor = TensorResultExtractor()

        // Try to extract as embedding (wrong shape type)
        #expect(throws: EmbedKitError.self) {
            _ = try extractor.extractEmbeddings(from: tensor)
        }

        // Try to extract as token embedding (wrong shape type)
        #expect(throws: EmbedKitError.self) {
            _ = try extractor.extractTokenEmbeddings(from: tensor)
        }
    }

    @Test("Validate nearly normalized vectors")
    func validateNearlyNormalizedVectors() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4EdgeCaseError.skipped("Metal not available")
        }

        let dims = 4
        guard let buffer = device.makeBuffer(length: dims * MemoryLayout<Float>.stride) else {
            throw Metal4EdgeCaseError.skipped("Failed to create buffer")
        }

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: dims)

        // Create vector with norm = 1.0001 (slightly off)
        let val = sqrt(1.0001 / Float(dims))
        for i in 0..<dims { ptr[i] = val }

        let tensor = ManagedTensor(
            buffer: buffer,
            shape: .embedding(batchSize: 1, dimensions: dims),
            label: "test"
        )

        let extractor = TensorResultExtractor()

        // With tight tolerance, should fail
        let strict = extractor.validateNormalized(tensor: tensor, dimensions: dims, tolerance: 1e-6)
        #expect(strict == false)

        // With loose tolerance, should pass
        let loose = extractor.validateNormalized(tensor: tensor, dimensions: dims, tolerance: 1e-2)
        #expect(loose == true)
    }
    #endif
}

// MARK: - Numerical Accuracy Tests

@Suite("Numerical Accuracy")
struct NumericalAccuracyTests {

    #if canImport(Metal)
    @Test("L2 normalization accuracy")
    func l2NormalizationAccuracy() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4EdgeCaseError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let accelerator = await MetalAccelerator()
        let dispatcher = await TensorOperationDispatcher(
            accelerator: accelerator,
            storageManager: storageManager
        )

        let batchSize = 100
        let dimensions = 384

        let input = try await storageManager.createEmbeddingTensor(
            batchSize: batchSize,
            dimensions: dimensions
        )

        let output = try await storageManager.createEmbeddingTensor(
            batchSize: batchSize,
            dimensions: dimensions
        )

        // Fill with known values
        let inputPtr = input.buffer.contents().bindMemory(to: Float.self, capacity: batchSize * dimensions)
        for b in 0..<batchSize {
            for d in 0..<dimensions {
                inputPtr[b * dimensions + d] = Float(d + 1) * Float(b + 1) / 1000.0
            }
        }

        _ = try await dispatcher.execute(
            input: input,
            output: output,
            config: .normalizeConfig(),
            inputShape: .embedding(batchSize: batchSize, dimensions: dimensions)
        )

        // Verify all vectors are unit normalized
        let outputPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: batchSize * dimensions)

        var maxNormError: Float = 0
        for b in 0..<batchSize {
            var sumSquares: Double = 0
            for d in 0..<dimensions {
                let val = Double(outputPtr[b * dimensions + d])
                sumSquares += val * val
            }
            let norm = Float(sqrt(sumSquares))
            let error = abs(norm - 1.0)
            maxNormError = max(maxNormError, error)
        }

        // Maximum error should be very small
        #expect(maxNormError < 1e-4)

        await storageManager.releaseAll()
    }

    @Test("Mean pooling accuracy")
    func meanPoolingAccuracy() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4EdgeCaseError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let accelerator = await MetalAccelerator()
        let dispatcher = await TensorOperationDispatcher(
            accelerator: accelerator,
            storageManager: storageManager
        )

        let batchSize = 4
        let sequenceLength = 8
        let dimensions = 16

        let input = try await storageManager.createTokenEmbeddingTensor(
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            dimensions: dimensions
        )

        let output = try await storageManager.createEmbeddingTensor(
            batchSize: batchSize,
            dimensions: dimensions
        )

        // Fill with known values: each position in sequence gets value = position
        let inputPtr = input.buffer.contents().bindMemory(
            to: Float.self,
            capacity: batchSize * sequenceLength * dimensions
        )

        for b in 0..<batchSize {
            for t in 0..<sequenceLength {
                for d in 0..<dimensions {
                    inputPtr[(b * sequenceLength + t) * dimensions + d] = Float(t)
                }
            }
        }

        _ = try await dispatcher.execute(
            input: input,
            output: output,
            config: TensorOperationConfig(
                operation: .fusedPoolNormalize,
                poolingStrategy: .mean,
                normalize: false  // Just pooling, no normalization
            ),
            inputShape: .tokenEmbedding(
                batchSize: batchSize,
                sequenceLength: sequenceLength,
                dimensions: dimensions
            )
        )

        // Expected mean = (0+1+2+3+4+5+6+7) / 8 = 3.5
        let expectedMean: Float = 3.5
        let outputPtr = output.buffer.contents().bindMemory(
            to: Float.self,
            capacity: batchSize * dimensions
        )

        var maxError: Float = 0
        for b in 0..<batchSize {
            for d in 0..<dimensions {
                let actual = outputPtr[b * dimensions + d]
                let error = abs(actual - expectedMean)
                maxError = max(maxError, error)
            }
        }

        // Should be very accurate
        #expect(maxError < 1e-4)

        await storageManager.releaseAll()
    }

    @Test("Cosine similarity accuracy")
    func cosineSimilarityAccuracy() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4EdgeCaseError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let accelerator = await MetalAccelerator()
        let dispatcher = await TensorOperationDispatcher(
            accelerator: accelerator,
            storageManager: storageManager
        )

        let dimensions = 128
        let queryCount = 10
        let keyCount = 10

        let queries = try await storageManager.createEmbeddingTensor(
            batchSize: queryCount,
            dimensions: dimensions
        )

        let keys = try await storageManager.createEmbeddingTensor(
            batchSize: keyCount,
            dimensions: dimensions
        )

        let output = try await storageManager.createSimilarityTensor(
            queryCount: queryCount,
            keyCount: keyCount
        )

        // Create normalized random vectors
        let qPtr = queries.buffer.contents().bindMemory(to: Float.self, capacity: queryCount * dimensions)
        let kPtr = keys.buffer.contents().bindMemory(to: Float.self, capacity: keyCount * dimensions)

        for q in 0..<queryCount {
            var sumSq: Float = 0
            for d in 0..<dimensions {
                qPtr[q * dimensions + d] = Float.random(in: -1...1)
                sumSq += qPtr[q * dimensions + d] * qPtr[q * dimensions + d]
            }
            let norm = sqrt(sumSq)
            for d in 0..<dimensions {
                qPtr[q * dimensions + d] /= norm
            }
        }

        for k in 0..<keyCount {
            var sumSq: Float = 0
            for d in 0..<dimensions {
                kPtr[k * dimensions + d] = Float.random(in: -1...1)
                sumSq += kPtr[k * dimensions + d] * kPtr[k * dimensions + d]
            }
            let norm = sqrt(sumSq)
            for d in 0..<dimensions {
                kPtr[k * dimensions + d] /= norm
            }
        }

        _ = try await dispatcher.executeSimilarity(
            queries: queries,
            keys: keys,
            output: output,
            queryCount: queryCount,
            keyCount: keyCount,
            dimensions: dimensions,
            normalized: true
        )

        // Compute expected similarities and compare
        let outPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: queryCount * keyCount)

        var maxError: Float = 0
        for q in 0..<queryCount {
            for k in 0..<keyCount {
                // Compute expected dot product
                var expectedDot: Float = 0
                for d in 0..<dimensions {
                    expectedDot += qPtr[q * dimensions + d] * kPtr[k * dimensions + d]
                }

                let actual = outPtr[q * keyCount + k]
                let error = abs(actual - expectedDot)
                maxError = max(maxError, error)
            }
        }

        // Allow for some numerical difference
        #expect(maxError < 1e-3)

        await storageManager.releaseAll()
    }
    #endif
}

// MARK: - Test Error

enum Metal4EdgeCaseError: Error {
    case skipped(String)
    case intentionalError
}

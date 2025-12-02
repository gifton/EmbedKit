// EmbedKit - Tensor Lifecycle Manager Tests
//
// Tests for tensor lifecycle management, scopes, pools, and auto-release.

import Testing
import Foundation
@testable import EmbedKit

#if canImport(Metal)
import Metal
#endif

// MARK: - TensorScope Tests

@Suite("TensorScope")
struct TensorScopeTests {

    #if canImport(Metal)
    @Test("Scope creates tensors correctly")
    func createsCorrectly() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorLifecycleTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let scope = TensorScope(storageManager: storageManager, label: "test_scope")

        let tensor = try await scope.createEmbeddingTensor(
            batchSize: 8,
            dimensions: 64,
            label: "scoped_tensor"
        )

        #expect(tensor.label == "scoped_tensor")
        #expect(scope.tensorCount == 1)
    }

    @Test("Scope tracks multiple tensors")
    func tracksMultipleTensors() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorLifecycleTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let scope = TensorScope(storageManager: storageManager)

        _ = try await scope.createEmbeddingTensor(batchSize: 8, dimensions: 64)
        _ = try await scope.createTokenEmbeddingTensor(batchSize: 2, sequenceLength: 32, dimensions: 64)
        _ = try await scope.createSimilarityTensor(queryCount: 10, keyCount: 20)

        #expect(scope.tensorCount == 3)
        #expect(scope.totalBytes > 0)
    }

    @Test("Scope releases tensors on finalize")
    func releasesOnFinalize() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorLifecycleTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let scope = TensorScope(storageManager: storageManager)

        let tensor = try await scope.createEmbeddingTensor(batchSize: 8, dimensions: 64)

        var usage = await storageManager.getMemoryUsage()
        #expect(usage.tensorCount == 1)

        await scope.finalize()

        usage = await storageManager.getMemoryUsage()
        #expect(usage.tensorCount == 0)
        #expect(tensor.state == .released)
    }

    @Test("Scope can release individual tensors")
    func releasesIndividualTensors() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorLifecycleTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let scope = TensorScope(storageManager: storageManager)

        let tensor1 = try await scope.createEmbeddingTensor(batchSize: 8, dimensions: 64)
        _ = try await scope.createEmbeddingTensor(batchSize: 16, dimensions: 128)

        #expect(scope.tensorCount == 2)

        await scope.release(tensor1)

        #expect(scope.tensorCount == 1)
    }

    @Test("Scope finalize is idempotent")
    func finalizeIsIdempotent() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorLifecycleTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let scope = TensorScope(storageManager: storageManager)

        _ = try await scope.createEmbeddingTensor(batchSize: 8, dimensions: 64)

        await scope.finalize()
        await scope.finalize()  // Should not crash or double-free
        await scope.finalize()

        let usage = await storageManager.getMemoryUsage()
        #expect(usage.tensorCount == 0)
    }
    #endif
}

// MARK: - TensorLifecycleManager Tests

@Suite("TensorLifecycleManager")
struct TensorLifecycleManagerTests {

    #if canImport(Metal)
    @Test("Lifecycle manager initializes correctly")
    func initializesCorrectly() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorLifecycleTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let lifecycleManager = TensorLifecycleManager(storageManager: storageManager)

        let status = await lifecycleManager.getStatus()
        #expect(status.isMonitoring == false)
        #expect(status.activeScopeCount == 0)
    }

    @Test("Lifecycle manager starts and stops monitoring")
    func startsAndStopsMonitoring() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorLifecycleTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let lifecycleManager = TensorLifecycleManager(storageManager: storageManager)

        await lifecycleManager.startMonitoring()
        var status = await lifecycleManager.getStatus()
        #expect(status.isMonitoring == true)

        await lifecycleManager.stopMonitoring()
        status = await lifecycleManager.getStatus()
        #expect(status.isMonitoring == false)
    }

    @Test("Creates and finalizes scopes")
    func createsAndFinalizesScopes() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorLifecycleTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let lifecycleManager = TensorLifecycleManager(storageManager: storageManager)

        let scope = await lifecycleManager.createScope(label: "test_scope")
        #expect(await lifecycleManager.activeScopeCount == 1)

        await lifecycleManager.finalizeScope(scope)
        #expect(await lifecycleManager.activeScopeCount == 0)

        let stats = await lifecycleManager.getStatistics()
        #expect(stats.scopesCreated == 1)
        #expect(stats.scopesFinalized == 1)
    }

    @Test("WithScope automatically cleans up")
    func withScopeAutomaticallyCleanups() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorLifecycleTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let lifecycleManager = TensorLifecycleManager(storageManager: storageManager)

        let tensorId: UUID = try await lifecycleManager.withScope(label: "auto_scope") { scope in
            let tensor = try await scope.createEmbeddingTensor(batchSize: 8, dimensions: 64)

            let usage = await storageManager.getMemoryUsage()
            #expect(usage.tensorCount == 1)

            return tensor.id
        }

        // After scope exits, tensor should be released
        // Give a moment for the deferred cleanup
        try? await Task.sleep(nanoseconds: 100_000_000)  // 100ms

        let usage = await storageManager.getMemoryUsage()
        #expect(usage.tensorCount == 0)

        // Tensor should no longer be retrievable
        let retrieved = await storageManager.getTensor(id: tensorId)
        #expect(retrieved == nil)
    }

    @Test("Performs cleanup cycle")
    func performsCleanupCycle() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorLifecycleTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let config = TensorLifecycleManager.Configuration(
            cleanupInterval: 1.0,
            memoryThreshold: 0.1  // Very low threshold to trigger cleanup
        )
        let lifecycleManager = TensorLifecycleManager(
            storageManager: storageManager,
            configuration: config
        )

        // Create tensors and mark as cached
        let tensor = try await storageManager.createEmbeddingTensor(batchSize: 64, dimensions: 384)
        await storageManager.markCached(tensor)

        await lifecycleManager.performCleanupCycle()

        let stats = await lifecycleManager.getStatistics()
        #expect(stats.cleanupCycles == 1)
    }
    #endif
}

// MARK: - TensorLifecycleManager Configuration Tests

@Suite("TensorLifecycleManagerConfiguration")
struct TensorLifecycleManagerConfigurationTests {

    @Test("Default configuration values")
    func defaultConfigurationValues() {
        let config = TensorLifecycleManager.Configuration.default

        #expect(config.cleanupInterval == 30.0)
        #expect(config.autoStartMonitoring == false)
        #expect(config.memoryThreshold == 0.8)
    }

    @Test("Aggressive configuration values")
    func aggressiveConfigurationValues() {
        let config = TensorLifecycleManager.Configuration.aggressive

        #expect(config.cleanupInterval == 10.0)
        #expect(config.autoStartMonitoring == true)
        #expect(config.memoryThreshold == 0.6)
    }

    @Test("Relaxed configuration values")
    func relaxedConfigurationValues() {
        let config = TensorLifecycleManager.Configuration.relaxed

        #expect(config.cleanupInterval == 60.0)
        #expect(config.autoStartMonitoring == false)
        #expect(config.memoryThreshold == 0.9)
    }

    @Test("Configuration enforces bounds")
    func configurationEnforcesBounds() {
        let config = TensorLifecycleManager.Configuration(
            cleanupInterval: 0.1,  // Too low
            memoryThreshold: 2.0   // Too high
        )

        #expect(config.cleanupInterval >= 1.0)
        #expect(config.memoryThreshold <= 1.0)
    }
}

// MARK: - TensorPool Tests

@Suite("TensorPool")
struct TensorPoolTests {

    #if canImport(Metal)
    @Test("Pool initializes with pre-allocated tensors")
    func initializesWithPreallocatedTensors() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorLifecycleTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let pool = try await TensorPool(
            storageManager: storageManager,
            shape: .embedding(batchSize: 8, dimensions: 64),
            poolSize: 4,
            label: "test_pool"
        )

        #expect(await pool.availableCount == 4)
        #expect(await pool.inUseCount == 0)
    }

    @Test("Pool acquire returns tensor")
    func acquireReturnsTensor() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorLifecycleTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let pool = try await TensorPool(
            storageManager: storageManager,
            shape: .embedding(batchSize: 8, dimensions: 64),
            poolSize: 2
        )

        let tensor = await pool.acquire()
        #expect(tensor != nil)
        #expect(await pool.availableCount == 1)
        #expect(await pool.inUseCount == 1)
    }

    @Test("Pool release returns tensor to pool")
    func releaseReturnsTensorToPool() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorLifecycleTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let pool = try await TensorPool(
            storageManager: storageManager,
            shape: .embedding(batchSize: 8, dimensions: 64),
            poolSize: 2
        )

        let tensor = await pool.acquire()
        #expect(await pool.availableCount == 1)

        await pool.release(tensor!)
        #expect(await pool.availableCount == 2)
        #expect(await pool.inUseCount == 0)
    }

    @Test("Pool returns nil when exhausted")
    func returnsNilWhenExhausted() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorLifecycleTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let pool = try await TensorPool(
            storageManager: storageManager,
            shape: .embedding(batchSize: 8, dimensions: 64),
            poolSize: 2
        )

        _ = await pool.acquire()
        _ = await pool.acquire()

        let third = await pool.acquire()
        #expect(third == nil)
    }

    @Test("Pool tracks statistics")
    func tracksStatistics() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorLifecycleTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let pool = try await TensorPool(
            storageManager: storageManager,
            shape: .embedding(batchSize: 8, dimensions: 64),
            poolSize: 2
        )

        let tensor1 = await pool.acquire()
        let tensor2 = await pool.acquire()
        _ = await pool.acquire()  // Miss

        await pool.release(tensor1!)
        await pool.release(tensor2!)

        let stats = await pool.statistics
        #expect(stats.acquireCount == 3)
        #expect(stats.releaseCount == 2)
        #expect(stats.missCount == 1)
        #expect(stats.hitRate < 1.0)
    }

    @Test("Pool can be cleared")
    func canBeCleared() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorLifecycleTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let pool = try await TensorPool(
            storageManager: storageManager,
            shape: .embedding(batchSize: 8, dimensions: 64),
            poolSize: 3
        )

        #expect(await pool.availableCount == 3)

        await pool.clear()

        #expect(await pool.availableCount == 0)
    }
    #endif
}

// MARK: - AutoReleaseTensor Tests

@Suite("AutoReleaseTensor")
struct AutoReleaseTensorTests {

    #if canImport(Metal)
    @Test("Auto-release tensor initializes correctly")
    func initializesCorrectly() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorLifecycleTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)

        let autoTensor = try await AutoReleaseTensor(
            batchSize: 8,
            dimensions: 64,
            label: "auto_tensor",
            storageManager: storageManager
        )

        #expect(autoTensor.tensor.label == "auto_tensor")
        if case .embedding(let batch, let dims) = autoTensor.shape {
            #expect(batch == 8)
            #expect(dims == 64)
        } else {
            #expect(Bool(false), "Expected embedding shape")
        }
    }

    @Test("Auto-release tensor provides buffer access")
    func providesBufferAccess() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorLifecycleTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)

        let autoTensor = try await AutoReleaseTensor(
            batchSize: 8,
            dimensions: 64,
            storageManager: storageManager
        )

        #expect(autoTensor.buffer.length > 0)
        #expect(autoTensor.sizeInBytes == autoTensor.tensor.sizeInBytes)
    }

    @Test("Auto-release tensor with token embeddings")
    func withTokenEmbeddings() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorLifecycleTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)

        let autoTensor = try await AutoReleaseTensor(
            batchSize: 2,
            sequenceLength: 32,
            dimensions: 64,
            storageManager: storageManager
        )

        if case .tokenEmbedding(let batch, let seq, let dims) = autoTensor.shape {
            #expect(batch == 2)
            #expect(seq == 32)
            #expect(dims == 64)
        } else {
            #expect(Bool(false), "Expected token embedding shape")
        }
    }

    @Test("Auto-release tensor wraps existing tensor")
    func wrapsExistingTensor() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorLifecycleTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)

        let tensor = try await storageManager.createEmbeddingTensor(
            batchSize: 8,
            dimensions: 64
        )

        let autoTensor = AutoReleaseTensor(tensor: tensor, storageManager: storageManager)

        #expect(autoTensor.tensor.id == tensor.id)
    }
    #endif
}

// MARK: - TensorStorageManager.withScope Tests

@Suite("TensorStorageManagerWithScope")
struct TensorStorageManagerWithScopeTests {

    #if canImport(Metal)
    @Test("WithScope extension works correctly")
    func withScopeExtensionWorks() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorLifecycleTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)

        let result: UUID = try await storageManager.withScope(label: "test") { scope in
            let tensor = try await scope.createEmbeddingTensor(batchSize: 8, dimensions: 64)

            // Tensor exists during scope
            let usage = await storageManager.getMemoryUsage()
            #expect(usage.tensorCount == 1)

            return tensor.id
        }

        // After scope, tensor should be released
        let usage = await storageManager.getMemoryUsage()
        #expect(usage.tensorCount == 0)
        // Verify we got a valid UUID (result is non-optional now)
        #expect(result.uuidString.isEmpty == false)
    }

    @Test("WithScope cleans up on error")
    func withScopeCleansUpOnError() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorLifecycleTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)

        do {
            _ = try await storageManager.withScope { scope in
                _ = try await scope.createEmbeddingTensor(batchSize: 8, dimensions: 64)

                throw TensorLifecycleTestError.intentionalError
            }
            #expect(Bool(false), "Should have thrown")
        } catch is TensorLifecycleTestError {
            // Expected
        }

        // Tensor should still be cleaned up
        let usage = await storageManager.getMemoryUsage()
        #expect(usage.tensorCount == 0)
    }
    #endif
}

// MARK: - Integration Tests

@Suite("TensorStorageLifecycleIntegration")
struct TensorStorageLifecycleIntegrationTests {

    #if canImport(Metal)
    @Test("Full lifecycle integration")
    func fullLifecycleIntegration() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorLifecycleTestError.skipped("Metal not available")
        }

        // Setup
        let storageManager = TensorStorageManager(device: device)
        let lifecycleManager = TensorLifecycleManager(storageManager: storageManager)

        // Create tensors via scope
        try await lifecycleManager.withScope(label: "processing") { scope in
            let input = try await scope.createTokenEmbeddingTensor(
                batchSize: 4,
                sequenceLength: 128,
                dimensions: 384,
                label: "input"
            )

            let output = try await scope.createEmbeddingTensor(
                batchSize: 4,
                dimensions: 384,
                label: "output"
            )

            // Verify both tensors exist
            let usage = await storageManager.getMemoryUsage()
            #expect(usage.tensorCount == 2)

            // Simulate processing
            input.markAccessed()
            output.markAccessed()
        }

        // After scope, all tensors should be released
        try? await Task.sleep(nanoseconds: 100_000_000)
        let finalUsage = await storageManager.getMemoryUsage()
        #expect(finalUsage.tensorCount == 0)

        // Check statistics
        let stats = await lifecycleManager.getStatistics()
        #expect(stats.scopesCreated == 1)
        #expect(stats.scopesFinalized == 1)
    }

    @Test("Pool integration with storage manager")
    func poolIntegrationWithStorageManager() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorLifecycleTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)

        // Create a pool
        let pool = try await TensorPool(
            storageManager: storageManager,
            shape: .embedding(batchSize: 32, dimensions: 384),
            poolSize: 3,
            label: "embedding_pool"
        )

        // Storage manager should track all pool tensors
        let usage = await storageManager.getMemoryUsage()
        #expect(usage.tensorCount == 3)

        // Acquire and release
        let tensor1 = await pool.acquire()
        let tensor2 = await pool.acquire()

        #expect(await pool.inUseCount == 2)

        await pool.release(tensor1!)
        await pool.release(tensor2!)

        #expect(await pool.availableCount == 3)

        // Clear pool
        await pool.clear()

        // Tensors should be released from storage manager
        let finalUsage = await storageManager.getMemoryUsage()
        #expect(finalUsage.tensorCount == 0)
    }
    #endif
}

// MARK: - Test Helper

enum TensorLifecycleTestError: Error {
    case skipped(String)
    case intentionalError
}

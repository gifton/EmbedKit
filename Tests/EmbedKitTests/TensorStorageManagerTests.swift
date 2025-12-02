// EmbedKit - Tensor Storage Manager Tests
//
// Tests for residency-aware tensor storage and lifecycle management.

import Testing
import Foundation
@testable import EmbedKit

#if canImport(Metal)
import Metal
#endif

// MARK: - ManagedTensor Tests

@Suite("ManagedTensor")
struct ManagedTensorTests {

    #if canImport(Metal)
    @Test("Managed tensor initializes correctly")
    func initializesCorrectly() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorStorageTestError.skipped("Metal not available")
        }

        let buffer = device.makeBuffer(length: 1024, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.embedding(batchSize: 4, dimensions: 64)
        let tensor = ManagedTensor(buffer: buffer, shape: shape, label: "test_tensor")

        #expect(tensor.label == "test_tensor")
        #expect(tensor.state == .active)
        #expect(tensor.isResident == false)
        #expect(tensor.accessCount == 0)
    }

    @Test("Managed tensor tracks access")
    func tracksAccess() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorStorageTestError.skipped("Metal not available")
        }

        let buffer = device.makeBuffer(length: 1024, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.embedding(batchSize: 4, dimensions: 64)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        #expect(tensor.accessCount == 0)

        tensor.markAccessed()
        #expect(tensor.accessCount == 1)

        tensor.markAccessed()
        tensor.markAccessed()
        #expect(tensor.accessCount == 3)
    }

    @Test("Managed tensor lifecycle state transitions")
    func lifecycleStateTransitions() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorStorageTestError.skipped("Metal not available")
        }

        let buffer = device.makeBuffer(length: 1024, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.embedding(batchSize: 4, dimensions: 64)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        #expect(tensor.state == .active)

        tensor.markCached()
        #expect(tensor.state == .cached)

        tensor.markReleased()
        #expect(tensor.state == .released)
    }

    @Test("Managed tensor residency tracking")
    func residencyTracking() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorStorageTestError.skipped("Metal not available")
        }

        let buffer = device.makeBuffer(length: 1024, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.embedding(batchSize: 4, dimensions: 64)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        #expect(tensor.isResident == false)
        #expect(tensor.residencySetName == nil)

        tensor.markResident(inSet: "embeddings")
        #expect(tensor.isResident == true)
        #expect(tensor.residencySetName == "embeddings")

        tensor.markNonResident()
        #expect(tensor.isResident == false)
        #expect(tensor.residencySetName == nil)
    }
    #endif
}

// MARK: - TensorShape Tests

@Suite("TensorShape")
struct TensorShapeTests {

    @Test("Embedding shape calculates correctly")
    func embeddingShapeCalculates() {
        let shape = ManagedTensor.TensorShape.embedding(batchSize: 32, dimensions: 384)

        #expect(shape.elementCount == 32 * 384)
        #expect(shape.sizeInBytes == 32 * 384 * MemoryLayout<Float>.stride)
    }

    @Test("Token embedding shape calculates correctly")
    func tokenEmbeddingShapeCalculates() {
        let shape = ManagedTensor.TensorShape.tokenEmbedding(
            batchSize: 4,
            sequenceLength: 128,
            dimensions: 384
        )

        #expect(shape.elementCount == 4 * 128 * 384)
        #expect(shape.sizeInBytes == 4 * 128 * 384 * MemoryLayout<Float>.stride)
    }

    @Test("Similarity matrix shape calculates correctly")
    func similarityMatrixShapeCalculates() {
        let shape = ManagedTensor.TensorShape.similarityMatrix(queryCount: 100, keyCount: 1000)

        #expect(shape.elementCount == 100 * 1000)
        #expect(shape.sizeInBytes == 100 * 1000 * MemoryLayout<Float>.stride)
    }

    @Test("Buffer shape calculates correctly")
    func bufferShapeCalculates() {
        let shape = ManagedTensor.TensorShape.buffer(length: 5000)

        #expect(shape.elementCount == 5000)
        #expect(shape.sizeInBytes == 5000 * MemoryLayout<Float>.stride)
    }

    @Test("Shapes are equatable")
    func shapesAreEquatable() {
        let shape1 = ManagedTensor.TensorShape.embedding(batchSize: 32, dimensions: 384)
        let shape2 = ManagedTensor.TensorShape.embedding(batchSize: 32, dimensions: 384)
        let shape3 = ManagedTensor.TensorShape.embedding(batchSize: 64, dimensions: 384)

        #expect(shape1 == shape2)
        #expect(shape1 != shape3)
    }

    @Test("Shapes are hashable")
    func shapesAreHashable() {
        let shape1 = ManagedTensor.TensorShape.embedding(batchSize: 32, dimensions: 384)
        let shape2 = ManagedTensor.TensorShape.embedding(batchSize: 32, dimensions: 384)

        var set: Set<ManagedTensor.TensorShape> = []
        set.insert(shape1)
        set.insert(shape2)

        #expect(set.count == 1)  // Same shape, should dedupe
    }
}

// MARK: - TensorStorageManager Tests

@Suite("TensorStorageManager")
struct TensorStorageManagerTests {

    #if canImport(Metal)
    @Test("Storage manager initializes correctly")
    func initializesCorrectly() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorStorageTestError.skipped("Metal not available")
        }

        let manager = TensorStorageManager(device: device)
        let usage = await manager.getMemoryUsage()

        #expect(usage.allocatedBytes == 0)
        #expect(usage.tensorCount == 0)
    }

    @Test("Creates embedding tensor")
    func createsEmbeddingTensor() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorStorageTestError.skipped("Metal not available")
        }

        let manager = TensorStorageManager(device: device)

        let tensor = try await manager.createEmbeddingTensor(
            batchSize: 32,
            dimensions: 384,
            label: "test_embeddings"
        )

        #expect(tensor.label == "test_embeddings")
        if case .embedding(let batch, let dims) = tensor.shape {
            #expect(batch == 32)
            #expect(dims == 384)
        } else {
            #expect(Bool(false), "Expected embedding shape")
        }

        let usage = await manager.getMemoryUsage()
        #expect(usage.tensorCount == 1)
        #expect(usage.allocatedBytes == tensor.sizeInBytes)
    }

    @Test("Creates token embedding tensor")
    func createsTokenEmbeddingTensor() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorStorageTestError.skipped("Metal not available")
        }

        let manager = TensorStorageManager(device: device)

        let tensor = try await manager.createTokenEmbeddingTensor(
            batchSize: 4,
            sequenceLength: 128,
            dimensions: 384,
            label: "token_embeddings"
        )

        if case .tokenEmbedding(let batch, let seq, let dims) = tensor.shape {
            #expect(batch == 4)
            #expect(seq == 128)
            #expect(dims == 384)
        } else {
            #expect(Bool(false), "Expected token embedding shape")
        }
    }

    @Test("Creates similarity tensor")
    func createsSimilarityTensor() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorStorageTestError.skipped("Metal not available")
        }

        let manager = TensorStorageManager(device: device)

        let tensor = try await manager.createSimilarityTensor(
            queryCount: 100,
            keyCount: 1000,
            label: "similarity_matrix"
        )

        if case .similarityMatrix(let q, let k) = tensor.shape {
            #expect(q == 100)
            #expect(k == 1000)
        } else {
            #expect(Bool(false), "Expected similarity matrix shape")
        }
    }

    @Test("Retrieves tensor by ID")
    func retrievesTensorById() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorStorageTestError.skipped("Metal not available")
        }

        let manager = TensorStorageManager(device: device)

        let tensor = try await manager.createEmbeddingTensor(
            batchSize: 8,
            dimensions: 64
        )

        let retrieved = await manager.getTensor(id: tensor.id)
        #expect(retrieved != nil)
        #expect(retrieved?.id == tensor.id)
    }

    @Test("Retrieves tensor by label")
    func retrievesTensorByLabel() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorStorageTestError.skipped("Metal not available")
        }

        let manager = TensorStorageManager(device: device)

        _ = try await manager.createEmbeddingTensor(
            batchSize: 8,
            dimensions: 64,
            label: "my_tensor"
        )

        let retrieved = await manager.getTensor(label: "my_tensor")
        #expect(retrieved != nil)
        #expect(retrieved?.label == "my_tensor")
    }

    @Test("Releases tensor correctly")
    func releasesTensorCorrectly() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorStorageTestError.skipped("Metal not available")
        }

        let manager = TensorStorageManager(device: device)

        let tensor = try await manager.createEmbeddingTensor(
            batchSize: 8,
            dimensions: 64,
            label: "to_release"
        )

        var usage = await manager.getMemoryUsage()
        #expect(usage.tensorCount == 1)

        await manager.release(tensor)

        usage = await manager.getMemoryUsage()
        #expect(usage.tensorCount == 0)
        #expect(usage.allocatedBytes == 0)

        // Tensor should be marked as released
        #expect(tensor.state == .released)
    }

    @Test("Releases tensor by ID")
    func releasesTensorById() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorStorageTestError.skipped("Metal not available")
        }

        let manager = TensorStorageManager(device: device)

        let tensor = try await manager.createEmbeddingTensor(
            batchSize: 8,
            dimensions: 64
        )

        await manager.release(id: tensor.id)

        let usage = await manager.getMemoryUsage()
        #expect(usage.tensorCount == 0)
    }

    @Test("Releases tensor by label")
    func releasesTensorByLabel() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorStorageTestError.skipped("Metal not available")
        }

        let manager = TensorStorageManager(device: device)

        _ = try await manager.createEmbeddingTensor(
            batchSize: 8,
            dimensions: 64,
            label: "labeled_tensor"
        )

        await manager.release(label: "labeled_tensor")

        let usage = await manager.getMemoryUsage()
        #expect(usage.tensorCount == 0)
    }

    @Test("Releases all tensors")
    func releasesAllTensors() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorStorageTestError.skipped("Metal not available")
        }

        let manager = TensorStorageManager(device: device)

        _ = try await manager.createEmbeddingTensor(batchSize: 8, dimensions: 64)
        _ = try await manager.createEmbeddingTensor(batchSize: 16, dimensions: 128)
        _ = try await manager.createEmbeddingTensor(batchSize: 32, dimensions: 256)

        var usage = await manager.getMemoryUsage()
        #expect(usage.tensorCount == 3)

        await manager.releaseAll()

        usage = await manager.getMemoryUsage()
        #expect(usage.tensorCount == 0)
        #expect(usage.allocatedBytes == 0)
    }

    @Test("Tracks statistics correctly")
    func tracksStatisticsCorrectly() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorStorageTestError.skipped("Metal not available")
        }

        let manager = TensorStorageManager(device: device)

        let tensor1 = try await manager.createEmbeddingTensor(batchSize: 8, dimensions: 64)
        _ = try await manager.createEmbeddingTensor(batchSize: 16, dimensions: 128)

        await manager.release(tensor1)

        let stats = await manager.getStatistics()
        #expect(stats.totalCreated == 2)
        #expect(stats.totalReleased == 1)
        #expect(stats.totalBytesAllocated > 0)
        #expect(stats.totalBytesFreed > 0)
    }

    @Test("Memory usage percentage calculated correctly")
    func memoryUsagePercentageCalculated() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorStorageTestError.skipped("Metal not available")
        }

        let config = TensorStorageManager.Configuration(maxMemoryBytes: 1024 * 1024)  // 1 MB
        let manager = TensorStorageManager(device: device, configuration: config)

        // Create a tensor that's about 50% of max
        _ = try await manager.createEmbeddingTensor(
            batchSize: 64,
            dimensions: 2048  // ~512 KB
        )

        let usage = await manager.getMemoryUsage()
        #expect(usage.usagePercentage > 0.4)
        #expect(usage.usagePercentage < 0.6)
    }

    @Test("Handles memory pressure")
    func handlesMemoryPressure() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorStorageTestError.skipped("Metal not available")
        }

        let config = TensorStorageManager.Configuration(
            maxMemoryBytes: 10 * 1024 * 1024,
            evictionIdleThreshold: 0  // Immediate eviction eligibility
        )
        let manager = TensorStorageManager(device: device, configuration: config)

        // Create some tensors
        let tensor1 = try await manager.createEmbeddingTensor(batchSize: 32, dimensions: 384)
        let tensor2 = try await manager.createEmbeddingTensor(batchSize: 32, dimensions: 384)

        // Mark as cached (eligible for eviction)
        await manager.markCached(tensor1)
        await manager.markCached(tensor2)

        var usage = await manager.getMemoryUsage()
        let initialCount = usage.tensorCount

        // Handle critical memory pressure
        await manager.handleMemoryPressure(level: 2)

        usage = await manager.getMemoryUsage()
        #expect(usage.tensorCount < initialCount)
    }
    #endif
}

// MARK: - TensorStorageManager Configuration Tests

@Suite("TensorStorageManagerConfiguration")
struct TensorStorageManagerConfigurationTests {

    @Test("Default configuration values")
    func defaultConfigurationValues() {
        let config = TensorStorageManager.Configuration.default

        #expect(config.maxMemoryBytes == 512 * 1024 * 1024)
        #expect(config.autoResidency == true)
        #expect(config.defaultResidencySet == "embeddings")
        #expect(config.evictionIdleThreshold == 60.0)
    }

    @Test("High memory configuration values")
    func highMemoryConfigurationValues() {
        let config = TensorStorageManager.Configuration.highMemory

        #expect(config.maxMemoryBytes == 1024 * 1024 * 1024)
        #expect(config.evictionIdleThreshold == 120.0)
    }

    @Test("Low memory configuration values")
    func lowMemoryConfigurationValues() {
        let config = TensorStorageManager.Configuration.lowMemory

        #expect(config.maxMemoryBytes == 128 * 1024 * 1024)
        #expect(config.evictionIdleThreshold == 30.0)
    }

    @Test("Configuration enforces minimum memory")
    func configurationEnforcesMinimumMemory() {
        let config = TensorStorageManager.Configuration(maxMemoryBytes: 100)

        #expect(config.maxMemoryBytes >= 1024 * 1024)  // Min 1 MB
    }
}

// MARK: - TensorHandle Tests

@Suite("TensorHandle")
struct TensorHandleTests {

    #if canImport(Metal)
    @Test("Handle created from tensor")
    func handleCreatedFromTensor() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorStorageTestError.skipped("Metal not available")
        }

        let buffer = device.makeBuffer(length: 1024, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.embedding(batchSize: 4, dimensions: 64)
        let tensor = ManagedTensor(buffer: buffer, shape: shape, label: "test")

        let handle = TensorHandle(from: tensor)

        #expect(handle.id == tensor.id)
        #expect(handle.label == tensor.label)
        #expect(handle.shape == tensor.shape)
    }

    @Test("Handles are hashable")
    func handlesAreHashable() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorStorageTestError.skipped("Metal not available")
        }

        let buffer1 = device.makeBuffer(length: 1024, options: .storageModeShared)!
        let buffer2 = device.makeBuffer(length: 1024, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.embedding(batchSize: 4, dimensions: 64)

        let tensor1 = ManagedTensor(buffer: buffer1, shape: shape)
        let tensor2 = ManagedTensor(buffer: buffer2, shape: shape)

        let handle1 = TensorHandle(from: tensor1)
        let handle2 = TensorHandle(from: tensor2)

        var set: Set<TensorHandle> = []
        set.insert(handle1)
        set.insert(handle2)

        #expect(set.count == 2)  // Different tensors, different handles
    }
    #endif
}

// MARK: - Test Helper

enum TensorStorageTestError: Error {
    case skipped(String)
}

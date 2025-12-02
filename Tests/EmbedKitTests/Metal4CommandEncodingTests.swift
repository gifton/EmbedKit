// EmbedKit - Metal 4 Command Encoding Tests

import Testing
import Foundation
@testable import EmbedKit

#if canImport(Metal)
import Metal
#endif

// MARK: - MetalOperation Tests

@Suite("MetalOperation")
struct MetalOperationTests {

    #if canImport(Metal)
    @Test("MetalOperation stores all properties correctly")
    func metaOperationProperties() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4TestError.skipped("Metal not available")
        }

        // Create a simple function for testing
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void test_kernel(device float* data [[buffer(0)]],
                                uint id [[thread_position_in_grid]]) {
            data[id] = data[id] * 2.0;
        }
        """

        let library = try await device.makeLibrary(source: source, options: nil)
        let function = library.makeFunction(name: "test_kernel")!
        let pso = try await device.makeComputePipelineState(function: function)

        let buffer = device.makeBuffer(length: 1024, options: .storageModeShared)!

        let operation = MetalOperation(
            pipelineState: pso,
            bufferBindings: [(index: 0, buffer: buffer, offset: 0)],
            threadgroups: MTLSize(width: 16, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1),
            dependsOnPrevious: true
        )

        #expect(operation.threadgroups.width == 16)
        #expect(operation.threadsPerThreadgroup.width == 64)
        #expect(operation.dependsOnPrevious == true)
        #expect(operation.bufferBindings.count == 1)
        #expect(operation.bufferBindings[0].index == 0)
    }

    @Test("MetalOperation default dependency is true")
    func defaultDependency() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4TestError.skipped("Metal not available")
        }

        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void noop() {}
        """

        let library = try await device.makeLibrary(source: source, options: nil)
        let function = library.makeFunction(name: "noop")!
        let pso = try await device.makeComputePipelineState(function: function)

        let operation = MetalOperation(
            pipelineState: pso,
            bufferBindings: [],
            threadgroups: MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
        )

        #expect(operation.dependsOnPrevious == true)
    }
    #endif
}

// MARK: - Metal4UnifiedEncoder Tests

@Suite("Metal4UnifiedEncoder")
struct Metal4UnifiedEncoderTests {

    #if canImport(Metal)
    @Test("Unified encoder initializes correctly")
    func initializesCorrectly() async throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let cmd = queue.makeCommandBuffer() else {
            throw Metal4TestError.skipped("Metal not available")
        }

        let encoder = Metal4UnifiedEncoder(commandBuffer: cmd)

        #expect(encoder.count == 0)
        #expect(encoder.finalized == false)
    }

    @Test("Unified encoder adds operations")
    func addsOperations() async throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let cmd = queue.makeCommandBuffer() else {
            throw Metal4TestError.skipped("Metal not available")
        }

        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void noop() {}
        """

        let library = try await device.makeLibrary(source: source, options: nil)
        let function = library.makeFunction(name: "noop")!
        let pso = try await device.makeComputePipelineState(function: function)

        let encoder = Metal4UnifiedEncoder(commandBuffer: cmd)

        let op1 = MetalOperation(
            pipelineState: pso,
            bufferBindings: [],
            threadgroups: MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1),
            dependsOnPrevious: false
        )

        let op2 = MetalOperation(
            pipelineState: pso,
            bufferBindings: [],
            threadgroups: MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1),
            dependsOnPrevious: true
        )

        try encoder.addOperation(op1)
        #expect(encoder.count == 1)

        try encoder.addOperation(op2)
        #expect(encoder.count == 2)

        encoder.finalize()
        #expect(encoder.finalized == true)
    }

    @Test("Cannot add operations after finalize")
    func cannotAddAfterFinalize() async throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let cmd = queue.makeCommandBuffer() else {
            throw Metal4TestError.skipped("Metal not available")
        }

        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void noop() {}
        """

        let library = try await device.makeLibrary(source: source, options: nil)
        let function = library.makeFunction(name: "noop")!
        let pso = try await device.makeComputePipelineState(function: function)

        let encoder = Metal4UnifiedEncoder(commandBuffer: cmd)
        encoder.finalize()

        let operation = MetalOperation(
            pipelineState: pso,
            bufferBindings: [],
            threadgroups: MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
        )

        #expect(throws: EmbedKitError.self) {
            try encoder.addOperation(operation)
        }
    }
    #endif
}

// MARK: - CommandAllocatorPool Tests

@Suite("CommandAllocatorPool")
struct CommandAllocatorPoolTests {

    #if canImport(Metal)
    @Test("Pool initializes with correct size")
    func initializesWithCorrectSize() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4TestError.skipped("Metal not available")
        }

        let pool = CommandAllocatorPool(device: device, poolSize: 5, heapSizeMB: 8)
        let stats = await pool.getStatistics()

        #expect(stats.totalAllocators == 5)
        #expect(stats.availableAllocators == 5)
        #expect(stats.inUseAllocators == 0)
        #expect(stats.heapSizePerAllocator == 8 * 1024 * 1024)
    }

    @Test("Acquire and release allocators")
    func acquireAndRelease() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4TestError.skipped("Metal not available")
        }

        let pool = CommandAllocatorPool(device: device, poolSize: 3)

        // Acquire first allocator
        let alloc1 = await pool.acquire()
        var stats = await pool.getStatistics()
        #expect(stats.availableAllocators == 2)
        #expect(stats.inUseAllocators == 1)

        // Acquire second allocator
        let alloc2 = await pool.acquire()
        stats = await pool.getStatistics()
        #expect(stats.availableAllocators == 1)
        #expect(stats.inUseAllocators == 2)

        // Release first allocator
        await pool.release(alloc1)
        stats = await pool.getStatistics()
        #expect(stats.availableAllocators == 2)
        #expect(stats.inUseAllocators == 1)

        // Release second allocator
        await pool.release(alloc2)
        stats = await pool.getStatistics()
        #expect(stats.availableAllocators == 3)
        #expect(stats.inUseAllocators == 0)
    }

    @Test("Try acquire returns nil when pool exhausted")
    func tryAcquireReturnsNil() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4TestError.skipped("Metal not available")
        }

        let pool = CommandAllocatorPool(device: device, poolSize: 2)

        // Exhaust the pool
        _ = await pool.acquire()
        _ = await pool.acquire()

        // Try acquire should return nil
        let result = await pool.tryAcquire()
        #expect(result == nil)
    }

    @Test("Release all resets entire pool")
    func releaseAllResetsPool() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4TestError.skipped("Metal not available")
        }

        let pool = CommandAllocatorPool(device: device, poolSize: 3)

        // Acquire all allocators
        _ = await pool.acquire()
        _ = await pool.acquire()
        _ = await pool.acquire()

        var stats = await pool.getStatistics()
        #expect(stats.inUseAllocators == 3)

        // Release all
        await pool.releaseAll()

        stats = await pool.getStatistics()
        #expect(stats.availableAllocators == 3)
        #expect(stats.inUseAllocators == 0)
    }

    @Test("Statistics track acquisitions and resets")
    func statisticsTrackUsage() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4TestError.skipped("Metal not available")
        }

        let pool = CommandAllocatorPool(device: device, poolSize: 2)

        let alloc1 = await pool.acquire()
        let alloc2 = await pool.acquire()
        await pool.release(alloc1)
        await pool.release(alloc2)

        let stats = await pool.getStatistics()
        #expect(stats.totalAcquisitions == 2)
        #expect(stats.totalResets == 2)
    }

    @Test("Default configuration is reasonable")
    func defaultConfiguration() async throws {
        let config = CommandAllocatorPool.Configuration.default

        #expect(config.poolSize == 3)
        #expect(config.heapSizePerAllocator == 16 * 1024 * 1024)
    }

    @Test("Large batch configuration has larger heaps")
    func largeBatchConfiguration() async throws {
        let config = CommandAllocatorPool.Configuration.largeBatch

        #expect(config.poolSize == 4)
        #expect(config.heapSizePerAllocator == 64 * 1024 * 1024)
    }
    #endif
}

// MARK: - SimpleCommandAllocator Tests

@Suite("SimpleCommandAllocator")
struct SimpleCommandAllocatorTests {

    #if canImport(Metal)
    @Test("Simple allocator tracks usage state")
    func tracksUsageState() throws {
        let allocator = SimpleCommandAllocator(heapSize: 1024)

        #expect(allocator.isInUse == false)
        #expect(allocator.heapSize == 1024)

        allocator.markInUse()
        #expect(allocator.isInUse == true)

        allocator.reset()
        #expect(allocator.isInUse == false)
    }

    @Test("Simple allocator uses default heap size")
    func defaultHeapSize() throws {
        let allocator = SimpleCommandAllocator()

        #expect(allocator.heapSize == 16 * 1024 * 1024)
    }
    #endif
}

// MARK: - Test Helper

enum Metal4TestError: Error {
    case skipped(String)
}

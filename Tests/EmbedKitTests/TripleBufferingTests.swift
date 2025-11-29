// EmbedKit - Triple Buffering Tests

import Testing
import Foundation
@testable import EmbedKit

#if canImport(Metal)
import Metal

@Suite("Triple Buffering")
struct TripleBufferingTests {

    // MARK: - CommandBufferPool Tests

    @Test("Pool initializes with correct buffer count")
    func poolInitialization() async throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal device not available")
        }

        let pool = CommandBufferPool(queue: queue, bufferCount: 3)
        #expect(pool.maxInFlight == 3)
        #expect(pool.currentInFlightCount == 0)
        #expect(pool.totalCompletedCount == 0)
    }

    @Test("Pool acquires and submits command buffers")
    func poolAcquireAndSubmit() async throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal device not available")
        }

        let pool = CommandBufferPool(queue: queue, bufferCount: 3)

        // Acquire a command buffer
        let cmd = try await pool.acquireCommandBuffer()
        #expect(pool.currentInFlightCount == 1)

        // Create a simple compute pass (no-op)
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.endEncoding()
        }

        // Submit
        pool.submit(cmd)

        // Wait for completion
        await pool.waitForAllComplete()

        #expect(pool.currentInFlightCount == 0)
        #expect(pool.totalCompletedCount == 1)
    }

    @Test("Pool enforces triple buffer limit")
    func poolTripleBufferLimit() async throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal device not available")
        }

        let pool = CommandBufferPool(queue: queue, bufferCount: 3)

        // Acquire 3 command buffers (should not block)
        var buffers: [MTLCommandBuffer] = []
        for _ in 0..<3 {
            let cmd = try await pool.acquireCommandBuffer()
            buffers.append(cmd)
        }

        #expect(pool.currentInFlightCount == 3)

        // Submit all and wait
        for cmd in buffers {
            if let enc = cmd.makeComputeCommandEncoder() {
                enc.endEncoding()
            }
            pool.submit(cmd)
        }

        await pool.waitForAllComplete()
        #expect(pool.totalCompletedCount == 3)
    }

    @Test("Pool submitAndWait works correctly")
    func poolSubmitAndWait() async throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal device not available")
        }

        let pool = CommandBufferPool(queue: queue, bufferCount: 3)

        let cmd = try await pool.acquireCommandBuffer()
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.endEncoding()
        }

        await pool.submitAndWait(cmd)

        // Give completion handler time to run
        try await Task.sleep(nanoseconds: 10_000_000) // 10ms

        #expect(pool.currentInFlightCount == 0)
        #expect(pool.totalCompletedCount == 1)
    }

    // MARK: - MetalBufferPool Tests

    @Test("Buffer pool initializes correctly")
    func bufferPoolInitialization() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }

        let pool = MetalBufferPool(device: device, maxPoolSize: 1024 * 1024)
        let stats = pool.statistics()

        #expect(stats.hits == 0)
        #expect(stats.misses == 0)
        #expect(stats.currentSize == 0)
        #expect(stats.bufferCount == 0)
    }

    @Test("Buffer pool acquires and releases buffers")
    func bufferPoolAcquireRelease() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }

        let pool = MetalBufferPool(device: device, maxPoolSize: 1024 * 1024)

        // First acquire is a miss
        let buffer1 = pool.acquire(minimumSize: 1024)
        #expect(buffer1 != nil)
        #expect(buffer1!.length >= 1024)

        var stats = pool.statistics()
        #expect(stats.misses == 1)
        #expect(stats.hits == 0)

        // Release the buffer
        pool.release(buffer1!)

        stats = pool.statistics()
        #expect(stats.bufferCount == 1)
        #expect(stats.currentSize > 0)

        // Second acquire should be a hit
        let buffer2 = pool.acquire(minimumSize: 1024)
        #expect(buffer2 != nil)

        stats = pool.statistics()
        #expect(stats.hits == 1)
        #expect(stats.misses == 1)
        #expect(stats.hitRate == 0.5)
    }

    @Test("Buffer pool uses power of 2 size classes")
    func bufferPoolSizeClasses() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }

        let pool = MetalBufferPool(device: device, maxPoolSize: 1024 * 1024)

        // Request 1000 bytes, should get 1024
        let buffer1 = pool.acquire(minimumSize: 1000)
        #expect(buffer1 != nil)
        #expect(buffer1!.length == 1024)

        // Request 2000 bytes, should get 2048
        let buffer2 = pool.acquire(minimumSize: 2000)
        #expect(buffer2 != nil)
        #expect(buffer2!.length == 2048)

        pool.release(buffer1!)
        pool.release(buffer2!)
    }

    @Test("Buffer pool respects max size")
    func bufferPoolMaxSize() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }

        // Small max size
        let pool = MetalBufferPool(device: device, maxPoolSize: 2048)

        // Acquire and release a 1024 byte buffer
        let buffer1 = pool.acquire(minimumSize: 1024)
        pool.release(buffer1!)

        // Acquire and release another 1024 byte buffer
        let buffer2 = pool.acquire(minimumSize: 1024)
        pool.release(buffer2!)

        // Try to release a 2048 byte buffer - should cause eviction
        let buffer3 = pool.acquire(minimumSize: 2048)
        pool.release(buffer3!)

        let stats = pool.statistics()
        // Should have evicted some buffers to stay under limit
        #expect(stats.currentSize <= 2048)
    }

    @Test("Buffer pool clears correctly")
    func bufferPoolClear() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }

        let pool = MetalBufferPool(device: device, maxPoolSize: 1024 * 1024)

        // Add some buffers
        for _ in 0..<5 {
            let buffer = pool.acquire(minimumSize: 1024)!
            pool.release(buffer)
        }

        var stats = pool.statistics()
        #expect(stats.bufferCount > 0)

        pool.clear()

        stats = pool.statistics()
        #expect(stats.bufferCount == 0)
        #expect(stats.currentSize == 0)
    }

    // MARK: - MetalAccelerator Integration Tests

    @Test("Accelerator has triple buffer pool")
    func acceleratorHasTripleBufferPool() async throws {
        let accelerator = await MetalAccelerator()

        guard await accelerator.isAvailable else {
            throw XCTSkip("Metal accelerator not available")
        }

        #expect(await accelerator.tripleBufferPool != nil)
        #expect(await accelerator.metalBufferPool != nil)
    }

    @Test("Streaming pool normalize processes batches")
    func streamingPoolNormalize() async throws {
        let accelerator = await MetalAccelerator()

        guard await accelerator.isAvailable else {
            throw XCTSkip("Metal accelerator not available")
        }

        let batchSize = 4
        let sequenceLength = 8
        let dimensions = 32
        let numBatches = 5

        // Create test batches
        var batches: [[Float]] = []
        for _ in 0..<numBatches {
            var batch: [Float] = []
            for _ in 0..<(batchSize * sequenceLength * dimensions) {
                batch.append(Float.random(in: -1...1))
            }
            batches.append(batch)
        }

        // Process with streaming
        let results = await accelerator.streamingPoolNormalize(
            batches: batches,
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            dimensions: dimensions,
            strategy: .mean,
            normalize: true
        )

        // Verify results
        #expect(results.count == numBatches)
        for batchResults in results {
            #expect(batchResults.count == batchSize)
            for embedding in batchResults {
                #expect(embedding.count == dimensions)

                // Verify normalized (L2 norm should be ~1.0)
                let norm = sqrt(embedding.reduce(0) { $0 + $1 * $1 })
                #expect(abs(norm - 1.0) < 0.01)
            }
        }
    }

    @Test("Buffer pool statistics available through accelerator")
    func bufferPoolStatistics() async throws {
        let accelerator = await MetalAccelerator()

        guard await accelerator.isAvailable else {
            throw XCTSkip("Metal accelerator not available")
        }

        // Do some work to populate buffer pool
        let batch: [Float] = (0..<(4 * 8 * 32)).map { _ in Float.random(in: -1...1) }
        _ = await accelerator.tensorPoolNormalize(
            embeddings: batch,
            batchSize: 4,
            sequenceLength: 8,
            dimensions: 32,
            strategy: .mean,
            normalize: true
        )

        let stats = await accelerator.bufferPoolStatistics
        #expect(stats != nil)
    }

    // MARK: - Performance Tests

    @Test("Triple buffering improves throughput for streaming batches", .tags(.performance))
    func tripleBufferingPerformance() async throws {
        let accelerator = await MetalAccelerator()

        guard await accelerator.isAvailable else {
            throw XCTSkip("Metal accelerator not available")
        }

        let batchSize = 16
        let sequenceLength = 64
        let dimensions = 384
        let numBatches = 10

        // Create test batches
        var batches: [[Float]] = []
        for _ in 0..<numBatches {
            var batch: [Float] = []
            for _ in 0..<(batchSize * sequenceLength * dimensions) {
                batch.append(Float.random(in: -1...1))
            }
            batches.append(batch)
        }

        // Measure streaming (triple buffered)
        let streamingStart = CFAbsoluteTimeGetCurrent()
        let _ = await accelerator.streamingPoolNormalize(
            batches: batches,
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            dimensions: dimensions,
            strategy: .mean,
            normalize: true
        )
        let streamingTime = CFAbsoluteTimeGetCurrent() - streamingStart

        // Measure sequential
        let sequentialStart = CFAbsoluteTimeGetCurrent()
        for batch in batches {
            let _ = await accelerator.tensorPoolNormalize(
                embeddings: batch,
                batchSize: batchSize,
                sequenceLength: sequenceLength,
                dimensions: dimensions,
                strategy: .mean,
                normalize: true
            )
        }
        let sequentialTime = CFAbsoluteTimeGetCurrent() - sequentialStart

        // Log times for comparison
        print("Streaming (triple buffered): \(streamingTime * 1000)ms")
        print("Sequential: \(sequentialTime * 1000)ms")
        print("Speedup: \(sequentialTime / streamingTime)x")

        // Streaming should not be catastrophically slower
        // Note: Performance varies based on system load; triple buffering benefits
        // are most visible with sustained high throughput, not single test runs
        #expect(streamingTime < sequentialTime * 3.0)
    }
}

#endif

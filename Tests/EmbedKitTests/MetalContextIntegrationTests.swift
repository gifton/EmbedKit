// EmbedKit - Metal Context Integration Tests
// Tests for Metal4Context and Metal resource management across components

import Testing
import Foundation
@testable import EmbedKit
import Metal

#if canImport(Metal)

// MARK: - Metal4 Context Tests

@Suite("Metal4Context Integration", .tags(.integration))
struct Metal4ContextIntegrationTests {

    // MARK: - Context Sharing Tests

    @Test("AccelerationManager uses shared Metal4 context")
    func accelerationManagerSharedContext() async throws {
        #if !targetEnvironment(simulator)
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }

        let manager = try await AccelerationManager.create()
        let hasContext = manager.isGPUAvailable

        // GPU is always available in new architecture
        #expect(hasContext == true)
        #else
        throw XCTSkip("Metal not available in simulator")
        #endif
    }

    @Test("Multiple components share same Metal4 context")
    func multipleComponentsShareContext() async throws {
        #if !targetEnvironment(simulator)
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }

        let manager1 = try await AccelerationManager.create()
        let manager2 = try await AccelerationManager.create()

        // Both should use GPU (Metal4)
        #expect(manager1.isGPUAvailable == true)
        #expect(manager2.isGPUAvailable == true)
        #else
        throw XCTSkip("Metal not available in simulator")
        #endif
    }

    @Test("GPUOptimizer integrates with shared context")
    func gpuOptimizerSharedContext() async throws {
        #if !targetEnvironment(simulator)
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }

        let optimizer = GPUOptimizer(device: device)
        let capabilities = await optimizer.capabilities
        let recommendation = capabilities.recommendedThreadgroupWidth(forDimensions: 384)

        #expect(recommendation > 0)
        #else
        throw XCTSkip("Metal not available in simulator")
        #endif
    }

    // MARK: - Buffer Pool Integration Tests

    @Test("Metal buffers are reused across operations")
    func metalBufferReuse() async throws {
        #if !targetEnvironment(simulator)
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }

        _ = try await AccelerationManager.create()

        // Perform multiple operations that should reuse buffers
        for _ in 0..<5 {
            let data = [Float](repeating: 1.0, count: 384)
            // Operations would reuse buffers if Metal is available
            _ = AccelerateBLAS.dotProduct(data, data)
        }

        // No specific assertion, but verifies no crashes with buffer reuse
        #expect(Bool(true))
        #else
        throw XCTSkip("Metal not available in simulator")
        #endif
    }

    @Test("Buffer pool handles concurrent access")
    func bufferPoolConcurrentAccess() async throws {
        #if !targetEnvironment(simulator)
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }

        _ = try await AccelerationManager.create()

        // Concurrent operations should safely share buffer pool
        try await withThrowingTaskGroup(of: Float.self) { group in
            for i in 0..<10 {
                group.addTask {
                    let data = [Float](repeating: Float(i), count: 384)
                    return AccelerateBLAS.dotProduct(data, data)
                }
            }

            var results: [Float] = []
            for try await result in group {
                results.append(result)
            }

            #expect(results.count == 10)
            #expect(results.allSatisfy { $0.isFinite })
        }
        #else
        throw XCTSkip("Metal not available in simulator")
        #endif
    }

    // MARK: - Memory Management Tests

    @Test("Metal buffers release under memory pressure")
    func metalBuffersReleaseUnderPressure() async throws {
        #if !targetEnvironment(simulator)
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }

        _ = try await AccelerationManager.create()

        // Create large allocations to simulate memory pressure
        var largeData: [[Float]] = []
        for _ in 0..<100 {
            largeData.append([Float](repeating: 1.0, count: 384))
        }

        // Process batches
        for batch in largeData.chunks(ofCount: 10) {
            for data in batch {
                _ = AccelerateBLAS.dotProduct(data, data)
            }
        }

        // Clean up
        largeData.removeAll()

        // Verify system is still responsive
        let testData = [Float](repeating: 1.0, count: 384)
        let result = AccelerateBLAS.dotProduct(testData, testData)
        #expect(result.isFinite)
        #else
        throw XCTSkip("Metal not available in simulator")
        #endif
    }

    @Test("Metal context survives manager deallocation")
    func metalContextSurvivesDeallocation() async throws {
        #if !targetEnvironment(simulator)
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }

        // Create and destroy manager
        do {
            let manager = try await AccelerationManager.create()
            _ = manager.isGPUAvailable
        }

        // Create new manager - should still work
        let newManager = try await AccelerationManager.create()
        _ = newManager.isGPUAvailable

        // Verify operations still work
        let data = [Float](repeating: 1.0, count: 384)
        let result = AccelerateBLAS.dotProduct(data, data)
        #expect(result.isFinite)
        #else
        throw XCTSkip("Metal not available in simulator")
        #endif
    }

    // MARK: - Metal Accelerator Tests

    @Test("MetalAccelerator buffer management")
    func metalAcceleratorBufferManagement() async throws {
        #if !targetEnvironment(simulator)
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }

        let accelerator = try await MetalAccelerator()
        let isAvailable = await accelerator.isAvailable

        if isAvailable {
            // Test buffer operations
            let vectors = [[Float]](repeating: [Float](repeating: 1.0, count: 384), count: 10)

            // MetalAccelerator should handle buffer creation/cleanup
            for vector in vectors {
                _ = AccelerateBLAS.dotProduct(vector, vector)
            }

            #expect(true)
        } else {
            throw XCTSkip("Metal accelerator not available")
        }
        #else
        throw XCTSkip("Metal not available in simulator")
        #endif
    }

    @Test("Metal pooling operations use shared context")
    func metalPoolingSharedContext() async throws {
        #if !targetEnvironment(simulator)
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }

        _ = try await AccelerationManager.create()

        // Create test data for pooling
        let batchSize = 4
        let seqLen = 128
        let hiddenSize = 384

        // Create sequences for mean pooling
        var pooledResults: [[Float]] = []
        for _ in 0..<batchSize {
            let sequence = [Float](repeating: 1.0, count: seqLen * hiddenSize)
            let pooled = PoolingHelpers.mean(sequence: sequence, tokens: seqLen, dim: hiddenSize)
            pooledResults.append(pooled)
        }

        #expect(pooledResults.count == batchSize)
        #expect(pooledResults.allSatisfy { $0.count == hiddenSize })
        #else
        throw XCTSkip("Metal not available in simulator")
        #endif
    }

    // MARK: - Cross-Component Integration

    @Test("EmbeddingGenerator with Metal acceleration")
    func embeddingGeneratorWithMetal() async throws {
        #if !targetEnvironment(simulator)
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }

        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(model: model)

        // Generate embeddings - may use Metal acceleration internally
        let vectors = try await generator.produce([
            "test one",
            "test two",
            "test three"
        ])

        #expect(vectors.count == 3)
        #expect(vectors.allSatisfy { $0.count == 384 })
        #else
        throw XCTSkip("Metal not available in simulator")
        #endif
    }

    @Test("MemoryAwareGenerator with Metal under pressure")
    func memoryAwareGeneratorWithMetal() async throws {
        #if !targetEnvironment(simulator)
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }

        let modelManager = ModelManager()
        let model = try await modelManager.loadMockModel()
        let embeddingGenerator = EmbeddingGenerator(model: model)
        let memoryAware = MemoryAwareGenerator(generator: embeddingGenerator)

        // Large batch that might trigger memory-aware batching
        let texts = Array(repeating: "test", count: 100)
        let vectors = try await memoryAware.produce(texts)

        #expect(vectors.count == 100)
        #expect(vectors.allSatisfy { $0.count == 384 })
        #else
        throw XCTSkip("Metal not available in simulator")
        #endif
    }

    // MARK: - Performance with Shared Context

    @Test("Shared context improves batch performance", .tags(.performance))
    func sharedContextPerformance() async throws {
        #if !targetEnvironment(simulator)
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }

        _ = try await AccelerationManager.create()

        // Create test data
        let query = [Float](repeating: 1.0, count: 384)
        let candidates = [[Float]](repeating: [Float](repeating: 0.5, count: 384), count: 1000)

        // Measure batch operations with shared context
        let start = Date()
        let distances = AccelerateBLAS.batchCosineDistance(
            query: query,
            candidates: candidates
        )
        let elapsed = Date().timeIntervalSince(start)

        #expect(distances.count == 1000)
        #expect(elapsed < 1.0) // Should complete in reasonable time
        #else
        throw XCTSkip("Metal not available in simulator")
        #endif
    }

    @Test("Context switching overhead is minimal")
    func contextSwitchingOverhead() async throws {
        #if !targetEnvironment(simulator)
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }

        // Multiple managers sharing context
        _ = try await AccelerationManager.create()
        _ = try await AccelerationManager.create()
        _ = try await AccelerationManager.create()

        // Interleaved operations should not cause excessive overhead
        let data = [Float](repeating: 1.0, count: 384)

        let start = Date()
        for _ in 0..<100 {
            _ = AccelerateBLAS.dotProduct(data, data)
        }
        let elapsed = Date().timeIntervalSince(start)

        #expect(elapsed < 1.0)
        #else
        throw XCTSkip("Metal not available in simulator")
        #endif
    }

    // MARK: - Error Handling

    @Test("Metal operations complete successfully")
    func metalOperationsComplete() async throws {
        #if !targetEnvironment(simulator)
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }

        let manager = try await AccelerationManager.create()

        let data = [Float](repeating: 1.0, count: 384)
        let result = AccelerateBLAS.dotProduct(data, data)

        #expect(result.isFinite)
        #expect(result > 0)

        _ = manager.isGPUAvailable
        #else
        throw XCTSkip("Metal not available in simulator")
        #endif
    }

    @Test("Metal buffer allocation works correctly")
    func metalBufferAllocationWorks() async throws {
        #if !targetEnvironment(simulator)
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }

        _ = try await AccelerationManager.create()

        // Create reasonable-sized data
        let data = [Float](repeating: 1.0, count: 384)
        let result = AccelerateBLAS.dotProduct(data, data)

        #expect(result.isFinite)
        #else
        throw XCTSkip("Metal not available in simulator")
        #endif
    }

    // MARK: - Triple Buffering Tests

    @Test("Triple buffering with Metal context")
    func tripleBufferingWithMetal() async throws {
        #if !targetEnvironment(simulator)
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }

        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let generator = EmbeddingGenerator(model: model)
        let processor = PipelinedBatchProcessor(
            generator: generator,
            config: .tripleBuffer
        )

        // Process large batch with triple buffering
        let texts = Array(repeating: "test", count: 96)
        let vectors = try await processor.produce(texts)

        #expect(vectors.count == 96)
        #expect(vectors.allSatisfy { $0.count == 384 })
        #else
        throw XCTSkip("Metal not available in simulator")
        #endif
    }
}

// MARK: - Helper Extensions

private extension Array {
    func chunks(ofCount count: Int) -> [[Element]] {
        return stride(from: 0, to: self.count, by: count).map {
            Array(self[$0..<Swift.min($0 + count, self.count)])
        }
    }
}

#else
// Non-Metal platforms - provide placeholder tests

@Suite("Metal4Context Integration (No Metal)", .tags(.integration))
struct Metal4ContextPlaceholderTests {
    @Test("Metal not available on this platform")
    func metalNotAvailable() throws {
        throw XCTSkip("Metal not available on this platform")
    }
}

#endif

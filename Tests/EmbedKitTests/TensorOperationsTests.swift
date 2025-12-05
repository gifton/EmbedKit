// EmbedKit - Tensor Operations Tests
//
// Tests for Metal 4 tensor operations:
// - TensorTypes (EmbeddingTensor, TokenEmbeddingTensor)
// - tensorPoolNormalize()
// - tensorSimilarityMatrix()
// - Fused operations

import Testing
import Foundation
@testable import EmbedKit

#if canImport(Metal)
import Metal
#endif

// MARK: - Tensor Types Tests

@Suite("Tensor Types")
struct TensorTypesTests {

    #if canImport(Metal)
    @Test("EmbeddingTensor creates from embeddings")
    func embeddingTensorFromEmbeddings() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorTestError.metalNotAvailable
        }

        let embeddings: [[Float]] = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ]

        let tensor = try EmbeddingTensor(embeddings: embeddings, device: device)

        #expect(tensor.batchSize == 3)
        #expect(tensor.dimensions == 4)
        #expect(tensor.totalElements == 12)
    }

    @Test("EmbeddingTensor reads back correctly")
    func embeddingTensorReadback() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorTestError.metalNotAvailable
        }

        let original: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]

        let tensor = try EmbeddingTensor(embeddings: original, device: device)
        let readback = tensor.toEmbeddings()

        #expect(readback.count == original.count)
        for (orig, read) in zip(original, readback) {
            for (a, b) in zip(orig, read) {
                #expect(abs(a - b) < 1e-6)
            }
        }
    }

    @Test("EmbeddingTensor accesses single embedding")
    func embeddingTensorSingleAccess() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorTestError.metalNotAvailable
        }

        let embeddings: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]

        let tensor = try EmbeddingTensor(embeddings: embeddings, device: device)

        let second = tensor.embedding(at: 1)
        #expect(second == [4.0, 5.0, 6.0])
    }

    @Test("TokenEmbeddingTensor creates from tokens")
    func tokenEmbeddingTensorFromTokens() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorTestError.metalNotAvailable
        }

        // [batchSize=2][seqLen=3][dims=4]
        let tokens: [[[Float]]] = [
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]
        ]

        let tensor = try TokenEmbeddingTensor(tokens: tokens, device: device)

        #expect(tensor.batchSize == 2)
        #expect(tensor.sequenceLength == 3)
        #expect(tensor.dimensions == 4)
        #expect(tensor.totalElements == 24)
    }

    @Test("TokenEmbeddingTensor reads back correctly")
    func tokenEmbeddingTensorReadback() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorTestError.metalNotAvailable
        }

        let original: [[[Float]]] = [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ]

        let tensor = try TokenEmbeddingTensor(tokens: original, device: device)
        let readback = tensor.toTokens()

        #expect(readback.count == 2)
        #expect(readback[0].count == 2)
        #expect(readback[0][0] == [1.0, 2.0])
        #expect(readback[1][1] == [7.0, 8.0])
    }

    @Test("Tensor parameter structs have correct size")
    func tensorParamSizes() {
        // These must match Metal shader expectations
        #expect(MemoryLayout<TensorPoolingParams>.size == 16)
        #expect(MemoryLayout<TensorNormalizationParams>.size == 16)
        #expect(MemoryLayout<TensorSimilarityParams>.size == 16)
        #expect(MemoryLayout<FusedPoolNormParams>.size == 32)
        #expect(MemoryLayout<EmbeddingPipelineParams>.size == 32)
    }
    #endif
}

// MARK: - Tensor Pool Normalize Tests

@Suite("Tensor Pool Normalize")
struct TensorPoolNormalizeTests {

    @Test("CPU fallback produces correct results")
    func cpuFallbackCorrect() async throws {
        let accelerator = await MetalAccelerator()

        // Small batch that will use CPU fallback
        let embeddings: [Float] = [
            // Batch 0: seq 0
            1, 2, 3, 4,
            // Batch 0: seq 1
            5, 6, 7, 8
        ]

        let results = await accelerator.tensorPoolNormalize(
            embeddings: embeddings,
            batchSize: 1,
            sequenceLength: 2,
            dimensions: 4,
            strategy: .mean
        )

        // Mean of [1,2,3,4] and [5,6,7,8] = [3,4,5,6]
        // Then normalized
        #expect(results.count == 1)
        #expect(results[0].count == 4)

        // Verify normalization (L2 norm should be ~1)
        let norm = sqrt(results[0].reduce(0) { $0 + $1 * $1 })
        #expect(abs(norm - 1.0) < 0.01)
    }

    @Test("Mean pooling averages correctly")
    func meanPoolingAverages() async throws {
        let accelerator = await MetalAccelerator()

        // 2 batches, 2 tokens each, 2 dimensions
        let embeddings: [Float] = [
            // Batch 0
            2, 4,  // token 0
            4, 6,  // token 1
            // Batch 1
            10, 20,  // token 0
            30, 40   // token 1
        ]

        let results = await accelerator.tensorPoolNormalize(
            embeddings: embeddings,
            batchSize: 2,
            sequenceLength: 2,
            dimensions: 2,
            normalize: false  // Skip normalization to check raw pooling
        )

        #expect(results.count == 2)

        // Batch 0: mean of [2,4] and [4,6] = [3, 5]
        #expect(abs(results[0][0] - 3.0) < 0.01)
        #expect(abs(results[0][1] - 5.0) < 0.01)

        // Batch 1: mean of [10,20] and [30,40] = [20, 30]
        #expect(abs(results[1][0] - 20.0) < 0.01)
        #expect(abs(results[1][1] - 30.0) < 0.01)
    }

    @Test("Max pooling selects maximum")
    func maxPoolingSelectsMax() async throws {
        let accelerator = await MetalAccelerator()

        let embeddings: [Float] = [
            // Batch 0
            1, 10,  // token 0
            5, 2,   // token 1
        ]

        let results = await accelerator.tensorPoolNormalize(
            embeddings: embeddings,
            batchSize: 1,
            sequenceLength: 2,
            dimensions: 2,
            strategy: .max,
            normalize: false
        )

        // Max of [1,10] and [5,2] = [5, 10]
        #expect(abs(results[0][0] - 5.0) < 0.01)
        #expect(abs(results[0][1] - 10.0) < 0.01)
    }

    @Test("CLS pooling extracts first token")
    func clsPoolingExtractsFirst() async throws {
        let accelerator = await MetalAccelerator()

        let embeddings: [Float] = [
            // Batch 0
            100, 200,  // CLS token
            1, 2,      // other token
            3, 4,      // other token
        ]

        let results = await accelerator.tensorPoolNormalize(
            embeddings: embeddings,
            batchSize: 1,
            sequenceLength: 3,
            dimensions: 2,
            strategy: .cls,
            normalize: false
        )

        // Should be the first token [100, 200]
        #expect(abs(results[0][0] - 100.0) < 0.01)
        #expect(abs(results[0][1] - 200.0) < 0.01)
    }

    @Test("Normalization produces unit vectors")
    func normalizationProducesUnitVectors() async throws {
        let accelerator = await MetalAccelerator()

        let embeddings: [Float] = Array(repeating: Float.random(in: -10...10), count: 4 * 128 * 384)

        let results = await accelerator.tensorPoolNormalize(
            embeddings: embeddings,
            batchSize: 4,
            sequenceLength: 128,
            dimensions: 384,
            normalize: true
        )

        #expect(results.count == 4)

        for result in results {
            let norm = sqrt(result.reduce(0) { $0 + $1 * $1 })
            #expect(abs(norm - 1.0) < 0.01, "Vector should be unit length, got \(norm)")
        }
    }

    @Test("Handles attention masks")
    func handlesAttentionMasks() async throws {
        let accelerator = await MetalAccelerator()

        // 2 tokens, but mask only first
        let embeddings: [Float] = [
            10, 20,  // token 0 (valid)
            100, 200 // token 1 (masked)
        ]

        let masks: [Int32] = [1, 0]  // Only first token valid

        let results = await accelerator.tensorPoolNormalize(
            embeddings: embeddings,
            batchSize: 1,
            sequenceLength: 2,
            dimensions: 2,
            masks: masks,
            normalize: false
        )

        // Should only use first token [10, 20]
        #expect(abs(results[0][0] - 10.0) < 0.01)
        #expect(abs(results[0][1] - 20.0) < 0.01)
    }
}

// MARK: - Tensor Similarity Matrix Tests

@Suite("Tensor Similarity Matrix")
struct TensorSimilarityMatrixTests {

    @Test("CPU fallback computes correct similarities")
    func cpuFallbackCorrect() async throws {
        let accelerator = await MetalAccelerator()

        // Two normalized vectors
        let v1: Float = 1.0 / sqrt(2.0)
        let queries: [Float] = [v1, v1, 0, 0]  // Two 2D vectors
        let keys: [Float] = [v1, v1, 0, 0]

        let result = await accelerator.tensorSimilarityMatrix(
            queries: queries,
            keys: keys,
            queryBatchSize: 2,
            keyBatchSize: 2,
            dimensions: 2,
            normalized: true
        )

        #expect(result.count == 2)
        #expect(result[0].count == 2)

        // Same vectors should have similarity ~1
        #expect(abs(result[0][0] - 1.0) < 0.01)
    }

    @Test("Orthogonal vectors have zero similarity")
    func orthogonalVectorsZeroSimilarity() async throws {
        let accelerator = await MetalAccelerator()

        // Orthogonal unit vectors
        let queries: [Float] = [1, 0]  // [1, 0]
        let keys: [Float] = [0, 1]     // [0, 1]

        let result = await accelerator.tensorSimilarityMatrix(
            queries: queries,
            keys: keys,
            queryBatchSize: 1,
            keyBatchSize: 1,
            dimensions: 2,
            normalized: true
        )

        #expect(abs(result[0][0]) < 0.01)
    }

    @Test("Produces correct matrix shape")
    func correctMatrixShape() async throws {
        let accelerator = await MetalAccelerator()

        let queries = [Float](repeating: 0.1, count: 5 * 128)
        let keys = [Float](repeating: 0.1, count: 10 * 128)

        let result = await accelerator.tensorSimilarityMatrix(
            queries: queries,
            keys: keys,
            queryBatchSize: 5,
            keyBatchSize: 10,
            dimensions: 128,
            normalized: true
        )

        #expect(result.count == 5)
        #expect(result[0].count == 10)
    }

    @Test("Unnormalized similarity computes cosine")
    func unnormalizedComputesCosine() async throws {
        let accelerator = await MetalAccelerator()

        // Non-unit vectors
        let queries: [Float] = [3, 4]  // norm = 5
        let keys: [Float] = [6, 8]     // norm = 10, same direction as queries

        let result = await accelerator.tensorSimilarityMatrix(
            queries: queries,
            keys: keys,
            queryBatchSize: 1,
            keyBatchSize: 1,
            dimensions: 2,
            normalized: false  // Compute full cosine
        )

        // Same direction = similarity 1.0
        #expect(abs(result[0][0] - 1.0) < 0.01)
    }
}

// MARK: - Metal Accelerator Integration Tests

@Suite("Metal Accelerator Tensor Integration")
struct MetalAcceleratorTensorIntegrationTests {

    @Test("Tensor pipelines availability check")
    func tensorPipelinesAvailability() async throws {
        let accelerator = await MetalAccelerator()

        // Note: Pipelines won't be available until metallib is compiled
        // This test verifies the property exists
        let available = await accelerator.tensorPipelinesAvailable
        // Just verify it doesn't crash
        _ = available
    }

    #if canImport(Metal)
    @Test("GPU capabilities are detected")
    func gpuCapabilitiesDetected() async throws {
        let accelerator = await MetalAccelerator()

        let caps = await accelerator.gpuCapabilities
        if let caps = caps {
            #expect(caps.maxThreadsPerThreadgroup > 0)
            #expect(caps.maxBufferLength > 0)
        }
    }

    @Test("Kernel selection works")
    func kernelSelectionWorks() async throws {
        let accelerator = await MetalAccelerator()

        let choice = await accelerator.selectKernel(
            for: .poolAndNormalize,
            batchSize: 16,
            sequenceLength: 128,
            dimensions: 384
        )

        // Should get a valid choice
        switch choice {
        case .cpu, .fused, .separate, .progressive:
            break  // All valid
        }
    }

    @Test("Optimal dispatch returns valid parameters")
    func optimalDispatchReturnsValid() async throws {
        let accelerator = await MetalAccelerator()

        let params = await accelerator.getOptimalDispatch(
            operation: .fusedPoolNorm,
            batchSize: 16,
            sequenceLength: 128,
            dimensions: 384
        )

        if let params = params {
            #expect(params.threadgroupSize.width > 0)
            #expect(params.gridSize.height > 0)
        }
    }

    @Test("Similarity tiles calculated for large batches")
    func similarityTilesCalculated() async throws {
        let accelerator = await MetalAccelerator()

        let tiles = await accelerator.getSimilarityTiles(
            queryBatchSize: 5000,
            keyBatchSize: 5000
        )

        if let tiles = tiles {
            #expect(tiles.count > 0)

            // Tiles should cover entire matrix
            var coverage = 0
            for tile in tiles {
                coverage += tile.elementCount
            }
            #expect(coverage == 5000 * 5000)
        }
    }
    #endif
}

// MARK: - Performance Tests

@Suite("Tensor Operations Performance", .tags(.performance))
struct TensorOperationsPerformanceTests {

    @Test("Large batch pool+normalize performance")
    func largeBatchPerformance() async throws {
        let accelerator = await MetalAccelerator()

        // Generate test data: 100 sequences, 128 tokens, 384 dims
        let embeddings = (0..<(100 * 128 * 384)).map { _ in Float.random(in: -1...1) }

        let start = Date()
        let results = await accelerator.tensorPoolNormalize(
            embeddings: embeddings,
            batchSize: 100,
            sequenceLength: 128,
            dimensions: 384
        )
        let elapsed = Date().timeIntervalSince(start)

        #expect(results.count == 100)
        #expect(elapsed < 5.0, "Should complete in reasonable time, took \(elapsed)s")
    }
}

// MARK: - Test Helpers

enum TensorTestError: Error {
    case metalNotAvailable
    case invalidDimensions
}

// Tags for test categorization
extension Tag {
    @Tag static var performance: Self
    @Tag static var integration: Self
}

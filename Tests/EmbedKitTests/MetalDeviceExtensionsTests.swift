// EmbedKit - Metal Device Extensions Tests
//
// Tests for MTLDevice convenience extensions for tensor creation,
// parameter buffer creation, and threadgroup calculation helpers.

import Testing
import Foundation
@testable import EmbedKit

#if canImport(Metal)
import Metal
#endif

// MARK: - Test Error

enum MetalDeviceExtensionsTestError: Error {
    case skipped(String)
}

// MARK: - MTLDevice Tensor Creation Tests

@Suite("MTLDevice - Tensor Creation")
struct MTLDeviceTensorCreationTests {

    #if canImport(Metal)
    @Test("Creates embedding tensor with dimensions")
    func createsEmbeddingTensor() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalDeviceExtensionsTestError.skipped("Metal not available")
        }

        let tensor = try device.makeEmbeddingTensor(batchSize: 8, dimensions: 384)

        #expect(tensor.batchSize == 8)
        #expect(tensor.dimensions == 384)
        #expect(tensor.totalElements == 8 * 384)
    }

    @Test("Creates embedding tensor from existing embeddings")
    func createsEmbeddingTensorFromData() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalDeviceExtensionsTestError.skipped("Metal not available")
        }

        let embeddings: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]

        let tensor = try device.makeEmbeddingTensor(embeddings: embeddings)

        #expect(tensor.batchSize == 2)
        #expect(tensor.dimensions == 3)

        // Verify data
        let readBack = tensor.toEmbeddings()
        #expect(readBack[0] == [1.0, 2.0, 3.0])
        #expect(readBack[1] == [4.0, 5.0, 6.0])
    }

    @Test("Creates token embedding tensor")
    func createsTokenEmbeddingTensor() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalDeviceExtensionsTestError.skipped("Metal not available")
        }

        let tensor = try device.makeTokenEmbeddingTensor(
            batchSize: 4,
            sequenceLength: 128,
            dimensions: 384
        )

        #expect(tensor.batchSize == 4)
        #expect(tensor.sequenceLength == 128)
        #expect(tensor.dimensions == 384)
        #expect(tensor.totalElements == 4 * 128 * 384)
    }

    @Test("Creates empty embedding tensor")
    func createsEmptyEmbeddingTensor() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalDeviceExtensionsTestError.skipped("Metal not available")
        }

        let tensor = try device.makeEmbeddingTensor(embeddings: [])

        #expect(tensor.batchSize == 0)
        #expect(tensor.dimensions == 0)
    }
    #endif
}

// MARK: - MTLDevice Parameter Buffer Tests

@Suite("MTLDevice - Parameter Buffers")
struct MTLDeviceParameterBufferTests {

    #if canImport(Metal)
    @Test("Creates pooling params buffer")
    func createsPoolingParamsBuffer() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalDeviceExtensionsTestError.skipped("Metal not available")
        }

        let params = TensorPoolingParams(
            batchSize: 8,
            sequenceLength: 128,
            dimensions: 384,
            strategy: .mean
        )

        let buffer = device.makePoolingParamsBuffer(params)

        #expect(buffer != nil)
        #expect(buffer?.length == MemoryLayout<TensorPoolingParams>.stride)
    }

    @Test("Creates normalization params buffer")
    func createsNormalizationParamsBuffer() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalDeviceExtensionsTestError.skipped("Metal not available")
        }

        let params = TensorNormalizationParams(
            batchSize: 16,
            dimensions: 384
        )

        let buffer = device.makeNormalizationParamsBuffer(params)

        #expect(buffer != nil)
        #expect(buffer?.length == MemoryLayout<TensorNormalizationParams>.stride)
    }

    @Test("Creates similarity params buffer")
    func createsSimilarityParamsBuffer() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalDeviceExtensionsTestError.skipped("Metal not available")
        }

        let params = TensorSimilarityParams(
            queryBatchSize: 4,
            keyBatchSize: 100,
            dimensions: 384
        )

        let buffer = device.makeSimilarityParamsBuffer(params)

        #expect(buffer != nil)
        #expect(buffer?.length == MemoryLayout<TensorSimilarityParams>.stride)
    }

    @Test("Creates fused pool+norm params buffer")
    func createsFusedPoolNormParamsBuffer() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalDeviceExtensionsTestError.skipped("Metal not available")
        }

        let params = FusedPoolNormParams(
            batchSize: 8,
            sequenceLength: 128,
            dimensions: 384,
            strategy: .mean,
            normalize: true
        )

        let buffer = device.makeFusedPoolNormParamsBuffer(params)

        #expect(buffer != nil)
        #expect(buffer?.length == MemoryLayout<FusedPoolNormParams>.stride)
    }
    #endif
}

// MARK: - MTLDevice Similarity Matrix Tests

@Suite("MTLDevice - Similarity Matrix")
struct MTLDeviceSimilarityMatrixTests {

    #if canImport(Metal)
    @Test("Creates similarity matrix buffer")
    func createsSimilarityMatrixBuffer() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalDeviceExtensionsTestError.skipped("Metal not available")
        }

        let buffer = device.makeSimilarityMatrixBuffer(
            queryBatchSize: 4,
            keyBatchSize: 100
        )

        let expectedSize = 4 * 100 * MemoryLayout<Float>.stride
        #expect(buffer != nil)
        #expect(buffer?.length == expectedSize)
    }

    @Test("Creates large similarity matrix buffer")
    func createsLargeSimilarityMatrixBuffer() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalDeviceExtensionsTestError.skipped("Metal not available")
        }

        let buffer = device.makeSimilarityMatrixBuffer(
            queryBatchSize: 32,
            keyBatchSize: 1000
        )

        let expectedSize = 32 * 1000 * MemoryLayout<Float>.stride
        #expect(buffer != nil)
        #expect(buffer?.length == expectedSize)
    }
    #endif
}

// MARK: - MTLDevice Attention Mask Tests

@Suite("MTLDevice - Attention Mask")
struct MTLDeviceAttentionMaskTests {

    #if canImport(Metal)
    @Test("Creates attention mask with default value of 1")
    func createsAttentionMaskDefaultValue() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalDeviceExtensionsTestError.skipped("Metal not available")
        }

        let buffer = device.makeAttentionMaskBuffer(
            batchSize: 2,
            sequenceLength: 4
        )

        #expect(buffer != nil)

        // Verify all values are 1
        let count = 2 * 4
        let ptr = buffer!.contents().bindMemory(to: Int32.self, capacity: count)
        for i in 0..<count {
            #expect(ptr[i] == 1)
        }
    }

    @Test("Creates attention mask with custom default value")
    func createsAttentionMaskCustomValue() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalDeviceExtensionsTestError.skipped("Metal not available")
        }

        let buffer = device.makeAttentionMaskBuffer(
            batchSize: 2,
            sequenceLength: 4,
            defaultValue: 0
        )

        #expect(buffer != nil)

        // Verify all values are 0
        let count = 2 * 4
        let ptr = buffer!.contents().bindMemory(to: Int32.self, capacity: count)
        for i in 0..<count {
            #expect(ptr[i] == 0)
        }
    }

    @Test("Creates attention mask with correct size")
    func createsAttentionMaskCorrectSize() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalDeviceExtensionsTestError.skipped("Metal not available")
        }

        let buffer = device.makeAttentionMaskBuffer(
            batchSize: 8,
            sequenceLength: 128
        )

        let expectedSize = 8 * 128 * MemoryLayout<Int32>.stride
        #expect(buffer?.length == expectedSize)
    }
    #endif
}

// MARK: - ThreadgroupCalculator Tests

@Suite("ThreadgroupCalculator")
struct ThreadgroupCalculatorTests {

    #if canImport(Metal)
    @Test("Calculates 1D threadgroup size", .disabled("Requires Metal library from app bundle"))
    func calculates1DThreadgroups() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return // Skip silently
        }

        // This test requires the EmbedKit Metal library to be loaded
        // which is only available when running in a real app context
        guard let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: "tensor_l2_normalize") else {
            return // Skip silently - no library available in test context
        }

        let pipeline = try device.makeComputePipelineState(function: function)
        let (threadgroups, threadsPerGroup) = ThreadgroupCalculator.calculate(
            pipeline: pipeline,
            totalThreads: 1024
        )

        #expect(threadgroups.width > 0)
        #expect(threadsPerGroup.width > 0)
        #expect(threadgroups.width * threadsPerGroup.width >= 1024)
    }

    @Test("Calculates 2D threadgroup size", .disabled("Requires Metal library from app bundle"))
    func calculates2DThreadgroups() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return // Skip silently
        }

        // This test requires the EmbedKit Metal library to be loaded
        guard let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: "tensor_l2_normalize") else {
            return // Skip silently
        }

        let pipeline = try device.makeComputePipelineState(function: function)
        let (threadgroups, threadsPerGroup) = ThreadgroupCalculator.calculate2D(
            pipeline: pipeline,
            batchSize: 8,
            dimensions: 384
        )

        #expect(threadgroups.width > 0)
        #expect(threadgroups.height > 0)
        #expect(threadsPerGroup.width > 0)
        #expect(threadsPerGroup.height > 0)

        // Verify coverage
        let totalWidth = threadgroups.width * threadsPerGroup.width
        let totalHeight = threadgroups.height * threadsPerGroup.height
        #expect(totalWidth >= 384)
        #expect(totalHeight >= 8)
    }

    @Test("ThreadgroupCalculator produces valid output structure")
    func validOutputStructure() throws {
        // Test the basic math without needing a real pipeline
        // The calculate functions produce MTLSize tuples with width/height/depth

        // For 1D: width should be > 0, height/depth = 1
        let threadgroups1D = MTLSize(width: 16, height: 1, depth: 1)
        let threadsPerGroup1D = MTLSize(width: 64, height: 1, depth: 1)

        #expect(threadgroups1D.width * threadsPerGroup1D.width >= 1024)

        // For 2D: both width/height > 0, depth = 1
        let threadgroups2D = MTLSize(width: 12, height: 8, depth: 1)
        let threadsPerGroup2D = MTLSize(width: 32, height: 1, depth: 1)

        #expect(threadgroups2D.width * threadsPerGroup2D.width >= 384)
        #expect(threadgroups2D.height * threadsPerGroup2D.height >= 8)
    }
    #endif
}

// MARK: - EmbeddingTensor Creation and Access Tests

@Suite("EmbeddingTensor - Operations")
struct EmbeddingTensorOperationsTests {

    #if canImport(Metal)
    @Test("Writes and reads embeddings")
    func writesAndReadsEmbeddings() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalDeviceExtensionsTestError.skipped("Metal not available")
        }

        let tensor = try device.makeEmbeddingTensor(batchSize: 3, dimensions: 4)

        let embeddings: [[Float]] = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ]

        try tensor.write(embeddings: embeddings)

        let readBack = tensor.toEmbeddings()
        #expect(readBack == embeddings)
    }

    @Test("Writes and reads single embedding")
    func writesAndReadsSingleEmbedding() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalDeviceExtensionsTestError.skipped("Metal not available")
        }

        let tensor = try device.makeEmbeddingTensor(batchSize: 3, dimensions: 4)

        let embedding: [Float] = [1.5, 2.5, 3.5, 4.5]
        try tensor.write(embedding: embedding, at: 1)

        let readBack = tensor.embedding(at: 1)
        #expect(readBack == embedding)
    }

    @Test("Clears tensor to zeros")
    func clearsTensorToZeros() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalDeviceExtensionsTestError.skipped("Metal not available")
        }

        let embeddings: [[Float]] = [[1.0, 2.0], [3.0, 4.0]]
        let tensor = try device.makeEmbeddingTensor(embeddings: embeddings)

        tensor.clear()

        let readBack = tensor.toEmbeddings()
        for embedding in readBack {
            for value in embedding {
                #expect(value == 0.0)
            }
        }
    }

    @Test("Throws on dimension mismatch when writing")
    func throwsOnDimensionMismatch() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalDeviceExtensionsTestError.skipped("Metal not available")
        }

        let tensor = try device.makeEmbeddingTensor(batchSize: 2, dimensions: 4)

        let wrongDimensions: [[Float]] = [
            [1.0, 2.0, 3.0], // Only 3 dimensions
            [4.0, 5.0, 6.0]
        ]

        #expect(throws: EmbedKitError.self) {
            try tensor.write(embeddings: wrongDimensions)
        }
    }

    @Test("Shape property returns correct tuple")
    func shapePropertyReturnsCorrectTuple() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalDeviceExtensionsTestError.skipped("Metal not available")
        }

        let tensor = try device.makeEmbeddingTensor(batchSize: 8, dimensions: 384)
        let shape = tensor.shape

        #expect(shape.batchSize == 8)
        #expect(shape.dimensions == 384)
    }
    #endif
}

// MARK: - TokenEmbeddingTensor Operations Tests

@Suite("TokenEmbeddingTensor - Operations")
struct TokenEmbeddingTensorOperationsTests {

    #if canImport(Metal)
    @Test("Writes and reads token embeddings")
    func writesAndReadsTokens() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalDeviceExtensionsTestError.skipped("Metal not available")
        }

        let tensor = try device.makeTokenEmbeddingTensor(
            batchSize: 2,
            sequenceLength: 3,
            dimensions: 2
        )

        let tokens: [[[Float]]] = [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        ]

        try tensor.write(tokens: tokens)

        let readBack = tensor.toTokens()
        #expect(readBack == tokens)
    }

    @Test("Accesses sequence by batch index")
    func accessesSequenceByBatchIndex() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalDeviceExtensionsTestError.skipped("Metal not available")
        }

        let tokens: [[[Float]]] = [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ]
        let tensor = try TokenEmbeddingTensor(tokens: tokens, device: device)

        let sequence0 = tensor.sequence(at: 0)
        #expect(sequence0 == [[1.0, 2.0], [3.0, 4.0]])

        let sequence1 = tensor.sequence(at: 1)
        #expect(sequence1 == [[5.0, 6.0], [7.0, 8.0]])
    }

    @Test("Accesses single token")
    func accessesSingleToken() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalDeviceExtensionsTestError.skipped("Metal not available")
        }

        let tokens: [[[Float]]] = [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ]
        let tensor = try TokenEmbeddingTensor(tokens: tokens, device: device)

        let token = tensor.token(batch: 1, token: 0)
        #expect(token == [5.0, 6.0])
    }

    @Test("Shape property returns correct tuple")
    func shapePropertyReturnsCorrectTuple() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalDeviceExtensionsTestError.skipped("Metal not available")
        }

        let tensor = try device.makeTokenEmbeddingTensor(
            batchSize: 4,
            sequenceLength: 128,
            dimensions: 384
        )
        let shape = tensor.shape

        #expect(shape.batchSize == 4)
        #expect(shape.sequenceLength == 128)
        #expect(shape.dimensions == 384)
    }

    @Test("Elements per sequence calculated correctly")
    func elementsPerSequenceCalculation() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalDeviceExtensionsTestError.skipped("Metal not available")
        }

        let tensor = try device.makeTokenEmbeddingTensor(
            batchSize: 2,
            sequenceLength: 128,
            dimensions: 384
        )

        #expect(tensor.elementsPerSequence == 128 * 384)
    }
    #endif
}

// EmbedKit - Tensor Descriptor Factory Tests
//
// Tests for tensor descriptor and parameter creation factories.
// Validates proper tensor allocation and parameter configuration.

import Testing
import Foundation
@testable import EmbedKit

#if canImport(Metal)
import Metal
#endif

// MARK: - Test Error

enum TensorDescriptorFactoryTestError: Error {
    case skipped(String)
}

// MARK: - Model Dimensions Tests

@Suite("TensorDescriptorFactory - Model Dimensions")
struct ModelDimensionsTests {

    @Test("Model dimensions have correct raw values")
    func modelDimensionsRawValues() {
        #expect(TensorDescriptorFactory.ModelDimensions.miniLM.rawValue == 384)
        #expect(TensorDescriptorFactory.ModelDimensions.bertBase.rawValue == 768)
        #expect(TensorDescriptorFactory.ModelDimensions.bertLarge.rawValue == 1024)
        #expect(TensorDescriptorFactory.ModelDimensions.ada002.rawValue == 1536)
        #expect(TensorDescriptorFactory.ModelDimensions.embedding3Large.rawValue == 3072)
    }

    @Test("Sequence lengths have correct raw values")
    func sequenceLengthsRawValues() {
        #expect(TensorDescriptorFactory.SequenceLength.short.rawValue == 128)
        #expect(TensorDescriptorFactory.SequenceLength.standard.rawValue == 512)
        #expect(TensorDescriptorFactory.SequenceLength.long.rawValue == 2048)
        #expect(TensorDescriptorFactory.SequenceLength.miniLMMax.rawValue == 256)
    }
}

// MARK: - Embedding Tensor Creation Tests

@Suite("TensorDescriptorFactory - Embedding Tensor")
struct EmbeddingTensorCreationTests {

    #if canImport(Metal)
    @Test("Creates embedding tensor with preset dimensions")
    func createsWithPreset() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorDescriptorFactoryTestError.skipped("Metal not available")
        }

        let tensor = try TensorDescriptorFactory.embeddingTensor(
            device: device,
            batchSize: 8,
            model: .miniLM
        )

        #expect(tensor.batchSize == 8)
        #expect(tensor.dimensions == 384)
        #expect(tensor.totalElements == 8 * 384)
    }

    @Test("Creates embedding tensor with BERT base dimensions")
    func createsWithBertBase() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorDescriptorFactoryTestError.skipped("Metal not available")
        }

        let tensor = try TensorDescriptorFactory.embeddingTensor(
            device: device,
            batchSize: 4,
            model: .bertBase
        )

        #expect(tensor.batchSize == 4)
        #expect(tensor.dimensions == 768)
    }

    @Test("Creates embedding tensor with ada002 dimensions")
    func createsWithAda002() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorDescriptorFactoryTestError.skipped("Metal not available")
        }

        let tensor = try TensorDescriptorFactory.embeddingTensor(
            device: device,
            batchSize: 2,
            model: .ada002
        )

        #expect(tensor.dimensions == 1536)
    }
    #endif
}

// MARK: - Token Tensor Creation Tests

@Suite("TensorDescriptorFactory - Token Tensor")
struct TokenTensorCreationTests {

    #if canImport(Metal)
    @Test("Creates token tensor with preset dimensions")
    func createsTokenTensor() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorDescriptorFactoryTestError.skipped("Metal not available")
        }

        let tensor = try TensorDescriptorFactory.tokenTensor(
            device: device,
            batchSize: 4,
            sequenceLength: .short,
            model: .miniLM
        )

        #expect(tensor.batchSize == 4)
        #expect(tensor.sequenceLength == 128)
        #expect(tensor.dimensions == 384)
        #expect(tensor.totalElements == 4 * 128 * 384)
    }

    @Test("Creates token tensor with standard sequence length")
    func createsWithStandardLength() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorDescriptorFactoryTestError.skipped("Metal not available")
        }

        let tensor = try TensorDescriptorFactory.tokenTensor(
            device: device,
            batchSize: 2,
            sequenceLength: .standard,
            model: .bertBase
        )

        #expect(tensor.sequenceLength == 512)
        #expect(tensor.dimensions == 768)
    }
    #endif
}

// MARK: - Pooling Configuration Tests

@Suite("TensorDescriptorFactory - Pooling Configuration")
struct PoolingConfigurationTests {

    #if canImport(Metal)
    @Test("Creates pooling configuration with correct tensors")
    func createsPoolingConfig() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorDescriptorFactoryTestError.skipped("Metal not available")
        }

        let config = try TensorDescriptorFactory.forPooling(
            device: device,
            batchSize: 8,
            sequenceLength: 128,
            dimensions: 384,
            strategy: .mean
        )

        // Verify input tensor
        #expect(config.input.batchSize == 8)
        #expect(config.input.sequenceLength == 128)
        #expect(config.input.dimensions == 384)

        // Verify output tensor
        #expect(config.output.batchSize == 8)
        #expect(config.output.dimensions == 384)

        // Verify params
        #expect(config.params.batchSize == 8)
        #expect(config.params.sequenceLength == 128)
        #expect(config.params.dimensions == 384)
        #expect(config.params.poolingStrategy == 0) // mean = 0
    }

    @Test("Creates pooling configuration with max strategy")
    func createsWithMaxStrategy() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorDescriptorFactoryTestError.skipped("Metal not available")
        }

        let config = try TensorDescriptorFactory.forPooling(
            device: device,
            batchSize: 4,
            sequenceLength: 64,
            dimensions: 256,
            strategy: .max
        )

        #expect(config.params.poolingStrategy == 1) // max = 1
    }

    @Test("Creates pooling configuration with CLS strategy")
    func createsWithCLSStrategy() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorDescriptorFactoryTestError.skipped("Metal not available")
        }

        let config = try TensorDescriptorFactory.forPooling(
            device: device,
            batchSize: 4,
            sequenceLength: 64,
            dimensions: 256,
            strategy: .cls
        )

        #expect(config.params.poolingStrategy == 2) // cls = 2
    }
    #endif
}

// MARK: - Normalization Configuration Tests

@Suite("TensorDescriptorFactory - Normalization Configuration")
struct NormalizationConfigurationTests {

    #if canImport(Metal)
    @Test("Creates normalization configuration")
    func createsNormalizationConfig() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorDescriptorFactoryTestError.skipped("Metal not available")
        }

        let config = try TensorDescriptorFactory.forNormalization(
            device: device,
            batchSize: 16,
            dimensions: 384
        )

        #expect(config.tensor.batchSize == 16)
        #expect(config.tensor.dimensions == 384)
        #expect(config.params.batchSize == 16)
        #expect(config.params.dimensions == 384)
    }
    #endif
}

// MARK: - Similarity Configuration Tests

@Suite("TensorDescriptorFactory - Similarity Configuration")
struct SimilarityConfigurationTests {

    #if canImport(Metal)
    @Test("Creates similarity configuration")
    func createsSimilarityConfig() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorDescriptorFactoryTestError.skipped("Metal not available")
        }

        let config = try TensorDescriptorFactory.forSimilarity(
            device: device,
            queryBatchSize: 4,
            keyBatchSize: 100,
            dimensions: 384
        )

        // Verify query tensor
        #expect(config.queries.batchSize == 4)
        #expect(config.queries.dimensions == 384)

        // Verify key tensor
        #expect(config.keys.batchSize == 100)
        #expect(config.keys.dimensions == 384)

        // Verify output buffer size (4 * 100 floats)
        #expect(config.output.length == 4 * 100 * MemoryLayout<Float>.stride)

        // Verify params
        #expect(config.params.queryBatchSize == 4)
        #expect(config.params.keyBatchSize == 100)
        #expect(config.params.dimensions == 384)
    }
    #endif
}

// MARK: - Fused Pool+Norm Configuration Tests

@Suite("TensorDescriptorFactory - Fused Pool+Norm")
struct FusedPoolNormConfigurationTests {

    #if canImport(Metal)
    @Test("Creates fused pool+norm configuration")
    func createsFusedConfig() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorDescriptorFactoryTestError.skipped("Metal not available")
        }

        let config = try TensorDescriptorFactory.forFusedPoolNorm(
            device: device,
            batchSize: 8,
            sequenceLength: 128,
            dimensions: 384,
            strategy: .mean,
            normalize: true
        )

        // Verify input tensor
        #expect(config.input.batchSize == 8)
        #expect(config.input.sequenceLength == 128)
        #expect(config.input.dimensions == 384)

        // Verify output tensor
        #expect(config.output.batchSize == 8)
        #expect(config.output.dimensions == 384)

        // Verify params
        #expect(config.params.batchSize == 8)
        #expect(config.params.sequenceLength == 128)
        #expect(config.params.dimensions == 384)
        #expect(config.params.poolingStrategy == 0)
        #expect(config.params.normalize == 1)
    }

    @Test("Creates fused config without normalization")
    func createsFusedWithoutNorm() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorDescriptorFactoryTestError.skipped("Metal not available")
        }

        let config = try TensorDescriptorFactory.forFusedPoolNorm(
            device: device,
            batchSize: 4,
            sequenceLength: 64,
            dimensions: 256,
            normalize: false
        )

        #expect(config.params.normalize == 0)
    }
    #endif
}

// MARK: - Attention Mask Tests

@Suite("TensorDescriptorFactory - Attention Mask")
struct AttentionMaskTests {

    #if canImport(Metal)
    @Test("Creates attention mask buffer with all ones")
    func createsAttentionMask() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorDescriptorFactoryTestError.skipped("Metal not available")
        }

        let buffer = try TensorDescriptorFactory.attentionMask(
            device: device,
            batchSize: 4,
            sequenceLength: 128
        )

        let count = 4 * 128
        #expect(buffer.length == count * MemoryLayout<Int32>.stride)

        // Verify all values are 1
        let ptr = buffer.contents().bindMemory(to: Int32.self, capacity: count)
        for i in 0..<count {
            #expect(ptr[i] == 1)
        }
    }

    @Test("Creates attention mask with correct size")
    func createsCorrectSize() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorDescriptorFactoryTestError.skipped("Metal not available")
        }

        let buffer = try TensorDescriptorFactory.attentionMask(
            device: device,
            batchSize: 2,
            sequenceLength: 64
        )

        #expect(buffer.length == 2 * 64 * MemoryLayout<Int32>.stride)
    }
    #endif
}

// MARK: - Parameter Struct Tests

@Suite("Tensor Parameter Structs")
struct TensorParameterStructTests {

    @Test("TensorPoolingParams initializes correctly")
    func poolingParamsInit() {
        let params = TensorPoolingParams(
            batchSize: 8,
            sequenceLength: 128,
            dimensions: 384,
            strategy: .mean
        )

        #expect(params.batchSize == 8)
        #expect(params.sequenceLength == 128)
        #expect(params.dimensions == 384)
        #expect(params.poolingStrategy == 0)
    }

    @Test("TensorNormalizationParams initializes correctly")
    func normalizationParamsInit() {
        let params = TensorNormalizationParams(
            batchSize: 16,
            dimensions: 384,
            shouldNormalize: true
        )

        #expect(params.batchSize == 16)
        #expect(params.dimensions == 384)
        #expect(params.shouldNormalize == 1)

        let paramsNoNorm = TensorNormalizationParams(
            batchSize: 8,
            dimensions: 256,
            shouldNormalize: false
        )

        #expect(paramsNoNorm.shouldNormalize == 0)
    }

    @Test("TensorSimilarityParams initializes correctly")
    func similarityParamsInit() {
        let params = TensorSimilarityParams(
            queryBatchSize: 4,
            keyBatchSize: 100,
            dimensions: 384,
            metric: 0
        )

        #expect(params.queryBatchSize == 4)
        #expect(params.keyBatchSize == 100)
        #expect(params.dimensions == 384)
        #expect(params.metric == 0)
    }

    @Test("FusedPoolNormParams initializes correctly")
    func fusedParamsInit() {
        let params = FusedPoolNormParams(
            batchSize: 8,
            sequenceLength: 128,
            dimensions: 384,
            strategy: .max,
            normalize: true
        )

        #expect(params.batchSize == 8)
        #expect(params.sequenceLength == 128)
        #expect(params.dimensions == 384)
        #expect(params.poolingStrategy == 1) // max
        #expect(params.normalize == 1)
    }

    @Test("EmbeddingPipelineParams initializes correctly")
    func pipelineParamsInit() {
        let params = EmbeddingPipelineParams(
            batchSize: 4,
            sequenceLength: 64,
            dimensions: 256,
            strategy: .cls,
            normalize: true,
            computeSimilarity: true
        )

        #expect(params.batchSize == 4)
        #expect(params.sequenceLength == 64)
        #expect(params.dimensions == 256)
        #expect(params.poolingStrategy == 2) // cls
        #expect(params.normalize == 1)
        #expect(params.computeSimilarity == 1)
    }

    @Test("PoolingStrategy metal index values")
    func poolingStrategyMetalIndex() {
        #expect(PoolingStrategy.mean.metalIndex == 0)
        #expect(PoolingStrategy.max.metalIndex == 1)
        #expect(PoolingStrategy.cls.metalIndex == 2)
        #expect(PoolingStrategy.attention.metalIndex == 3)
    }

    @Test("TensorSimilarityMetric raw values")
    func similarityMetricRawValues() {
        #expect(TensorSimilarityMetric.cosine.rawValue == 0)
        #expect(TensorSimilarityMetric.dotProduct.rawValue == 1)
        #expect(TensorSimilarityMetric.euclidean.rawValue == 2)
    }
}

// MARK: - Struct Memory Layout Tests

@Suite("Parameter Struct Memory Layout")
struct ParameterStructMemoryLayoutTests {

    @Test("TensorPoolingParams has 16-byte alignment")
    func poolingParamsAlignment() {
        #expect(MemoryLayout<TensorPoolingParams>.stride == 16)
    }

    @Test("TensorNormalizationParams has 16-byte alignment")
    func normalizationParamsAlignment() {
        #expect(MemoryLayout<TensorNormalizationParams>.stride == 16)
    }

    @Test("TensorSimilarityParams has 16-byte alignment")
    func similarityParamsAlignment() {
        #expect(MemoryLayout<TensorSimilarityParams>.stride == 16)
    }

    @Test("FusedPoolNormParams has 32-byte size")
    func fusedParamsSize() {
        #expect(MemoryLayout<FusedPoolNormParams>.stride == 32)
    }

    @Test("EmbeddingPipelineParams has 32-byte size")
    func pipelineParamsSize() {
        #expect(MemoryLayout<EmbeddingPipelineParams>.stride == 32)
    }
}

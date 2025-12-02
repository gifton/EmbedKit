// EmbedKit - Tensor Result Extractor Tests
//
// Tests for extracting results from GPU buffers into Swift types
// with shape validation and error handling.

import Testing
import Foundation
@testable import EmbedKit

#if canImport(Metal)
import Metal
#endif

// MARK: - Test Error

enum TensorResultExtractorTestError: Error {
    case skipped(String)
}

// MARK: - Flat Extraction Tests

@Suite("TensorResultExtractor - Flat Extraction")
struct TensorResultExtractorFlatTests {

    #if canImport(Metal)
    @Test("Extracts flat array from tensor")
    func extractsFlat() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        let data: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let buffer = device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.buffer(length: data.count)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        let extractor = TensorResultExtractor()
        let result = extractor.extractFlat(from: tensor)

        #expect(result.count == data.count)
        for i in 0..<data.count {
            #expect(result[i] == data[i])
        }
    }

    @Test("Extracts partial flat array with offset and count")
    func extractsPartialFlat() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        let data: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        let buffer = device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.buffer(length: data.count)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        let extractor = TensorResultExtractor()
        let result = try extractor.extractFlat(from: tensor, offset: 2, count: 4)

        #expect(result.count == 4)
        #expect(result == [3.0, 4.0, 5.0, 6.0])
    }

    @Test("Throws on invalid extraction range")
    func throwsOnInvalidRange() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        let data: [Float] = [1.0, 2.0, 3.0, 4.0]
        let buffer = device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.buffer(length: data.count)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        let extractor = TensorResultExtractor()

        #expect(throws: EmbedKitError.self) {
            _ = try extractor.extractFlat(from: tensor, offset: 2, count: 10)
        }

        #expect(throws: EmbedKitError.self) {
            _ = try extractor.extractFlat(from: tensor, offset: -1, count: 2)
        }
    }
    #endif
}

// MARK: - 2D Extraction Tests

@Suite("TensorResultExtractor - 2D Extraction")
struct TensorResultExtractor2DTests {

    #if canImport(Metal)
    @Test("Extracts 2D array from tensor")
    func extracts2D() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        // 3 rows x 4 columns
        let data: [Float] = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0
        ]
        let buffer = device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.embedding(batchSize: 3, dimensions: 4)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        let extractor = TensorResultExtractor()
        let result = try extractor.extract2D(from: tensor, rows: 3, columns: 4)

        #expect(result.count == 3)
        #expect(result[0] == [1.0, 2.0, 3.0, 4.0])
        #expect(result[1] == [5.0, 6.0, 7.0, 8.0])
        #expect(result[2] == [9.0, 10.0, 11.0, 12.0])
    }

    @Test("Throws on shape mismatch for 2D")
    func throwsOnShapeMismatch2D() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        let data: [Float] = [1.0, 2.0, 3.0, 4.0]
        let buffer = device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.buffer(length: data.count)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        let extractor = TensorResultExtractor()

        #expect(throws: EmbedKitError.self) {
            _ = try extractor.extract2D(from: tensor, rows: 10, columns: 10)
        }
    }

    @Test("Extracts embeddings using shape information")
    func extractsEmbeddings() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        let data: [Float] = [
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6
        ]
        let buffer = device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.embedding(batchSize: 2, dimensions: 3)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        let extractor = TensorResultExtractor()
        let result = try extractor.extractEmbeddings(from: tensor)

        #expect(result.count == 2)
        #expect(result[0].count == 3)
        #expect(result[1].count == 3)
    }

    @Test("Throws on wrong shape for embeddings extraction")
    func throwsOnWrongShapeForEmbeddings() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        let data: [Float] = [1.0, 2.0, 3.0, 4.0]
        let buffer = device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.buffer(length: data.count)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        let extractor = TensorResultExtractor()

        #expect(throws: EmbedKitError.self) {
            _ = try extractor.extractEmbeddings(from: tensor)
        }
    }
    #endif
}

// MARK: - 3D Extraction Tests

@Suite("TensorResultExtractor - 3D Extraction")
struct TensorResultExtractor3DTests {

    #if canImport(Metal)
    @Test("Extracts 3D array from tensor")
    func extracts3D() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        // 2 batches x 3 sequences x 2 dimensions
        let data: [Float] = [
            1.0, 2.0,  3.0, 4.0,  5.0, 6.0,    // batch 0
            7.0, 8.0,  9.0, 10.0, 11.0, 12.0   // batch 1
        ]
        let buffer = device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.tokenEmbedding(batchSize: 2, sequenceLength: 3, dimensions: 2)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        let extractor = TensorResultExtractor()
        let result = try extractor.extract3D(from: tensor, dim0: 2, dim1: 3, dim2: 2)

        #expect(result.count == 2)
        #expect(result[0].count == 3)
        #expect(result[0][0] == [1.0, 2.0])
        #expect(result[0][1] == [3.0, 4.0])
        #expect(result[0][2] == [5.0, 6.0])
        #expect(result[1][0] == [7.0, 8.0])
        #expect(result[1][1] == [9.0, 10.0])
        #expect(result[1][2] == [11.0, 12.0])
    }

    @Test("Extracts token embeddings using shape information")
    func extractsTokenEmbeddings() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        let data: [Float] = Array(repeating: 0.5, count: 2 * 4 * 3)
        let buffer = device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.tokenEmbedding(batchSize: 2, sequenceLength: 4, dimensions: 3)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        let extractor = TensorResultExtractor()
        let result = try extractor.extractTokenEmbeddings(from: tensor)

        #expect(result.count == 2)
        #expect(result[0].count == 4)
        #expect(result[0][0].count == 3)
    }
    #endif
}

// MARK: - Single Vector Extraction Tests

@Suite("TensorResultExtractor - Vector Extraction")
struct TensorResultExtractorVectorTests {

    #if canImport(Metal)
    @Test("Extracts single vector by index")
    func extractsVector() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        let data: [Float] = [
            1.0, 2.0, 3.0,  // vector 0
            4.0, 5.0, 6.0,  // vector 1
            7.0, 8.0, 9.0   // vector 2
        ]
        let buffer = device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.embedding(batchSize: 3, dimensions: 3)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        let extractor = TensorResultExtractor()

        let v0 = try extractor.extractVector(from: tensor, at: 0, dimensions: 3)
        #expect(v0 == [1.0, 2.0, 3.0])

        let v1 = try extractor.extractVector(from: tensor, at: 1, dimensions: 3)
        #expect(v1 == [4.0, 5.0, 6.0])

        let v2 = try extractor.extractVector(from: tensor, at: 2, dimensions: 3)
        #expect(v2 == [7.0, 8.0, 9.0])
    }

    @Test("Throws on out of range vector index")
    func throwsOnOutOfRangeIndex() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        let data: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let buffer = device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.embedding(batchSize: 2, dimensions: 3)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        let extractor = TensorResultExtractor()

        #expect(throws: EmbedKitError.self) {
            _ = try extractor.extractVector(from: tensor, at: 5, dimensions: 3)
        }
    }

    @Test("Extracts first embedding from tensor")
    func extractsFirstEmbedding() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        let data: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        let buffer = device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.embedding(batchSize: 2, dimensions: 3)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        let extractor = TensorResultExtractor()
        let result = try extractor.extractFirstEmbedding(from: tensor)

        #expect(result == [0.1, 0.2, 0.3])
    }
    #endif
}

// MARK: - Scalar Extraction Tests

@Suite("TensorResultExtractor - Scalar Extraction")
struct TensorResultExtractorScalarTests {

    #if canImport(Metal)
    @Test("Extracts scalar at index")
    func extractsScalar() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        let data: [Float] = [1.5, 2.5, 3.5, 4.5]
        let buffer = device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.buffer(length: data.count)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        let extractor = TensorResultExtractor()

        #expect(try extractor.extractScalar(from: tensor, at: 0) == 1.5)
        #expect(try extractor.extractScalar(from: tensor, at: 1) == 2.5)
        #expect(try extractor.extractScalar(from: tensor, at: 2) == 3.5)
        #expect(try extractor.extractScalar(from: tensor, at: 3) == 4.5)
    }

    @Test("Throws on out of range scalar index")
    func throwsOnOutOfRangeScalar() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        let data: [Float] = [1.0, 2.0]
        let buffer = device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.buffer(length: data.count)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        let extractor = TensorResultExtractor()

        #expect(throws: EmbedKitError.self) {
            _ = try extractor.extractScalar(from: tensor, at: 10)
        }

        #expect(throws: EmbedKitError.self) {
            _ = try extractor.extractScalar(from: tensor, at: -1)
        }
    }
    #endif
}

// MARK: - Statistics Tests

@Suite("TensorResultExtractor - Statistics")
struct TensorResultExtractorStatisticsTests {

    #if canImport(Metal)
    @Test("Computes tensor statistics")
    func computesStatistics() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        let data: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let buffer = device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.buffer(length: data.count)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        let extractor = TensorResultExtractor()
        let stats = extractor.extractStatistics(from: tensor)

        #expect(stats.count == 5)
        #expect(stats.min == 1.0)
        #expect(stats.max == 5.0)
        #expect(stats.mean == 3.0)
        #expect(stats.range == 4.0)
        #expect(stats.standardDeviation > 0)
        #expect(stats.l2Norm > 0)
    }

    @Test("Handles minimal tensor statistics")
    func handlesMinimalStatistics() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        // Note: Metal doesn't allow 0-length buffers, so test with minimal size
        // The source code handles empty by checking buffer.length / stride == 0
        let data: [Float] = [42.0]
        let buffer = device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.buffer(length: 1)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        let extractor = TensorResultExtractor()
        let stats = extractor.extractStatistics(from: tensor)

        #expect(stats.count == 1)
        #expect(stats.min == 42.0)
        #expect(stats.max == 42.0)
        #expect(stats.mean == 42.0)
    }
    #endif
}

// MARK: - Validation Tests

@Suite("TensorResultExtractor - Validation")
struct TensorResultExtractorValidationTests {

    #if canImport(Metal)
    @Test("Validates normalized vectors")
    func validatesNormalized() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        // Create two unit vectors (L2 norm = 1)
        let unitVec1: [Float] = [1.0, 0.0, 0.0]
        let unitVec2: [Float] = [0.0, 1.0, 0.0]
        let data = unitVec1 + unitVec2

        let buffer = device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.embedding(batchSize: 2, dimensions: 3)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        let extractor = TensorResultExtractor()
        let isNormalized = extractor.validateNormalized(tensor: tensor, dimensions: 3)

        #expect(isNormalized == true)
    }

    @Test("Detects non-normalized vectors")
    func detectsNonNormalized() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        // Non-unit vectors
        let data: [Float] = [2.0, 0.0, 0.0, 0.0, 3.0, 0.0]

        let buffer = device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.embedding(batchSize: 2, dimensions: 3)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        let extractor = TensorResultExtractor()
        let isNormalized = extractor.validateNormalized(tensor: tensor, dimensions: 3)

        #expect(isNormalized == false)
    }

    @Test("Checks validity for NaN values")
    func checksValidityForNaN() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        let data: [Float] = [1.0, Float.nan, 3.0, Float.infinity]
        let buffer = device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.buffer(length: data.count)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        let extractor = TensorResultExtractor()
        let validity = extractor.checkValidity(tensor: tensor)

        #expect(validity.totalElements == 4)
        #expect(validity.nanCount == 1)
        #expect(validity.positiveInfinityCount == 1)
        #expect(validity.isValid == false)
        #expect(validity.invalidCount == 2)
    }

    @Test("Reports valid tensor")
    func reportsValidTensor() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        let data: [Float] = [1.0, 2.0, 3.0, 4.0]
        let buffer = device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.buffer(length: data.count)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        let extractor = TensorResultExtractor()
        let validity = extractor.checkValidity(tensor: tensor)

        #expect(validity.isValid == true)
        #expect(validity.invalidCount == 0)
    }
    #endif
}

// MARK: - ManagedTensor Extension Tests

@Suite("ManagedTensor - Convenience Extensions")
struct ManagedTensorExtensionTests {

    #if canImport(Metal)
    @Test("toArray returns flat array")
    func toArrayReturnsFlat() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        let data: [Float] = [1.0, 2.0, 3.0, 4.0]
        let buffer = device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.buffer(length: data.count)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        let result = tensor.toArray()
        #expect(result == data)
    }

    @Test("to2DArray returns 2D array for embeddings")
    func to2DArrayReturns2D() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        let data: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let buffer = device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.embedding(batchSize: 2, dimensions: 3)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        let result = try tensor.to2DArray()
        #expect(result.count == 2)
        #expect(result[0] == [1.0, 2.0, 3.0])
        #expect(result[1] == [4.0, 5.0, 6.0])
    }

    @Test("statistics returns TensorStatistics")
    func statisticsReturns() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        let data: [Float] = [1.0, 2.0, 3.0, 4.0]
        let buffer = device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.buffer(length: data.count)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        let stats = tensor.statistics()
        #expect(stats.count == 4)
        #expect(stats.min == 1.0)
        #expect(stats.max == 4.0)
    }

    @Test("checkValidity returns ValidityCheck")
    func checkValidityReturns() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        let data: [Float] = [1.0, 2.0, 3.0]
        let buffer = device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let shape = ManagedTensor.TensorShape.buffer(length: data.count)
        let tensor = ManagedTensor(buffer: buffer, shape: shape)

        let validity = tensor.checkValidity()
        #expect(validity.isValid == true)
    }
    #endif
}

// MARK: - Batch Result Extractor Tests

@Suite("BatchResultExtractor Unit")
struct BatchResultExtractorUnitTests {

    #if canImport(Metal)
    @Test("Extracts embeddings from multiple tensors")
    func extractsFromMultiple() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        let data1: [Float] = [1.0, 2.0, 3.0]
        let buffer1 = device.makeBuffer(bytes: data1, length: data1.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let tensor1 = ManagedTensor(buffer: buffer1, shape: .embedding(batchSize: 1, dimensions: 3))

        let data2: [Float] = [4.0, 5.0, 6.0]
        let buffer2 = device.makeBuffer(bytes: data2, length: data2.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let tensor2 = ManagedTensor(buffer: buffer2, shape: .embedding(batchSize: 1, dimensions: 3))

        let batchExtractor = BatchResultExtractor()
        let result = try batchExtractor.extractEmbeddings(from: [tensor1, tensor2])

        #expect(result.count == 2)
        #expect(result[0] == [1.0, 2.0, 3.0])
        #expect(result[1] == [4.0, 5.0, 6.0])
    }

    @Test("Concatenates flat arrays from multiple tensors")
    func concatenatesFlat() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorResultExtractorTestError.skipped("Metal not available")
        }

        let data1: [Float] = [1.0, 2.0]
        let buffer1 = device.makeBuffer(bytes: data1, length: data1.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let tensor1 = ManagedTensor(buffer: buffer1, shape: .buffer(length: 2))

        let data2: [Float] = [3.0, 4.0, 5.0]
        let buffer2 = device.makeBuffer(bytes: data2, length: data2.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let tensor2 = ManagedTensor(buffer: buffer2, shape: .buffer(length: 3))

        let batchExtractor = BatchResultExtractor()
        let result = batchExtractor.extractConcatenated(from: [tensor1, tensor2])

        #expect(result == [1.0, 2.0, 3.0, 4.0, 5.0])
    }
    #endif
}

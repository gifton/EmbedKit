// EmbedKit - Embedding Quantization
// GPU-accelerated quantization using VectorAccelerate Metal4 kernels

import Foundation
import Accelerate
import VectorAccelerate

// MARK: - Quantization Format

/// Supported quantization formats for embedding vectors.
public enum QuantizationFormat: String, CaseIterable, Codable, Sendable {
    /// 8-bit signed integer quantization (4x compression vs Float32)
    case int8

    /// 4-bit integer quantization (8x compression vs Float32)
    case int4

    /// 16-bit floating point (2x compression vs Float32)
    case float16

    /// 1-bit binary quantization (32x compression vs Float32)
    case binary

    /// Memory size per element in bytes (approximate for sub-byte formats)
    public var bytesPerElement: Float {
        switch self {
        case .int8: return 1.0
        case .int4: return 0.5
        case .float16: return 2.0
        case .binary: return 1.0 / 8.0  // 1 bit per dimension
        }
    }

    /// Compression ratio vs Float32
    public var compressionRatio: Float {
        Float(MemoryLayout<Float>.size) / bytesPerElement
    }

    /// Description of the format
    public var formatDescription: String {
        switch self {
        case .int8: return "8-bit integer (4x compression)"
        case .int4: return "4-bit integer (8x compression)"
        case .float16: return "16-bit float (2x compression)"
        case .binary: return "1-bit binary (32x compression)"
        }
    }
}

// MARK: - Quantization Parameters

/// Parameters needed to dequantize a vector back to Float32.
///
/// For int8/int4 quantization: `float_value = (quant_value * scale) + offset`
/// For float16/binary: scale=1.0, offset=0.0 (direct conversion)
public struct QuantizationParams: Codable, Sendable, Equatable {
    /// Scale factor for dequantization
    public let scale: Float

    /// Offset (zero point) for dequantization
    public let offset: Float

    /// Minimum value in the original vector (for validation)
    public let minValue: Float

    /// Maximum value in the original vector (for validation)
    public let maxValue: Float

    public init(scale: Float, offset: Float, minValue: Float, maxValue: Float) {
        self.scale = scale
        self.offset = offset
        self.minValue = minValue
        self.maxValue = maxValue
    }

    /// Identity parameters (no transformation)
    public static let identity = QuantizationParams(scale: 1.0, offset: 0.0, minValue: 0, maxValue: 0)
}

// MARK: - Quantized Vector

/// A quantized embedding vector with efficient storage.
public struct QuantizedVector: Sendable {
    /// The quantization format used
    public let format: QuantizationFormat

    /// Parameters for dequantization
    public let params: QuantizationParams

    /// Number of dimensions
    public let dimensions: Int

    /// Raw quantized data
    public let data: Data

    /// Memory size in bytes
    public var sizeInBytes: Int { data.count }

    /// Compression ratio achieved
    public var compressionRatio: Float {
        let originalSize = dimensions * MemoryLayout<Float>.size
        return Float(originalSize) / Float(max(1, sizeInBytes))
    }

    public init(format: QuantizationFormat, params: QuantizationParams, dimensions: Int, data: Data) {
        self.format = format
        self.params = params
        self.dimensions = dimensions
        self.data = data
    }
}

// MARK: - Quantized Embedding

/// A quantized embedding with metadata preserved.
public struct QuantizedEmbedding: Sendable {
    /// The quantized vector data
    public let vector: QuantizedVector

    /// Original embedding metadata
    public let metadata: EmbeddingMetadata

    /// Number of dimensions
    public var dimensions: Int { vector.dimensions }

    /// Memory size of the quantized vector
    public var sizeInBytes: Int { vector.sizeInBytes }

    /// Compression ratio vs original Float32
    public var compressionRatio: Float { vector.compressionRatio }

    public init(vector: QuantizedVector, metadata: EmbeddingMetadata) {
        self.vector = vector
        self.metadata = metadata
    }

    /// Dequantize back to a regular Embedding.
    public func dequantize() async throws -> Embedding {
        let floatVector = try await GPUQuantizer.shared().dequantize(vector)
        return Embedding(vector: floatVector, metadata: metadata)
    }

    /// Dequantize using CPU (synchronous).
    public func dequantizeCPU() -> Embedding {
        let floatVector = CPUQuantizer.dequantize(vector)
        return Embedding(vector: floatVector, metadata: metadata)
    }
}

// MARK: - Quantization Statistics

/// Statistics about quantization accuracy.
public struct QuantizationStats: Sendable {
    /// Maximum absolute error between original and dequantized values
    public let maxError: Float

    /// Mean absolute error
    public let meanError: Float

    /// Root mean squared error
    public let rmse: Float

    /// Cosine similarity between original and dequantized vectors
    public let cosineSimilarity: Float

    /// Compression ratio achieved
    public let compressionRatio: Float

    /// Original size in bytes
    public let originalSizeBytes: Int

    /// Quantized size in bytes
    public let quantizedSizeBytes: Int

    public init(
        maxError: Float,
        meanError: Float,
        rmse: Float,
        cosineSimilarity: Float,
        compressionRatio: Float,
        originalSizeBytes: Int,
        quantizedSizeBytes: Int
    ) {
        self.maxError = maxError
        self.meanError = meanError
        self.rmse = rmse
        self.cosineSimilarity = cosineSimilarity
        self.compressionRatio = compressionRatio
        self.originalSizeBytes = originalSizeBytes
        self.quantizedSizeBytes = quantizedSizeBytes
    }
}

// MARK: - Quantization Errors

/// Errors that can occur during quantization.
public enum QuantizationError: Error, LocalizedError, Sendable {
    case emptyVector
    case invalidData(String)
    case dimensionMismatch(expected: Int, got: Int)
    case unsupportedFormat(QuantizationFormat)
    case gpuError(Error)

    public var errorDescription: String? {
        switch self {
        case .emptyVector:
            return "Cannot quantize empty vector"
        case .invalidData(let msg):
            return "Invalid quantization data: \(msg)"
        case .dimensionMismatch(let expected, let got):
            return "Dimension mismatch: expected \(expected), got \(got)"
        case .unsupportedFormat(let format):
            return "Unsupported quantization format: \(format)"
        case .gpuError(let error):
            return "GPU quantization error: \(error.localizedDescription)"
        }
    }
}

// MARK: - GPU Quantizer

/// GPU-accelerated quantization using VectorAccelerate Metal4 kernels.
///
/// Provides high-performance batch quantization for embedding vectors
/// with support for int8, int4, float16, and binary formats.
public actor GPUQuantizer {

    // MARK: - Singleton

    /// Get the shared GPU quantizer instance.
    public static func shared() async throws -> GPUQuantizer {
        try await SharedGPUQuantizer.instance
    }

    // MARK: - Properties

    private let context: Metal4Context
    private let scalarKernel: ScalarQuantizationKernel
    private let binaryKernel: BinaryQuantizationKernel

    // MARK: - Initialization

    /// Create a GPU quantizer with a new Metal4Context.
    public init() async throws {
        self.context = try await Metal4ContextManager.shared()
        self.scalarKernel = try await ScalarQuantizationKernel(context: context)
        self.binaryKernel = try await BinaryQuantizationKernel(context: context)
    }

    /// Create a GPU quantizer with a shared Metal4Context.
    public init(context: Metal4Context) async throws {
        self.context = context
        self.scalarKernel = try await ScalarQuantizationKernel(context: context)
        self.binaryKernel = try await BinaryQuantizationKernel(context: context)
    }

    // MARK: - Quantize Single

    /// Quantize an embedding to the specified format using GPU.
    public func quantize(_ embedding: Embedding, format: QuantizationFormat) async throws -> QuantizedEmbedding {
        let quantizedVector = try await quantize(embedding.vector, format: format)
        return QuantizedEmbedding(vector: quantizedVector, metadata: embedding.metadata)
    }

    /// Quantize a float vector to the specified format using GPU.
    public func quantize(_ vector: [Float], format: QuantizationFormat) async throws -> QuantizedVector {
        guard !vector.isEmpty else {
            throw QuantizationError.emptyVector
        }

        switch format {
        case .int8:
            return try await quantizeToInt8GPU(vector)
        case .int4:
            return try await quantizeToInt4GPU(vector)
        case .float16:
            return CPUQuantizer.quantizeToFloat16(vector)
        case .binary:
            return try await quantizeToBinaryGPU(vector)
        }
    }

    // MARK: - Quantize Batch

    /// Quantize multiple embeddings efficiently using GPU.
    public func quantizeBatch(
        _ embeddings: [Embedding],
        format: QuantizationFormat
    ) async throws -> [QuantizedEmbedding] {
        guard !embeddings.isEmpty else { return [] }

        let vectors = embeddings.map { $0.vector }
        let quantizedVectors = try await quantizeBatch(vectors, format: format)

        return zip(quantizedVectors, embeddings).map { qv, emb in
            QuantizedEmbedding(vector: qv, metadata: emb.metadata)
        }
    }

    /// Quantize multiple vectors efficiently using GPU.
    public func quantizeBatch(
        _ vectors: [[Float]],
        format: QuantizationFormat
    ) async throws -> [QuantizedVector] {
        guard !vectors.isEmpty else { return [] }

        switch format {
        case .int8:
            return try await quantizeBatchToInt8GPU(vectors)
        case .int4:
            return try await quantizeBatchToInt4GPU(vectors)
        case .float16:
            return vectors.map { CPUQuantizer.quantizeToFloat16($0) }
        case .binary:
            return try await quantizeBatchToBinaryGPU(vectors)
        }
    }

    // MARK: - Dequantize

    /// Dequantize a vector back to Float32.
    public func dequantize(_ quantized: QuantizedVector) async throws -> [Float] {
        switch quantized.format {
        case .int8:
            return try await dequantizeFromInt8GPU(quantized)
        case .int4:
            return try await dequantizeFromInt4GPU(quantized)
        case .float16:
            return CPUQuantizer.dequantizeFromFloat16(quantized)
        case .binary:
            return try await dequantizeFromBinaryGPU(quantized)
        }
    }

    // MARK: - Int8 GPU Implementation

    private func quantizeToInt8GPU(_ vector: [Float]) async throws -> QuantizedVector {
        let result = try await scalarKernel.quantize(vector, bitWidth: .int8, type: .symmetric)
        return convertScalarResult(result, format: .int8, dimension: vector.count)
    }

    private func quantizeBatchToInt8GPU(_ vectors: [[Float]]) async throws -> [QuantizedVector] {
        var results: [QuantizedVector] = []
        results.reserveCapacity(vectors.count)

        for vector in vectors {
            let result = try await scalarKernel.quantize(vector, bitWidth: .int8, type: .symmetric)
            results.append(convertScalarResult(result, format: .int8, dimension: vector.count))
        }

        return results
    }

    private func dequantizeFromInt8GPU(_ quantized: QuantizedVector) async throws -> [Float] {
        // Use CPU dequantization (GPU dequantization requires internal Metal4QuantizationResult)
        CPUQuantizer.dequantize(quantized)
    }

    // MARK: - Int4 GPU Implementation

    private func quantizeToInt4GPU(_ vector: [Float]) async throws -> QuantizedVector {
        let result = try await scalarKernel.quantize(vector, bitWidth: .int4, type: .symmetric)
        return convertScalarResult(result, format: .int4, dimension: vector.count)
    }

    private func quantizeBatchToInt4GPU(_ vectors: [[Float]]) async throws -> [QuantizedVector] {
        var results: [QuantizedVector] = []
        results.reserveCapacity(vectors.count)

        for vector in vectors {
            let result = try await scalarKernel.quantize(vector, bitWidth: .int4, type: .symmetric)
            results.append(convertScalarResult(result, format: .int4, dimension: vector.count))
        }

        return results
    }

    private func dequantizeFromInt4GPU(_ quantized: QuantizedVector) async throws -> [Float] {
        // Use CPU dequantization (GPU dequantization requires internal Metal4QuantizationResult)
        CPUQuantizer.dequantize(quantized)
    }

    // MARK: - Binary GPU Implementation

    private func quantizeToBinaryGPU(_ vector: [Float]) async throws -> QuantizedVector {
        let result = try await binaryKernel.quantize(vector: vector)
        return convertBinaryVectorResult(result, dimension: vector.count)
    }

    private func quantizeBatchToBinaryGPU(_ vectors: [[Float]]) async throws -> [QuantizedVector] {
        let batchResult = try await binaryKernel.quantize(vectors: vectors)
        return convertBinaryBatchResult(batchResult)
    }

    private func dequantizeFromBinaryGPU(_ quantized: QuantizedVector) async throws -> [Float] {
        // Binary dequantization: 0 -> -1, 1 -> +1 (sign bit convention)
        let numWords = (quantized.dimensions + 31) / 32
        let words: [UInt32] = quantized.data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: UInt32.self).prefix(numWords))
        }

        var result = [Float](repeating: 0, count: quantized.dimensions)
        for i in 0..<quantized.dimensions {
            let wordIndex = i / 32
            let bitIndex = i % 32
            let bit = (words[wordIndex] >> bitIndex) & 1
            result[i] = bit == 1 ? 1.0 : -1.0
        }

        return result
    }

    // MARK: - Conversion Helpers

    private func convertScalarResult(
        _ result: Metal4QuantizationResult,
        format: QuantizationFormat,
        dimension: Int
    ) -> QuantizedVector {
        // Extract min/max from original data if available, otherwise use defaults
        let params = QuantizationParams(
            scale: result.scale,
            offset: Float(result.zeroPoint ?? 0),
            minValue: 0,  // Not tracked by Metal4QuantizationResult
            maxValue: 0   // Not tracked by Metal4QuantizationResult
        )

        return QuantizedVector(
            format: format,
            params: params,
            dimensions: dimension,
            data: result.quantizedData
        )
    }

    private func convertBinaryVectorResult(_ result: Metal4BinaryVector, dimension: Int) -> QuantizedVector {
        let data = result.data.withUnsafeBytes { Data($0) }

        return QuantizedVector(
            format: .binary,
            params: .identity,
            dimensions: dimension,
            data: data
        )
    }

    private func convertBinaryBatchResult(_ result: Metal4BinaryQuantizationResult) -> [QuantizedVector] {
        var vectors: [QuantizedVector] = []
        vectors.reserveCapacity(result.binaryVectors.count)

        for i in 0..<result.binaryVectors.count {
            if let binaryVec = result.binaryVectors.vector(at: i) {
                vectors.append(convertBinaryVectorResult(binaryVec, dimension: binaryVec.dimension))
            }
        }

        return vectors
    }
}

// MARK: - Shared Instance Helper

/// Actor for thread-safe singleton management of GPUQuantizer.
private actor SharedGPUQuantizer {
    static let holder = SharedGPUQuantizer()
    private var quantizer: GPUQuantizer?
    private var initError: Error?

    static var instance: GPUQuantizer {
        get async throws {
            if let existing = await holder.quantizer {
                return existing
            }
            if let error = await holder.initError {
                throw error
            }

            do {
                let new = try await GPUQuantizer()
                await holder.setQuantizer(new)
                return new
            } catch {
                await holder.setError(error)
                throw error
            }
        }
    }

    private func setQuantizer(_ quantizer: GPUQuantizer) {
        self.quantizer = quantizer
    }

    private func setError(_ error: Error) {
        self.initError = error
    }
}

// MARK: - CPU Quantizer (Fallback)

/// CPU-based quantization utilities using Accelerate.
///
/// Used for float16 quantization and as fallback for systems without Metal4.
public enum CPUQuantizer {

    // MARK: - Float16 Implementation

    public static func quantizeToFloat16(_ vector: [Float]) -> QuantizedVector {
        var float16Data = [UInt16](repeating: 0, count: vector.count)

        vector.withUnsafeBufferPointer { srcPtr in
            float16Data.withUnsafeMutableBufferPointer { dstPtr in
                var src = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: srcPtr.baseAddress!),
                    height: 1,
                    width: vImagePixelCount(vector.count),
                    rowBytes: vector.count * MemoryLayout<Float>.size
                )
                var dst = vImage_Buffer(
                    data: dstPtr.baseAddress!,
                    height: 1,
                    width: vImagePixelCount(vector.count),
                    rowBytes: vector.count * MemoryLayout<UInt16>.size
                )
                vImageConvert_PlanarFtoPlanar16F(&src, &dst, 0)
            }
        }

        let data = float16Data.withUnsafeBytes { Data($0) }

        var minVal: Float = 0
        var maxVal: Float = 0
        vDSP_minv(vector, 1, &minVal, vDSP_Length(vector.count))
        vDSP_maxv(vector, 1, &maxVal, vDSP_Length(vector.count))

        let params = QuantizationParams(
            scale: 1.0,
            offset: 0.0,
            minValue: minVal,
            maxValue: maxVal
        )

        return QuantizedVector(
            format: .float16,
            params: params,
            dimensions: vector.count,
            data: data
        )
    }

    public static func dequantizeFromFloat16(_ quantized: QuantizedVector) -> [Float] {
        let count = quantized.dimensions

        var float16Values: [UInt16] = quantized.data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: UInt16.self))
        }

        guard float16Values.count == count else {
            return [Float](repeating: 0, count: count)
        }

        var result = [Float](repeating: 0, count: count)

        float16Values.withUnsafeMutableBufferPointer { srcPtr in
            result.withUnsafeMutableBufferPointer { dstPtr in
                var src = vImage_Buffer(
                    data: srcPtr.baseAddress!,
                    height: 1,
                    width: vImagePixelCount(count),
                    rowBytes: count * MemoryLayout<UInt16>.size
                )
                var dst = vImage_Buffer(
                    data: dstPtr.baseAddress!,
                    height: 1,
                    width: vImagePixelCount(count),
                    rowBytes: count * MemoryLayout<Float>.size
                )
                vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
            }
        }

        return result
    }

    // MARK: - General Dequantize (for legacy support)

    public static func dequantize(_ quantized: QuantizedVector) -> [Float] {
        switch quantized.format {
        case .float16:
            return dequantizeFromFloat16(quantized)
        case .int8:
            return dequantizeFromInt8(quantized)
        case .int4:
            return dequantizeFromInt4(quantized)
        case .binary:
            return dequantizeFromBinary(quantized)
        }
    }

    private static func dequantizeFromInt8(_ quantized: QuantizedVector) -> [Float] {
        let count = quantized.dimensions

        let int8Values: [Int8] = quantized.data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Int8.self))
        }

        guard int8Values.count == count else {
            return [Float](repeating: 0, count: count)
        }

        var floatValues = int8Values.map { Float($0) }
        var scale = quantized.params.scale
        vDSP_vsmul(floatValues, 1, &scale, &floatValues, 1, vDSP_Length(count))

        var offset = quantized.params.offset
        vDSP_vsadd(floatValues, 1, &offset, &floatValues, 1, vDSP_Length(count))

        return floatValues
    }

    private static func dequantizeFromInt4(_ quantized: QuantizedVector) -> [Float] {
        let count = quantized.dimensions
        let byteCount = (count + 1) / 2

        let bytes: [UInt8] = quantized.data.withUnsafeBytes { buffer in
            Array(buffer.prefix(byteCount))
        }

        var floatValues = [Float](repeating: 0, count: count)

        for i in 0..<count {
            let byteIndex = i / 2
            let isHighNibble = (i % 2) == 1
            let nibble = isHighNibble ? (bytes[byteIndex] >> 4) : (bytes[byteIndex] & 0x0F)
            // Convert from unsigned 4-bit to signed (-7 to +7)
            let signedValue = Int(nibble) - 8
            floatValues[i] = Float(signedValue)
        }

        var scale = quantized.params.scale
        vDSP_vsmul(floatValues, 1, &scale, &floatValues, 1, vDSP_Length(count))

        var offset = quantized.params.offset
        vDSP_vsadd(floatValues, 1, &offset, &floatValues, 1, vDSP_Length(count))

        return floatValues
    }

    private static func dequantizeFromBinary(_ quantized: QuantizedVector) -> [Float] {
        let count = quantized.dimensions
        let numWords = (count + 31) / 32

        let words: [UInt32] = quantized.data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: UInt32.self).prefix(numWords))
        }

        var result = [Float](repeating: 0, count: count)
        for i in 0..<count {
            let wordIndex = i / 32
            let bitIndex = i % 32
            let bit = (words[wordIndex] >> bitIndex) & 1
            result[i] = bit == 1 ? 1.0 : -1.0
        }

        return result
    }
}

// MARK: - Legacy Quantizer (Compatibility)

/// High-performance quantization utilities.
///
/// This enum provides static methods for backward compatibility.
/// For GPU-accelerated quantization, use `GPUQuantizer.shared()`.
public enum Quantizer {

    /// Quantize an embedding using GPU.
    public static func quantize(_ embedding: Embedding, format: QuantizationFormat) async throws -> QuantizedEmbedding {
        try await GPUQuantizer.shared().quantize(embedding, format: format)
    }

    /// Quantize a float vector using GPU.
    public static func quantize(_ vector: [Float], format: QuantizationFormat) async throws -> QuantizedVector {
        try await GPUQuantizer.shared().quantize(vector, format: format)
    }

    /// Dequantize a vector using GPU.
    public static func dequantize(_ quantized: QuantizedVector) async throws -> [Float] {
        try await GPUQuantizer.shared().dequantize(quantized)
    }

    /// Dequantize using CPU (synchronous).
    public static func dequantizeCPU(_ quantized: QuantizedVector) -> [Float] {
        CPUQuantizer.dequantize(quantized)
    }

    /// Quantize multiple embeddings using GPU.
    public static func quantizeBatch(
        _ embeddings: [Embedding],
        format: QuantizationFormat
    ) async throws -> [QuantizedEmbedding] {
        try await GPUQuantizer.shared().quantizeBatch(embeddings, format: format)
    }

    /// Dequantize multiple embeddings.
    public static func dequantizeBatch(
        _ quantized: [QuantizedEmbedding]
    ) async throws -> [Embedding] {
        var results: [Embedding] = []
        results.reserveCapacity(quantized.count)
        for q in quantized {
            results.append(try await q.dequantize())
        }
        return results
    }

    /// Compute statistics about quantization accuracy.
    public static func computeStats(original: [Float], quantized: QuantizedVector) async throws -> QuantizationStats {
        let restored = try await GPUQuantizer.shared().dequantize(quantized)

        guard original.count == restored.count, !original.isEmpty else {
            return QuantizationStats(
                maxError: .infinity,
                meanError: .infinity,
                rmse: .infinity,
                cosineSimilarity: 0,
                compressionRatio: 0,
                originalSizeBytes: 0,
                quantizedSizeBytes: 0
            )
        }

        // Compute errors using Accelerate
        var errors = [Float](repeating: 0, count: original.count)
        vDSP_vsub(restored, 1, original, 1, &errors, 1, vDSP_Length(original.count))

        var absErrors = [Float](repeating: 0, count: original.count)
        vDSP_vabs(errors, 1, &absErrors, 1, vDSP_Length(original.count))

        var maxError: Float = 0
        vDSP_maxv(absErrors, 1, &maxError, vDSP_Length(absErrors.count))

        var meanError: Float = 0
        vDSP_meanv(absErrors, 1, &meanError, vDSP_Length(absErrors.count))

        var squaredErrors = [Float](repeating: 0, count: original.count)
        vDSP_vsq(errors, 1, &squaredErrors, 1, vDSP_Length(errors.count))
        var mse: Float = 0
        vDSP_meanv(squaredErrors, 1, &mse, vDSP_Length(squaredErrors.count))
        let rmse = sqrt(mse)

        let cosineSim = cosineSimilarity(original, restored)

        let originalSize = original.count * MemoryLayout<Float>.size
        let quantizedSize = quantized.sizeInBytes

        return QuantizationStats(
            maxError: maxError,
            meanError: meanError,
            rmse: rmse,
            cosineSimilarity: cosineSim,
            compressionRatio: Float(originalSize) / Float(quantizedSize),
            originalSizeBytes: originalSize,
            quantizedSizeBytes: quantizedSize
        )
    }

    private static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }

        var dot: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(a.count))

        var magASquared: Float = 0
        vDSP_svesq(a, 1, &magASquared, vDSP_Length(a.count))

        var magBSquared: Float = 0
        vDSP_svesq(b, 1, &magBSquared, vDSP_Length(b.count))

        let magA = sqrt(max(1e-12, magASquared))
        let magB = sqrt(max(1e-12, magBSquared))

        return dot / (magA * magB)
    }
}

// MARK: - Embedding Extension

public extension Embedding {
    /// Quantize this embedding to the specified format using GPU.
    func quantized(to format: QuantizationFormat) async throws -> QuantizedEmbedding {
        try await GPUQuantizer.shared().quantize(self, format: format)
    }
}

// MARK: - Array Extension

public extension Array where Element == Embedding {
    /// Quantize all embeddings in the array using GPU.
    func quantized(to format: QuantizationFormat) async throws -> [QuantizedEmbedding] {
        try await GPUQuantizer.shared().quantizeBatch(self, format: format)
    }
}

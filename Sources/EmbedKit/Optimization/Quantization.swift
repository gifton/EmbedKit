// EmbedKit - Embedding Quantization
// Reduce memory footprint with 8-bit and 16-bit quantization

import Foundation
import Accelerate

// MARK: - Quantization Format

/// Supported quantization formats for embedding vectors.
public enum QuantizationFormat: String, CaseIterable, Codable, Sendable {
    /// 8-bit signed integer quantization (4x compression vs Float32)
    case int8

    /// 16-bit floating point (2x compression vs Float32)
    case float16

    /// Memory size per element in bytes
    public var bytesPerElement: Int {
        switch self {
        case .int8: return 1
        case .float16: return 2
        }
    }

    /// Compression ratio vs Float32
    public var compressionRatio: Float {
        Float(MemoryLayout<Float>.size) / Float(bytesPerElement)
    }
}

// MARK: - Quantization Parameters

/// Parameters needed to dequantize a vector back to Float32.
///
/// For int8 quantization: `float_value = (int8_value * scale) + offset`
/// For float16: scale=1.0, offset=0.0 (direct conversion)
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
        return Float(originalSize) / Float(sizeInBytes)
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
    public func dequantize() -> Embedding {
        let floatVector = Quantizer.dequantize(vector)
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
        }
    }
}

// MARK: - Quantizer

/// High-performance quantization utilities.
///
/// Provides SIMD-optimized quantization and dequantization for embedding vectors.
///
/// Example:
/// ```swift
/// let embedding = try await model.embed("Hello world")
///
/// // Quantize to int8 (4x compression)
/// let quantized = try Quantizer.quantize(embedding, format: .int8)
/// print("Compression: \(quantized.compressionRatio)x")
///
/// // Dequantize when needed
/// let restored = quantized.dequantize()
/// ```
public enum Quantizer {

    // MARK: - Quantize Embedding

    /// Quantize an embedding to the specified format.
    ///
    /// - Parameters:
    ///   - embedding: The embedding to quantize
    ///   - format: Target quantization format
    /// - Returns: Quantized embedding
    /// - Throws: QuantizationError if quantization fails
    public static func quantize(_ embedding: Embedding, format: QuantizationFormat) throws -> QuantizedEmbedding {
        let quantizedVector = try quantize(embedding.vector, format: format)
        return QuantizedEmbedding(vector: quantizedVector, metadata: embedding.metadata)
    }

    /// Quantize a float vector to the specified format.
    ///
    /// - Parameters:
    ///   - vector: Float32 vector to quantize
    ///   - format: Target quantization format
    /// - Returns: Quantized vector
    /// - Throws: QuantizationError if quantization fails
    public static func quantize(_ vector: [Float], format: QuantizationFormat) throws -> QuantizedVector {
        guard !vector.isEmpty else {
            throw QuantizationError.emptyVector
        }

        switch format {
        case .int8:
            return try quantizeToInt8(vector)
        case .float16:
            return try quantizeToFloat16(vector)
        }
    }

    // MARK: - Dequantize

    /// Dequantize a vector back to Float32.
    ///
    /// - Parameter quantized: The quantized vector
    /// - Returns: Float32 vector
    public static func dequantize(_ quantized: QuantizedVector) -> [Float] {
        switch quantized.format {
        case .int8:
            return dequantizeFromInt8(quantized)
        case .float16:
            return dequantizeFromFloat16(quantized)
        }
    }

    // MARK: - Statistics

    /// Compute statistics about quantization accuracy.
    ///
    /// - Parameters:
    ///   - original: Original Float32 vector
    ///   - quantized: Quantized vector
    /// - Returns: Quantization statistics
    public static func computeStats(original: [Float], quantized: QuantizedVector) -> QuantizationStats {
        let restored = dequantize(quantized)

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

        // Absolute errors
        var absErrors = [Float](repeating: 0, count: original.count)
        vDSP_vabs(errors, 1, &absErrors, 1, vDSP_Length(original.count))

        // Max error
        var maxError: Float = 0
        vDSP_maxv(absErrors, 1, &maxError, vDSP_Length(absErrors.count))

        // Mean error
        var meanError: Float = 0
        vDSP_meanv(absErrors, 1, &meanError, vDSP_Length(absErrors.count))

        // RMSE
        var squaredErrors = [Float](repeating: 0, count: original.count)
        vDSP_vsq(errors, 1, &squaredErrors, 1, vDSP_Length(errors.count))
        var mse: Float = 0
        vDSP_meanv(squaredErrors, 1, &mse, vDSP_Length(squaredErrors.count))
        let rmse = sqrt(mse)

        // Cosine similarity
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

    // MARK: - Batch Operations

    /// Quantize multiple embeddings efficiently.
    ///
    /// - Parameters:
    ///   - embeddings: Array of embeddings to quantize
    ///   - format: Target quantization format
    /// - Returns: Array of quantized embeddings
    /// - Throws: QuantizationError if any quantization fails
    public static func quantizeBatch(_ embeddings: [Embedding], format: QuantizationFormat) throws -> [QuantizedEmbedding] {
        try embeddings.map { try quantize($0, format: format) }
    }

    /// Dequantize multiple embeddings.
    ///
    /// - Parameter quantized: Array of quantized embeddings
    /// - Returns: Array of regular embeddings
    public static func dequantizeBatch(_ quantized: [QuantizedEmbedding]) -> [Embedding] {
        quantized.map { $0.dequantize() }
    }

    // MARK: - Int8 Implementation

    private static func quantizeToInt8(_ vector: [Float]) throws -> QuantizedVector {
        // Find min/max for scaling
        var minVal: Float = 0
        var maxVal: Float = 0
        vDSP_minv(vector, 1, &minVal, vDSP_Length(vector.count))
        vDSP_maxv(vector, 1, &maxVal, vDSP_Length(vector.count))

        // Handle constant vector (all same values)
        let range = maxVal - minVal
        let scale: Float
        let offset: Float

        if range < 1e-10 {
            // Constant vector - use identity-like params
            scale = 1.0 / 127.0
            offset = minVal
        } else {
            // Scale to [-127, 127] range (symmetric quantization)
            scale = range / 254.0
            offset = minVal + 127.0 * scale
        }

        // Quantize: int8_value = (float_value - offset) / scale
        var normalized = [Float](repeating: 0, count: vector.count)
        var negOffset = -offset
        vDSP_vsadd(vector, 1, &negOffset, &normalized, 1, vDSP_Length(vector.count))

        var invScale = 1.0 / scale
        vDSP_vsmul(normalized, 1, &invScale, &normalized, 1, vDSP_Length(vector.count))

        // Convert to Int8 with clamping
        var int8Values = [Int8](repeating: 0, count: vector.count)
        for i in 0..<vector.count {
            let clamped = max(-127, min(127, normalized[i]))
            int8Values[i] = Int8(clamped.rounded())
        }

        let data = int8Values.withUnsafeBytes { Data($0) }

        let params = QuantizationParams(
            scale: scale,
            offset: offset,
            minValue: minVal,
            maxValue: maxVal
        )

        return QuantizedVector(
            format: .int8,
            params: params,
            dimensions: vector.count,
            data: data
        )
    }

    private static func dequantizeFromInt8(_ quantized: QuantizedVector) -> [Float] {
        let count = quantized.dimensions

        // Extract Int8 values
        let int8Values: [Int8] = quantized.data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Int8.self))
        }

        guard int8Values.count == count else {
            return [Float](repeating: 0, count: count)
        }

        // Convert to Float
        var floatValues = int8Values.map { Float($0) }

        // Dequantize: float_value = (int8_value * scale) + offset
        var scale = quantized.params.scale
        vDSP_vsmul(floatValues, 1, &scale, &floatValues, 1, vDSP_Length(count))

        var offset = quantized.params.offset
        vDSP_vsadd(floatValues, 1, &offset, &floatValues, 1, vDSP_Length(count))

        return floatValues
    }

    // MARK: - Float16 Implementation

    private static func quantizeToFloat16(_ vector: [Float]) throws -> QuantizedVector {
        // Convert Float32 to Float16 using vDSP
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

        // Find min/max for stats
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

    private static func dequantizeFromFloat16(_ quantized: QuantizedVector) -> [Float] {
        let count = quantized.dimensions

        // Extract UInt16 (Float16 bit pattern)
        var float16Values: [UInt16] = quantized.data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: UInt16.self))
        }

        guard float16Values.count == count else {
            return [Float](repeating: 0, count: count)
        }

        // Convert back to Float32 using vImage
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

    // MARK: - Helpers

    private static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }

        // Dot product
        var dot: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(a.count))

        // Magnitudes
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
    /// Quantize this embedding to the specified format.
    ///
    /// - Parameter format: Target quantization format
    /// - Returns: Quantized embedding
    /// - Throws: QuantizationError if quantization fails
    func quantized(to format: QuantizationFormat) throws -> QuantizedEmbedding {
        try Quantizer.quantize(self, format: format)
    }
}

// MARK: - Array Extension

public extension Array where Element == Embedding {
    /// Quantize all embeddings in the array.
    ///
    /// - Parameter format: Target quantization format
    /// - Returns: Array of quantized embeddings
    /// - Throws: QuantizationError if any quantization fails
    func quantized(to format: QuantizationFormat) throws -> [QuantizedEmbedding] {
        try Quantizer.quantizeBatch(self, format: format)
    }
}

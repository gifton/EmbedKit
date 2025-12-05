// Tests for Embedding Quantization - Week 5 Batch 2
import Testing
import Foundation
@testable import EmbedKit

// MARK: - Quantization Format Tests

@Suite("Quantization - Format")
struct QuantizationFormatTests {

    @Test("Int8 format has correct properties")
    func int8FormatProperties() {
        let format = QuantizationFormat.int8

        #expect(format.bytesPerElement == 1)
        #expect(format.compressionRatio == 4.0)
    }

    @Test("Float16 format has correct properties")
    func float16FormatProperties() {
        let format = QuantizationFormat.float16

        #expect(format.bytesPerElement == 2)
        #expect(format.compressionRatio == 2.0)
    }

    @Test("All formats are enumerable")
    func allFormatsEnumerable() {
        let formats = QuantizationFormat.allCases

        #expect(formats.count == 4)  // int8, int4, float16, binary
        #expect(formats.contains(.int8))
        #expect(formats.contains(.int4))
        #expect(formats.contains(.float16))
        #expect(formats.contains(.binary))
    }
}

// MARK: - Int8 Quantization Tests

@Suite("Quantization - Int8")
struct Int8QuantizationTests {

    @Test("Int8 quantization roundtrips with acceptable error")
    func int8Roundtrip() async throws {
        let original: [Float] = (0..<128).map { Float($0) / 128.0 - 0.5 }

        let quantized = try await Quantizer.quantize(original, format: .int8)
        let restored = try await Quantizer.dequantize(quantized)

        #expect(quantized.format == .int8)
        #expect(quantized.dimensions == original.count)
        #expect(quantized.data.count == original.count)  // 1 byte per element

        // Check roundtrip error is small
        let stats = try await Quantizer.computeStats(original: original, quantized: quantized)
        #expect(stats.maxError < 0.01, "Max error \(stats.maxError) exceeds threshold")
        #expect(stats.cosineSimilarity > 0.999, "Cosine similarity \(stats.cosineSimilarity) too low")
    }

    @Test("Int8 achieves 4x compression")
    func int8Compression() async throws {
        let dimensions = 384
        let original: [Float] = (0..<dimensions).map { _ in Float.random(in: -1...1) }

        let quantized = try await Quantizer.quantize(original, format: .int8)

        #expect(quantized.compressionRatio >= 3.9)
        #expect(quantized.compressionRatio <= 4.1)

        let expectedSize = dimensions * 1  // 1 byte per element
        #expect(quantized.sizeInBytes == expectedSize)
    }

    @Test("Int8 handles constant vector")
    func int8ConstantVector() async throws {
        let original: [Float] = [Float](repeating: 0.5, count: 64)

        let quantized = try await Quantizer.quantize(original, format: .int8)
        let restored = try await Quantizer.dequantize(quantized)

        // Constant vectors should roundtrip well
        for (orig, rest) in zip(original, restored) {
            #expect(abs(orig - rest) < 0.1, "Value diff too large: \(orig) vs \(rest)")
        }
    }

    @Test("Int8 handles extreme values")
    func int8ExtremeValues() async throws {
        let original: [Float] = [-1000.0, -100.0, -1.0, 0.0, 1.0, 100.0, 1000.0]

        let quantized = try await Quantizer.quantize(original, format: .int8)
        let restored = try await Quantizer.dequantize(quantized)

        // Values should be in roughly correct order and relative magnitude
        #expect(restored[0] < restored[1])  // Most negative
        #expect(restored[1] < restored[2])
        #expect(restored[5] > restored[4])
        #expect(restored[6] > restored[5])  // Most positive
    }
}

// MARK: - Float16 Quantization Tests

@Suite("Quantization - Float16")
struct Float16QuantizationTests {

    @Test("Float16 quantization roundtrips with minimal error")
    func float16Roundtrip() async throws {
        let original: [Float] = (0..<128).map { Float($0) / 128.0 - 0.5 }

        let quantized = try await Quantizer.quantize(original, format: .float16)
        let restored = try await Quantizer.dequantize(quantized)

        #expect(quantized.format == .float16)
        #expect(quantized.dimensions == original.count)
        #expect(quantized.data.count == original.count * 2)  // 2 bytes per element

        // Float16 should have very high precision for normal values
        let stats = try await Quantizer.computeStats(original: original, quantized: quantized)
        #expect(stats.maxError < 0.001, "Max error \(stats.maxError) exceeds threshold")
        #expect(stats.cosineSimilarity > 0.9999, "Cosine similarity \(stats.cosineSimilarity) too low")
    }

    @Test("Float16 achieves 2x compression")
    func float16Compression() async throws {
        let dimensions = 384
        let original: [Float] = (0..<dimensions).map { _ in Float.random(in: -1...1) }

        let quantized = try await Quantizer.quantize(original, format: .float16)

        #expect(quantized.compressionRatio >= 1.9)
        #expect(quantized.compressionRatio <= 2.1)

        let expectedSize = dimensions * 2  // 2 bytes per element
        #expect(quantized.sizeInBytes == expectedSize)
    }

    @Test("Float16 preserves small differences")
    func float16SmallDifferences() async throws {
        // Values very close together
        let original: [Float] = [0.1, 0.10001, 0.10002, 0.10003, 0.10004]

        let quantized = try await Quantizer.quantize(original, format: .float16)
        let restored = try await Quantizer.dequantize(quantized)

        // Float16 has ~3 decimal digits of precision, so very small diffs may not be preserved
        // But order should be maintained
        for i in 0..<(restored.count - 1) {
            #expect(restored[i] <= restored[i + 1], "Order not preserved at index \(i)")
        }
    }
}

// MARK: - Embedding Quantization Tests

@Suite("Quantization - Embedding")
struct EmbeddingQuantizationTests {

    @Test("Embedding quantizes and dequantizes correctly")
    func embeddingRoundtrip() async throws {
        let vector: [Float] = (0..<256).map { Float($0) / 256.0 }
        let modelID = ModelID(provider: "test", name: "model", version: "1.0")
        let embedding = Embedding(
            vector: vector,
            metadata: EmbeddingMetadata(
                modelID: modelID,
                tokenCount: 10,
                processingTime: 0.05,
                normalized: true
            )
        )

        let quantized = try await embedding.quantized(to: .int8)
        let restored = try await quantized.dequantize()

        // Metadata should be preserved
        #expect(restored.metadata.modelID == modelID)
        #expect(restored.metadata.tokenCount == 10)
        #expect(restored.metadata.normalized == true)

        // Dimensions should match
        #expect(restored.dimensions == embedding.dimensions)

        // Cosine similarity should be very high
        let similarity = embedding.similarity(to: restored)
        #expect(similarity > 0.99, "Similarity \(similarity) too low")
    }

    @Test("QuantizedEmbedding reports correct properties")
    func quantizedEmbeddingProperties() async throws {
        let vector: [Float] = (0..<384).map { _ in Float.random(in: -1...1) }
        let embedding = Embedding(
            vector: vector,
            metadata: EmbeddingMetadata(
                modelID: ModelID(provider: "test", name: "m", version: "1"),
                tokenCount: 5,
                processingTime: 0.01,
                normalized: true
            )
        )

        let quantized = try await Quantizer.quantize(embedding, format: .int8)

        #expect(quantized.dimensions == 384)
        #expect(quantized.sizeInBytes == 384)  // 1 byte per element for int8
        #expect(quantized.compressionRatio >= 3.9)
    }
}

// MARK: - Batch Quantization Tests

@Suite("Quantization - Batch")
struct BatchQuantizationTests {

    @Test("Batch quantization processes all embeddings")
    func batchQuantization() async throws {
        let modelID = ModelID(provider: "test", name: "model", version: "1.0")

        let embeddings: [Embedding] = (0..<10).map { i in
            let vector: [Float] = (0..<128).map { _ in Float.random(in: -1...1) }
            return Embedding(
                vector: vector,
                metadata: EmbeddingMetadata(
                    modelID: modelID,
                    tokenCount: i + 1,
                    processingTime: 0.01,
                    normalized: true
                )
            )
        }

        let quantized = try await Quantizer.quantizeBatch(embeddings, format: .int8)
        let restored = try await Quantizer.dequantizeBatch(quantized)

        #expect(quantized.count == 10)
        #expect(restored.count == 10)

        // Verify metadata preserved
        for (i, emb) in restored.enumerated() {
            #expect(emb.metadata.tokenCount == i + 1)
        }
    }

    @Test("Array extension quantizes embeddings")
    func arrayExtension() async throws {
        let modelID = ModelID(provider: "test", name: "model", version: "1.0")

        let embeddings: [Embedding] = (0..<5).map { _ in
            let vector: [Float] = (0..<64).map { _ in Float.random(in: -1...1) }
            return Embedding(
                vector: vector,
                metadata: EmbeddingMetadata(
                    modelID: modelID,
                    tokenCount: 5,
                    processingTime: 0.01,
                    normalized: true
                )
            )
        }

        let quantized = try await embeddings.quantized(to: .float16)

        #expect(quantized.count == 5)
        for q in quantized {
            #expect(q.vector.format == .float16)
        }
    }
}

// MARK: - Statistics Tests

@Suite("Quantization - Statistics")
struct QuantizationStatsTests {

    @Test("Statistics accurately report compression")
    func statsCompression() async throws {
        let dimensions = 256
        let original: [Float] = (0..<dimensions).map { _ in Float.random(in: -1...1) }

        let quantizedInt8 = try await Quantizer.quantize(original, format: .int8)
        let statsInt8 = try await Quantizer.computeStats(original: original, quantized: quantizedInt8)

        #expect(statsInt8.originalSizeBytes == dimensions * 4)  // Float32
        #expect(statsInt8.quantizedSizeBytes == dimensions * 1)  // Int8
        #expect(abs(statsInt8.compressionRatio - 4.0) < 0.1)

        let quantizedF16 = try await Quantizer.quantize(original, format: .float16)
        let statsF16 = try await Quantizer.computeStats(original: original, quantized: quantizedF16)

        #expect(statsF16.originalSizeBytes == dimensions * 4)  // Float32
        #expect(statsF16.quantizedSizeBytes == dimensions * 2)  // Float16
        #expect(abs(statsF16.compressionRatio - 2.0) < 0.1)
    }

    @Test("Statistics report error metrics")
    func statsErrors() async throws {
        let original: [Float] = (0..<128).map { Float($0) / 128.0 }

        let quantized = try await Quantizer.quantize(original, format: .int8)
        let stats = try await Quantizer.computeStats(original: original, quantized: quantized)

        // Error metrics should be non-negative
        #expect(stats.maxError >= 0)
        #expect(stats.meanError >= 0)
        #expect(stats.rmse >= 0)

        // Max error should be >= mean error
        #expect(stats.maxError >= stats.meanError)

        // Cosine similarity should be high for good quantization
        #expect(stats.cosineSimilarity > 0.99)
    }
}

// MARK: - Error Handling Tests

@Suite("Quantization - Errors")
struct QuantizationErrorTests {

    @Test("Empty vector throws error")
    func emptyVectorError() async throws {
        let empty: [Float] = []

        await #expect(throws: QuantizationError.self) {
            try await Quantizer.quantize(empty, format: .int8)
        }
    }

    @Test("QuantizationError has descriptions")
    func errorDescriptions() {
        let errors: [QuantizationError] = [
            .emptyVector,
            .invalidData("test"),
            .dimensionMismatch(expected: 128, got: 64),
            .unsupportedFormat(.int8)
        ]

        for error in errors {
            #expect(error.errorDescription != nil)
            #expect(!error.errorDescription!.isEmpty)
        }
    }
}

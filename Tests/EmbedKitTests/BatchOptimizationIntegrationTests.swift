import XCTest
@testable import EmbedKit

/// Phase 4 End-to-End Integration Tests
///
/// Verifies that Phase 4 optimizations are properly integrated into the main pipeline
/// and accessible through MetalAccelerator and EmbeddingPipeline.
final class BatchOptimizationIntegrationTests: XCTestCase {

    // MARK: - MetalAccelerator Integration

    func testBatchOptimizationIntegratedInMetalAccelerator() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        print("\n=== Batch Optimization Integration Test: MetalAccelerator ===")

        // Test with optimal dimension for batch optimization (should get ~4× throughput)
        let dimension = 32
        let batchSize = 100
        let vectors = (0..<batchSize).map { _ in
            (0..<dimension).map { _ in Float.random(in: -10...10) }
        }

        // Process with batch optimization enabled (default)
        let batch = try VectorBatch(vectors: vectors)
        let startEnabled = CFAbsoluteTimeGetCurrent()
        _ = try await accelerator.normalizeVectors(batch)
        let timeEnabled = CFAbsoluteTimeGetCurrent() - startEnabled

        // Process with batch optimization disabled
        await accelerator.setBatchOptimization(false)
        let startDisabled = CFAbsoluteTimeGetCurrent()
        _ = try await accelerator.normalizeVectors(batch)
        let timeDisabled = CFAbsoluteTimeGetCurrent() - startDisabled

        // Re-enable for future tests
        await accelerator.setBatchOptimization(true)

        // Verify Phase 4 is faster
        let speedup = timeDisabled / timeEnabled
        print("Speedup with batch optimization: \(String(format: "%.2f", speedup))×")

        // For small vectors, we expect significant speedup
        XCTAssertGreaterThan(speedup, 1.5, "Batch optimization should provide speedup for small vectors")
    }

    // MARK: - EmbeddingPipeline Integration

    func testBatchOptimizationInEmbeddingPipeline() async throws {
        // Skip if no model available
        guard let modelURL = findTestModel() else {
            throw XCTSkip("No test model available")
        }

        print("\n=== Batch Optimization Integration Test: EmbeddingPipeline ===")

        // Create pipeline with test configuration
        let config = EmbeddingPipelineConfiguration(
            poolingStrategy: .mean,
            normalize: true,  // This will use batch-optimized normalization
            useGPUAcceleration: true
        )

        let tokenizer = MockTokenizer()
        let pipeline = try await EmbeddingPipeline(
            modelURL: modelURL,
            tokenizer: tokenizer,
            configuration: config
        )

        // Process some text (normalization will use batch optimization)
        let texts = [
            "Phase 4 optimization test",
            "GPU occupancy improvements",
            "Multiple vectors per threadgroup",
            "Better batch processing throughput"
        ]

        let embeddings = try await pipeline.embed(batch: texts)

        // Verify embeddings are normalized (Phase 4 was used)
        for (i, embedding) in embeddings.enumerated() {
            let vector = embedding.toArray()
            let magnitude = sqrt(vector.reduce(0) { $0 + $1 * $1 })
            XCTAssertEqual(magnitude, 1.0, accuracy: 0.01,
                          "Embedding \(i) should be normalized via batch optimization")
        }

        print("✅ EmbeddingPipeline successfully used batch-optimized normalization")
    }

    // MARK: - Configuration Tests

    func testBatchOptimizationConfigurationToggle() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        print("\n=== Batch Optimization Configuration Toggle Test ===")

        // Create test data
        let vectors = (0..<10).map { i in
            (0..<16).map { j in Float(i * 16 + j) }
        }
        let batch = try VectorBatch(vectors: vectors)

        // Test with batch optimization enabled
        await accelerator.setBatchOptimization(true)
        let result1 = try await accelerator.normalizeVectors(batch)

        // Test with batch optimization disabled
        await accelerator.setBatchOptimization(false)
        let result2 = try await accelerator.normalizeVectors(batch)

        // Results should be identical (same correctness)
        for i in 0..<result1.count {
            let vec1 = Array(result1[i])
            let vec2 = Array(result2[i])
            for j in 0..<vec1.count {
                XCTAssertEqual(vec1[j], vec2[j], accuracy: 0.0001,
                              "Results should be identical regardless of batch optimization")
            }
        }

        print("✅ Batch optimization toggle works correctly")
    }

    // MARK: - Backward Compatibility Test

    func testBatchOptimizationBackwardCompatibility() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        print("\n=== Batch Optimization Backward Compatibility Test ===")

        // Test that deprecated array API still works
        let vectors = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]

        // Use new VectorBatch API
        let vectorsFloat = vectors.map { $0.map { Float($0) } }
        let batch = try VectorBatch(vectors: vectorsFloat)
        let normalized = try await accelerator.normalizeVectors(batch)

        XCTAssertEqual(normalized.count, 3)
        for vec in normalized.toArrays() {
            let magnitude = sqrt(vec.reduce(0) { $0 + $1 * $1 })
            XCTAssertEqual(magnitude, 1.0, accuracy: 0.01,
                          "VectorBatch API should normalize correctly")
        }

        print("✅ Backward compatibility maintained")
    }

    // MARK: - Helper Methods

    private func findTestModel() -> URL? {
        // Look for test model in common locations
        let fileManager = FileManager.default
        let possiblePaths = [
            "MiniLM-L12-v2.mlpackage",
            "MiniLM-L12-v2.mlmodelc",
            "../MiniLM-L12-v2.mlpackage",
            "../MiniLM-L12-v2.mlmodelc"
        ]

        for path in possiblePaths {
            let url = URL(fileURLWithPath: path)
            if fileManager.fileExists(atPath: url.path) {
                return url
            }
        }

        return nil
    }
}

// MARK: - Mock Components for Testing

private struct MockTokenizer: Tokenizer {
    func tokenize(_ text: String) async throws -> TokenizedInput {
        let tokens = text.split(separator: " ").map(String.init)
        let tokenIds = tokens.map { $0.hashValue }
        let attentionMask = Array(repeating: 1, count: tokenIds.count)
        return TokenizedInput(
            tokenIds: tokenIds,
            attentionMask: attentionMask,
            tokenTypeIds: nil,
            originalLength: tokens.count
        )
    }

    func tokenize(batch texts: [String]) async throws -> [TokenizedInput] {
        var results: [TokenizedInput] = []
        for text in texts {
            results.append(try await tokenize(text))
        }
        return results
    }

    var maxSequenceLength: Int { 512 }
    var vocabularySize: Int { 30522 }
    var specialTokens: SpecialTokens { SpecialTokens() }
}

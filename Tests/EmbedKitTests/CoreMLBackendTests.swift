import XCTest
import CoreML
@testable import EmbedKit

/// Comprehensive tests for CoreML backend functionality and correctness
///
/// Test Coverage:
/// - Model loading and validation
/// - Input tensor creation and validation
/// - Single inference correctness
/// - Batch inference correctness
/// - Output shape and value validation
/// - Error handling and edge cases
/// - Actor isolation verification
///
/// Priority: P0 (Critical)
/// Reference: TEST_PLAN.md Section 3.2
final class CoreMLBackendTests: XCTestCase {

    // MARK: - Test Configuration

    /// Path to test model (MiniLM-L12-v2 compiled)
    private let testModelURL = URL(fileURLWithPath: "/Users/goftin/dev/gsuite/VSK/EmbedKit/MiniLM-L12-v2.mlmodelc")

    /// Expected dimensions for MiniLM-L12-v2
    private let expectedSequenceLength = 512
    private let expectedEmbeddingDimensions = 384

    // MARK: - Test Fixtures

    /// Create a valid tokenized input for testing
    private func createValidInput(sequenceLength: Int = 512) -> TokenizedInput {
        // Typical BERT input: [CLS] + tokens + [SEP] + padding
        var tokenIds = [101] // [CLS]
        tokenIds.append(contentsOf: Array(repeating: 100, count: min(sequenceLength - 2, 10))) // Some tokens
        tokenIds.append(102) // [SEP]
        tokenIds.append(contentsOf: Array(repeating: 0, count: max(0, sequenceLength - tokenIds.count))) // PAD

        // Attention mask: 1 for real tokens, 0 for padding
        let realTokenCount = 12 // [CLS] + 10 tokens + [SEP]
        var attentionMask = Array(repeating: 1, count: min(realTokenCount, sequenceLength))
        attentionMask.append(contentsOf: Array(repeating: 0, count: max(0, sequenceLength - attentionMask.count)))

        // Token type IDs (all 0 for single sequence)
        let tokenTypeIds = Array(repeating: 0, count: sequenceLength)

        return TokenizedInput(
            tokenIds: tokenIds,
            attentionMask: attentionMask,
            tokenTypeIds: tokenTypeIds,
            originalLength: realTokenCount
        )
    }

    /// Create multiple valid inputs for batch testing
    private func createBatchInputs(count: Int, sequenceLength: Int = 512) -> [TokenizedInput] {
        return (0..<count).map { _ in createValidInput(sequenceLength: sequenceLength) }
    }

    // MARK: - Setup & Teardown

    override func setUp() {
        super.setUp()

        // Verify test model exists
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: testModelURL.path),
            "Test model not found at: \(testModelURL.path). Run model conversion first."
        )
    }

    // MARK: - T-FUNC-002a: Model Loading and Metadata Extraction

    func testModelLoading() async throws {
        let backend = CoreMLBackend()

        // Initially not loaded
        let isLoadedBefore = await backend.isLoaded
        XCTAssertFalse(isLoadedBefore, "Model should not be loaded initially")

        // Load model
        try await backend.loadModel(from: testModelURL)

        // Now should be loaded
        let isLoadedAfter = await backend.isLoaded
        XCTAssertTrue(isLoadedAfter, "Model should be loaded after loadModel()")

        // Verify metadata exists
        let metadata = await backend.metadata
        XCTAssertNotNil(metadata, "Metadata should be extracted after loading")

        // Verify metadata values
        if let metadata = metadata {
            XCTAssertEqual(
                metadata.embeddingDimensions,
                expectedEmbeddingDimensions,
                "Should extract correct embedding dimensions"
            )
            XCTAssertEqual(
                metadata.maxSequenceLength,
                expectedSequenceLength,
                "Should extract correct sequence length"
            )
        }

        // Verify input/output dimensions methods
        let inputDims = await backend.inputDimensions()
        XCTAssertNotNil(inputDims, "Input dimensions should be available")
        XCTAssertEqual(inputDims?.sequence, expectedSequenceLength)
        XCTAssertEqual(inputDims?.features, expectedEmbeddingDimensions)

        let outputDims = await backend.outputDimensions()
        XCTAssertNotNil(outputDims, "Output dimensions should be available")
        XCTAssertEqual(outputDims, expectedEmbeddingDimensions)
    }

    func testModelLoadingWithCustomConfiguration() async throws {
        // Test with CPU-only configuration
        let cpuConfig = CoreMLConfiguration(
            useNeuralEngine: false,
            allowCPUFallback: true,
            maxBatchSize: 16
        )

        let backend = CoreMLBackend(configuration: cpuConfig)
        try await backend.loadModel(from: testModelURL)

        let isLoaded = await backend.isLoaded
        XCTAssertTrue(isLoaded, "Model should load with custom configuration")

        let metadata = await backend.metadata
        XCTAssertNotNil(metadata, "Metadata should be available")
    }

    func testModelLoadingFailsWithInvalidPath() async throws {
        let backend = CoreMLBackend()
        let invalidURL = URL(fileURLWithPath: "/nonexistent/model.mlpackage")

        do {
            try await backend.loadModel(from: invalidURL)
            XCTFail("Should throw error for invalid model path")
        } catch let error as CoreMLError {
            switch error {
            case .modelLoadingFailed(let url, _):
                XCTAssertEqual(url, invalidURL)
            default:
                XCTFail("Expected modelLoadingFailed error, got: \(error)")
            }
        } catch {
            XCTFail("Expected CoreMLError, got: \(error)")
        }

        // Model should not be loaded
        let isLoaded = await backend.isLoaded
        XCTAssertFalse(isLoaded, "Model should not be loaded after failed load")
    }

    // MARK: - T-FUNC-002b: Single Inference Correctness

    func testSingleInference() async throws {
        let backend = CoreMLBackend()
        try await backend.loadModel(from: testModelURL)

        let input = createValidInput()

        // Run inference
        let output = try await backend.generateEmbeddings(for: input)

        // Validate output structure
        XCTAssertEqual(
            output.tokenEmbeddings.count,
            expectedSequenceLength,
            "Should return embeddings for all tokens"
        )

        // Validate embedding dimensions
        for (idx, embedding) in output.tokenEmbeddings.enumerated() {
            XCTAssertEqual(
                embedding.count,
                expectedEmbeddingDimensions,
                "Token \(idx) embedding should have correct dimensions"
            )
        }

        // Validate all embeddings are finite (no NaN or Inf)
        for (tokenIdx, embedding) in output.tokenEmbeddings.enumerated() {
            for (dimIdx, value) in embedding.enumerated() {
                XCTAssertTrue(
                    value.isFinite,
                    "Token \(tokenIdx) dimension \(dimIdx) has non-finite value: \(value)"
                )
            }
        }

        // Verify metadata
        XCTAssertNotNil(output.metadata["hiddenSize"])
        XCTAssertNotNil(output.metadata["tokenCount"])
        XCTAssertEqual(output.metadata["tokenCount"], String(expectedSequenceLength))
    }

    func testSingleInferenceProducesNonZeroEmbeddings() async throws {
        let backend = CoreMLBackend()
        try await backend.loadModel(from: testModelURL)

        let input = createValidInput()
        let output = try await backend.generateEmbeddings(for: input)

        // At least some embeddings should be non-zero
        // (checking first token, which is [CLS])
        let firstEmbedding = output.tokenEmbeddings[0]
        let nonZeroCount = firstEmbedding.filter { abs($0) > 1e-10 }.count

        XCTAssertGreaterThan(
            nonZeroCount,
            0,
            "Embeddings should contain non-zero values"
        )

        // Check that embeddings have reasonable magnitude
        // (not all zeros, not exploding)
        let magnitude = sqrt(firstEmbedding.reduce(0) { $0 + $1 * $1 })
        XCTAssertGreaterThan(magnitude, 0.1, "Embedding magnitude too small")
        XCTAssertLessThan(magnitude, 1000.0, "Embedding magnitude too large")
    }

    func testInferenceFailsWithoutLoadedModel() async throws {
        let backend = CoreMLBackend()

        // Don't load model
        let input = createValidInput()

        do {
            _ = try await backend.generateEmbeddings(for: input)
            XCTFail("Should throw error when model not loaded")
        } catch let error as CoreMLError {
            switch error {
            case .unsupportedModelType(let message):
                XCTAssertTrue(message.contains("No model loaded"))
            default:
                XCTFail("Expected unsupportedModelType error, got: \(error)")
            }
        } catch {
            XCTFail("Expected CoreMLError, got: \(error)")
        }
    }

    // MARK: - T-FUNC-002c: Batch Inference Correctness

    func testBatchInference() async throws {
        let backend = CoreMLBackend()
        try await backend.loadModel(from: testModelURL)

        let batchSize = 5
        let inputs = createBatchInputs(count: batchSize)

        // Run batch inference
        let outputs = try await backend.generateEmbeddings(for: inputs)

        // Validate output count
        XCTAssertEqual(
            outputs.count,
            batchSize,
            "Should return output for each input"
        )

        // Validate each output
        for (idx, output) in outputs.enumerated() {
            XCTAssertEqual(
                output.tokenEmbeddings.count,
                expectedSequenceLength,
                "Output \(idx) should have correct token count"
            )

            for embedding in output.tokenEmbeddings {
                XCTAssertEqual(
                    embedding.count,
                    expectedEmbeddingDimensions,
                    "Output \(idx) embeddings should have correct dimensions"
                )

                XCTAssertTrue(
                    embedding.allSatisfy { $0.isFinite },
                    "Output \(idx) should have all finite values"
                )
            }
        }
    }

    func testBatchInferenceWithLargeBatch() async throws {
        let backend = CoreMLBackend()
        try await backend.loadModel(from: testModelURL)

        // Test with batch larger than default maxBatchSize
        let largeBatchSize = 50
        let inputs = createBatchInputs(count: largeBatchSize)

        let outputs = try await backend.generateEmbeddings(for: inputs)

        XCTAssertEqual(
            outputs.count,
            largeBatchSize,
            "Should handle large batches by chunking"
        )

        // All outputs should be valid
        for output in outputs {
            XCTAssertEqual(output.tokenEmbeddings.count, expectedSequenceLength)
            XCTAssertTrue(output.tokenEmbeddings.allSatisfy { embedding in
                embedding.count == expectedEmbeddingDimensions &&
                embedding.allSatisfy { $0.isFinite }
            })
        }
    }

    func testBatchInferenceConsistency() async throws {
        let backend = CoreMLBackend()
        try await backend.loadModel(from: testModelURL)

        // Create identical inputs
        let identicalInput = createValidInput()
        let inputs = Array(repeating: identicalInput, count: 3)

        let outputs = try await backend.generateEmbeddings(for: inputs)

        // All outputs should be identical (within floating-point tolerance)
        let reference = outputs[0].tokenEmbeddings
        for (idx, output) in outputs.dropFirst().enumerated() {
            for (tokenIdx, (refToken, outToken)) in zip(reference, output.tokenEmbeddings).enumerated() {
                for (dimIdx, (refVal, outVal)) in zip(refToken, outToken).enumerated() {
                    XCTAssertEqual(
                        refVal, outVal,
                        accuracy: 1e-5,
                        "Output \(idx+1) token \(tokenIdx) dim \(dimIdx) differs from reference"
                    )
                }
            }
        }
    }

    func testEmptyBatch() async throws {
        let backend = CoreMLBackend()
        try await backend.loadModel(from: testModelURL)

        let emptyInputs: [TokenizedInput] = []
        let outputs = try await backend.generateEmbeddings(for: emptyInputs)

        XCTAssertEqual(outputs.count, 0, "Empty batch should return empty results")
    }

    // MARK: - T-FUNC-002d: Input Validation

    func testInferenceWithDifferentSequenceLengths() async throws {
        let backend = CoreMLBackend()
        try await backend.loadModel(from: testModelURL)

        // Note: MiniLM-L12-v2 requires fixed 512-token sequences
        // Test with full sequence
        let fullInput = createValidInput(sequenceLength: 512)
        let fullOutput = try await backend.generateEmbeddings(for: fullInput)

        XCTAssertEqual(
            fullOutput.tokenEmbeddings.count,
            512,
            "Should handle full-length sequences"
        )

        // Verify all embeddings are valid
        for embedding in fullOutput.tokenEmbeddings {
            XCTAssertEqual(embedding.count, expectedEmbeddingDimensions)
            XCTAssertTrue(embedding.allSatisfy { $0.isFinite })
        }
    }

    func testInferenceWithOnlySpecialTokens() async throws {
        let backend = CoreMLBackend()
        try await backend.loadModel(from: testModelURL)

        // Input with only [CLS] and [SEP], padded to 512
        var tokenIds = [101, 102] // [CLS], [SEP]
        tokenIds.append(contentsOf: Array(repeating: 0, count: 510)) // PAD to 512

        var attentionMask = [1, 1] // Only special tokens are real
        attentionMask.append(contentsOf: Array(repeating: 0, count: 510))

        let specialOnlyInput = TokenizedInput(
            tokenIds: tokenIds,
            attentionMask: attentionMask,
            tokenTypeIds: Array(repeating: 0, count: 512),
            originalLength: 2
        )

        let output = try await backend.generateEmbeddings(for: specialOnlyInput)

        XCTAssertEqual(
            output.tokenEmbeddings.count,
            512,
            "Should handle minimal input (padded to 512)"
        )

        // Check first two embeddings (special tokens) are non-zero
        let clsEmbedding = output.tokenEmbeddings[0]
        let nonZeroCount = clsEmbedding.filter { abs($0) > 1e-6 }.count
        XCTAssertGreaterThan(nonZeroCount, 0, "Special token embeddings should be non-zero")
    }

    func testInferenceRequiresTokenTypeIds() async throws {
        let backend = CoreMLBackend()
        try await backend.loadModel(from: testModelURL)

        // MiniLM-L12-v2 REQUIRES token_type_ids (not optional)
        // This test verifies proper error handling when they're missing
        var tokenIds = [101, 100, 100, 102] // [CLS], tokens, [SEP]
        tokenIds.append(contentsOf: Array(repeating: 0, count: 508)) // PAD

        var attentionMask = [1, 1, 1, 1]
        attentionMask.append(contentsOf: Array(repeating: 0, count: 508))

        let input = TokenizedInput(
            tokenIds: tokenIds,
            attentionMask: attentionMask,
            tokenTypeIds: nil, // Missing required input
            originalLength: 4
        )

        // Should fail with proper error message
        do {
            _ = try await backend.generateEmbeddings(for: input)
            XCTFail("Should have thrown error for missing token_type_ids")
        } catch {
            // Expected - verify error message mentions token_type_ids
            let errorDescription = String(describing: error)
            XCTAssertTrue(
                errorDescription.contains("token_type_ids"),
                "Error should mention missing token_type_ids feature"
            )
        }
    }

    // MARK: - T-FUNC-002e: Model Lifecycle Management

    func testModelUnload() async throws {
        let backend = CoreMLBackend()

        // Load model
        try await backend.loadModel(from: testModelURL)
        var isLoaded = await backend.isLoaded
        XCTAssertTrue(isLoaded)

        // Unload model
        try await backend.unloadModel()
        isLoaded = await backend.isLoaded
        XCTAssertFalse(isLoaded, "Model should be unloaded")

        // Metadata should be cleared
        let metadata = await backend.metadata
        XCTAssertNil(metadata, "Metadata should be nil after unload")

        // Inference should fail after unload
        let input = createValidInput()
        do {
            _ = try await backend.generateEmbeddings(for: input)
            XCTFail("Should fail to generate embeddings after unload")
        } catch {
            // Expected
        }
    }

    func testModelReload() async throws {
        let backend = CoreMLBackend()

        // Load
        try await backend.loadModel(from: testModelURL)
        let input = createValidInput()
        let output1 = try await backend.generateEmbeddings(for: input)

        // Unload
        try await backend.unloadModel()

        // Reload
        try await backend.loadModel(from: testModelURL)
        let output2 = try await backend.generateEmbeddings(for: input)

        // Outputs should be identical
        XCTAssertEqual(output1.tokenEmbeddings.count, output2.tokenEmbeddings.count)

        for (token1, token2) in zip(output1.tokenEmbeddings, output2.tokenEmbeddings) {
            for (val1, val2) in zip(token1, token2) {
                XCTAssertEqual(val1, val2, accuracy: 1e-5)
            }
        }
    }

    // MARK: - T-CONCUR-002a: Actor Isolation

    func testConcurrentModelAccess() async throws {
        let backend = CoreMLBackend()
        try await backend.loadModel(from: testModelURL)

        let inputs = createBatchInputs(count: 10)

        // Launch concurrent inference tasks
        try await withThrowingTaskGroup(of: ModelOutput.self) { group in
            for input in inputs {
                group.addTask {
                    try await backend.generateEmbeddings(for: input)
                }
            }

            var outputs: [ModelOutput] = []
            for try await output in group {
                outputs.append(output)
            }

            // All outputs should be valid
            XCTAssertEqual(outputs.count, inputs.count)
            for output in outputs {
                XCTAssertEqual(output.tokenEmbeddings.count, expectedSequenceLength)
                XCTAssertTrue(output.tokenEmbeddings.allSatisfy { embedding in
                    embedding.count == expectedEmbeddingDimensions &&
                    embedding.allSatisfy { $0.isFinite }
                })
            }
        }
    }

    func testConcurrentLoadUnload() async throws {
        let backend = CoreMLBackend()

        // This test ensures actor isolation prevents race conditions
        // during concurrent load/unload operations

        let modelURL = testModelURL  // Capture in local variable for Sendable
        try await withThrowingTaskGroup(of: Void.self) { group in
            // Task 1: Load model
            group.addTask {
                try await backend.loadModel(from: modelURL)
            }

            // Task 2: Check if loaded (may execute before or after load)
            group.addTask {
                _ = await backend.isLoaded
            }

            // Task 3: Get metadata (may execute before or after load)
            group.addTask {
                _ = await backend.metadata
            }

            // Wait for all tasks
            try await group.waitForAll()
        }

        // Model should be in consistent state
        let isLoaded = await backend.isLoaded
        let metadata = await backend.metadata

        // If loaded, metadata should exist
        if isLoaded {
            XCTAssertNotNil(metadata, "Loaded model should have metadata")
        }
    }

    // MARK: - Additional Edge Cases

    func testAttentionMaskHandling() async throws {
        let backend = CoreMLBackend()
        try await backend.loadModel(from: testModelURL)

        // Test with sparse attention mask (many padding tokens)
        var tokenIds = [101] // [CLS]
        tokenIds.append(contentsOf: [100, 200, 300]) // Real tokens
        tokenIds.append(102) // [SEP]
        tokenIds.append(contentsOf: Array(repeating: 0, count: 512 - 5)) // Padding

        var attentionMask = [1, 1, 1, 1, 1] // Real tokens
        attentionMask.append(contentsOf: Array(repeating: 0, count: 512 - 5)) // Padding

        let input = TokenizedInput(
            tokenIds: tokenIds,
            attentionMask: attentionMask,
            tokenTypeIds: Array(repeating: 0, count: 512),
            originalLength: 5
        )

        let output = try await backend.generateEmbeddings(for: input)

        // Should successfully process despite sparse mask
        XCTAssertEqual(output.tokenEmbeddings.count, 512)

        // Padding token embeddings should still be computed
        // (they just won't be used in pooling)
        let paddingEmbedding = output.tokenEmbeddings[10] // Some padding token
        XCTAssertTrue(paddingEmbedding.allSatisfy { $0.isFinite })
    }

    func testIdentifierProperty() async throws {
        let backend = CoreMLBackend()
        let identifier = await backend.identifier

        XCTAssertEqual(identifier, "CoreML", "Backend should have correct identifier")
    }

    func testMetadataBeforeLoading() async throws {
        let backend = CoreMLBackend()

        // Metadata should be nil before loading
        let metadata = await backend.metadata
        XCTAssertNil(metadata, "Metadata should be nil before model is loaded")

        // Dimension methods should return nil
        let inputDims = await backend.inputDimensions()
        XCTAssertNil(inputDims, "Input dimensions should be nil before loading")

        let outputDims = await backend.outputDimensions()
        XCTAssertNil(outputDims, "Output dimensions should be nil before loading")
    }
}

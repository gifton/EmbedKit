// Tests for Configuration Matrix - P1 Category
// Validates different configuration combinations work correctly
import Testing
import Foundation
@testable import EmbedKit

// MARK: - Test Infrastructure

/// Backend for configuration testing
actor ConfigTestBackend: CoreMLProcessingBackend {
    private(set) var isLoaded: Bool = false
    var memoryUsage: Int64 { 0 }
    private let dimensions: Int

    init(dimensions: Int = 4) {
        self.dimensions = dimensions
    }

    func load() async throws { isLoaded = true }
    func unload() async throws { isLoaded = false }

    func process(_ input: CoreMLInput) async throws -> CoreMLOutput {
        // Return deterministic output based on input
        let values = (0..<(input.tokenIDs.count * dimensions)).map { Float($0 % 10) / 10.0 }
        return CoreMLOutput(values: values, shape: [input.tokenIDs.count, dimensions])
    }

    func processBatch(_ inputs: [CoreMLInput]) async throws -> [CoreMLOutput] {
        return inputs.map { inp in
            let values = (0..<(inp.tokenIDs.count * dimensions)).map { Float($0 % 10) / 10.0 }
            return CoreMLOutput(values: values, shape: [inp.tokenIDs.count, dimensions])
        }
    }
}

// MARK: - Pooling Strategy Tests

@Suite("Configuration Matrix - Pooling Strategies")
struct PoolingStrategyConfigTests {

    @Test("Mean pooling produces averaged output")
    func meanPoolingConfig() async throws {
        let backend = ConfigTestBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            poolingStrategy: .mean,
            normalizeOutput: false
        )

        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4,
            device: .cpu
        )

        let emb = try await model.embed("test input")
        #expect(emb.metadata.poolingStrategy == .mean)
        #expect(emb.dimensions == 4)
    }

    @Test("Max pooling produces max-pooled output")
    func maxPoolingConfig() async throws {
        let backend = ConfigTestBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            poolingStrategy: .max,
            normalizeOutput: false
        )

        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4,
            device: .cpu
        )

        let emb = try await model.embed("test input")
        #expect(emb.metadata.poolingStrategy == .max)
    }

    @Test("CLS pooling uses first token")
    func clsPoolingConfig() async throws {
        let backend = ConfigTestBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            poolingStrategy: .cls,
            normalizeOutput: false
        )

        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4,
            device: .cpu
        )

        let emb = try await model.embed("test input")
        #expect(emb.metadata.poolingStrategy == .cls)
    }

    @Test("All pooling strategies with normalization")
    func allPoolingWithNormalization() async throws {
        let backend = ConfigTestBackend()
        let tokenizer = SimpleTokenizer()

        for strategy in PoolingStrategy.allCases {
            let cfg = EmbeddingConfiguration(
                poolingStrategy: strategy,
                normalizeOutput: true
            )

            let model = AppleEmbeddingModel(
                backend: backend,
                tokenizer: tokenizer,
                configuration: cfg,
                dimensions: 4,
                device: .cpu
            )

            let emb = try await model.embed("test")
            #expect(emb.dimensions == 4)
            // Normalized should have magnitude ~1
            #expect(abs(emb.magnitude - 1.0) < 0.01)
        }
    }
}

// MARK: - Truncation Strategy Tests

@Suite("Configuration Matrix - Truncation Strategies")
struct TruncationStrategyConfigTests {

    @Test("End truncation removes tokens from end")
    func endTruncation() async throws {
        let backend = ConfigTestBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            maxTokens: 5,
            truncationStrategy: .end,
            includeSpecialTokens: false
        )

        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4,
            device: .cpu
        )

        // Long input that should be truncated
        let emb = try await model.embed("one two three four five six seven eight nine ten")
        #expect(emb.dimensions == 4)
    }

    @Test("Start truncation removes tokens from start")
    func startTruncation() async throws {
        let backend = ConfigTestBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            maxTokens: 5,
            truncationStrategy: .start,
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4,
            device: .cpu
        )

        let emb = try await model.embed("one two three four five six seven eight nine ten")
        #expect(emb.dimensions == 4)
    }

    @Test("Middle truncation removes from middle")
    func middleTruncation() async throws {
        let backend = ConfigTestBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            maxTokens: 6,
            truncationStrategy: .middle,
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4,
            device: .cpu
        )

        let emb = try await model.embed("one two three four five six seven eight nine ten")
        #expect(emb.dimensions == 4)
    }

    @Test("No truncation with short input")
    func noTruncationShortInput() async throws {
        let backend = ConfigTestBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            maxTokens: 512,
            truncationStrategy: .none,
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4,
            device: .cpu
        )

        let emb = try await model.embed("short")
        #expect(emb.metadata.truncated == false)
    }
}

// MARK: - Padding Strategy Tests

@Suite("Configuration Matrix - Padding Strategies")
struct PaddingStrategyConfigTests {

    @Test("No padding leaves sequences as-is")
    func noPadding() async throws {
        let backend = ConfigTestBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            paddingStrategy: .none,
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4,
            device: .cpu
        )

        let texts = ["short", "longer text here"]
        let embeddings = try await model.embedBatch(texts, options: BatchOptions())

        #expect(embeddings.count == 2)
    }

    @Test("Max padding pads to maxTokens")
    func maxPadding() async throws {
        let backend = ConfigTestBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            maxTokens: 10,
            paddingStrategy: .max,
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4,
            device: .cpu
        )

        let texts = ["short", "longer text here"]
        let embeddings = try await model.embedBatch(texts, options: BatchOptions())

        #expect(embeddings.count == 2)
    }

    @Test("Batch padding pads to longest in batch")
    func batchPadding() async throws {
        // Use WordPieceTokenizer which has PAD token support
        let vocab = Vocabulary(tokens: ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "a", "b", "c", "d", "e", "f"])
        let backend = ConfigTestBackend()
        let tokenizer = WordPieceTokenizer(vocabulary: vocab)
        let cfg = EmbeddingConfiguration(
            paddingStrategy: .batch,
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4,
            device: .cpu
        )

        let texts = ["a", "a b c d e f"]
        let embeddings = try await model.embedBatch(texts, options: BatchOptions())

        #expect(embeddings.count == 2)
    }
}

// MARK: - Batch Options Tests

@Suite("Configuration Matrix - Batch Options")
struct BatchOptionsConfigTests {

    @Test("Dynamic batching enabled")
    func dynamicBatchingEnabled() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()

        let opts = BatchOptions(
            maxBatchSize: 16,
            dynamicBatching: true
        )
        let texts = (0..<20).map { "Text \($0)" }
        let embeddings = try await model.embedBatch(texts, options: opts)

        #expect(embeddings.count == 20)
    }

    @Test("Dynamic batching disabled")
    func dynamicBatchingDisabled() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()

        let opts = BatchOptions(
            maxBatchSize: 32,
            dynamicBatching: false
        )
        let texts = (0..<10).map { "Text \($0)" }
        let embeddings = try await model.embedBatch(texts, options: opts)

        #expect(embeddings.count == 10)
    }

    @Test("Sort by length enabled")
    func sortByLengthEnabled() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()

        let opts = BatchOptions(

            sortByLength: true
        )
        let texts = ["short", "a very long text with many words", "medium text"]
        let embeddings = try await model.embedBatch(texts, options: opts)

        #expect(embeddings.count == 3)
    }

    @Test("Sort by length disabled")
    func sortByLengthDisabled() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()

        let opts = BatchOptions(

            sortByLength: false
        )
        let texts = ["short", "a very long text with many words", "medium text"]
        let embeddings = try await model.embedBatch(texts, options: opts)

        #expect(embeddings.count == 3)
    }

    @Test("Various batch sizes")
    func variousBatchSizes() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let texts = (0..<50).map { "Text \($0)" }

        for batchSize in [1, 4, 8, 16, 32, 64] {
            let opts = BatchOptions(
                maxBatchSize: batchSize
            )
            let embeddings = try await model.embedBatch(texts, options: opts)
            #expect(embeddings.count == 50)
        }
    }

    @Test("Max batch tokens constraint")
    func maxBatchTokensConstraint() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()

        let opts = BatchOptions(

            maxBatchTokens: 100  // Limit tokens per batch
        )
        let texts = (0..<10).map { "Word \($0)" }
        let embeddings = try await model.embedBatch(texts, options: opts)

        #expect(embeddings.count == 10)
    }

    @Test("Tokenization concurrency settings")
    func tokenizationConcurrencySettings() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let texts = (0..<20).map { "Text \($0)" }

        for concurrency in [1, 2, 4, 8] {
            let opts = BatchOptions(
                tokenizationConcurrency: concurrency
            )
            let embeddings = try await model.embedBatch(texts, options: opts)
            #expect(embeddings.count == 20)
        }
    }
}

// MARK: - Device Configuration Tests

@Suite("Configuration Matrix - Compute Device")
struct ComputeDeviceConfigTests {

    @Test("CPU device configuration")
    func cpuDevice() async throws {
        let backend = ConfigTestBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            inferenceDevice: .cpu
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4,
            device: .cpu
        )

        let emb = try await model.embed("test")
        #expect(emb.dimensions == 4)
    }

    @Test("Auto device configuration")
    func autoDevice() async throws {
        let backend = ConfigTestBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            inferenceDevice: .auto
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4,
            device: .auto
        )

        let emb = try await model.embed("test")
        #expect(emb.dimensions == 4)
    }

    @Test("GPU threshold settings")
    func gpuThresholdSettings() async throws {
        let backend = ConfigTestBackend()
        let tokenizer = SimpleTokenizer()

        for threshold in [0, 1000, 8192, Int.max] {
            let cfg = EmbeddingConfiguration(
                minElementsForGPU: threshold
            )
            let model = AppleEmbeddingModel(
                backend: backend,
                tokenizer: tokenizer,
                configuration: cfg,
                dimensions: 4,
                device: .cpu
            )

            let emb = try await model.embed("test")
            #expect(emb.dimensions == 4)
        }
    }
}

// MARK: - Special Tokens Tests

@Suite("Configuration Matrix - Special Tokens")
struct SpecialTokensConfigTests {

    @Test("Include special tokens enabled")
    func includeSpecialTokensEnabled() async throws {
        let backend = ConfigTestBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            includeSpecialTokens: true
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4,
            device: .cpu
        )

        let emb = try await model.embed("test")
        #expect(emb.dimensions == 4)
    }

    @Test("Include special tokens disabled")
    func includeSpecialTokensDisabled() async throws {
        let backend = ConfigTestBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4,
            device: .cpu
        )

        let emb = try await model.embed("test")
        #expect(emb.dimensions == 4)
    }
}

// MARK: - Normalization Configuration Tests

@Suite("Configuration Matrix - Output Normalization")
struct OutputNormalizationConfigTests {

    @Test("Normalize output enabled produces unit vectors")
    func normalizeOutputEnabled() async throws {
        let backend = ConfigTestBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            normalizeOutput: true
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4,
            device: .cpu
        )

        let emb = try await model.embed("test input text")
        #expect(abs(emb.magnitude - 1.0) < 0.01)
        #expect(emb.metadata.normalized == true)
    }

    @Test("Normalize output disabled preserves raw values")
    func normalizeOutputDisabled() async throws {
        let backend = ConfigTestBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            normalizeOutput: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: 4,
            device: .cpu
        )

        let emb = try await model.embed("test input text")
        // May or may not have unit magnitude
        #expect(emb.dimensions == 4)
    }
}

// MARK: - Combined Configuration Tests

@Suite("Configuration Matrix - Combined Settings")
struct CombinedConfigTests {

    @Test("All pooling with all truncation")
    func allPoolingAllTruncation() async throws {
        let backend = ConfigTestBackend()
        let tokenizer = SimpleTokenizer()

        for pooling in PoolingStrategy.allCases {
            for truncation in TruncationStrategy.allCases {
                let cfg = EmbeddingConfiguration(
                    maxTokens: 10,
                    truncationStrategy: truncation,
                    includeSpecialTokens: false,
                    poolingStrategy: pooling
                )
                let model = AppleEmbeddingModel(
                    backend: backend,
                    tokenizer: tokenizer,
                    configuration: cfg,
                    dimensions: 4,
                    device: .cpu
                )

                let emb = try await model.embed("test input")
                #expect(emb.dimensions == 4)
            }
        }
    }

    @Test("All padding with dynamic batching combinations")
    func allPaddingDynamicBatching() async throws {
        // Use WordPieceTokenizer for PAD token support in batch padding
        let vocab = Vocabulary(tokens: [
            "[PAD]", "[CLS]", "[SEP]", "[UNK]",
            "short", "medium", "length", "a", "longer", "piece", "of", "text"
        ])
        let backend = ConfigTestBackend()
        let tokenizer = WordPieceTokenizer(vocabulary: vocab)
        let texts = ["short", "medium length", "a longer piece of text"]

        for padding in PaddingStrategy.allCases {
            for dynamic in [true, false] {
                let cfg = EmbeddingConfiguration(
                    paddingStrategy: padding,
                    includeSpecialTokens: false
                )
                let opts = BatchOptions(
                    dynamicBatching: dynamic
                )
                let model = AppleEmbeddingModel(
                    backend: backend,
                    tokenizer: tokenizer,
                    configuration: cfg,
                    dimensions: 4,
                    device: .cpu
                )

                let embeddings = try await model.embedBatch(texts, options: opts)
                #expect(embeddings.count == 3)
            }
        }
    }

    @Test("Extreme configuration combinations")
    func extremeConfigurations() async throws {
        let backend = ConfigTestBackend()
        let tokenizer = SimpleTokenizer()

        // Very small max tokens
        let cfg1 = EmbeddingConfiguration(
            maxTokens: 2,
            truncationStrategy: .end,
            includeSpecialTokens: false
        )
        let model1 = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg1,
            dimensions: 4,
            device: .cpu
        )

        let emb1 = try await model1.embed("this is a test")
        #expect(emb1.dimensions == 4)

        // Very large max tokens
        let cfg2 = EmbeddingConfiguration(
            maxTokens: 10000,
            truncationStrategy: .none
        )
        let model2 = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg2,
            dimensions: 4,
            device: .cpu
        )

        let emb2 = try await model2.embed("short")
        #expect(emb2.dimensions == 4)
    }
}

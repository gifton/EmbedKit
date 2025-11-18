// EmbedKit Usage Examples - Clean, Powerful, Customizable

import Foundation

// ============================================================================
// MARK: - Basic Usage
// ============================================================================

func basicUsage() async throws {
    let manager = ModelManager()

    // Load Apple's on-device model
    let model = try await manager.loadAppleModel()

    // Simple embedding
    let embedding = try await model.embed("Hello, world!")
    print("Dimensions: \(embedding.dimensions)")
    print("Normalized: \(embedding.isNormalized)")
}

// ============================================================================
// MARK: - Full Customization
// ============================================================================

func advancedCustomization() async throws {
    let manager = ModelManager()

    // Fully customized configuration
    var config = EmbeddingConfiguration()
    config.maxTokens = 256
    config.truncationStrategy = .middle
    config.poolingStrategy = .attention
    config.normalizeOutput = false
    config.batchSize = 64
    config.preferredDevice = .ane
    config.useMixedPrecision = true

    // Load with custom config
    let model = try await manager.loadAppleModel(
        variant: .large,
        configuration: config
    )

    // Generate embedding
    let embedding = try await model.embed("Complex technical document...")

    // Access detailed metadata
    print("Model: \(embedding.metadata.modelID)")
    print("Tokens processed: \(embedding.metadata.tokenCount)")
    print("Processing time: \(embedding.metadata.processingTime)ms")
    print("Truncated: \(embedding.metadata.truncated)")
}

// ============================================================================
// MARK: - Batch Processing
// ============================================================================

func batchProcessing() async throws {
    let manager = ModelManager()
    let model = try await manager.loadAppleModel()

    let documents = [
        "First document about machine learning",
        "Second document about Swift programming",
        "Third document about iOS development",
        // ... hundreds more
    ]

    // Custom batch options
    var batchOptions = BatchOptions()
    batchOptions.maxBatchSize = 128
    batchOptions.sortByLength = true  // Optimize padding
    batchOptions.dynamicBatching = true
    batchOptions.timeout = 30.0

    // Process in optimized batches
    let embeddings = try await model.embedBatch(
        documents,
        options: batchOptions
    )

    // Compute similarity matrix
    for (i, emb1) in embeddings.enumerated() {
        for (j, emb2) in embeddings.enumerated() where j > i {
            let similarity = emb1.similarity(to: emb2)
            print("Doc \(i) <-> Doc \(j): \(similarity)")
        }
    }
}

// ============================================================================
// MARK: - Multiple Models
// ============================================================================

func multipleModels() async throws {
    let manager = ModelManager()

    // Load different models
    let appleModel = try await manager.loadAppleModel(variant: .base)
    let multilingualModel = try await manager.loadAppleModel(variant: .multilingual)

    // Load a custom local model
    let localModel = try await manager.loadLocalModel(
        at: URL(fileURLWithPath: "/path/to/model.mlmodelc"),
        tokenizer: CustomTokenizer()
    )

    // Use different models for different purposes
    let englishEmbedding = try await appleModel.embed("English text")
    let multilingualEmbedding = try await multilingualModel.embed("多言語テキスト")
    let customEmbedding = try await localModel.embed("Domain-specific text")

    // Compare embeddings across models
    print("Cross-model similarity: \(englishEmbedding.similarity(to: customEmbedding))")
}

// ============================================================================
// MARK: - Custom Tokenizer
// ============================================================================

struct CustomTokenizer: Tokenizer {
    var vocabularySize: Int = 50000

    var specialTokens: SpecialTokens {
        SpecialTokens(
            cls: SpecialTokens.Token(text: "[CLS]", id: 101),
            sep: SpecialTokens.Token(text: "[SEP]", id: 102),
            pad: SpecialTokens.Token(text: "[PAD]", id: 0),
            unk: SpecialTokens.Token(text: "[UNK]", id: 100),
            mask: nil,
            bos: nil,
            eos: nil
        )
    }

    func encode(_ text: String, config: TokenizerConfig) async throws -> TokenizedText {
        // Custom tokenization logic
        let tokens = text.split(separator: " ").map(String.init)
        let ids = tokens.map { $0.hashValue % vocabularySize }
        let mask = Array(repeating: 1, count: tokens.count)

        return TokenizedText(
            ids: ids,
            tokens: tokens,
            attentionMask: mask,
            typeIds: nil,
            specialTokenMask: Array(repeating: false, count: tokens.count),
            offsets: nil
        )
    }

    func decode(_ ids: [Int]) async throws -> String {
        // Reverse tokenization
        ids.map { "token_\($0)" }.joined(separator: " ")
    }
}

// ============================================================================
// MARK: - Resource Management
// ============================================================================

func resourceManagement() async throws {
    let manager = ModelManager()

    // Load model
    let modelID = ModelID(
        provider: "apple",
        name: "text-embedding",
        version: "1.0.0",
        variant: "base"
    )

    let spec = ModelSpecification(
        id: modelID,
        source: .system,
        format: .coreml,
        preload: true  // Preload into memory
    )

    let model = try await manager.loadModel(spec)

    // Use model
    let embeddings = try await model.embedBatch(
        ["text1", "text2"],
        options: BatchOptions()
    )

    // Check metrics
    let metrics = await model.metrics
    print("Total requests: \(metrics.totalRequests)")
    print("Average latency: \(metrics.averageLatency)ms")
    print("Memory usage: \(metrics.memoryUsage / 1024 / 1024)MB")

    // Unload when done
    try await manager.unloadModel(modelID)
}

// ============================================================================
// MARK: - Error Handling
// ============================================================================

func robustErrorHandling() async throws {
    let manager = ModelManager()

    do {
        // Configure with strict limits
        var config = EmbeddingConfiguration()
        config.maxTokens = 128
        config.truncationStrategy = .none  // Error on long input

        let model = try await manager.loadAppleModel(configuration: config)

        // This might fail if text is too long
        let embedding = try await model.embed(String(repeating: "word ", count: 1000))

    } catch EmbedKitError.inputTooLong(let length, let max) {
        print("Input too long: \(length) tokens (max: \(max))")
        // Handle by truncating or splitting

    } catch EmbedKitError.deviceNotAvailable(let device) {
        print("Device \(device) not available, falling back...")
        // Retry with different device

    } catch EmbedKitError.processingTimeout {
        print("Processing timed out")
        // Handle timeout

    } catch {
        print("Unexpected error: \(error)")
    }
}

// ============================================================================
// MARK: - Semantic Search
// ============================================================================

func semanticSearch() async throws {
    let manager = ModelManager()
    let model = try await manager.loadAppleModel()

    // Document corpus
    let documents = [
        "Swift is a powerful programming language",
        "iOS development requires knowledge of UIKit or SwiftUI",
        "Machine learning models can run on device",
        "Core Data provides persistent storage",
        "Combine framework enables reactive programming"
    ]

    // Generate embeddings for all documents
    let docEmbeddings = try await model.embedBatch(
        documents,
        options: BatchOptions(sortByLength: true)
    )

    // Query
    let query = "How to build iOS apps?"
    let queryEmbedding = try await model.embed(query)

    // Find most similar documents
    let similarities = docEmbeddings.enumerated().map { index, docEmb in
        (index: index, similarity: queryEmbedding.similarity(to: docEmb))
    }.sorted { $0.similarity > $1.similarity }

    print("Top results for '\(query)':")
    for (index, similarity) in similarities.prefix(3) {
        print("  [\(similarity)]: \(documents[index])")
    }
}

// ============================================================================
// MARK: - Performance Monitoring
// ============================================================================

func performanceMonitoring() async throws {
    let manager = ModelManager()

    var config = EmbeddingConfiguration()
    config.cacheTokenization = true  // Enable caching
    config.useMixedPrecision = true  // Use FP16 where possible

    let model = try await manager.loadAppleModel(configuration: config)

    // Warmup
    try await model.warmup()

    // Benchmark
    let texts = (0..<100).map { "Test document number \($0)" }

    let start = Date()
    let embeddings = try await model.embedBatch(
        texts,
        options: BatchOptions(maxBatchSize: 50)
    )
    let elapsed = Date().timeIntervalSince(start)

    print("Processed \(texts.count) documents in \(elapsed)s")
    print("Throughput: \(Double(texts.count) / elapsed) docs/sec")

    // Detailed metrics
    let metrics = await model.metrics
    print("Cache hit rate: \(metrics.cacheHitRate * 100)%")
    print("P95 latency: \(metrics.p95Latency)ms")
}

// ============================================================================
// MARK: - Custom Model Specification
// ============================================================================

func customModelSpec() async throws {
    let manager = ModelManager()

    // Define custom model specification
    let customSpec = ModelSpecification(
        id: ModelID(
            provider: "huggingface",
            name: "sentence-transformers",
            version: "2.0.0",
            variant: "all-MiniLM-L6-v2"
        ),
        source: .remote(URL(string: "https://example.com/model.mlmodelc")!),
        format: .coreml,
        preload: false
    )

    // Custom configuration
    var config = EmbeddingConfiguration()
    config.maxTokens = 384
    config.poolingStrategy = .meanSqrt
    config.normalizeOutput = true

    // Load and use
    let model = try await manager.loadModel(customSpec, configuration: config)
    let embedding = try await model.embed("Custom model test")

    print("Custom model output: \(embedding.dimensions) dimensions")
}
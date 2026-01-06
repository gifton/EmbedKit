// EmbedKit - v0.2.2 Integration Tests
// Tests integration between newly completed batches for v0.2.2 release
//
// Covers:
// - ConfigurationFactory + EmbeddingGenerator integration
// - BatchProgress + AdaptiveBatcher integration
// - VSKError protocol integration
// - PipelineConfiguration + toAdaptiveBatcherConfig() integration

import Testing
import Foundation
@testable import EmbedKit
import VectorCore

// MARK: - ConfigurationFactory + EmbeddingGenerator Integration

@Suite("ConfigurationFactory + EmbeddingGenerator Integration")
struct ConfigurationFactoryGeneratorTests {

    @Test("Factory semantic search config produces expected embedding behavior")
    func testSemanticSearchConfig() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 384)
        let config = GeneratorConfiguration.forSemanticSearch(maxLength: 256)
        let generator = EmbeddingGenerator(
            model: mockModel,
            configuration: config.embedding,
            batchOptions: config.batch
        )

        // Verify configuration flowed through
        #expect(generator.producesNormalizedVectors == true)
        #expect(generator.dimensions == 384)

        // Verify embedding works
        let vectors = try await generator.produce(["test query", "another query"])
        #expect(vectors.count == 2)
        #expect(vectors[0].count == 384)
    }

    @Test("Factory high throughput config increases batch size")
    func testHighThroughputConfig() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 128)
        let config = GeneratorConfiguration.highThroughput
        let generator = EmbeddingGenerator(
            model: mockModel,
            configuration: config.embedding,
            batchOptions: config.batch
        )

        // High throughput should have larger batch size
        let hints = await generator.hints
        #expect(hints.maxBatchSize == 64)
        #expect(hints.optimalBatchSize == 64)

        // Should still produce correct embeddings
        let vectors = try await generator.produce(["a", "b", "c"])
        #expect(vectors.count == 3)
    }

    @Test("Factory low latency config reduces batch size")
    func testLowLatencyConfig() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 128)
        let config = GeneratorConfiguration.lowLatency
        let generator = EmbeddingGenerator(
            model: mockModel,
            configuration: config.embedding,
            batchOptions: config.batch
        )

        // Low latency should have smaller batch size
        let hints = await generator.hints
        #expect(hints.maxBatchSize == 8)
    }

    @Test("Factory RAG config with custom chunk size")
    func testRAGConfig() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 768)
        let config = GeneratorConfiguration.forRAG(chunkSize: 512)
        let generator = EmbeddingGenerator(
            model: mockModel,
            configuration: config.embedding,
            batchOptions: config.batch
        )

        // RAG config should normalize (for cosine similarity)
        #expect(generator.producesNormalizedVectors == true)

        // Should process chunks correctly
        let chunks = (0..<10).map { "Chunk \($0) content here" }
        let vectors = try await generator.produce(chunks)
        #expect(vectors.count == 10)
    }

    @Test("EmbeddingConfiguration factories integrate with generator")
    func testEmbeddingConfigFactories() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 384)

        // Test each EmbeddingConfiguration factory
        let configs: [EmbeddingConfiguration] = [
            .forSemanticSearch(),
            .forRAG(),
            .forSimilarity(),
            .forDocuments(),
            .forShortText()
        ]

        for config in configs {
            let generator = EmbeddingGenerator(model: mockModel, configuration: config)
            let vector = try await generator.produce("test")
            #expect(vector.count == 384)
        }
    }

    @Test("Model-specific factories produce correct configurations")
    func testModelSpecificFactories() async throws {
        // MiniLM-style model (384 dims)
        let miniLMModel = MockEmbeddingModel(dimensions: 384)
        let miniLMConfig = EmbeddingConfiguration.forMiniLM(useCase: .semanticSearch)
        let miniLMGenerator = EmbeddingGenerator(model: miniLMModel, configuration: miniLMConfig)
        #expect(miniLMGenerator.producesNormalizedVectors == true)

        // BERT-style model (768 dims)
        let bertModel = MockEmbeddingModel(dimensions: 768)
        let bertConfig = EmbeddingConfiguration.forBERT(useCase: .rag)
        let bertGenerator = EmbeddingGenerator(model: bertModel, configuration: bertConfig)
        #expect(bertGenerator.producesNormalizedVectors == true)
    }
}

// MARK: - BatchProgress + AdaptiveBatcher Integration

@Suite("BatchProgress + AdaptiveBatcher Integration")
struct BatchProgressAdaptiveBatcherTests {

    @Test("Progress callback receives accurate batch index")
    func testProgressBatchIndex() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()

        var config = AdaptiveBatcherConfig()
        config.maxBatchSize = 3
        let batcher = AdaptiveBatcher(model: model, config: config)

        actor ProgressCollector {
            var batchIndices: [Int] = []
            func add(_ index: Int) { batchIndices.append(index) }
        }
        let collector = ProgressCollector()

        let texts = (0..<9).map { "Text \($0)" }  // 3 batches of 3

        _ = try await batcher.embedWithProgress(texts) { progress in
            Task { await collector.add(progress.batchIndex) }
        }

        // Allow callbacks to complete
        try await Task.sleep(for: .milliseconds(100))

        let indices = await collector.batchIndices
        // Should have batch indices 0, 1, 2 (plus start/complete)
        let uniqueIndices = Set(indices)
        #expect(uniqueIndices.contains(0))
        #expect(uniqueIndices.contains(1))
        #expect(uniqueIndices.contains(2))
    }

    @Test("Progress callback itemsPerSecond is reasonable")
    func testProgressItemsPerSecond() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let batcher = AdaptiveBatcher(model: model)

        actor ProgressCollector {
            var finalProgress: BatchProgress?
            func set(_ p: BatchProgress) { finalProgress = p }
        }
        let collector = ProgressCollector()

        let texts = (0..<20).map { "Text \($0)" }

        _ = try await batcher.embedWithProgress(texts) { progress in
            Task { await collector.set(progress) }
        }

        try await Task.sleep(for: .milliseconds(100))

        let final = await collector.finalProgress
        #expect(final != nil)
        #expect(final?.itemsPerSecond ?? 0 > 0, "Should have positive throughput")
        #expect(final?.isComplete == true)
    }

    @Test("Progress stream yields all items and reaches completion")
    func testProgressStreamCompletion() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let batcher = AdaptiveBatcher(model: model)

        let texts = ["first", "second", "third", "fourth", "fifth"]

        var totalEmbeddings = 0
        var finalProgress: BatchProgress?
        for try await (batchEmbeddings, progress) in await batcher.embedBatchStream(texts) {
            totalEmbeddings += batchEmbeddings.count
            finalProgress = progress
        }

        // Should have processed all 5 texts
        #expect(totalEmbeddings == 5)
        // Final progress should show all items complete
        #expect(finalProgress?.current == 5)
        #expect(finalProgress?.total == 5)
        #expect(finalProgress?.isComplete == true)
    }

    @Test("BatchProgress.started and .completed factory methods work")
    func testBatchProgressFactories() async throws {
        let started = BatchProgress.started(total: 100, totalBatches: 10)
        #expect(started.current == 0)
        #expect(started.total == 100)
        #expect(started.totalBatches == 10)
        #expect(started.phase == "Starting")
        #expect(started.isComplete == false)

        let completed = BatchProgress.completed(
            total: 100,
            totalBatches: 10,
            tokensProcessed: 5000,
            itemsPerSecond: 50.0
        )
        #expect(completed.current == 100)
        #expect(completed.isComplete == true)
        #expect(completed.phase == "Complete")
        #expect(completed.tokensProcessed == 5000)
        #expect(completed.itemsPerSecond == 50.0)
    }

    @Test("BatchProgress wraps OperationProgress correctly")
    func testBatchProgressOperationProgress() {
        let batchProgress = BatchProgress(
            current: 50,
            total: 100,
            batchIndex: 2,
            totalBatches: 4,
            phase: "Processing",
            message: "Batch 3/4",
            itemsPerSecond: 25.0,
            tokensProcessed: 1000,
            currentBatchSize: 25,
            estimatedTimeRemaining: 2.0
        )

        // Check OperationProgress properties are accessible
        #expect(batchProgress.current == 50)
        #expect(batchProgress.total == 100)
        #expect(batchProgress.fraction == 0.5)
        #expect(batchProgress.percentage == 50)
        #expect(batchProgress.phase == "Processing")

        // Check BatchProgress-specific properties
        #expect(batchProgress.batchIndex == 2)
        #expect(batchProgress.totalBatches == 4)
        #expect(batchProgress.itemsPerSecond == 25.0)
        #expect(batchProgress.tokensProcessed == 1000)
    }
}

// MARK: - VSKError Integration

@Suite("VSKError Protocol Integration")
struct VSKErrorIntegrationTests {

    @Test("EmbedKitError can be caught as VSKError")
    func testCatchAsVSKError() async throws {
        let error: any Error = EmbedKitError.modelNotFound(ModelID(provider: "test", name: "missing", version: "1.0"))

        // Should be catchable as VSKError
        if let vskError = error as? any VSKError {
            #expect(vskError.domain == "EmbedKit")
            #expect(vskError.errorCode >= 2000)
            #expect(vskError.errorCode < 3000)
        } else {
            Issue.record("EmbedKitError should conform to VSKError")
        }
    }

    @Test("Error code ranges don't overlap within EmbedKit")
    func testErrorCodeRangesNoOverlap() {
        // Collect all error codes
        let errors: [EmbedKitError] = [
            .modelNotFound(ModelID(provider: "test", name: "test", version: "1.0")),
            .modelLoadFailed("test model load failed"),
            .tokenizationFailed("test"),
            .inputTooLong(length: 100, max: 50),
            .inferenceFailed("test"),
            .processingTimeout,
            .batchSizeExceeded(size: 100, max: 50),
            .dimensionMismatch(expected: 384, got: 768),
            .deviceNotAvailable(.gpu),
            .invalidConfiguration("test"),
            .metalDeviceUnavailable,
            .metalBufferFailed,
            .metalPipelineNotFound("test"),
            .metalEncoderFailed,
            .metalTensorFailed("test")
        ]

        var codes = Set<Int>()
        for error in errors {
            let code = error.errorCode
            #expect(!codes.contains(code), "Duplicate error code: \(code)")
            codes.insert(code)
        }
    }

    @Test("Error context contains debugging info")
    func testErrorContextInfo() {
        let error = EmbedKitError.modelNotFound(ModelID(provider: "huggingface", name: "bert-base", version: "1.0"))
        let context = error.context

        #expect(context.additionalInfo["modelID"] != nil)
        #expect(context.additionalInfo["modelID"]?.contains("huggingface") == true)
    }

    @Test("VSKError description format is consistent")
    func testVSKErrorDescription() {
        let error = EmbedKitError.tokenizationFailed("Invalid UTF-8 sequence")

        // VSKError description should include domain and code
        let description = error.description
        #expect(description.contains("EmbedKit"))
        #expect(description.contains("2100"))  // Tokenization error code
    }

    @Test("Recoverable errors are correctly identified")
    func testRecoverableErrors() {
        let recoverable: [EmbedKitError] = [
            .processingTimeout,
            .metalBufferFailed,
            .metalEncoderFailed,
            .metalTensorFailed("test")
        ]

        let nonRecoverable: [EmbedKitError] = [
            .modelNotFound(ModelID(provider: "test", name: "test", version: "1.0")),
            .dimensionMismatch(expected: 384, got: 768),
            .invalidConfiguration("test")
        ]

        for error in recoverable {
            #expect(error.isRecoverable == true, "\(error) should be recoverable")
        }

        for error in nonRecoverable {
            #expect(error.isRecoverable == false, "\(error) should not be recoverable")
        }
    }

    @Test("VSKErrorCodeRange.embedKit is correct")
    func testEmbedKitCodeRange() {
        #expect(VSKErrorCodeRange.embedKit == 2000..<3000)

        // All EmbedKit errors should be in this range
        let error = EmbedKitError.inferenceFailed("test")
        #expect(VSKErrorCodeRange.embedKit.contains(error.errorCode))
    }
}

// MARK: - PipelineConfiguration Integration

@Suite("PipelineConfiguration Integration")
struct PipelineConfigurationIntegrationTests {

    @Test("PipelineConfiguration.toAdaptiveBatcherConfig creates valid config")
    func testToAdaptiveBatcherConfig() async throws {
        let pipelineConfig = ConfigurationFactory.highThroughput()
        let batcherConfig = pipelineConfig.toAdaptiveBatcherConfig()

        #expect(batcherConfig.maxBatchSize == pipelineConfig.batch.maxBatchSize)

        // Create batcher with config
        let manager = ModelManager()
        let model = try await manager.loadMockModel()
        let batcher = AdaptiveBatcher(model: model, config: batcherConfig)

        // Should work correctly (mock model defaults to 384 dimensions)
        let result = try await batcher.embed("test")
        #expect(result.vector.count == 384)
    }

    @Test("Factory presets create working pipeline configs", .timeLimit(.minutes(1)))
    func testFactoryPresetsCreateWorkingConfigs() async throws {
        let presets: [PipelineConfiguration] = [
            ConfigurationFactory.default(),
            ConfigurationFactory.highThroughput(),
            ConfigurationFactory.lowLatency(),
            ConfigurationFactory.gpuOptimized(),
            ConfigurationFactory.memoryEfficient(),
            ConfigurationFactory.batteryEfficient()
        ]

        let manager = ModelManager()
        let model = try await manager.loadMockModel()

        for preset in presets {
            let batcherConfig = preset.toAdaptiveBatcherConfig()
            let batcher = AdaptiveBatcher(model: model, config: batcherConfig)

            let result = try await batcher.embed("test text")
            #expect(result.vector.count == 384)  // Mock model defaults to 384 dimensions
        }
    }

    @Test("Use-case factories create valid configs")
    func testUseCaseFactories() async throws {
        let useCaseConfigs: [PipelineConfiguration] = [
            ConfigurationFactory.forSemanticSearch(),
            ConfigurationFactory.forRAG(),
            ConfigurationFactory.forRealTimeSearch(),
            ConfigurationFactory.forBatchIndexing()
        ]

        let manager = ModelManager()
        let model = try await manager.loadMockModel()

        for config in useCaseConfigs {
            let batcherConfig = config.toAdaptiveBatcherConfig()
            let batcher = AdaptiveBatcher(model: model, config: batcherConfig)

            let texts = ["query one", "query two", "query three"]
            let results = try await batcher.embedBatch(texts)
            #expect(results.count == 3)
        }
    }

    @Test("Model-specific factories create valid configs")
    func testModelSpecificPipelineFactories() async throws {
        // MiniLM config
        let miniLMConfig = ConfigurationFactory.forMiniLM(useCase: .semanticSearch)
        #expect(miniLMConfig.embedding.normalizeOutput == true)

        // BERT config
        let bertConfig = ConfigurationFactory.forBERT(useCase: .rag)
        #expect(bertConfig.embedding.normalizeOutput == true)

        // Large model config
        let largeConfig = ConfigurationFactory.forLargeModel(dimensions: 1024)
        #expect(largeConfig.batch.maxBatchSize <= 32)  // Should limit batch size for large models
    }

    @Test("withCache modifier adds cache configuration")
    func testWithCacheModifier() {
        let base = ConfigurationFactory.default()
        #expect(base.cache == nil)

        let withCache = base.withCache(.default)
        #expect(withCache.cache != nil)
    }

    @Test("withMemoryBudget modifier sets budget")
    func testWithMemoryBudgetModifier() {
        let base = ConfigurationFactory.default()
        #expect(base.memoryBudget == nil)

        let withBudget = base.withMemoryBudget(mb: 256)
        #expect(withBudget.memoryBudget == 256 * 1024 * 1024)
    }

    @Test("ComputeConfiguration presets have correct values")
    func testComputeConfigurationPresets() {
        let gpuOptimized = ComputeConfiguration.gpuOptimized()
        #expect(gpuOptimized.useFusedKernels == true)

        let memEfficient = ComputeConfiguration.memoryEfficient()
        #expect(memEfficient.maxResidentMemoryMB < 256)
    }
}

// MARK: - MemoryAwareGenerator + ConfigurationFactory Integration

@Suite("MemoryAwareGenerator + ConfigurationFactory Integration")
struct MemoryAwareConfigurationTests {

    @Test("Memory efficient config works with MemoryAwareGenerator")
    func testMemoryEfficientWithMemoryAware() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 128)
        let pipelineConfig = ConfigurationFactory.memoryEfficient()

        let generator = EmbeddingGenerator(
            model: mockModel,
            configuration: pipelineConfig.embedding,
            batchOptions: pipelineConfig.batch
        )

        let memoryAware = await generator.memoryAware(config: .conservative)

        // Should produce correct results
        let texts = (0..<10).map { "Text \($0)" }
        let results = try await memoryAware.produce(texts)
        #expect(results.count == 10)
    }

    @Test("MemoryAwareGenerator respects factory batch options")
    func testMemoryAwareRespectsBatchOptions() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 64)
        let pipelineConfig = ConfigurationFactory.lowLatency()

        let generator = EmbeddingGenerator(
            model: mockModel,
            configuration: pipelineConfig.embedding,
            batchOptions: pipelineConfig.batch
        )

        let memoryAware = await generator.memoryAware(
            config: MemoryAwareConfig(baseBatchSize: 4, minBatchSize: 2)
        )

        // Under normal pressure, should use base batch size
        await memoryAware.setPressure(.normal)
        let normalBatch = await memoryAware.effectiveBatchSize
        #expect(normalBatch == 4)

        // Under critical pressure, should reduce
        await memoryAware.setPressure(.critical)
        let criticalBatch = await memoryAware.effectiveBatchSize
        #expect(criticalBatch == 2)  // min batch size
    }
}

// MARK: - VectorProducer Protocol Integration

@Suite("v0.2.2 VectorProducer Protocol")
struct V022VectorProducerTests {

    @Test("EmbeddingGenerator can be used as VectorProducer existential")
    func testAsVectorProducerExistential() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 256)
        let generator = EmbeddingGenerator(model: mockModel)

        // Use as existential type
        let producer: any VectorProducer = generator

        #expect(producer.dimensions == 256)
        #expect(producer.producesNormalizedVectors == true)

        let vectors = try await producer.produce(["test one", "test two"])
        #expect(vectors.count == 2)
        #expect(vectors[0].count == 256)
    }

    @Test("MemoryAwareGenerator can be used as VectorProducer")
    func testMemoryAwareAsVectorProducer() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: mockModel)
        let memoryAware = MemoryAwareGenerator(generator: generator)

        let producer: any VectorProducer = memoryAware

        #expect(producer.dimensions == 128)

        let vector = try await producer.produce("single text")
        #expect(vector.count == 128)
    }

    @Test("VectorProducerHints are accurate")
    func testVectorProducerHints() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 384)
        let generator = EmbeddingGenerator(
            model: mockModel,
            batchOptions: BatchOptions(maxBatchSize: 64)
        )

        let hints = await generator.hints

        #expect(hints.dimensions == 384)
        #expect(hints.isNormalized == true)
        #expect(hints.optimalBatchSize == 64)
        #expect(hints.maxBatchSize == 64)
    }

    @Test("Multiple VectorProducers can be used interchangeably")
    func testMultipleProducersInterchangeable() async throws {
        let mockModel1 = MockEmbeddingModel(dimensions: 128)
        let mockModel2 = MockEmbeddingModel(dimensions: 256)

        let generator1 = EmbeddingGenerator(model: mockModel1)
        let generator2 = EmbeddingGenerator(model: mockModel2)

        // Function that accepts any VectorProducer
        func embedWithProducer(_ producer: any VectorProducer, text: String) async throws -> [Float] {
            try await producer.produce(text)
        }

        let vec1 = try await embedWithProducer(generator1, text: "test")
        let vec2 = try await embedWithProducer(generator2, text: "test")

        #expect(vec1.count == 128)
        #expect(vec2.count == 256)
    }
}

// MARK: - OperationProgress Streaming Integration

@Suite("OperationProgress Streaming Integration")
struct OperationProgressStreamingTests {

    @Test("ProgressStream iteration completes correctly")
    func testProgressStreamIteration() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: mockModel)

        let texts = ["a", "b", "c", "d", "e"]

        var count = 0
        var lastProgress: OperationProgress?

        for try await (vector, progress) in await generator.generateWithProgress(texts) {
            #expect(vector.count == 64)
            count += 1
            lastProgress = progress
        }

        #expect(count == 5)
        #expect(lastProgress?.isComplete == true)
        #expect(lastProgress?.percentage == 100)
    }

    @Test("ProgressStream handles empty input")
    func testProgressStreamEmptyInput() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: mockModel)

        var count = 0
        for try await _ in await generator.generateWithProgress([]) {
            count += 1
        }

        #expect(count == 0)
    }

    @Test("ProgressStream ETA calculation is reasonable")
    func testProgressStreamETA() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: mockModel)

        let texts = (0..<10).map { "Text \($0)" }

        var midpointETA: TimeInterval?

        for try await (_, progress) in await generator.generateWithProgress(texts) {
            if progress.current == 5 {
                midpointETA = progress.estimatedTimeRemaining
            }
        }

        // At midpoint, should have some ETA (or nil if too fast to estimate)
        if let eta = midpointETA {
            #expect(eta >= 0, "ETA should be non-negative")
        }
    }

    @Test("ProgressStream can be cancelled early")
    func testProgressStreamCancellation() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: mockModel)

        let texts = (0..<100).map { "Text \($0)" }

        var count = 0
        for try await _ in await generator.generateWithProgress(texts) {
            count += 1
            if count >= 5 {
                break  // Early termination
            }
        }

        // Should have terminated early
        #expect(count == 5)
    }
}

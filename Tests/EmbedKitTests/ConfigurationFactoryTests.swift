import Testing
@testable import EmbedKit

@Suite("Configuration Factories")
struct ConfigurationFactoryTests {

    @Test("forSemanticSearch creates valid config")
    func semanticSearchConfig() {
        let config = EmbeddingConfiguration.forSemanticSearch()
        #expect(config.maxTokens == 512)
        #expect(config.includeSpecialTokens == true)
        #expect(config.truncationStrategy == .end)
        #expect(config.paddingStrategy == .batch)
        #expect(config.poolingStrategy == .mean)
        #expect(config.normalizeOutput == true)
    }

    @Test("forSemanticSearch with custom params")
    func semanticSearchCustom() {
        let config = EmbeddingConfiguration.forSemanticSearch(
            maxLength: 256,
            normalize: false
        )
        #expect(config.maxTokens == 256)
        #expect(config.normalizeOutput == false)
        #expect(config.paddingStrategy == .batch)
    }

    @Test("forRAG creates valid config")
    func ragConfig() {
        let config = EmbeddingConfiguration.forRAG(chunkSize: 512)
        #expect(config.maxTokens == 512)
        #expect(config.normalizeOutput == true)
        #expect(config.truncationStrategy == .end)
        #expect(config.paddingStrategy == .batch)
    }

    @Test("forRAG with default chunk size")
    func ragConfigDefault() {
        let config = EmbeddingConfiguration.forRAG()
        #expect(config.maxTokens == 256)
        #expect(config.includeSpecialTokens == true)
    }

    @Test("forClustering creates valid config")
    func clusteringConfig() {
        let config = EmbeddingConfiguration.forClustering()
        #expect(config.maxTokens == 128)
        #expect(config.normalizeOutput == true)
        #expect(config.poolingStrategy == .mean)
    }

    @Test("forClustering with custom length")
    func clusteringConfigCustom() {
        let config = EmbeddingConfiguration.forClustering(maxLength: 256)
        #expect(config.maxTokens == 256)
        #expect(config.paddingStrategy == .batch)
    }

    @Test("forSimilarity creates valid config")
    func similarityConfig() {
        let config = EmbeddingConfiguration.forSimilarity()
        #expect(config.maxTokens == 256)
        #expect(config.normalizeOutput == true)
        #expect(config.truncationStrategy == .end)
    }

    @Test("forSimilarity with custom length")
    func similarityConfigCustom() {
        let config = EmbeddingConfiguration.forSimilarity(maxLength: 512)
        #expect(config.maxTokens == 512)
        #expect(config.includeSpecialTokens == true)
    }

    @Test("forDocuments creates valid config")
    func documentsConfig() {
        let config = EmbeddingConfiguration.forDocuments()
        #expect(config.maxTokens == 2048)
        #expect(config.paddingStrategy == .none)
        #expect(config.normalizeOutput == true)
    }

    @Test("forDocuments with custom length")
    func documentsConfigCustom() {
        let config = EmbeddingConfiguration.forDocuments(maxLength: 1024)
        #expect(config.maxTokens == 1024)
        #expect(config.paddingStrategy == .none)
    }

    @Test("forShortText creates valid config")
    func shortTextConfig() {
        let config = EmbeddingConfiguration.forShortText()
        #expect(config.maxTokens == 64)
        #expect(config.paddingStrategy == .max)
        #expect(config.normalizeOutput == true)
    }

    @Test("forShortText with custom length")
    func shortTextConfigCustom() {
        let config = EmbeddingConfiguration.forShortText(maxLength: 32)
        #expect(config.maxTokens == 32)
        #expect(config.paddingStrategy == .max)
    }

    @Test("forMiniLM with different use cases")
    func miniLMConfigs() {
        let search = EmbeddingConfiguration.forMiniLM(useCase: .semanticSearch)
        let rag = EmbeddingConfiguration.forMiniLM(useCase: .rag)
        let cluster = EmbeddingConfiguration.forMiniLM(useCase: .clustering)
        let similarity = EmbeddingConfiguration.forMiniLM(useCase: .similarity)

        #expect(search.maxTokens == 256)
        #expect(rag.maxTokens == 256)
        #expect(cluster.maxTokens == 128)
        #expect(similarity.maxTokens == 256)

        // All should normalize
        #expect(search.normalizeOutput == true)
        #expect(rag.normalizeOutput == true)
        #expect(cluster.normalizeOutput == true)
        #expect(similarity.normalizeOutput == true)
    }

    @Test("forMiniLM default use case")
    func miniLMDefault() {
        let config = EmbeddingConfiguration.forMiniLM()
        #expect(config.maxTokens == 256)
        #expect(config.normalizeOutput == true)
    }

    @Test("forBERT with different use cases")
    func bertConfigs() {
        let search = EmbeddingConfiguration.forBERT(useCase: .semanticSearch)
        let rag = EmbeddingConfiguration.forBERT(useCase: .rag)
        let cluster = EmbeddingConfiguration.forBERT(useCase: .clustering)
        let similarity = EmbeddingConfiguration.forBERT(useCase: .similarity)

        #expect(search.maxTokens == 512)
        #expect(rag.maxTokens == 384)
        #expect(cluster.maxTokens == 256)
        #expect(similarity.maxTokens == 512)

        // All should normalize
        #expect(search.normalizeOutput == true)
        #expect(rag.normalizeOutput == true)
        #expect(cluster.normalizeOutput == true)
        #expect(similarity.normalizeOutput == true)
    }

    @Test("forBERT default use case")
    func bertDefault() {
        let config = EmbeddingConfiguration.forBERT()
        #expect(config.maxTokens == 512)
        #expect(config.normalizeOutput == true)
    }

    @Test("UseCase is CaseIterable")
    func useCaseIterable() {
        let cases = EmbeddingConfiguration.UseCase.allCases
        #expect(cases.count == 4)
        #expect(cases.contains(.semanticSearch))
        #expect(cases.contains(.rag))
        #expect(cases.contains(.clustering))
        #expect(cases.contains(.similarity))
    }

    @Test("UseCase is Sendable")
    func useCaseSendable() {
        // This test verifies the UseCase enum conforms to Sendable
        // by attempting to use it in a Sendable context
        let useCase: EmbeddingConfiguration.UseCase = .semanticSearch
        let config = EmbeddingConfiguration.forMiniLM(useCase: useCase)
        #expect(config.maxTokens > 0)
    }

    @Test("UseCase has String raw values")
    func useCaseRawValues() {
        #expect(EmbeddingConfiguration.UseCase.semanticSearch.rawValue == "semanticSearch")
        #expect(EmbeddingConfiguration.UseCase.rag.rawValue == "rag")
        #expect(EmbeddingConfiguration.UseCase.clustering.rawValue == "clustering")
        #expect(EmbeddingConfiguration.UseCase.similarity.rawValue == "similarity")
    }

    @Test("All factory configs have valid maxTokens")
    func allFactoriesHaveValidMaxTokens() {
        let configs: [EmbeddingConfiguration] = [
            .forSemanticSearch(),
            .forRAG(),
            .forClustering(),
            .forSimilarity(),
            .forDocuments(),
            .forShortText(),
            .forMiniLM(),
            .forBERT()
        ]

        for config in configs {
            #expect(config.maxTokens > 0)
        }
    }

    @Test("All factory configs include special tokens")
    func allFactoriesIncludeSpecialTokens() {
        let configs: [EmbeddingConfiguration] = [
            .forSemanticSearch(),
            .forRAG(),
            .forClustering(),
            .forSimilarity(),
            .forDocuments(),
            .forShortText(),
            .forMiniLM(),
            .forBERT()
        ]

        for config in configs {
            #expect(config.includeSpecialTokens == true)
        }
    }

    @Test("All factory configs use mean pooling")
    func allFactoriesUseMeanPooling() {
        let configs: [EmbeddingConfiguration] = [
            .forSemanticSearch(),
            .forRAG(),
            .forClustering(),
            .forSimilarity(),
            .forDocuments(),
            .forShortText(),
            .forMiniLM(),
            .forBERT()
        ]

        for config in configs {
            #expect(config.poolingStrategy == .mean)
        }
    }

    @Test("Different padding strategies are used appropriately")
    func paddingStrategies() {
        // Batch padding for most cases
        #expect(EmbeddingConfiguration.forSemanticSearch().paddingStrategy == .batch)
        #expect(EmbeddingConfiguration.forRAG().paddingStrategy == .batch)
        #expect(EmbeddingConfiguration.forClustering().paddingStrategy == .batch)
        #expect(EmbeddingConfiguration.forSimilarity().paddingStrategy == .batch)

        // No padding for documents
        #expect(EmbeddingConfiguration.forDocuments().paddingStrategy == .none)

        // Max padding for short text
        #expect(EmbeddingConfiguration.forShortText().paddingStrategy == .max)
    }

    @Test("Truncation strategy is consistent")
    func truncationStrategy() {
        let configs: [EmbeddingConfiguration] = [
            .forSemanticSearch(),
            .forRAG(),
            .forClustering(),
            .forSimilarity(),
            .forDocuments(),
            .forShortText(),
            .forMiniLM(),
            .forBERT()
        ]

        // All should use end truncation
        for config in configs {
            #expect(config.truncationStrategy == .end)
        }
    }
}

// MARK: - Pipeline Configuration Tests

@Suite("Pipeline Configuration")
struct PipelineConfigurationTests {

    // MARK: - Basic Construction

    @Test("Default pipeline configuration has sensible defaults")
    func defaultPipelineConfig() {
        let config = PipelineConfiguration()

        #expect(config.embedding.maxTokens == 512)
        #expect(config.batch.maxBatchSize == 32)
        #expect(config.cache == nil)
        #expect(config.memoryBudget == nil)
    }

    @Test("Pipeline configuration bundles all components")
    func pipelineConfigComponents() {
        let config = PipelineConfiguration(
            embedding: .forSemanticSearch(),
            batch: .highThroughput,
            compute: .gpuOptimized(),
            cache: .default,
            memoryBudget: 256 * 1024 * 1024
        )

        #expect(config.embedding.maxTokens == 512)
        #expect(config.batch.maxBatchSize == 64)
        #expect(config.cache != nil)
        #expect(config.memoryBudget == 256 * 1024 * 1024)
    }

    // MARK: - Compute Configuration

    @Test("Compute configuration default values")
    func computeConfigDefault() {
        let config = ComputeConfiguration.default

        #expect(config.useFusedKernels == true)
        #expect(config.adaptiveKernelSelection == true)
        #expect(config.maxResidentMemoryMB == 512)
    }

    @Test("GPU optimized compute configuration")
    func computeConfigGPUOptimized() {
        let config = ComputeConfiguration.gpuOptimized()

        #expect(config.useFusedKernels == true)
        #expect(config.maxResidentMemoryMB == 1024)
    }

    @Test("Memory efficient compute configuration")
    func computeConfigMemoryEfficient() {
        let config = ComputeConfiguration.memoryEfficient()

        #expect(config.maxResidentMemoryMB == 128)
    }

    // MARK: - Convenience Methods

    @Test("withCache adds caching")
    func withCacheMethod() {
        let config = PipelineConfiguration()
        #expect(config.cache == nil)

        let withCache = config.withCache()
        #expect(withCache.cache != nil)
        #expect(withCache.cache?.maxEntries == 100_000)
    }

    @Test("withMemoryBudget sets budget")
    func withMemoryBudgetMethod() {
        let config = PipelineConfiguration()
        #expect(config.memoryBudget == nil)

        let withBudget = config.withMemoryBudget(mb: 256)
        #expect(withBudget.memoryBudget == 256 * 1024 * 1024)
    }

    @Test("toAdaptiveBatcherConfig creates valid config")
    func toAdaptiveBatcherConfigMethod() {
        let pipeline = ConfigurationFactory.highThroughput()
        let batcherConfig = pipeline.toAdaptiveBatcherConfig()

        #expect(batcherConfig.maxBatchSize == 128)
        #expect(batcherConfig.autoFlush == true)
        #expect(batcherConfig.batchOptions.sortByLength == true)
    }

    // MARK: - Description

    @Test("PipelineConfiguration has description")
    func pipelineConfigDescription() {
        let config = ConfigurationFactory.highThroughput()
        let desc = config.description

        #expect(desc.contains("embedding"))
        #expect(desc.contains("batch"))
        #expect(desc.contains("compute"))
    }

    @Test("ComputeConfiguration has description")
    func computeConfigDescription() {
        let config = ComputeConfiguration.gpuOptimized()
        let desc = config.description

        #expect(desc.contains("fused"))
    }
}

// MARK: - Configuration Factory Tests

@Suite("Configuration Factory Presets")
struct ConfigurationFactoryPresetsTests {

    // MARK: - Performance-Oriented Presets

    @Test("default factory creates balanced config")
    func defaultFactory() {
        let config = ConfigurationFactory.default()

        #expect(config.embedding.maxTokens == 512)
        #expect(config.batch.maxBatchSize == 32)
        #expect(config.cache == nil)
    }

    @Test("highThroughput optimizes for batch processing")
    func highThroughputFactory() {
        let config = ConfigurationFactory.highThroughput()

        // Large batches
        #expect(config.batch.maxBatchSize == 128)
        #expect(config.batch.dynamicBatching == true)
        #expect(config.batch.sortByLength == true)

        // Aggressive GPU
        #expect(config.embedding.minElementsForGPU == 2048)

        // Token limits
        #expect(config.batch.maxBatchTokens == 16384)
    }

    @Test("lowLatency optimizes for single items")
    func lowLatencyFactory() {
        let config = ConfigurationFactory.lowLatency()

        // Small batches, no waiting
        #expect(config.batch.maxBatchSize == 8)
        #expect(config.batch.dynamicBatching == false)
        #expect(config.batch.sortByLength == false)

        // Conservative GPU (CPU often faster for small)
        #expect(config.embedding.minElementsForGPU == 16384)

        // Short timeout
        #expect(config.batch.timeout == 5.0)
    }

    @Test("gpuOptimized maximizes GPU utilization")
    func gpuOptimizedFactory() {
        let config = ConfigurationFactory.gpuOptimized()

        // GPU settings
        #expect(config.embedding.inferenceDevice == .gpu)
        #expect(config.embedding.minElementsForGPU == 1024)

        // Compute settings
        #expect(config.compute.useFusedKernels == true)
        #expect(config.compute.maxResidentMemoryMB == 1024)

        // Large token batches
        #expect(config.batch.maxBatchTokens == 32768)
    }

    // MARK: - Resource-Oriented Presets

    @Test("memoryEfficient reduces memory usage")
    func memoryEfficientFactory() {
        let config = ConfigurationFactory.memoryEfficient()

        // Small batches
        #expect(config.batch.maxBatchSize == 8)
        #expect(config.batch.maxBatchTokens == 1024)

        // Shorter sequences
        #expect(config.embedding.maxTokens == 256)

        // No padding overhead
        #expect(config.embedding.paddingStrategy == .none)

        // Low GPU memory
        #expect(config.compute.maxResidentMemoryMB == 128)

        // No cache
        #expect(config.cache == nil)
    }

    @Test("memoryEfficient accepts custom budget")
    func memoryEfficientCustomBudget() {
        let config = ConfigurationFactory.memoryEfficient(memoryBudgetMB: 64)

        #expect(config.memoryBudget == 64 * 1024 * 1024)
    }

    @Test("batteryEfficient uses ANE and conservative settings")
    func batteryEfficientFactory() {
        let config = ConfigurationFactory.batteryEfficient()

        // ANE for power efficiency
        #expect(config.embedding.inferenceDevice == .ane)

        // Conservative GPU threshold
        #expect(config.embedding.minElementsForGPU == 32768)

        // Moderate batches
        #expect(config.batch.maxBatchSize == 16)

        // Fewer concurrent threads
        #expect(config.batch.tokenizationConcurrency == 2)
    }

    // MARK: - Use-Case Oriented Presets

    @Test("forSemanticSearch creates search config")
    func forSemanticSearchFactory() {
        let config = ConfigurationFactory.forSemanticSearch()

        #expect(config.embedding.maxTokens == 512)
        #expect(config.embedding.normalizeOutput == true)
        #expect(config.batch.maxBatchSize == 32)
        #expect(config.cache == nil)
    }

    @Test("forSemanticSearch with caching")
    func forSemanticSearchWithCache() {
        let config = ConfigurationFactory.forSemanticSearch(enableCache: true)

        #expect(config.cache != nil)
    }

    @Test("forRAG creates RAG pipeline config")
    func forRAGFactory() {
        let config = ConfigurationFactory.forRAG()

        #expect(config.embedding.maxTokens == 256)  // Default chunk size
        #expect(config.batch.maxBatchSize == 64)
        #expect(config.cache != nil)  // Cache enabled by default
        #expect(config.cache?.maxEntries == 50_000)
    }

    @Test("forRAG with custom chunk size")
    func forRAGCustomChunkSize() {
        let config = ConfigurationFactory.forRAG(chunkSize: 384)

        #expect(config.embedding.maxTokens == 384)
        #expect(config.batch.maxBatchTokens == 384 * 64)
    }

    @Test("forRealTimeSearch optimizes for interactivity")
    func forRealTimeSearchFactory() {
        let config = ConfigurationFactory.forRealTimeSearch()

        // Short queries
        #expect(config.embedding.maxTokens == 128)

        // Very small batches, no waiting
        #expect(config.batch.maxBatchSize == 4)
        #expect(config.batch.dynamicBatching == false)

        // Short timeout
        #expect(config.batch.timeout == 2.0)

        // Cache with TTL
        #expect(config.cache != nil)
        #expect(config.cache?.ttlSeconds == 3600)
        #expect(config.cache?.enableSemanticDedup == true)
    }

    @Test("forBatchIndexing optimizes for bulk processing")
    func forBatchIndexingFactory() {
        let config = ConfigurationFactory.forBatchIndexing()

        // Very large batches
        #expect(config.batch.maxBatchSize == 128)
        #expect(config.batch.minBatchSize == 32)
        #expect(config.batch.maxBatchTokens == 65536)

        // Long timeout
        #expect(config.batch.timeout == 120.0)

        // Large cache
        #expect(config.cache != nil)
        #expect(config.cache?.maxEntries == 1_000_000)
    }

    // MARK: - Model-Specific Presets

    @Test("forMiniLM optimizes for 384-dim models")
    func forMiniLMFactory() {
        let config = ConfigurationFactory.forMiniLM()

        // Larger batches (smaller model)
        #expect(config.batch.maxBatchSize == 64)
    }

    @Test("forMiniLM with different use cases")
    func forMiniLMUseCases() {
        let search = ConfigurationFactory.forMiniLM(useCase: .semanticSearch)
        let rag = ConfigurationFactory.forMiniLM(useCase: .rag)
        let cluster = ConfigurationFactory.forMiniLM(useCase: .clustering)

        #expect(search.embedding.maxTokens == 256)
        #expect(rag.embedding.maxTokens == 256)
        #expect(cluster.embedding.maxTokens == 128)
    }

    @Test("forBERT optimizes for 768-dim models")
    func forBERTFactory() {
        let config = ConfigurationFactory.forBERT()

        // Smaller batches (larger model)
        #expect(config.batch.maxBatchSize == 32)
        #expect(config.batch.maxBatchTokens == 8192)
        #expect(config.compute.maxResidentMemoryMB == 512)
    }

    @Test("forBERT with different use cases")
    func forBERTUseCases() {
        let search = ConfigurationFactory.forBERT(useCase: .semanticSearch)
        let rag = ConfigurationFactory.forBERT(useCase: .rag)
        let cluster = ConfigurationFactory.forBERT(useCase: .clustering)

        #expect(search.embedding.maxTokens == 512)
        #expect(rag.embedding.maxTokens == 384)
        #expect(cluster.embedding.maxTokens == 256)
    }

    @Test("forLargeModel handles high-dimensional embeddings")
    func forLargeModelFactory() {
        let config = ConfigurationFactory.forLargeModel(dimensions: 1024)

        #expect(config.compute.maxResidentMemoryMB == 1024)

        // Batch size adjusted for dimensions
        #expect(config.batch.maxBatchSize <= 24)
    }

    @Test("forLargeModel with caching")
    func forLargeModelWithCache() {
        let config = ConfigurationFactory.forLargeModel(enableCache: true)

        #expect(config.cache != nil)
        #expect(config.cache?.maxEntries == 25_000)  // Fewer entries
    }

    // MARK: - Invariants

    @Test("All factory presets produce valid configurations")
    func allPresetsValid() {
        let presets: [PipelineConfiguration] = [
            ConfigurationFactory.default(),
            ConfigurationFactory.highThroughput(),
            ConfigurationFactory.lowLatency(),
            ConfigurationFactory.gpuOptimized(),
            ConfigurationFactory.memoryEfficient(),
            ConfigurationFactory.batteryEfficient(),
            ConfigurationFactory.forSemanticSearch(),
            ConfigurationFactory.forRAG(),
            ConfigurationFactory.forRealTimeSearch(),
            ConfigurationFactory.forBatchIndexing(),
            ConfigurationFactory.forMiniLM(),
            ConfigurationFactory.forBERT(),
            ConfigurationFactory.forLargeModel()
        ]

        for config in presets {
            // All have valid embedding config
            #expect(config.embedding.maxTokens > 0)

            // All have valid batch config
            #expect(config.batch.maxBatchSize > 0)

            // All have valid compute config
            #expect(config.compute.maxResidentMemoryMB >= 0)
        }
    }

    @Test("High throughput has larger batches than low latency")
    func throughputVsLatencyBatchSize() {
        let highThroughput = ConfigurationFactory.highThroughput()
        let lowLatency = ConfigurationFactory.lowLatency()

        #expect(highThroughput.batch.maxBatchSize > lowLatency.batch.maxBatchSize)
    }

    @Test("GPU optimized has lower GPU element threshold than memory efficient")
    func gpuVsMemoryThresholds() {
        let gpuOptimized = ConfigurationFactory.gpuOptimized()
        let memoryEfficient = ConfigurationFactory.memoryEfficient()

        #expect(gpuOptimized.embedding.minElementsForGPU < memoryEfficient.embedding.minElementsForGPU)
    }

    @Test("Batch indexing has largest batch sizes")
    func batchIndexingLargestBatches() {
        let presets: [PipelineConfiguration] = [
            ConfigurationFactory.default(),
            ConfigurationFactory.highThroughput(),
            ConfigurationFactory.forSemanticSearch(),
            ConfigurationFactory.forRAG()
        ]

        let batchIndexing = ConfigurationFactory.forBatchIndexing()

        for config in presets {
            #expect(batchIndexing.batch.maxBatchSize >= config.batch.maxBatchSize)
        }
    }
}

// MARK: - Sendability Tests

@Suite("Configuration Sendability")
struct ConfigurationSendabilityTests {

    @Test("PipelineConfiguration is Sendable")
    func pipelineConfigSendable() async {
        let config = ConfigurationFactory.highThroughput()

        // Pass across isolation boundary
        await Task {
            #expect(config.batch.maxBatchSize == 128)
        }.value
    }

    @Test("ComputeConfiguration is Sendable")
    func computeConfigSendable() async {
        let config = ComputeConfiguration.gpuOptimized()

        await Task {
            #expect(config.useFusedKernels == true)
        }.value
    }
}

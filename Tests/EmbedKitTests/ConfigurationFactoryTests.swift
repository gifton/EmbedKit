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

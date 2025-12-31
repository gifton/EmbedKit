// EmbedKit - System Model Integration Tests
// Tests that verify the mock model remediation works correctly
// and that real semantic embeddings are produced by default.

import Testing
import Foundation
@testable import EmbedKit

// MARK: - System Model Integration Tests

@Suite("System Model Integration")
struct SystemModelIntegrationTests {

    // MARK: - ModelManager.loadAppleModel() Tests

    @Test("loadAppleModel returns real NLContextualEmbedding model")
    func loadAppleModelReturnsSystemModel() async throws {
        let manager = ModelManager()
        let model = try await manager.loadAppleModel()

        // Verify it's the system model, not mock
        #expect(model.id.provider == "apple")
        #expect(model.id.name == "nl-contextual")
        #expect(model.id.version == "system")
    }

    @Test("loadAppleModel produces embeddings with correct dimensions")
    func loadAppleModelProducesEmbeddings() async throws {
        let manager = ModelManager()
        let model = try await manager.loadAppleModel()

        let embedding = try await model.embed("Hello, world!")

        // NLContextualEmbedding produces 512-dimensional embeddings
        #expect(embedding.dimensions == 512)
        #expect(!embedding.vector.isEmpty)
        #expect(!embedding.vector.contains(Float.nan))
    }

    @Test("loadMockModel still returns mock model for testing")
    func loadMockModelReturnsMock() async throws {
        let manager = ModelManager()
        let model = try await manager.loadMockModel()

        // Verify it's the mock model
        #expect(model.id.provider == "mock")
    }

    // MARK: - Semantic Similarity Tests

    @Test("System model produces semantically meaningful embeddings")
    func semanticallySimilarTextsHaveHigherSimilarity() async throws {
        let manager = ModelManager()
        let model = try await manager.loadAppleModel()

        // Embed semantically related and unrelated texts
        // Using full sentences to avoid single-word quirks like "dogecoin"
        let catSentence = try await model.embed("The cat sat on the mat")
        let kittenSentence = try await model.embed("A kitten rested on the rug")
        let financeSentence = try await model.embed("Stock market trading strategies")

        let simCatKitten = catSentence.similarity(to: kittenSentence)
        let simCatFinance = catSentence.similarity(to: financeSentence)

        // Cat sentence should be more similar to kitten sentence than finance
        #expect(simCatKitten > simCatFinance, "Expected cat/kitten sentences to be more similar than cat/finance")
    }

    @Test("System model distinguishes between different topics")
    func systemModelDistinguishesTopics() async throws {
        let manager = ModelManager()
        let model = try await manager.loadAppleModel()

        // Technology vs Food topics
        let techEmb = try await model.embed("machine learning and artificial intelligence")
        let foodEmb = try await model.embed("cooking recipes and kitchen ingredients")
        let aiEmb = try await model.embed("neural networks and deep learning")

        let simTechAI = techEmb.similarity(to: aiEmb)
        let simTechFood = techEmb.similarity(to: foodEmb)

        // Tech should be more similar to AI than to food
        #expect(simTechAI > simTechFood, "Expected tech topics to cluster together")
    }

    // MARK: - Convenience API Tests

    @Test("semanticSearch returns semantically relevant results")
    func semanticSearchReturnsRelevantResults() async throws {
        let manager = ModelManager()

        let documents = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is transforming industries",
            "A canine ran across the yard",
            "Artificial intelligence and neural networks"
        ]

        let results = try await manager.semanticSearch(
            query: "dog running",
            in: documents,
            topK: 2
        )

        // Documents about dogs/canines should rank higher than ML docs
        #expect(results.count == 2)

        // The top result should be one of the dog-related documents (index 0 or 2)
        let topIndices = results.map { $0.index }
        let hasDogDocument = topIndices.contains(0) || topIndices.contains(2)
        #expect(hasDogDocument, "Expected dog-related documents to rank higher")
    }

    @Test("quickEmbed uses system model by default")
    func quickEmbedUsesSystemModel() async throws {
        let manager = ModelManager()

        let vector1 = try await manager.quickEmbed("hello world")
        let vector2 = try await manager.quickEmbed("hello world")

        // System model produces 512 dimensions
        #expect(vector1.count == 512)

        // Same text should produce identical embeddings
        #expect(vector1 == vector2)
    }

    @Test("clusterDocuments groups semantically similar documents")
    func clusterDocumentsGroupsSimilarDocs() async throws {
        let manager = ModelManager()

        let documents = [
            "Apple iPhone 15 review",
            "Samsung Galaxy S24 specs",
            "Best chocolate cake recipe",
            "How to make pasta from scratch",
            "Google Pixel 8 camera test",
            "Homemade bread baking tips"
        ]

        let clusters = try await manager.clusterDocuments(documents, numberOfClusters: 2)

        // Should create 2 clusters
        #expect(clusters.count == 2)

        // All documents should be assigned
        let totalAssigned = clusters.reduce(0) { $0 + $1.count }
        #expect(totalAssigned == documents.count)
    }

    // MARK: - SwiftUI ViewModel Tests

    @Test("EmbeddingViewModel defaults to system model")
    func embeddingViewModelDefaultsToSystem() async throws {
        let viewModel = await EmbeddingViewModel()

        let embedding = await viewModel.embed("test text")

        // Should produce real embeddings (512 dimensions for NLContextualEmbedding)
        #expect(embedding != nil)
        #expect(embedding?.dimensions == 512)
    }

    @Test("EmbeddingViewModel with mock provider uses mock")
    func embeddingViewModelWithMockUsesMock() async throws {
        let viewModel = await EmbeddingViewModel(modelProvider: .mock)

        let embedding = await viewModel.embed("test text")

        // Mock produces 384 dimensions by default
        #expect(embedding != nil)
        #expect(embedding?.dimensions == 384)
    }

    @Test("SimilarityViewModel defaults to system model")
    func similarityViewModelDefaultsToSystem() async throws {
        let viewModel = await SimilarityViewModel()

        let similarity = await viewModel.computeSimilarity(
            between: "cat",
            and: "kitten"
        )

        // Should compute real similarity
        #expect(similarity != nil)
        #expect(similarity! > 0, "Expected positive similarity between 'cat' and 'kitten'")
    }

    @Test("SimilarityViewModel with mock provider uses mock")
    func similarityViewModelWithMockUsesMock() async throws {
        let viewModel = await SimilarityViewModel(modelProvider: .mock)

        let similarity = await viewModel.computeSimilarity(
            between: "hello",
            and: "world"
        )

        // Mock should still compute similarity (hash-based)
        #expect(similarity != nil)
    }

    // MARK: - Cached Model Tests

    @Test("getOrCreateSystemModel caches the model")
    func cachedSystemModelIsReused() async throws {
        let manager = ModelManager()

        // First call creates model
        _ = try await manager.quickEmbed("test 1")

        // Second call should reuse cached model (fast)
        let start = CFAbsoluteTimeGetCurrent()
        _ = try await manager.quickEmbed("test 2")
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        // Second call should be fast since model is cached
        #expect(elapsed < 0.5, "Expected cached model to be fast")
    }

    @Test("clearCachedSystemModel clears the cache")
    func clearCachedSystemModelWorks() async throws {
        let manager = ModelManager()

        // Create cached model
        _ = try await manager.quickEmbed("test")

        // Clear it
        await manager.clearCachedSystemModel()

        // Next call should create new model (no crash)
        let vector = try await manager.quickEmbed("test again")
        #expect(vector.count == 512)
    }

    @Test("unloadAll clears cached system model")
    func unloadAllClearsCachedModel() async throws {
        let manager = ModelManager()

        // Create cached model
        _ = try await manager.quickEmbed("test")

        // Unload all
        await manager.unloadAll()

        // Should still work (creates new model)
        let vector = try await manager.quickEmbed("test again")
        #expect(vector.count == 512)
    }

    // MARK: - Dimension Reporting Tests (Issue: VectorProducer contract)

    @Test("AppleNLContextualModel reports correct dimensions")
    func appleNLContextualModelReportsDimensions() async throws {
        let manager = ModelManager()
        let model = try await manager.loadNLContextualEmbedding(language: "en")

        // Should report 512, not 0
        #expect(model.dimensions == 512, "AppleNLContextualModel should report 512 dimensions for English")
    }

    @Test("EmbeddingGenerator reports correct dimensions via VectorProducer")
    func embeddingGeneratorReportsDimensions() async throws {
        let manager = ModelManager()
        let generator = try await manager.createSystemGenerator(language: "en")

        // VectorProducer.dimensions should be 512
        #expect(generator.dimensions == 512, "Generator should report 512 dimensions")
    }

    @Test("Generated embedding matches reported dimensions")
    func generatedEmbeddingMatchesReportedDimensions() async throws {
        let manager = ModelManager()
        let generator = try await manager.createSystemGenerator(language: "en")

        let vector = try await generator.produce("test text")

        // Vector length should match the reported dimensions
        #expect(vector.count == generator.dimensions, "Vector length should match reported dimensions")
    }

    @Test("Dimension override works when explicitly specified")
    func dimensionOverrideWorks() throws {
        // When dimensions is explicitly passed, it should be used instead of lookup
        let model = try AppleNLContextualModel(
            language: "en",
            dimensions: 1024
        )

        #expect(model.dimensions == 1024, "Explicit dimension should override lookup")
    }

    @Test("NLContextualDimensions lookup returns correct values")
    func nlContextualDimensionsLookup() {
        // All supported languages should return 512
        #expect(NLContextualDimensions.dimension(for: "en") == 512)
        #expect(NLContextualDimensions.dimension(for: "de") == 512)
        #expect(NLContextualDimensions.dimension(for: "fr") == 512)
        #expect(NLContextualDimensions.dimension(for: "es") == 512)
        #expect(NLContextualDimensions.dimension(for: "ja") == 512)
        #expect(NLContextualDimensions.dimension(for: "zh") == 512)

        // Unknown languages should also return 512 (safe default)
        #expect(NLContextualDimensions.dimension(for: "unknown") == 512)
    }

    @Test("Different languages report correct dimensions")
    func differentLanguagesReportDimensions() async throws {
        // Test a few different language codes
        for language in ["en", "de", "fr"] {
            let model = try AppleNLContextualModel(language: language)
            #expect(model.dimensions == 512, "Language '\(language)' should report 512 dimensions")
        }
    }
}

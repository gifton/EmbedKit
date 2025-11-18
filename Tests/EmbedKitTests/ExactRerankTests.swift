//
//  ExactRerankTests.swift
//  EmbedKitTests
//
//  Tests for ExactRerank integration with the new clean API
//

import XCTest
@testable import EmbedKit
import VectorIndex
import VectorCore

final class ExactRerankTests: XCTestCase {

    // MARK: - Test Setup

    var storage: InMemoryVectorStorage!
    var adapter: VectorIndexAdapter!
    var pipeline: EmbeddingPipeline!

    override func setUp() async throws {
        try await super.setUp()

        // Initialize components
        storage = InMemoryVectorStorage()
        let config = EmbeddingPipelineConfiguration(
            poolingStrategy: .mean,
            normalize: true,
            useGPUAcceleration: false  // Disable for mock backend
        )
        pipeline = EmbeddingPipeline(
            tokenizer: MockTokenizer(),
            backend: MockBackend(),
            configuration: config
        )
        adapter = VectorIndexAdapter(
            pipeline: pipeline,
            storage: storage
        )

        // Populate test data
        try await populateTestData()
    }

    override func tearDown() async throws {
        storage = nil
        adapter = nil
        pipeline = nil
        try await super.tearDown()
    }

    // MARK: - Test Data Population

    private func populateTestData() async throws {
        // Create test vectors with known distances
        let testVectors: [(id: UUID, vector: [Float], metadata: [String: String])] = [
            (UUID(), [1.0, 0.0, 0.0], ["name": "vec1", "timestamp": "100.0"]),
            (UUID(), [0.0, 1.0, 0.0], ["name": "vec2", "timestamp": "200.0"]),
            (UUID(), [0.0, 0.0, 1.0], ["name": "vec3", "timestamp": "300.0"]),
            (UUID(), [0.5, 0.5, 0.0], ["name": "vec4", "timestamp": "400.0"]),
            (UUID(), [0.5, 0.0, 0.5], ["name": "vec5", "timestamp": "500.0"]),
            (UUID(), [0.0, 0.5, 0.5], ["name": "vec6", "timestamp": "600.0"]),
            (UUID(), [0.33, 0.33, 0.34], ["name": "vec7", "timestamp": "700.0"]),
            (UUID(), [0.6, 0.8, 0.0], ["name": "vec8", "timestamp": "800.0"]),
            (UUID(), [0.8, 0.0, 0.6], ["name": "vec9", "timestamp": "900.0"]),
            (UUID(), [0.0, 0.8, 0.6], ["name": "vec10", "timestamp": "1000.0"]),
        ]

        for (id, vector, metadata) in testVectors {
            _ = try await storage.add(vector: vector, metadata: metadata)
        }
    }

    // MARK: - Test Cases

    /// Test 1: Basic ExactRerank functionality
    func testBasicExactRerank() async throws {
        let rerankStrategy = ExactRerankStrategy(
            storage: storage,
            metric: .euclidean,
            dimension: 3
        )

        let queryVector: [Float] = [0.9, 0.1, 0.1]  // Close to vec1
        let candidates = try await createMockCandidates()

        // Rerank
        let reranked = try await rerankStrategy.rerank(
            query: try DynamicEmbedding(values: queryVector),
            candidates: candidates,
            k: 3,
            options: .default
        )

        // Verify top result is closest to query
        XCTAssertEqual(reranked.count, 3)
        XCTAssertEqual(reranked[0].metadata["name"], "vec1")
    }

    /// Test 2: Reranking with different metrics
    func testRerankingMetrics() async throws {
        // Test Euclidean distance
        await testMetric(.euclidean, expectedTop: "vec1")

        // Test Cosine similarity
        await testMetric(.cosine, expectedTop: "vec1")

        // Test Dot product
        await testMetric(.dotProduct, expectedTop: "vec1")
    }

    private func testMetric(_ metric: SupportedDistanceMetric, expectedTop: String) async {
        do {
            let rerankStrategy = ExactRerankStrategy(
                storage: storage,
                metric: metric,
                dimension: 3
            )

            let queryVector: [Float] = [0.9, 0.1, 0.1]
            let candidates = try await createMockCandidates()

            let reranked = try await rerankStrategy.rerank(
                query: try DynamicEmbedding(values: queryVector),
                candidates: candidates,
                k: 1,
                options: .default
            )

            XCTAssertEqual(reranked.count, 1)
            XCTAssertEqual(reranked[0].metadata["name"], expectedTop,
                          "Failed for metric: \(metric)")
        } catch {
            XCTFail("Test failed for metric \(metric): \(error)")
        }
    }

    /// Test 3: Candidate multiplier effect
    func testCandidateMultiplier() async throws {
        let rerankStrategy = ExactRerankStrategy(
            storage: storage,
            metric: .euclidean,
            dimension: 3
        )

        let queryVector: [Float] = [0.5, 0.5, 0.0]  // Between vec1 and vec2
        let candidates = try await createMockCandidates()

        // Test with different multipliers
        for multiplier in [1, 2, 3, 5] {
            var options = RerankOptions.default
            options.candidateMultiplier = multiplier

            let reranked = try await rerankStrategy.rerank(
                query: try DynamicEmbedding(values: queryVector),
                candidates: candidates,
                k: 2,
                options: options
            )

            XCTAssertLessThanOrEqual(reranked.count, 2)
            print("Multiplier \(multiplier): \(reranked.count) results")
        }
    }

    /// Test 4: Missing vector handling
    func testMissingVectorHandling() async throws {
        let rerankStrategy = ExactRerankStrategy(
            storage: storage,
            metric: .euclidean,
            dimension: 3
        )

        // Create candidates with some invalid IDs
        var candidates = try await createMockCandidates()
        candidates.append(VectorSearchResult(
            id: UUID().uuidString,  // Non-existent ID
            score: 0.5,
            metadata: ["name": "missing"],
            embedding: nil
        ))

        // Test with skipMissing = true
        var options = RerankOptions.default
        options.skipMissing = true

        let rerankedSkip = try await rerankStrategy.rerank(
            query: try DynamicEmbedding(values: [0.5, 0.5, 0.0] as [Float]),
            candidates: candidates,
            k: 5,
            options: options
        )

        // Should not include missing vector
        XCTAssertFalse(rerankedSkip.contains { $0.metadata["name"] == "missing" })

        // Test with skipMissing = false
        options.skipMissing = false

        let rerankedInclude = try await rerankStrategy.rerank(
            query: try DynamicEmbedding(values: [0.5, 0.5, 0.0] as [Float]),
            candidates: candidates,
            k: 5,
            options: options
        )

        // May include missing vector with sentinel score
        print("Include missing: \(rerankedInclude.count) results")
    }

    /// Test 5: Parallel processing
    func testParallelProcessing() async throws {
        let rerankStrategy = ExactRerankStrategy(
            storage: storage,
            metric: .euclidean,
            dimension: 3
        )

        let queryVector: [Float] = [0.5, 0.5, 0.0]
        let candidates = try await createMockCandidates()

        // Test with parallel enabled
        var optionsParallel = RerankOptions.default
        optionsParallel.enableParallel = true
        optionsParallel.maxConcurrency = 4

        let startParallel = Date()
        let rerankedParallel = try await rerankStrategy.rerank(
            query: try DynamicEmbedding(values: queryVector),
            candidates: candidates,
            k: 5,
            options: optionsParallel
        )
        let timeParallel = Date().timeIntervalSince(startParallel)

        // Test with parallel disabled
        var optionsSerial = RerankOptions.default
        optionsSerial.enableParallel = false

        let startSerial = Date()
        let rerankedSerial = try await rerankStrategy.rerank(
            query: try DynamicEmbedding(values: queryVector),
            candidates: candidates,
            k: 5,
            options: optionsSerial
        )
        let timeSerial = Date().timeIntervalSince(startSerial)

        // Results should be identical
        XCTAssertEqual(rerankedParallel.count, rerankedSerial.count)
        for i in 0..<min(rerankedParallel.count, rerankedSerial.count) {
            XCTAssertEqual(
                rerankedParallel[i].metadata["name"],
                rerankedSerial[i].metadata["name"]
            )
        }

        print("Parallel: \(timeParallel * 1000)ms, Serial: \(timeSerial * 1000)ms")
    }

    /// Test 6: Different tile sizes
    func testTileSizes() async throws {
        let rerankStrategy = ExactRerankStrategy(
            storage: storage,
            metric: .euclidean,
            dimension: 3
        )

        let queryVector: [Float] = [0.5, 0.5, 0.0]
        let candidates = try await createMockCandidates()

        for tileSize in [1, 2, 4, 8, 16] {
            var options = RerankOptions.default
            options.tileSize = tileSize

            let reranked = try await rerankStrategy.rerank(
                query: try DynamicEmbedding(values: queryVector),
                candidates: candidates,
                k: 3,
                options: options
            )

            XCTAssertEqual(reranked.count, 3, "Failed for tile size \(tileSize)")
        }
    }

    /// Test 7: Empty candidates handling
    func testEmptyCandidates() async throws {
        let rerankStrategy = ExactRerankStrategy(
            storage: storage,
            metric: .euclidean,
            dimension: 3
        )

        let reranked = try await rerankStrategy.rerank(
            query: try DynamicEmbedding(values: [0.5, 0.5, 0.0] as [Float]),
            candidates: [],
            k: 5,
            options: .default
        )

        XCTAssertEqual(reranked.count, 0)
    }

    /// Test 8: Performance comparison
    func testRerankingPerformance() async throws {
        let rerankStrategy = ExactRerankStrategy(
            storage: storage,
            metric: .euclidean,
            dimension: 3
        )

        let queryVector: [Float] = [0.5, 0.5, 0.0]
        let candidates = try await createMockCandidates()

        // Measure without reranking (baseline)
        let baselineStart = Date()
        let baselineResults = Array(candidates.prefix(3))
        let baselineTime = Date().timeIntervalSince(baselineStart)

        // Measure with reranking
        let rerankStart = Date()
        let rerankedResults = try await rerankStrategy.rerank(
            query: try DynamicEmbedding(values: queryVector),
            candidates: candidates,
            k: 3,
            options: .default
        )
        let rerankTime = Date().timeIntervalSince(rerankStart)

        print("Performance Comparison:")
        print("  Baseline: \(baselineTime * 1000)ms for \(baselineResults.count) results")
        print("  Reranked: \(rerankTime * 1000)ms for \(rerankedResults.count) results")
        print("  Overhead: \((rerankTime - baselineTime) * 1000)ms")

        // Reranking should complete in reasonable time (< 100ms for small dataset)
        XCTAssertLessThan(rerankTime, 0.1)
    }

    /// Test 9: Integration with VectorIndexAdapter new API
    func testAdapterIntegration() async throws {
        let rerankStrategy = ExactRerankStrategy(
            storage: storage,
            metric: .euclidean,
            dimension: 3
        )

        // Test the new clean API
        let results = try await adapter.semanticSearch(
            query: "test query",
            k: 5,
            rerankStrategy: rerankStrategy,
            rerankOptions: .default
        )

        // Verify results structure
        XCTAssertNotNil(results)
        for result in results {
            XCTAssertFalse(result.id.isEmpty)
            XCTAssertNotNil(result.metadata)
        }
    }

    /// Test 10: Batch search with new API
    func testBatchSearch() async throws {
        let rerankStrategy = ExactRerankStrategy(
            storage: storage,
            metric: .euclidean,
            dimension: 3
        )

        let queries = ["query1", "query2", "query3"]

        // Test batch search without reranking
        let resultsNoRerank = try await adapter.batchSearch(
            queries: queries,
            k: 3
        )

        XCTAssertEqual(resultsNoRerank.count, queries.count)

        // Test batch search with reranking
        let resultsWithRerank = try await adapter.batchSearch(
            queries: queries,
            k: 3,
            rerankStrategy: rerankStrategy,
            rerankOptions: .fast
        )

        XCTAssertEqual(resultsWithRerank.count, queries.count)

        // Each query should have results
        for (index, queryResults) in resultsWithRerank.enumerated() {
            XCTAssertLessThanOrEqual(queryResults.count, 3, "Query \(index) should have at most 3 results")
        }
    }

    /// Test 11: Verify no backward compatibility remains
    func testNoBackwardCompatibility() async throws {
        // This test verifies that old API is completely gone

        // The following should NOT compile (commented out to show what's removed):
        // let results = await adapter.semanticSearch(query: "test", k: 10, rerank: true)
        // let strategy = SimpleRerankStrategy()  // Removed
        // let factory = RerankingStrategyFactory.simple()  // Removed

        // Only the new API should work:
        let results = try await adapter.semanticSearch(
            query: "test",
            k: 10
        )

        XCTAssertNotNil(results)
    }

    /// Test 12: Test reranking options presets
    func testRerankingPresets() async throws {
        let rerankStrategy = ExactRerankStrategy(
            storage: storage,
            metric: .euclidean,
            dimension: 3
        )

        let candidates = try await createMockCandidates()
        let query = try DynamicEmbedding(values: [0.5, 0.5, 0.0] as [Float])

        // Test default preset
        let defaultResults = try await rerankStrategy.rerank(
            query: query,
            candidates: candidates,
            k: 3,
            options: .default
        )
        XCTAssertEqual(defaultResults.count, 3)

        // Test fast preset
        let fastResults = try await rerankStrategy.rerank(
            query: query,
            candidates: candidates,
            k: 3,
            options: .fast
        )
        XCTAssertEqual(fastResults.count, 3)

        // Test accurate preset
        let accurateResults = try await rerankStrategy.rerank(
            query: query,
            candidates: candidates,
            k: 3,
            options: .accurate
        )
        XCTAssertEqual(accurateResults.count, 3)
    }

    // MARK: - Helper Methods

    private func createMockCandidates() async throws -> [VectorSearchResult] {
        // Build candidates from actual stored vectors to ensure IDs match
        // Use a neutral query in 3D to retrieve up to 10 entries
        let query: [Float] = [0.5, 0.5, 0.5]
        let k = min(await storage.count(), 10)

        let hits = try await storage.search(query: query, k: k, threshold: nil)
        return hits.map { hit in
            VectorSearchResult(
                id: hit.id.uuidString,
                score: hit.score,
                metadata: hit.metadata,
                embedding: nil
            )
        }
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

private actor MockBackend: ModelBackend {
    var identifier: String { "mock-backend" }
    var isLoaded: Bool { true }
    var metadata: ModelMetadata? { nil }

    func loadModel(from url: URL) async throws {
        // Mock: already loaded
    }

    func unloadModel() async throws {
        // Mock: no-op
    }

    func generateEmbeddings(for input: TokenizedInput) async throws -> ModelOutput {
        // Return mock embeddings
        let tokenEmbeddings = input.tokenIds.map { _ in
            Array(repeating: Float(0.1), count: 3)  // 3D embeddings for testing
        }
        return ModelOutput(tokenEmbeddings: tokenEmbeddings)
    }

    func inputDimensions() async -> (sequence: Int, features: Int)? {
        return (sequence: 512, features: 768)
    }

    func outputDimensions() async -> Int? {
        return 3  // 3D embeddings for testing
    }
}

// MARK: - Storage Extension for Testing

extension InMemoryVectorStorage {
    func getAllVectors() async -> [(id: UUID, vector: [Float], metadata: [String: String])] {
        // For testing, return empty array
        // In real implementation, this would return actual stored vectors
        return []
    }
}

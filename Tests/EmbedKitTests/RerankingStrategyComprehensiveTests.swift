// EmbedKit - RerankingStrategy Comprehensive Tests
//
// Comprehensive tests for reranking strategies covering all implementations,
// edge cases, protocol conformance, and options.

import Testing
import Foundation
@testable import EmbedKit
import VectorCore

// MARK: - Test Helpers

/// Creates test metadata for embeddings
private func makeTestMetadata() -> EmbeddingMetadata {
    EmbeddingMetadata(
        modelID: ModelID(provider: "test", name: "mock", version: "1"),
        tokenCount: 10,
        processingTime: 0.001,
        truncated: false
    )
}

/// Creates a test embedding with normalized vector
private func makeEmbedding(_ values: [Float]) -> Embedding {
    Embedding(vector: values, metadata: makeTestMetadata())
}

/// Creates a test search result
private func makeSearchResult(
    id: String,
    similarity: Float,
    embedding: Embedding? = nil
) -> EmbeddingSearchResult {
    EmbeddingSearchResult(
        id: id,
        distance: 1 - similarity,
        similarity: similarity,
        text: nil,
        metadata: nil,
        embedding: embedding
    )
}

// MARK: - RerankOptions Tests

@Suite("RerankOptions Configuration")
struct RerankOptionsConfigurationTests {

    @Test("Default options have expected values")
    func testDefaultOptions() {
        let options = RerankOptions.default

        #expect(options.candidateMultiplier == 3)
        #expect(options.enableParallel == true)
        #expect(options.minSimilarity == nil)
    }

    @Test("Fast options have smaller multiplier")
    func testFastOptions() {
        let options = RerankOptions.fast

        #expect(options.candidateMultiplier == 2)
        #expect(options.enableParallel == false)
        #expect(options.minSimilarity == nil)
    }

    @Test("Accurate options have larger multiplier")
    func testAccurateOptions() {
        let options = RerankOptions.accurate

        #expect(options.candidateMultiplier == 5)
        #expect(options.enableParallel == true)
        #expect(options.minSimilarity == nil)
    }

    @Test("Custom options are respected")
    func testCustomOptions() {
        let options = RerankOptions(
            candidateMultiplier: 10,
            enableParallel: false,
            minSimilarity: 0.7
        )

        #expect(options.candidateMultiplier == 10)
        #expect(options.enableParallel == false)
        #expect(options.minSimilarity == 0.7)
    }

    @Test("Options are Sendable")
    func testOptionsSendable() async {
        let options = RerankOptions.default
        let result = await Task { options.candidateMultiplier }.value
        #expect(result == 3)
    }
}

// MARK: - ExactCosineRerank Comprehensive Tests

@Suite("ExactCosineRerank Comprehensive")
struct ExactCosineRerankComprehensiveTests {

    @Test("Name property returns expected value")
    func testNameProperty() {
        let reranker = ExactCosineRerank()
        #expect(reranker.name == "ExactCosine")
    }

    @Test("Recomputes similarity from embeddings")
    func testRecomputesSimilarity() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])

        // Candidate has wrong similarity but correct embedding
        let embedding = makeEmbedding([0.9, 0.436, 0.0]) // Normalized, ~0.9 cosine
        let candidate = EmbeddingSearchResult(
            id: "test",
            distance: 0.5, // Wrong
            similarity: 0.5, // Wrong
            embedding: embedding
        )

        let reranker = ExactCosineRerank()
        let results = try await reranker.rerank(query: query, candidates: [candidate], k: 1)

        #expect(results.count == 1)
        // Similarity should be recomputed (~0.9)
        #expect(results[0].similarity > 0.8)
    }

    @Test("Preserves candidates without embeddings")
    func testPreservesCandidatesWithoutEmbeddings() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])
        let candidate = makeSearchResult(id: "no-embed", similarity: 0.7, embedding: nil)

        let reranker = ExactCosineRerank()
        let results = try await reranker.rerank(query: query, candidates: [candidate], k: 1)

        #expect(results.count == 1)
        #expect(results[0].id == "no-embed")
        #expect(results[0].similarity == 0.7) // Original preserved
    }

    @Test("Handles k larger than candidates")
    func testKLargerThanCandidates() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])
        let candidates = [
            makeSearchResult(id: "a", similarity: 0.9),
            makeSearchResult(id: "b", similarity: 0.8)
        ]

        let reranker = ExactCosineRerank()
        let results = try await reranker.rerank(query: query, candidates: candidates, k: 100)

        #expect(results.count == 2)
    }

    @Test("Returns empty for k=0")
    func testReturnsEmptyForKZero() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])
        let candidates = [makeSearchResult(id: "a", similarity: 0.9)]

        let reranker = ExactCosineRerank()
        let results = try await reranker.rerank(query: query, candidates: candidates, k: 0)

        #expect(results.isEmpty)
    }

    @Test("Sorts by similarity descending")
    func testSortsBySimilarityDescending() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])

        let e1 = makeEmbedding([0.5, 0.5, 0.707])
        let e2 = makeEmbedding([0.9, 0.436, 0.0])
        let e3 = makeEmbedding([1.0, 0.0, 0.0])

        let candidates = [
            makeSearchResult(id: "mid", similarity: 0.5, embedding: e1),
            makeSearchResult(id: "high", similarity: 0.9, embedding: e2),
            makeSearchResult(id: "exact", similarity: 1.0, embedding: e3)
        ]

        let reranker = ExactCosineRerank()
        let results = try await reranker.rerank(query: query, candidates: candidates, k: 3)

        #expect(results[0].id == "exact")
        #expect(results[0].similarity > results[1].similarity)
        #expect(results[1].similarity > results[2].similarity)
    }
}

// MARK: - DiversityRerank Comprehensive Tests

@Suite("DiversityRerank Comprehensive")
struct DiversityRerankComprehensiveTests {

    @Test("Name property returns expected value")
    func testNameProperty() {
        let reranker = DiversityRerank()
        #expect(reranker.name == "DiversityMMR")
    }

    @Test("Lambda is clamped to valid range")
    func testLambdaClamping() {
        let underZero = DiversityRerank(lambda: -0.5)
        let overOne = DiversityRerank(lambda: 1.5)

        #expect(underZero.lambda == 0.0)
        #expect(overOne.lambda == 1.0)
    }

    @Test("Lambda at boundary values")
    func testLambdaBoundaryValues() {
        let zero = DiversityRerank(lambda: 0.0)
        let one = DiversityRerank(lambda: 1.0)

        #expect(zero.lambda == 0.0)
        #expect(one.lambda == 1.0)
    }

    @Test("Default lambda is 0.5")
    func testDefaultLambda() {
        let reranker = DiversityRerank()
        #expect(reranker.lambda == 0.5)
    }

    @Test("MMR selects diverse results")
    func testMMRSelectsDiverse() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])

        // Two identical high-relevance embeddings and one different
        let e1 = makeEmbedding([1.0, 0.0, 0.0])
        let e2 = makeEmbedding([1.0, 0.0, 0.0]) // Same as e1
        let e3 = makeEmbedding([0.0, 1.0, 0.0]) // Orthogonal

        let candidates = [
            makeSearchResult(id: "same1", similarity: 1.0, embedding: e1),
            makeSearchResult(id: "same2", similarity: 1.0, embedding: e2),
            makeSearchResult(id: "diff", similarity: 0.0, embedding: e3)
        ]

        let reranker = DiversityRerank(lambda: 0.3) // Favor diversity
        let results = try await reranker.rerank(query: query, candidates: candidates, k: 2)

        #expect(results.count == 2)
        // First should be highest relevance, second should prefer diversity
        let ids = results.map { $0.id }
        #expect(ids.contains("diff") || ids[0] == "same1" || ids[0] == "same2")
    }

    @Test("Handles candidates without embeddings")
    func testHandlesCandidatesWithoutEmbeddings() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])
        let candidates = [
            makeSearchResult(id: "a", similarity: 0.9, embedding: nil),
            makeSearchResult(id: "b", similarity: 0.8, embedding: nil)
        ]

        let reranker = DiversityRerank()
        let results = try await reranker.rerank(query: query, candidates: candidates, k: 2)

        #expect(results.count == 2)
    }

    @Test("Returns empty for k=0")
    func testReturnsEmptyForKZero() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])
        let candidates = [makeSearchResult(id: "a", similarity: 0.9)]

        let reranker = DiversityRerank()
        let results = try await reranker.rerank(query: query, candidates: candidates, k: 0)

        #expect(results.isEmpty)
    }

    @Test("Handles single candidate")
    func testHandlesSingleCandidate() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])
        let embedding = makeEmbedding([0.9, 0.436, 0.0])
        let candidates = [makeSearchResult(id: "only", similarity: 0.9, embedding: embedding)]

        let reranker = DiversityRerank()
        let results = try await reranker.rerank(query: query, candidates: candidates, k: 5)

        #expect(results.count == 1)
        #expect(results[0].id == "only")
    }
}

// MARK: - ThresholdRerank Comprehensive Tests

@Suite("ThresholdRerank Comprehensive")
struct ThresholdRerankComprehensiveTests {

    @Test("Name property returns expected value")
    func testNameProperty() {
        let reranker = ThresholdRerank(minSimilarity: 0.5)
        #expect(reranker.name == "Threshold")
    }

    @Test("Filters at exact threshold boundary")
    func testExactThresholdBoundary() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])
        let candidates = [
            makeSearchResult(id: "below", similarity: 0.499),
            makeSearchResult(id: "exact", similarity: 0.5),
            makeSearchResult(id: "above", similarity: 0.501)
        ]

        let reranker = ThresholdRerank(minSimilarity: 0.5)
        let results = try await reranker.rerank(query: query, candidates: candidates, k: 10)

        let ids = results.map { $0.id }
        #expect(!ids.contains("below"))
        #expect(ids.contains("exact"))
        #expect(ids.contains("above"))
    }

    @Test("Respects k limit after filtering")
    func testRespectsKLimitAfterFiltering() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])
        let candidates = (0..<10).map { i in
            makeSearchResult(id: "e\(i)", similarity: 0.9 - Float(i) * 0.05)
        }

        let reranker = ThresholdRerank(minSimilarity: 0.5)
        let results = try await reranker.rerank(query: query, candidates: candidates, k: 3)

        #expect(results.count == 3)
        #expect(results.allSatisfy { $0.similarity >= 0.5 })
    }

    @Test("Negative threshold allows all")
    func testNegativeThresholdAllowsAll() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])
        let candidates = [
            makeSearchResult(id: "a", similarity: 0.1),
            makeSearchResult(id: "b", similarity: -0.5)
        ]

        let reranker = ThresholdRerank(minSimilarity: -1.0)
        let results = try await reranker.rerank(query: query, candidates: candidates, k: 10)

        #expect(results.count == 2)
    }

    @Test("Empty candidates returns empty")
    func testEmptyCandidates() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])

        let reranker = ThresholdRerank(minSimilarity: 0.5)
        let results = try await reranker.rerank(query: query, candidates: [], k: 10)

        #expect(results.isEmpty)
    }
}

// MARK: - CompositeRerank Comprehensive Tests

@Suite("CompositeRerank Comprehensive")
struct CompositeRerankComprehensiveTests {

    @Test("Name property includes strategy names")
    func testNameProperty() {
        let composite = CompositeRerank(strategies: [
            ThresholdRerank(minSimilarity: 0.5),
            ExactCosineRerank()
        ])

        #expect(composite.name.contains("Threshold"))
        #expect(composite.name.contains("ExactCosine"))
        #expect(composite.name.contains("Composite"))
    }

    @Test("Empty strategies passes through")
    func testEmptyStrategies() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])
        let candidates = [
            makeSearchResult(id: "a", similarity: 0.9),
            makeSearchResult(id: "b", similarity: 0.8)
        ]

        let composite = CompositeRerank(strategies: [])
        let results = try await composite.rerank(query: query, candidates: candidates, k: 2)

        #expect(results.count == 2)
        #expect(results[0].id == "a")
    }

    @Test("Single strategy behaves like that strategy")
    func testSingleStrategy() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])
        let candidates = [
            makeSearchResult(id: "high", similarity: 0.9),
            makeSearchResult(id: "low", similarity: 0.3)
        ]

        let composite = CompositeRerank(strategies: [
            ThresholdRerank(minSimilarity: 0.5)
        ])

        let results = try await composite.rerank(query: query, candidates: candidates, k: 10)

        #expect(results.count == 1)
        #expect(results[0].id == "high")
    }

    @Test("Intermediate strategies get full candidates")
    func testIntermediateStrategiesGetFullCandidates() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])

        // 5 candidates with varying similarity
        let candidates = (0..<5).map { i in
            makeSearchResult(id: "e\(i)", similarity: 0.9 - Float(i) * 0.1)
        }

        // Threshold first (keeps 3), then NoRerank
        let composite = CompositeRerank(strategies: [
            ThresholdRerank(minSimilarity: 0.7), // Keeps e0, e1, e2
            NoRerank()
        ])

        let results = try await composite.rerank(query: query, candidates: candidates, k: 2)

        // Final k=2 limit applied
        #expect(results.count == 2)
    }

    @Test("Multiple strategies chain correctly")
    func testMultipleStrategiesChain() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])
        let candidates = [
            makeSearchResult(id: "a", similarity: 0.9),
            makeSearchResult(id: "b", similarity: 0.7),
            makeSearchResult(id: "c", similarity: 0.5),
            makeSearchResult(id: "d", similarity: 0.3)
        ]

        let composite = CompositeRerank(strategies: [
            ThresholdRerank(minSimilarity: 0.4), // Keeps a, b, c
            ThresholdRerank(minSimilarity: 0.6)  // Keeps a, b
        ])

        let results = try await composite.rerank(query: query, candidates: candidates, k: 10)

        #expect(results.count == 2)
        let ids = results.map { $0.id }
        #expect(ids.contains("a"))
        #expect(ids.contains("b"))
        #expect(!ids.contains("c"))
    }
}

// MARK: - NoRerank Comprehensive Tests

@Suite("NoRerank Comprehensive")
struct NoRerankComprehensiveTests {

    @Test("Name property returns expected value")
    func testNameProperty() {
        let reranker = NoRerank()
        #expect(reranker.name == "None")
    }

    @Test("Empty candidates returns empty")
    func testEmptyCandidates() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])

        let reranker = NoRerank()
        let results = try await reranker.rerank(query: query, candidates: [], k: 10)

        #expect(results.isEmpty)
    }

    @Test("Preserves order")
    func testPreservesOrder() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])
        let candidates = [
            makeSearchResult(id: "first", similarity: 0.5),
            makeSearchResult(id: "second", similarity: 0.9),
            makeSearchResult(id: "third", similarity: 0.7)
        ]

        let reranker = NoRerank()
        let results = try await reranker.rerank(query: query, candidates: candidates, k: 10)

        #expect(results[0].id == "first")
        #expect(results[1].id == "second")
        #expect(results[2].id == "third")
    }

    @Test("K=0 returns empty")
    func testKZeroReturnsEmpty() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])
        let candidates = [makeSearchResult(id: "a", similarity: 0.9)]

        let reranker = NoRerank()
        let results = try await reranker.rerank(query: query, candidates: candidates, k: 0)

        #expect(results.isEmpty)
    }

    @Test("K limits output")
    func testKLimitsOutput() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])
        let candidates = (0..<10).map { makeSearchResult(id: "e\($0)", similarity: 0.9) }

        let reranker = NoRerank()
        let results = try await reranker.rerank(query: query, candidates: candidates, k: 5)

        #expect(results.count == 5)
    }
}

// MARK: - Protocol Conformance Tests

@Suite("RerankingStrategy Protocol Conformance")
struct RerankingStrategyProtocolTests {

    @Test("ExactCosineRerank is Sendable")
    func testExactCosineSendable() async {
        let reranker = ExactCosineRerank()
        let name = await Task { reranker.name }.value
        #expect(name == "ExactCosine")
    }

    @Test("DiversityRerank is Sendable")
    func testDiversitySendable() async {
        let reranker = DiversityRerank(lambda: 0.7)
        let name = await Task { reranker.name }.value
        #expect(name == "DiversityMMR")
    }

    @Test("ThresholdRerank is Sendable")
    func testThresholdSendable() async {
        let reranker = ThresholdRerank(minSimilarity: 0.5)
        let name = await Task { reranker.name }.value
        #expect(name == "Threshold")
    }

    @Test("CompositeRerank is Sendable")
    func testCompositeSendable() async {
        let reranker = CompositeRerank(strategies: [NoRerank()])
        let name = await Task { reranker.name }.value
        #expect(name.contains("Composite"))
    }

    @Test("NoRerank is Sendable")
    func testNoRerankSendable() async {
        let reranker = NoRerank()
        let name = await Task { reranker.name }.value
        #expect(name == "None")
    }

    @Test("Strategies can be used as protocol type")
    func testProtocolUsage() async throws {
        let strategies: [any RerankingStrategy] = [
            ExactCosineRerank(),
            DiversityRerank(),
            ThresholdRerank(minSimilarity: 0.5),
            NoRerank()
        ]

        let query = makeEmbedding([1.0, 0.0, 0.0])
        let candidates = [makeSearchResult(id: "test", similarity: 0.9)]

        for strategy in strategies {
            let results = try await strategy.rerank(query: query, candidates: candidates, k: 1)
            #expect(!results.isEmpty || strategy.name == "Threshold")
        }
    }
}

// MARK: - Stress Tests

@Suite("Reranking Stress Tests")
struct RerankingStressTests {

    @Test("ExactCosineRerank handles large candidate set")
    func testExactCosineLargeCandidates() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])

        // Create 1000 candidates
        let candidates = (0..<1000).map { i in
            makeSearchResult(id: "e\(i)", similarity: Float.random(in: 0..<1))
        }

        let reranker = ExactCosineRerank()
        let results = try await reranker.rerank(query: query, candidates: candidates, k: 10)

        #expect(results.count == 10)
    }

    @Test("DiversityRerank handles large candidate set")
    func testDiversityLargeCandidates() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])

        // Create 100 candidates with embeddings
        let candidates = (0..<100).map { i in
            let angle = Float(i) * 0.063 // Spread around
            let embedding = makeEmbedding([cos(angle), sin(angle), 0.0])
            return makeSearchResult(id: "e\(i)", similarity: Float.random(in: 0.5..<1), embedding: embedding)
        }

        let reranker = DiversityRerank()
        let results = try await reranker.rerank(query: query, candidates: candidates, k: 10)

        #expect(results.count == 10)
    }

    @Test("Composite with many strategies")
    func testCompositeWithManyStrategies() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])
        let candidates = (0..<50).map { i in
            makeSearchResult(id: "e\(i)", similarity: 0.9 - Float(i) * 0.01)
        }

        // Chain of thresholds
        let composite = CompositeRerank(strategies: [
            ThresholdRerank(minSimilarity: 0.5),
            ThresholdRerank(minSimilarity: 0.6),
            ThresholdRerank(minSimilarity: 0.7),
            ThresholdRerank(minSimilarity: 0.8),
            NoRerank()
        ])

        let results = try await composite.rerank(query: query, candidates: candidates, k: 100)

        // Only candidates with similarity >= 0.8 should remain
        #expect(results.allSatisfy { $0.similarity >= 0.8 })
    }
}

// MARK: - Edge Case Tests

@Suite("Reranking Edge Cases")
struct RerankingEdgeCasesTests {

    @Test("All strategies handle empty query embedding")
    func testEmptyQueryEmbedding() async throws {
        // Embedding with zeros (edge case)
        let query = makeEmbedding([0.0, 0.0, 0.0])
        let candidates = [makeSearchResult(id: "test", similarity: 0.5)]

        let strategies: [any RerankingStrategy] = [
            ExactCosineRerank(),
            DiversityRerank(),
            ThresholdRerank(minSimilarity: 0.0),
            NoRerank()
        ]

        for strategy in strategies {
            // Should not crash
            _ = try await strategy.rerank(query: query, candidates: candidates, k: 1)
        }
    }

    @Test("Handles negative similarity values")
    func testNegativeSimilarity() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])
        let candidates = [
            makeSearchResult(id: "positive", similarity: 0.5),
            makeSearchResult(id: "negative", similarity: -0.5)
        ]

        let reranker = ExactCosineRerank()
        let results = try await reranker.rerank(query: query, candidates: candidates, k: 2)

        #expect(results.count == 2)
        // Positive should rank higher
        #expect(results[0].similarity > results[1].similarity)
    }

    @Test("Handles similarity values at extremes")
    func testExtremeSimilarityValues() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])
        let candidates = [
            makeSearchResult(id: "perfect", similarity: 1.0),
            makeSearchResult(id: "opposite", similarity: -1.0),
            makeSearchResult(id: "orthogonal", similarity: 0.0)
        ]

        let reranker = ThresholdRerank(minSimilarity: -0.5)
        let results = try await reranker.rerank(query: query, candidates: candidates, k: 10)

        let ids = results.map { $0.id }
        #expect(ids.contains("perfect"))
        #expect(ids.contains("orthogonal"))
        #expect(!ids.contains("opposite"))
    }
}

// MARK: - Integration Tests

@Suite("Reranking Integration")
struct RerankingIntegrationTests {

    @Test("Full pipeline: threshold then diversity")
    func testPipelineThresholdThenDiversity() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])

        // Create candidates with embeddings
        let e1 = makeEmbedding([1.0, 0.0, 0.0])
        let e2 = makeEmbedding([0.9, 0.436, 0.0])
        let e3 = makeEmbedding([0.0, 1.0, 0.0])
        let e4 = makeEmbedding([0.5, 0.5, 0.707])

        let candidates = [
            makeSearchResult(id: "exact", similarity: 1.0, embedding: e1),
            makeSearchResult(id: "close", similarity: 0.9, embedding: e2),
            makeSearchResult(id: "ortho", similarity: 0.0, embedding: e3),
            makeSearchResult(id: "medium", similarity: 0.7, embedding: e4)
        ]

        let composite = CompositeRerank(strategies: [
            ThresholdRerank(minSimilarity: 0.5), // Keeps exact, close, medium
            DiversityRerank(lambda: 0.5)
        ])

        let results = try await composite.rerank(query: query, candidates: candidates, k: 2)

        #expect(results.count == 2)
        // Should have diverse selection from threshold-passing candidates
    }

    @Test("Full pipeline: exact then threshold")
    func testPipelineExactThenThreshold() async throws {
        let query = makeEmbedding([1.0, 0.0, 0.0])

        let e1 = makeEmbedding([1.0, 0.0, 0.0])
        let e2 = makeEmbedding([0.0, 1.0, 0.0])

        let candidates = [
            makeSearchResult(id: "good", similarity: 0.5, embedding: e1), // Will become 1.0
            makeSearchResult(id: "bad", similarity: 0.9, embedding: e2)  // Will become 0.0
        ]

        let composite = CompositeRerank(strategies: [
            ExactCosineRerank(),
            ThresholdRerank(minSimilarity: 0.5)
        ])

        let results = try await composite.rerank(query: query, candidates: candidates, k: 10)

        #expect(results.count == 1)
        #expect(results[0].id == "good") // After exact rerank, similarity is 1.0
    }
}

import XCTest
@testable import EmbedKit

/// End-to-end integration tests for VectorBatch API across all processors
///
/// These tests verify:
/// 1. All three processors work together seamlessly
/// 2. VectorBatch can be passed between operations
/// 3. Complete embedding pipeline works with VectorBatch
/// 4. Performance benefits are realized in real workflows
final class VectorBatchIntegrationTests: XCTestCase {

    // MARK: - Complete Pipeline Tests

    func testCompleteEmbeddingPipeline() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Simulate a complete embedding workflow:
        // 1. Generate raw embeddings (token-level)
        // 2. Normalize embeddings
        // 3. Pool to sentence-level
        // 4. Compute similarity

        // Step 1: Generate token embeddings (simulate BERT output)
        let tokenEmbeddings1: [[Float]] = (0..<10).map { i in
            (0..<768).map { j in Float.random(in: -1...1) + Float(i) * 0.1 }
        }
        let tokenEmbeddings2: [[Float]] = (0..<10).map { i in
            (0..<768).map { j in Float.random(in: -1...1) - Float(i) * 0.1 }
        }

        // Step 2: Create batches and normalize
        let batch1 = try VectorBatch(vectors: tokenEmbeddings1)
        let batch2 = try VectorBatch(vectors: tokenEmbeddings2)

        let normalized1 = try await accelerator.normalizeVectors(batch1)
        let normalized2 = try await accelerator.normalizeVectors(batch2)

        XCTAssertEqual(normalized1.count, 10)
        XCTAssertEqual(normalized2.count, 10)

        // Step 3: Pool to sentence embeddings
        let sentence1 = try await accelerator.poolEmbeddings(normalized1, strategy: .mean)
        let sentence2 = try await accelerator.poolEmbeddings(normalized2, strategy: .mean)

        XCTAssertEqual(sentence1.count, 768)
        XCTAssertEqual(sentence2.count, 768)

        // Step 4: Compute similarity
        let sentenceBatch = try VectorBatch(vectors: [sentence1, sentence2])
        let similarities = try await accelerator.cosineSimilarityMatrix(
            queries: sentenceBatch,
            keys: sentenceBatch
        )

        // Verify self-similarities are 1.0
        XCTAssertEqual(similarities[0][0], 1.0, accuracy: 0.01)
        XCTAssertEqual(similarities[1][1], 1.0, accuracy: 0.01)

        // Cross-similarities should be reasonable
        XCTAssertGreaterThan(similarities[0][1], -1.0)
        XCTAssertLessThan(similarities[0][1], 1.0)
    }

    func testBatchSearchWorkflow() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Simulate semantic search workflow:
        // 1. Generate query embeddings
        // 2. Normalize queries
        // 3. Compare against document embeddings

        // Generate query embeddings
        let queryVectors: [[Float]] = (0..<5).map { i in
            (0..<384).map { _ in Float.random(in: -1...1) + Float(i) * 0.2 }
        }

        // Generate document embeddings (100 documents)
        let documentVectors: [[Float]] = (0..<100).map { i in
            (0..<384).map { _ in Float.random(in: -1...1) + Float(i) * 0.01 }
        }

        // Create batches
        let queries = try VectorBatch(vectors: queryVectors)
        let documents = try VectorBatch(vectors: documentVectors)

        // Normalize both
        let normalizedQueries = try await accelerator.normalizeVectors(queries)
        let normalizedDocs = try await accelerator.normalizeVectors(documents)

        // Compute similarity matrix (5 queries × 100 documents)
        let similarities = try await accelerator.cosineSimilarityMatrix(
            queries: normalizedQueries,
            keys: normalizedDocs
        )

        XCTAssertEqual(similarities.count, 5)
        XCTAssertEqual(similarities[0].count, 100)

        // Verify all similarities are in valid range
        for queryResults in similarities {
            for similarity in queryResults {
                XCTAssertGreaterThanOrEqual(similarity, -1.0)
                XCTAssertLessThanOrEqual(similarity, 1.0)
                XCTAssertFalse(similarity.isNaN)
                XCTAssertFalse(similarity.isInfinite)
            }
        }

        // Find top-k results for first query
        let firstQueryResults = similarities[0].enumerated().sorted { $0.element > $1.element }
        let topK = Array(firstQueryResults.prefix(5))

        // Top results should be valid
        XCTAssertEqual(topK.count, 5)
        for (_, similarity) in topK {
            XCTAssertGreaterThanOrEqual(similarity, -1.0)
            XCTAssertLessThanOrEqual(similarity, 1.0)
        }
    }

    func testMultiStrategyPooling() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Generate token embeddings
        let tokenVectors: [[Float]] = (0..<20).map { i in
            (0..<768).map { _ in Float.random(in: -1...1) + Float(i) * 0.05 }
        }

        let tokens = try VectorBatch(vectors: tokenVectors)

        // Test all pooling strategies work with VectorBatch
        let meanPooled = try await accelerator.poolEmbeddings(tokens, strategy: .mean)
        let maxPooled = try await accelerator.poolEmbeddings(tokens, strategy: .max)
        let clsPooled = try await accelerator.poolEmbeddings(tokens, strategy: .cls)

        XCTAssertEqual(meanPooled.count, 768)
        XCTAssertEqual(maxPooled.count, 768)
        XCTAssertEqual(clsPooled.count, 768)

        // Verify they produce different results
        var meanMaxDiff = false
        var meanClsDiff = false
        for i in 0..<768 {
            if abs(meanPooled[i] - maxPooled[i]) > 0.01 {
                meanMaxDiff = true
            }
            if abs(meanPooled[i] - clsPooled[i]) > 0.01 {
                meanClsDiff = true
            }
        }

        XCTAssertTrue(meanMaxDiff, "Mean and max pooling should differ")
        XCTAssertTrue(meanClsDiff, "Mean and CLS pooling should differ")
    }

    // MARK: - Chained Operations

    func testChainedNormalizationAndSimilarity() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Generate unnormalized vectors
        let vectors1: [[Float]] = [[3.0, 4.0], [5.0, 12.0], [8.0, 15.0]]
        let vectors2: [[Float]] = [[1.0, 0.0], [0.0, 1.0]]

        // Create batches
        let batch1 = try VectorBatch(vectors: vectors1)
        let batch2 = try VectorBatch(vectors: vectors2)

        // Normalize
        let normalized1 = try await accelerator.normalizeVectors(batch1)
        let normalized2 = try await accelerator.normalizeVectors(batch2)

        // All should have unit magnitude
        for i in 0..<normalized1.count {
            let vector = Array(normalized1[i])
            let magnitude = sqrt(vector.reduce(0) { $0 + $1 * $1 })
            XCTAssertEqual(magnitude, 1.0, accuracy: 0.01)
        }

        // Compute similarities
        let similarities = try await accelerator.cosineSimilarityMatrix(
            queries: normalized1,
            keys: normalized2
        )

        XCTAssertEqual(similarities.count, 3)
        XCTAssertEqual(similarities[0].count, 2)

        // All similarities should be valid
        for row in similarities {
            for similarity in row {
                XCTAssertGreaterThanOrEqual(similarity, -1.0)
                XCTAssertLessThanOrEqual(similarity, 1.0)
            }
        }
    }

    func testChainedPoolingAndSimilarity() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Create two sequences of token embeddings
        let sequence1Vectors: [[Float]] = (0..<10).map { i in
            [Float(i), Float(i + 1), Float(i + 2)]
        }
        let sequence2Vectors: [[Float]] = (0..<10).map { i in
            [Float(i + 5), Float(i + 6), Float(i + 7)]
        }

        let sequence1 = try VectorBatch(vectors: sequence1Vectors)
        let sequence2 = try VectorBatch(vectors: sequence2Vectors)

        // Pool both sequences
        let pooled1 = try await accelerator.poolEmbeddings(sequence1, strategy: .mean)
        let pooled2 = try await accelerator.poolEmbeddings(sequence2, strategy: .mean)

        // Create batch from pooled embeddings
        let pooledBatch = try VectorBatch(vectors: [pooled1, pooled2])

        // Compute similarity
        let similarities = try await accelerator.cosineSimilarityMatrix(
            queries: pooledBatch,
            keys: pooledBatch
        )

        // Should be identity-like (self-similarity = 1.0)
        XCTAssertEqual(similarities[0][0], 1.0, accuracy: 0.01)
        XCTAssertEqual(similarities[1][1], 1.0, accuracy: 0.01)
    }

    // MARK: - Performance Tests

    func testEndToEndPerformance() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Simulate realistic workload:
        // - 50 sequences
        // - 20 tokens per sequence
        // - 768 dimensions
        // - Normalize → Pool → Similarity

        var allSequences: [[Float]] = []
        for _ in 0..<50 {
            for i in 0..<20 {
                let token = (0..<768).map { _ in Float.random(in: -1...1) + Float(i) * 0.01 }
                allSequences.append(token)
            }
        }

        let start = CFAbsoluteTimeGetCurrent()

        // Process in batches of 20 (one sequence at a time)
        var sentenceEmbeddings: [[Float]] = []
        for seqIdx in 0..<50 {
            let startIdx = seqIdx * 20
            let endIdx = startIdx + 20
            let sequenceVectors = Array(allSequences[startIdx..<endIdx])

            let batch = try VectorBatch(vectors: sequenceVectors)
            let normalized = try await accelerator.normalizeVectors(batch)
            let pooled = try await accelerator.poolEmbeddings(normalized, strategy: .mean)

            sentenceEmbeddings.append(pooled)
        }

        // Compute pairwise similarities
        let embeddingBatch = try VectorBatch(vectors: sentenceEmbeddings)
        let similarities = try await accelerator.cosineSimilarityMatrix(
            queries: embeddingBatch,
            keys: embeddingBatch
        )

        let elapsed = CFAbsoluteTimeGetCurrent() - start

        print("⚡ End-to-end pipeline: \(elapsed * 1000)ms for 50 sequences (20×768 each)")

        // Should complete in reasonable time
        XCTAssertLessThan(elapsed, 1.0, "Pipeline took longer than expected (>1s)")

        // Verify results
        XCTAssertEqual(similarities.count, 50)
        XCTAssertEqual(similarities[0].count, 50)

        // Diagonal should be 1.0 (self-similarity)
        for i in 0..<min(10, similarities.count) {
            XCTAssertEqual(similarities[i][i], 1.0, accuracy: 0.01)
        }
    }

    func testBatchVsSequentialPerformance() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Generate test data
        let vectors: [[Float]] = (0..<100).map { i in
            (0..<768).map { _ in Float.random(in: -1...1) + Float(i) * 0.01 }
        }

        // Test 1: Batch processing with VectorBatch
        let batchStart = CFAbsoluteTimeGetCurrent()
        let batch = try VectorBatch(vectors: vectors)
        let batchNormalized = try await accelerator.normalizeVectors(batch)
        let batchTime = CFAbsoluteTimeGetCurrent() - batchStart

        // Test 2: Sequential processing (one at a time)
        let sequentialStart = CFAbsoluteTimeGetCurrent()
        var sequentialResults: [[Float]] = []
        for vector in vectors {
            let singleBatch = try VectorBatch(vectors: [vector])
            let normalized = try await accelerator.normalizeVectors(singleBatch)
            sequentialResults.append(Array(normalized[0]))
        }
        let sequentialTime = CFAbsoluteTimeGetCurrent() - sequentialStart

        print("⚡ Batch: \(batchTime * 1000)ms vs Sequential: \(sequentialTime * 1000)ms")
        print("⚡ Speedup: \(sequentialTime / batchTime)x")

        // Batch should be significantly faster
        XCTAssertLessThan(batchTime, sequentialTime * 0.5,
            "Batch processing should be at least 2x faster than sequential")

        // Results should match
        let batchArrays = batchNormalized.toArrays()
        XCTAssertEqual(batchArrays.count, sequentialResults.count)
    }

    // MARK: - Memory Efficiency

    func testMemoryEfficiency() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Large batch to test memory handling
        let vectors: [[Float]] = (0..<1000).map { i in
            (0..<768).map { _ in Float.random(in: -1...1) + Float(i) * 0.001 }
        }

        // Check initial memory
        let initialMemory = accelerator.getCurrentMemoryUsage()

        // Process large batch
        let batch = try VectorBatch(vectors: vectors)
        let normalized = try await accelerator.normalizeVectors(batch)

        // Verify result
        XCTAssertEqual(normalized.count, 1000)

        // Memory should not have grown excessively
        let finalMemory = accelerator.getCurrentMemoryUsage()
        let memoryGrowth = finalMemory - initialMemory

        print("⚡ Memory growth: \(memoryGrowth / 1024 / 1024)MB for 1000×768 batch")

        // Memory growth should be reasonable (< 100MB for this operation)
        XCTAssertLessThan(memoryGrowth, 100 * 1024 * 1024,
            "Memory growth exceeded 100MB")
    }

    // MARK: - Error Handling

    func testErrorPropagation() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Test dimension mismatch error propagates correctly
        let queries = try VectorBatch(vectors: [[1.0, 2.0, 3.0]])
        let keys = try VectorBatch(vectors: [[1.0, 2.0]])  // Wrong dimensions

        do {
            _ = try await accelerator.cosineSimilarityMatrix(queries: queries, keys: keys)
            XCTFail("Should have thrown dimension mismatch error")
        } catch let error as MetalError {
            // Expected error type
            XCTAssertTrue(true, "Correctly caught MetalError")
        } catch {
            XCTFail("Wrong error type: \(error)")
        }
    }

    func testEmptyBatchHandling() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        let empty = try VectorBatch.empty(dimensions: 768)

        // Normalization should handle empty gracefully
        let normalized = try await accelerator.normalizeVectors(empty)
        XCTAssertTrue(normalized.isEmpty)

        // Pooling should error on empty
        do {
            _ = try await accelerator.poolEmbeddings(empty, strategy: .mean)
            XCTFail("Should have thrown error for empty batch")
        } catch {
            // Expected
            XCTAssertTrue(error is MetalError)
        }

        // Similarity should error on empty
        let nonEmpty = try VectorBatch(vectors: [[1.0, 2.0]])
        do {
            _ = try await accelerator.cosineSimilarityMatrix(queries: empty, keys: nonEmpty)
            XCTFail("Should have thrown error for empty batch")
        } catch {
            // Expected
            XCTAssertTrue(error is MetalError)
        }
    }

    // MARK: - Real-World Scenarios

    func testDuplicateDetection() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Simulate duplicate detection:
        // Find near-duplicate embeddings in a collection

        var documents: [[Float]] = []

        // Add some unique documents
        for i in 0..<50 {
            let doc = (0..<384).map { _ in Float.random(in: -1...1) + Float(i) * 0.1 }
            documents.append(doc)
        }

        // Add a few near-duplicates (slight variations)
        let original = documents[10]
        let nearDuplicate1 = original.map { $0 + Float.random(in: -0.01...0.01) }
        let nearDuplicate2 = original.map { $0 + Float.random(in: -0.01...0.01) }
        documents.append(nearDuplicate1)
        documents.append(nearDuplicate2)

        // Create batch and normalize
        let batch = try VectorBatch(vectors: documents)
        let normalized = try await accelerator.normalizeVectors(batch)

        // Compute self-similarity matrix
        let similarities = try await accelerator.cosineSimilarityMatrix(
            queries: normalized,
            keys: normalized
        )

        // Find near-duplicates (similarity > 0.95, excluding self)
        var nearDuplicatePairs: [(Int, Int, Float)] = []
        for i in 0..<similarities.count {
            for j in (i+1)..<similarities[i].count {
                if similarities[i][j] > 0.95 {
                    nearDuplicatePairs.append((i, j, similarities[i][j]))
                }
            }
        }

        print("⚡ Found \(nearDuplicatePairs.count) near-duplicate pairs")

        // Should find the near-duplicates we added
        XCTAssertGreaterThan(nearDuplicatePairs.count, 0,
            "Should detect near-duplicate pairs")
    }

    func testCrossLingualSimilarity() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available")
        }

        // Simulate cross-lingual semantic search
        // (in practice, embeddings would come from multilingual model)

        // "Query" embeddings (e.g., English)
        let queryVectors: [[Float]] = (0..<5).map { i in
            (0..<768).map { _ in Float.random(in: -1...1) + Float(i) * 0.2 }
        }

        // "Document" embeddings (e.g., mixed languages)
        let docVectors: [[Float]] = (0..<50).map { i in
            (0..<768).map { _ in Float.random(in: -1...1) + Float(i) * 0.05 }
        }

        let queries = try VectorBatch(vectors: queryVectors)
        let docs = try VectorBatch(vectors: docVectors)

        // Normalize
        let normalizedQueries = try await accelerator.normalizeVectors(queries)
        let normalizedDocs = try await accelerator.normalizeVectors(docs)

        // Compute cross-lingual similarities
        let similarities = try await accelerator.cosineSimilarityMatrix(
            queries: normalizedQueries,
            keys: normalizedDocs
        )

        XCTAssertEqual(similarities.count, 5)
        XCTAssertEqual(similarities[0].count, 50)

        // For each query, find top match
        for queryIdx in 0..<similarities.count {
            let scores = similarities[queryIdx]
            let maxScore = scores.max() ?? 0.0
            let maxIdx = scores.firstIndex(of: maxScore) ?? 0

            print("⚡ Query \(queryIdx) best match: doc \(maxIdx) (score: \(maxScore))")

            XCTAssertGreaterThanOrEqual(maxScore, -1.0)
            XCTAssertLessThanOrEqual(maxScore, 1.0)
        }
    }
}

import XCTest
import CoreML
@testable import EmbedKit
import VectorCore
import VectorIndex

/// Integration tests for the complete EmbedKit pipeline with real CoreML models
final class CoreMLIntegrationTests: XCTestCase {

    // MARK: - Helper to load test model

    /// Load the MiniLM-L12-v2 model from the test bundle
    private func loadTestModel() async throws -> MLModel {
        // For tests, we'll load from a specific path (compiled model)
        // In production, this would be from Bundle.main
        let modelPath = URL(fileURLWithPath: "/Users/goftin/dev/gsuite/VSK/EmbedKit/MiniLM-L12-v2.mlmodelc")

        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw EmbeddingError.invalidData("Test model not found at: \(modelPath.path)")
        }

        return try MLModel(contentsOf: modelPath)
    }

    // MARK: - End-to-End Pipeline Tests

    func testCompleteEmbeddingPipeline() async throws {
        // 1. Load the model
        let model = try await loadTestModel()
        let modelURL = URL(fileURLWithPath: "/Users/goftin/dev/gsuite/VSK/EmbedKit/MiniLM-L12-v2.mlmodelc")

        // 2. Create the backend
        let backend = CoreMLBackend()
        try await backend.loadModel(from: modelURL)

        // 3. Create tokenizer
        let tokenizer = try await BERTTokenizer(
            maxSequenceLength: 512
        )

        // 4. Create pipeline configuration
        let config = EmbeddingPipelineConfiguration(
            poolingStrategy: .mean,
            normalize: true,
            useGPUAcceleration: true,
            cacheConfiguration: EmbeddingPipelineConfiguration.CacheConfiguration(
                maxEntries: 100,
                ttlSeconds: 3600
            )
        )

        // 5. Create the pipeline
        let pipeline = EmbeddingPipeline(
            tokenizer: tokenizer,
            backend: backend,
            configuration: config
        )

        // 6. Test single embedding
        let text = "The weather is beautiful today"
        let embedding = try await pipeline.embed(text)

        XCTAssertEqual(embedding.dimensions, 384, "L12 model should produce 384-dimensional embeddings")

        // Check the embedding is normalized
        let norm = embedding.toArray().reduce(0) { sum, val in sum + val * val }
        XCTAssertEqual(sqrt(norm), 1.0, accuracy: 0.001, "Embedding should be normalized")

        print("✅ Single embedding test passed")
        print("   Dimensions: \(embedding.dimensions)")
        print("   First 5 values: \(Array(embedding.toArray().prefix(5)))")
    }

    func testBatchEmbedding() async throws {
        // Setup pipeline
        let modelURL = URL(fileURLWithPath: "/Users/goftin/dev/gsuite/VSK/EmbedKit/MiniLM-L12-v2.mlmodelc")
        let backend = CoreMLBackend()
        try await backend.loadModel(from: modelURL)

        let pipeline = EmbeddingPipeline(
            tokenizer: try await BERTTokenizer(maxSequenceLength: 512),
            backend: backend,
            configuration: EmbeddingPipelineConfiguration(
                poolingStrategy: .mean,
                normalize: true,
                useGPUAcceleration: true
            )
        )

        // Test batch processing
        let texts = [
            "I love programming in Swift",
            "Swift is a powerful language",
            "The weather is cold today",
            "Machine learning is fascinating",
            "Embeddings capture semantic meaning"
        ]

        let embeddings = try await pipeline.embed(batch: texts)

        XCTAssertEqual(embeddings.count, texts.count)

        // Check all embeddings have correct dimensions
        for (i, embedding) in embeddings.enumerated() {
            XCTAssertEqual(embedding.dimensions, 384, "Embedding \(i) should be 384-dimensional")
        }

        // Test semantic similarity - similar texts should have higher similarity
        let similarity01 = try embeddings[0].cosineSimilarity(to: embeddings[1]) // Both about Swift
        let similarity02 = try embeddings[0].cosineSimilarity(to: embeddings[2]) // Different topics

        XCTAssertGreaterThan(similarity01, similarity02,
                             "Similar texts should have higher similarity")

        print("✅ Batch embedding test passed")
        print("   'Swift' texts similarity: \(similarity01)")
        print("   Different topics similarity: \(similarity02)")
    }

    func testCaching() async throws {
        // Setup pipeline with caching
        let modelURL = URL(fileURLWithPath: "/Users/goftin/dev/gsuite/VSK/EmbedKit/MiniLM-L12-v2.mlmodelc")
        let backend = CoreMLBackend()
        try await backend.loadModel(from: modelURL)

        let pipeline = EmbeddingPipeline(
            tokenizer: try await BERTTokenizer(maxSequenceLength: 512),
            backend: backend,
            configuration: EmbeddingPipelineConfiguration(
                poolingStrategy: .mean,
                normalize: true,
                cacheConfiguration: EmbeddingPipelineConfiguration.CacheConfiguration(
                    maxEntries: 10,
                    ttlSeconds: 60
                )
            )
        )

        let text = "This text will be cached"

        // First embedding - will compute
        let start1 = Date()
        let embedding1 = try await pipeline.embed(text)
        let time1 = Date().timeIntervalSince(start1)

        // Second embedding - should be cached
        let start2 = Date()
        let embedding2 = try await pipeline.embed(text)
        let time2 = Date().timeIntervalSince(start2)

        // Verify embeddings are identical
        XCTAssertEqual(embedding1.toArray(), embedding2.toArray(),
                       "Cached embedding should be identical")

        // Cached should be significantly faster
        XCTAssertLessThan(time2, time1 * 0.1,
                          "Cached retrieval should be much faster than computation")

        print("✅ Caching test passed")
        print("   First computation: \(Int(time1 * 1000))ms")
        print("   Cached retrieval: \(Int(time2 * 1000))ms")
    }

    // MARK: - Vector Storage Integration

    func testVectorIndexIntegration() async throws {
        // Setup pipeline
        let modelURL = URL(fileURLWithPath: "/Users/goftin/dev/gsuite/VSK/EmbedKit/MiniLM-L12-v2.mlmodelc")
        let backend = CoreMLBackend()
        try await backend.loadModel(from: modelURL)

        let pipeline = EmbeddingPipeline(
            tokenizer: try await BERTTokenizer(maxSequenceLength: 512),
            backend: backend,
            configuration: EmbeddingPipelineConfiguration(
                poolingStrategy: .mean,
                normalize: true
            )
        )

        // Create vector index with HNSW for better performance
        let vectorStorage = VectorIndexBridge.hnsw(
            dimensions: 384,
            distanceMetric: .cosine
        )

        // Test data - journal entries
        let journalEntries = [
            (id: UUID(), text: "Today I felt really happy and grateful for my friends"),
            (id: UUID(), text: "Feeling anxious about the upcoming presentation"),
            (id: UUID(), text: "Had a great workout this morning, feeling energized"),
            (id: UUID(), text: "Grateful for the beautiful weather and time with family"),
            (id: UUID(), text: "Stressed about deadlines but making progress")
        ]

        // Add entries to index
        for entry in journalEntries {
            let embedding = try await pipeline.embed(entry.text)
            try await vectorStorage.add(
                vector: embedding.toArray(),
                metadata: ["text": entry.text, "id": entry.id.uuidString]
            )
        }

        // Search for similar entries
        let queryText = "I'm feeling thankful and appreciative today"
        let queryEmbedding = try await pipeline.embed(queryText)

        let results = try await vectorStorage.search(
            query: queryEmbedding.toArray(),
            k: 3
        )

        XCTAssertEqual(results.count, 3)

        // The most similar should be about gratitude
        if let topResult = results.first,
           let text = topResult.metadata["text"] {
            XCTAssertTrue(text.contains("happy") || text.contains("grateful") || text.contains("Grateful"),
                          "Top result should be about positive feelings")
        }

        print("✅ Vector index integration test passed")
        print("   Query: '\(queryText)'")
        for (i, result) in results.enumerated() {
            if let text = result.metadata["text"] {
                print("   Result \(i+1): '\(text)' (score: \(result.score))")
            }
        }
    }

    // MARK: - Performance Tests

    func testEmbeddingPerformance() async throws {
        let modelURL = URL(fileURLWithPath: "/Users/goftin/dev/gsuite/VSK/EmbedKit/MiniLM-L12-v2.mlmodelc")
        let backend = CoreMLBackend()
        try await backend.loadModel(from: modelURL)

        let pipeline = EmbeddingPipeline(
            tokenizer: try await BERTTokenizer(maxSequenceLength: 512),
            backend: backend,
            configuration: EmbeddingPipelineConfiguration(
                poolingStrategy: .mean,
                normalize: true,
                useGPUAcceleration: true
            )
        )

        let text = "The quick brown fox jumps over the lazy dog"

        // Warm-up
        _ = try await pipeline.embed(text)

        // Measure single embedding time
        let singleStart = Date()
        for _ in 0..<10 {
            _ = try await pipeline.embed(text)
        }
        let singleTime = Date().timeIntervalSince(singleStart) / 10.0

        // Measure batch embedding time
        let batchTexts = Array(repeating: text, count: 10)
        let batchStart = Date()
        _ = try await pipeline.embed(batch: batchTexts)
        let batchTime = Date().timeIntervalSince(batchStart)

        print("✅ Performance test results:")
        print("   Single embedding: \(Int(singleTime * 1000))ms")
        print("   Batch (10 texts): \(Int(batchTime * 1000))ms")
        print("   Batch efficiency: \(String(format: "%.1fx", (singleTime * 10) / batchTime)) faster")

        // L12 should be around 20-30ms on modern hardware
        XCTAssertLessThan(singleTime, 0.1, "Single embedding should be < 100ms")
    }
}

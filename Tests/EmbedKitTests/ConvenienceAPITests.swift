// Tests for ConvenienceAPI
import Testing
import Foundation
@testable import EmbedKit

@Suite("Convenience API")
struct ConvenienceAPITests {

    // MARK: - Quick Embed Tests

    @Test
    func quickEmbed_returnsVector() async throws {
        let manager = ModelManager()
        let vector = try await manager.quickEmbed("Hello world")

        #expect(!vector.isEmpty)
        #expect(vector.count == 384)  // Mock model default dimensions
        #expect(!vector.contains(Float.nan))
    }

    // MARK: - Semantic Search Tests

    @Test
    func semanticSearch_returnsResults() async throws {
        let manager = ModelManager()
        let documents = [
            "Swift programming language",
            "Python is versatile",
            "Machine learning basics",
            "Web development with JavaScript"
        ]

        let results = try await manager.semanticSearch(
            query: "programming",
            in: documents,
            topK: 2
        )

        #expect(results.count <= 2)
        for result in results {
            #expect(result.index >= 0 && result.index < documents.count)
            #expect(result.score >= -1.0 && result.score <= 1.0)
        }
    }

    @Test
    func semanticSearch_emptyDocuments_returnsEmpty() async throws {
        let manager = ModelManager()
        let results = try await manager.semanticSearch(
            query: "test",
            in: [],
            topK: 5
        )
        #expect(results.isEmpty)
    }

    @Test
    func semanticSearch_topKGreaterThanDocs() async throws {
        let manager = ModelManager()
        let documents = ["doc1", "doc2", "doc3"]

        let results = try await manager.semanticSearch(
            query: "test",
            in: documents,
            topK: 10
        )

        #expect(results.count == 3)
    }

    @Test
    func semanticSearch_resultsAreSorted() async throws {
        let manager = ModelManager()
        let documents = ["a", "b", "c", "d", "e"]

        let results = try await manager.semanticSearch(
            query: "test",
            in: documents,
            topK: 5
        )

        // Results should be sorted by descending score
        for i in 1..<results.count {
            #expect(results[i-1].score >= results[i].score)
        }
    }

    // MARK: - Find Most Similar Tests

    @Test
    func findMostSimilar_returnsBestMatch() async throws {
        let manager = ModelManager()
        let documents = ["apple fruit", "programming code", "ocean water"]

        let result = try await manager.findMostSimilar(
            query: "fruit",
            in: documents
        )

        #expect(result != nil)
        if let r = result {
            #expect(r.index >= 0 && r.index < documents.count)
            #expect(r.document == documents[r.index])
            #expect(r.score >= -1.0 && r.score <= 1.0)
        }
    }

    @Test
    func findMostSimilar_emptyDocuments_returnsNil() async throws {
        let manager = ModelManager()
        let result = try await manager.findMostSimilar(query: "test", in: [])
        #expect(result == nil)
    }

    // MARK: - Clustering Tests

    @Test
    func clusterDocuments_createsClusters() async throws {
        let manager = ModelManager()
        let documents = ["a", "b", "c", "d", "e", "f"]

        let clusters = try await manager.clusterDocuments(
            documents,
            numberOfClusters: 2
        )

        // Should have at most 2 clusters
        #expect(clusters.count <= 2)

        // All document indices should be covered
        let allIndices = Set(clusters.flatMap { $0 })
        #expect(allIndices.count == documents.count)

        // Each index should appear exactly once
        let flatIndices = clusters.flatMap { $0 }
        #expect(Set(flatIndices).count == flatIndices.count)
    }

    @Test
    func clusterDocuments_emptyInput_returnsEmpty() async throws {
        let manager = ModelManager()
        let clusters = try await manager.clusterDocuments([], numberOfClusters: 3)
        #expect(clusters.isEmpty)
    }

    @Test
    func clusterDocuments_kZero_returnsAllInOne() async throws {
        let manager = ModelManager()
        let documents = ["a", "b", "c"]

        let clusters = try await manager.clusterDocuments(documents, numberOfClusters: 0)

        #expect(clusters.count == 1)
        #expect(clusters[0].count == 3)
    }

    @Test
    func clusterDocuments_kGreaterThanDocs() async throws {
        let manager = ModelManager()
        let documents = ["a", "b", "c"]

        let clusters = try await manager.clusterDocuments(documents, numberOfClusters: 10)

        // Each doc should be in its own cluster
        #expect(clusters.count == 3)
        for cluster in clusters {
            #expect(cluster.count == 1)
        }
    }

    // MARK: - Similarity Matrix Tests

    @Test
    func similarityMatrix_correctDimensions() async throws {
        let manager = ModelManager()
        let documents = ["a", "b", "c", "d"]

        let matrix = try await manager.similarityMatrix(documents)

        #expect(matrix.count == 4)
        for row in matrix {
            #expect(row.count == 4)
        }
    }

    @Test
    func similarityMatrix_diagonalIsOne() async throws {
        let manager = ModelManager()
        let documents = ["test1", "test2", "test3"]

        let matrix = try await manager.similarityMatrix(documents)

        // Self-similarity should be ~1.0
        for i in 0..<matrix.count {
            #expect(abs(matrix[i][i] - 1.0) < 0.01)
        }
    }

    @Test
    func similarityMatrix_isSymmetric() async throws {
        let manager = ModelManager()
        let documents = ["alpha", "beta", "gamma"]

        let matrix = try await manager.similarityMatrix(documents)

        for i in 0..<matrix.count {
            for j in 0..<matrix.count {
                #expect(abs(matrix[i][j] - matrix[j][i]) < 1e-5)
            }
        }
    }

    @Test
    func similarityMatrix_emptyInput_returnsEmpty() async throws {
        let manager = ModelManager()
        let matrix = try await manager.similarityMatrix([])
        #expect(matrix.isEmpty)
    }

    // MARK: - Embed Sequence Tests

    @Test
    func embedSequence_processesAllTexts() async throws {
        let manager = ModelManager()
        let texts = ["one", "two", "three", "four"]

        // embedSequence is actor-isolated, so access it properly
        let stream = await manager.embedSequence(texts)

        var count = 0
        for try await embedding in stream {
            #expect(!embedding.vector.isEmpty)
            count += 1
        }

        #expect(count == 4)
    }

    // MARK: - Embedding Array Extensions Tests

    @Test
    func averaged_computesAverage() async throws {
        let metadata = EmbeddingMetadata(
            modelID: ModelID(provider: "test", name: "test", version: "1"),
            tokenCount: 5,
            processingTime: 0.01,
            normalized: true
        )
        let emb1 = Embedding(vector: [1, 0, 0], metadata: metadata)
        let emb2 = Embedding(vector: [0, 1, 0], metadata: metadata)
        let emb3 = Embedding(vector: [0, 0, 1], metadata: metadata)

        let embeddings = [emb1, emb2, emb3]
        let avg = embeddings.averaged(normalize: false)

        #expect(avg != nil)
        if let a = avg {
            // Average should be [1/3, 1/3, 1/3]
            let expected: Float = 1.0 / 3.0
            for v in a.vector {
                #expect(abs(v - expected) < 0.01)
            }
        }
    }

    @Test
    func averaged_normalized() async throws {
        let metadata = EmbeddingMetadata(
            modelID: ModelID(provider: "test", name: "test", version: "1"),
            tokenCount: 5,
            processingTime: 0.01,
            normalized: true
        )
        let emb1 = Embedding(vector: [2, 0], metadata: metadata)
        let emb2 = Embedding(vector: [0, 2], metadata: metadata)

        let embeddings = [emb1, emb2]
        let avg = embeddings.averaged(normalize: true)

        #expect(avg != nil)
        if let a = avg {
            let mag = sqrt(a.vector.reduce(0) { $0 + $1 * $1 })
            #expect(abs(mag - 1.0) < 0.01)
        }
    }

    @Test
    func averaged_emptyArray_returnsNil() async throws {
        let embeddings: [Embedding] = []
        let avg = embeddings.averaged()
        #expect(avg == nil)
    }

    @Test
    func mostSimilar_findsBestMatch() async throws {
        let metadata = EmbeddingMetadata(
            modelID: ModelID(provider: "test", name: "test", version: "1"),
            tokenCount: 5,
            processingTime: 0.01,
            normalized: true
        )
        let target = Embedding(vector: [1, 0, 0], metadata: metadata)
        let emb1 = Embedding(vector: [0.9, 0.1, 0], metadata: metadata)
        let emb2 = Embedding(vector: [0, 1, 0], metadata: metadata)
        let emb3 = Embedding(vector: [0, 0, 1], metadata: metadata)

        let embeddings = [emb1, emb2, emb3]
        let result = embeddings.mostSimilar(to: target)

        #expect(result != nil)
        if let r = result {
            #expect(r.index == 0)  // emb1 should be most similar to target
        }
    }

    @Test
    func mostSimilar_emptyArray_returnsNil() async throws {
        let metadata = EmbeddingMetadata(
            modelID: ModelID(provider: "test", name: "test", version: "1"),
            tokenCount: 5,
            processingTime: 0.01,
            normalized: true
        )
        let target = Embedding(vector: [1, 0, 0], metadata: metadata)
        let embeddings: [Embedding] = []

        let result = embeddings.mostSimilar(to: target)
        #expect(result == nil)
    }
}

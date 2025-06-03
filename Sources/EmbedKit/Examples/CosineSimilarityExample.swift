import Foundation

/// Example demonstrating Metal-accelerated cosine similarity calculations
public struct CosineSimilarityExample {
    
    public static func runExample() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            print("Metal not available on this device")
            return
        }
        
        print("=== Metal Cosine Similarity Example ===\n")
        
        // Example 1: Single vector pair similarity
        print("1. Single Vector Pair Similarity:")
        let vector1: [Float] = [1.0, 2.0, 3.0, 4.0]
        let vector2: [Float] = [0.5, 1.0, 1.5, 2.0]  // Half of vector1
        
        let similarity = try await accelerator.cosineSimilarity(vector1, vector2)
        print("   Similarity between vectors: \(similarity)")
        print("   (Expected: 1.0 for perfectly aligned vectors)\n")
        
        // Example 2: Query vs multiple keys
        print("2. Query vs Multiple Keys:")
        let query: [Float] = [1.0, 0.0, 0.0, 0.0]
        let keys: [[Float]] = [
            [1.0, 0.0, 0.0, 0.0],  // Identical to query
            [0.0, 1.0, 0.0, 0.0],  // Orthogonal
            [-1.0, 0.0, 0.0, 0.0], // Opposite
            [0.707, 0.707, 0.0, 0.0] // 45 degrees
        ]
        
        let similarities = try await accelerator.cosineSimilarity(query: query, keys: keys)
        print("   Query similarities: \(similarities)")
        print("   (Expected: [1.0, 0.0, -1.0, 0.707])\n")
        
        // Example 3: Batch processing for performance
        print("3. Batch Processing Performance:")
        let batchSize = 1000
        let dimensions = 768  // Common embedding dimension
        
        // Generate random vector pairs
        var vectorPairs: [([Float], [Float])] = []
        for _ in 0..<batchSize {
            let vectorA = (0..<dimensions).map { _ in Float.random(in: -1...1) }
            let vectorB = (0..<dimensions).map { _ in Float.random(in: -1...1) }
            vectorPairs.append((vectorA, vectorB))
        }
        
        // Measure batch performance
        let batchStart = CFAbsoluteTimeGetCurrent()
        let batchResults = try await accelerator.cosineSimilarityBatch(vectorPairs)
        let batchTime = CFAbsoluteTimeGetCurrent() - batchStart
        
        // Measure sequential performance
        let sequentialStart = CFAbsoluteTimeGetCurrent()
        var sequentialResults: [Float] = []
        for (vectorA, vectorB) in vectorPairs.prefix(100) { // Test subset for speed
            let similarity = try await accelerator.cosineSimilarity(vectorA, vectorB)
            sequentialResults.append(similarity)
        }
        let sequentialTime = (CFAbsoluteTimeGetCurrent() - sequentialStart) * 10 // Extrapolate
        
        print("   Batch processing time: \(String(format: "%.3f", batchTime))s")
        print("   Sequential time (estimated): \(String(format: "%.3f", sequentialTime))s")
        print("   Speedup: \(String(format: "%.1fx", sequentialTime/batchTime))")
        print("   Average similarity: \(batchResults.reduce(0, +) / Float(batchResults.count))\n")
        
        // Example 4: Similarity matrix for clustering
        print("4. Similarity Matrix for Clustering:")
        let documents: [[Float]] = [
            [1.0, 0.0, 0.0, 0.0],  // Document A
            [0.9, 0.1, 0.0, 0.0],  // Similar to A
            [0.0, 1.0, 0.0, 0.0],  // Document B
            [0.0, 0.9, 0.1, 0.0],  // Similar to B
            [0.5, 0.5, 0.0, 0.0]   // Between A and B
        ]
        
        let matrix = try await accelerator.cosineSimilarityMatrix(
            queries: documents,
            keys: documents
        )
        
        print("   Similarity Matrix:")
        for (i, row) in matrix.enumerated() {
            let formattedRow = row.map { String(format: "%.2f", $0) }.joined(separator: " ")
            print("   Doc \(i): [\(formattedRow)]")
        }
        
        print("\n=== Example Complete ===")
    }
    
    /// Example: Finding similar embeddings in a database
    public static func semanticSearchExample() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            print("Metal not available")
            return
        }
        
        print("=== Semantic Search Example ===\n")
        
        // Simulated embedding database
        let embeddingDatabase: [[Float]] = (0..<10000).map { _ in
            (0..<384).map { _ in Float.random(in: -1...1) }
        }
        
        // Query embedding
        let queryEmbedding = (0..<384).map { _ in Float.random(in: -1...1) }
        
        print("Searching \(embeddingDatabase.count) embeddings...")
        
        let searchStart = CFAbsoluteTimeGetCurrent()
        let similarities = try await accelerator.cosineSimilarity(
            query: queryEmbedding,
            keys: embeddingDatabase
        )
        let searchTime = CFAbsoluteTimeGetCurrent() - searchStart
        
        // Find top 5 most similar
        let indexed = similarities.enumerated().map { ($0.offset, $0.element) }
        let topResults = indexed.sorted { $0.1 > $1.1 }.prefix(5)
        
        print("Search completed in \(String(format: "%.3f", searchTime))s")
        print("\nTop 5 results:")
        for (index, similarity) in topResults {
            print("  Index: \(index), Similarity: \(String(format: "%.4f", similarity))")
        }
        
        print("\n=== Search Complete ===")
    }
}
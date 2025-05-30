import Foundation
import PipelineKit

/// Example demonstrating real-world usage of EmbedKit's PipelineKit integration
public struct PipelineIntegrationUsage {
    
    /// Example: Semantic Search Application
    public static func semanticSearchExample() async throws {
        print("🔍 Semantic Search Application Example\n")
        
        // 1. Initialize the pipeline
        let (pipeline, cleanup) = try await PipelineIntegration.quickStart(
            modelIdentifier: "all-MiniLM-L6-v2"
        )
        
        defer {
            Task { await cleanup() }
        }
        
        // 2. Index documents
        let documents = [
            (id: "doc1", text: "Machine learning is a subset of artificial intelligence."),
            (id: "doc2", text: "Deep learning uses neural networks with multiple layers."),
            (id: "doc3", text: "Natural language processing helps computers understand human language."),
            (id: "doc4", text: "Computer vision enables machines to interpret visual information."),
            (id: "doc5", text: "Reinforcement learning trains agents through rewards and penalties.")
        ]
        
        print("📚 Indexing \(documents.count) documents...")
        
        var documentEmbeddings: [(id: String, embedding: EmbeddingVector)] = []
        
        // Batch embed all documents
        let texts = documents.map { $0.text }
        let batchResult = try await pipeline.embedBatch(texts, batchSize: 10)
        
        for (index, doc) in documents.enumerated() {
            documentEmbeddings.append((
                id: doc.id,
                embedding: batchResult.embeddings[index]
            ))
        }
        
        print("✅ Indexed \(documentEmbeddings.count) documents")
        print("⏱️ Average embedding time: \(String(format: "%.3f", batchResult.averageDuration))s")
        
        // 3. Search queries
        let queries = [
            "What is neural network architecture?",
            "How do computers see images?",
            "Training AI with rewards"
        ]
        
        print("\n🔎 Processing search queries:")
        
        for query in queries {
            print("\nQuery: '\(query)'")
            
            // Embed the query
            let queryResult = try await pipeline.embed(query)
            let queryEmbedding = queryResult.embedding
            
            // Find most similar documents
            var similarities: [(id: String, score: Float)] = []
            
            for docEmbed in documentEmbeddings {
                let similarity = queryEmbedding.cosineSimilarity(with: docEmbed.embedding)
                similarities.append((id: docEmbed.id, score: similarity))
            }
            
            // Sort by similarity and show top results
            similarities.sort { $0.score > $1.score }
            
            print("Top matches:")
            for (rank, result) in similarities.prefix(3).enumerated() {
                let doc = documents.first { $0.id == result.id }!
                print("  \(rank + 1). [\(result.id)] (score: \(String(format: "%.3f", result.score)))")
                print("     \(doc.text)")
            }
        }
        
        // 4. Show cache statistics
        let cacheStats = await pipeline.getCacheStatistics()
        print("\n📊 Cache Statistics:")
        print("   Hit rate: \(String(format: "%.1f%%", cacheStats.hitRate * 100))")
        print("   Total hits: \(cacheStats.hits)")
        print("   Total misses: \(cacheStats.misses)")
    }
    
    /// Example: Real-time Document Processing Pipeline
    public static func documentProcessingExample() async throws {
        print("\n📄 Document Processing Pipeline Example\n")
        
        // Create a high-performance pipeline
        let modelManager = EmbeddingModelManager()
        let embedder = try await modelManager.loadModel(
            identifier: "all-MiniLM-L6-v2",
            configuration: EmbeddingConfiguration(
                batchSize: 64,
                useGPUAcceleration: true
            )
        )
        
        let pipeline = try await EmbeddingPipelineFactory.highPerformance(
            embedder: embedder,
            modelManager: modelManager
        )
        
        // Simulate incoming document stream
        let documentStream = createDocumentStream(count: 50, delay: 0.1)
        
        print("🌊 Processing document stream...")
        print("📈 Monitoring throughput...\n")
        
        var processedCount = 0
        var totalBytes = 0
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Process documents as they arrive
        let embeddingStream = try await pipeline.streamEmbeddings(
            from: documentStream,
            maxConcurrency: 20,
            bufferSize: 100
        )
        
        var embeddings: [(text: String, embedding: EmbeddingVector)] = []
        
        for try await result in embeddingStream {
            processedCount += 1
            totalBytes += result.text.count
            embeddings.append((text: result.text, embedding: result.embedding))
            
            // Print progress
            if processedCount % 10 == 0 {
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let docsPerSecond = Double(processedCount) / elapsed
                let mbPerSecond = Double(totalBytes) / elapsed / 1_000_000
                
                print("⚡ Processed: \(processedCount) docs | " +
                      "\(String(format: "%.1f", docsPerSecond)) docs/s | " +
                      "\(String(format: "%.2f", mbPerSecond)) MB/s")
            }
        }
        
        let totalTime = CFAbsoluteTimeGetCurrent() - startTime
        
        print("\n✅ Stream processing complete")
        print("📊 Final Statistics:")
        print("   Documents: \(processedCount)")
        print("   Total size: \(totalBytes / 1024) KB")
        print("   Total time: \(String(format: "%.2f", totalTime))s")
        print("   Throughput: \(String(format: "%.1f", Double(processedCount) / totalTime)) docs/s")
        
        // Perform clustering on embeddings
        print("\n🎯 Clustering documents...")
        let clusters = performSimpleClustering(embeddings: embeddings, k: 3)
        
        for (index, cluster) in clusters.enumerated() {
            print("\nCluster \(index + 1) (\(cluster.count) documents):")
            for doc in cluster.prefix(3) {
                print("  - \(doc.prefix(60))...")
            }
        }
    }
    
    /// Example: Multi-model Comparison
    public static func multiModelComparisonExample() async throws {
        print("\n🤖 Multi-Model Comparison Example\n")
        
        let modelManager = EmbeddingModelManager()
        let models = [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "sentence-transformers/all-distilroberta-v1"
        ]
        
        // Test texts representing different domains
        let testTexts = [
            "The quantum computer uses superposition to process information.",
            "El gato está durmiendo en el sofá.", // Spanish
            "import numpy as np\nmodel = Sequential()", // Code
            "The stock market showed significant volatility today.",
            "🚀 To the moon! 💎🙌" // Emojis
        ]
        
        var results: [String: [(text: String, embedding: EmbeddingVector, time: TimeInterval)]] = [:]
        
        for modelId in models {
            print("\n📊 Testing model: \(modelId)")
            
            do {
                // Create pipeline for this model
                let embedder = try await modelManager.loadModel(
                    identifier: modelId,
                    configuration: EmbeddingConfiguration()
                )
                
                let pipeline = try await EmbeddingPipelineFactory.minimal(
                    embedder: embedder,
                    modelManager: modelManager
                )
                
                var modelResults: [(String, EmbeddingVector, TimeInterval)] = []
                
                // Embed each test text
                for text in testTexts {
                    let startTime = CFAbsoluteTimeGetCurrent()
                    let result = try await pipeline.embed(text, useCache: false)
                    let duration = CFAbsoluteTimeGetCurrent() - startTime
                    
                    modelResults.append((text, result.embedding, duration))
                    print("  ✓ Embedded text \(testTexts.firstIndex(of: text)! + 1) " +
                          "(\(result.embedding.dimensions) dims, \(String(format: "%.3f", duration))s)")
                }
                
                results[modelId] = modelResults
                
                // Cleanup
                _ = try await pipeline.unloadModel()
                
            } catch {
                print("  ❌ Failed to load model: \(error.localizedDescription)")
            }
        }
        
        // Compare results
        print("\n📊 Model Comparison Summary:")
        print("┌─────────────────────────┬──────────┬────────────┬─────────────┐")
        print("│ Model                   │ Dims     │ Avg Time   │ Status      │")
        print("├─────────────────────────┼──────────┼────────────┼─────────────┤")
        
        for model in models {
            if let modelResults = results[model] {
                let dims = modelResults.first?.1.dimensions ?? 0
                let avgTime = modelResults.map { $0.2 }.reduce(0, +) / Double(modelResults.count)
                let modelName = String(model.prefix(23)).padding(toLength: 23, withPad: " ", startingAt: 0)
                print("│ \(modelName) │ \(String(format: "%8d", dims)) │ \(String(format: "%8.3fs", avgTime)) │ ✅ Success  │")
            } else {
                let modelName = String(model.prefix(23)).padding(toLength: 23, withPad: " ", startingAt: 0)
                print("│ \(modelName) │     N/A  │      N/A   │ ❌ Failed   │")
            }
        }
        print("└─────────────────────────┴──────────┴────────────┴─────────────┘")
        
        // Cross-model similarity comparison
        if results.count >= 2 {
            print("\n🔗 Cross-Model Similarity (for first test text):")
            let modelIds = Array(results.keys)
            
            for i in 0..<modelIds.count {
                for j in i+1..<modelIds.count {
                    if let embed1 = results[modelIds[i]]?.first?.1,
                       let embed2 = results[modelIds[j]]?.first?.1,
                       embed1.dimensions == embed2.dimensions {
                        let similarity = embed1.cosineSimilarity(with: embed2)
                        print("  \(modelIds[i]) vs \(modelIds[j]): \(String(format: "%.3f", similarity))")
                    }
                }
            }
        }
    }
    
    // MARK: - Helper Functions
    
    private static func createDocumentStream(count: Int, delay: TimeInterval) -> ArrayTextSource {
        let documents = (1...count).map { i in
            let topics = ["technology", "science", "business", "health", "education"]
            let topic = topics[i % topics.count]
            
            return "Document \(i): This is a \(topic) article discussing various aspects of \(topic). " +
                   "It contains detailed information about recent developments in the field of \(topic). " +
                   "The document explores how \(topic) impacts our daily lives and future prospects."
        }
        
        return ArrayTextSource(documents)
    }
    
    private static func performSimpleClustering(
        embeddings: [(text: String, embedding: EmbeddingVector)],
        k: Int
    ) -> [[String]] {
        // Simple k-means style clustering (simplified for example)
        var clusters: [[String]] = Array(repeating: [], count: k)
        
        // Randomly assign initial clusters
        for (index, item) in embeddings.enumerated() {
            clusters[index % k].append(item.text)
        }
        
        return clusters
    }
    
    // MARK: - Run All Examples
    
    public static func runAllExamples() async throws {
        print("🚀 EmbedKit Pipeline Integration Usage Examples\n")
        print("=" * 60)
        
        do {
            try await semanticSearchExample()
            print("\n" + "=" * 60)
            
            try await documentProcessingExample()
            print("\n" + "=" * 60)
            
            try await multiModelComparisonExample()
            print("\n" + "=" * 60)
            
            print("\n✅ All usage examples completed successfully!")
            
        } catch {
            print("\n❌ Example failed: \(error)")
            throw error
        }
    }
}

// String extension for demo formatting
private extension String {
    static func *(lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}
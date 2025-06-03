import Foundation
import PipelineKit

/// Example demonstrating basic EmbedKit usage
public struct EmbedKitExample {
    
    /// Example 1: Direct embedding using TextEmbedder
    public static func directEmbeddingExample() async throws {
        // Create a text embedder
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: .miniLM_L6_v2,
            configuration: Configuration(
                model: ModelConfiguration(
                    identifier: .miniLM_L6_v2,
                    maxSequenceLength: 128,
                    normalizeEmbeddings: true,
                    poolingStrategy: .mean
                )
            )
        )
        
        // Load the model
        try await embedder.loadModel()
        
        // Generate embeddings
        let text = "The quick brown fox jumps over the lazy dog"
        let embedding = try await embedder.embed(text)
        
        print("Generated embedding with \(embedding.dimensions) dimensions")
        
        // Batch embedding
        let texts = [
            "Machine learning is fascinating",
            "Natural language processing enables understanding",
            "Embeddings capture semantic meaning"
        ]
        
        let embeddings = try await embedder.embed(batch: texts)
        
        // Calculate similarities
        for i in 0..<embeddings.count {
            for j in (i+1)..<embeddings.count {
                let similarity = embeddings[i].cosineSimilarity(with: embeddings[j])
                print("Similarity between '\(texts[i])' and '\(texts[j])': \(similarity)")
            }
        }
    }
    
    /// Example 2: Using PipelineKit commands
    public static func pipelineCommandExample() async throws {
        // Create model manager and embedder
        let _ = DefaultEmbeddingModelManager()
        let embedder = MockTextEmbedder() // Using mock for example
        
        // Create command handler
        let cache = EmbeddingCache()
        let telemetry = TelemetrySystem()
        let handler = EmbedTextHandler(embedder: embedder, cache: cache, telemetry: telemetry)
        
        // Create and execute command
        let command = EmbedTextCommand(
            text: "Understanding context through embeddings"
        )
        
        let result = try await handler.handle(command)
        print("Command generated embedding with \(result.embedding.dimensions) dimensions")
    }
    
    /// Example 3: Batch processing with commands
    public static func batchCommandExample() async throws {
        // Create components
        let embedder = MockTextEmbedder()
        let cache = EmbeddingCache()
        let telemetry = TelemetrySystem()
        let batchHandler = BatchEmbedHandler(embedder: embedder, cache: cache, telemetry: telemetry)
        
        // Create batch command
        let batchCommand = BatchEmbedCommand(
            texts: [
                "First document about technology",
                "Second document about science", 
                "Third document about innovation"
            ]
        )
        
        let results = try await batchHandler.handle(batchCommand)
        print("Batch processed \(results.embeddings.count) embeddings")
    }
    
    /// Example 4: Streaming embeddings
    public static func streamingExample() async throws {
        let embedder = MockTextEmbedder()
        let cache = EmbeddingCache()
        let telemetry = TelemetrySystem()
        let streamHandler = StreamEmbedHandler(embedder: embedder, cache: cache, telemetry: telemetry)
        
        // Create array of texts for streaming
        let texts = (1...10).map { i in
            "Document \(i): This is sample text for streaming"
        }
        
        // Create stream command
        let streamCommand = StreamEmbedCommand(
            textSource: ArrayTextSource(texts),
            maxConcurrency: 3
        )
        
        // Process stream
        let resultStream = try await streamHandler.handle(streamCommand)
        
        var processedCount = 0
        do {
            for try await result in resultStream {
                processedCount += 1
                print("Processed embedding \(processedCount) with \(result.embedding.dimensions) dimensions")
            }
        } catch {
            print("Stream processing error: \(error)")
        }
    }
    
    /// Example 5: Similarity search
    public static func similaritySearchExample() async throws {
        let embedder = MockTextEmbedder(dimensions: 384)
        try await embedder.loadModel()
        
        // Reference documents
        let documents = [
            "The solar system consists of the Sun and celestial objects",
            "Machine learning algorithms can identify patterns in data",
            "The planets orbit around the Sun in elliptical paths",
            "Neural networks are inspired by biological neurons",
            "Mars is the fourth planet from the Sun"
        ]
        
        // Generate embeddings for documents
        let documentEmbeddings = try await embedder.embed(batch: documents)
        
        // Query
        let query = "Tell me about planets and space"
        let queryEmbedding = try await embedder.embed(query)
        
        // Find most similar documents
        var similarities: [(index: Int, score: Float)] = []
        
        for (index, docEmbedding) in documentEmbeddings.enumerated() {
            let similarity = queryEmbedding.cosineSimilarity(with: docEmbedding)
            similarities.append((index: index, score: similarity))
        }
        
        // Sort by similarity (descending)
        similarities.sort { $0.score > $1.score }
        
        print("\nQuery: '\(query)'")
        print("Most similar documents:")
        for (rank, result) in similarities.prefix(3).enumerated() {
            print("\(rank + 1). '\(documents[result.index])' (score: \(result.score))")
        }
    }
}

// Note: MockTextEmbedder is defined in ComprehensiveBenchmarks.swift
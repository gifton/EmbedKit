import Foundation
import PipelineKit

/// Example demonstrating basic EmbedKit usage
public struct EmbedKitExample {
    
    /// Example 1: Direct embedding using TextEmbedder
    public static func directEmbeddingExample() async throws {
        // Create a text embedder
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: "all-MiniLM-L6-v2",
            configuration: EmbeddingConfiguration(
                maxSequenceLength: 128,
                normalizeEmbeddings: true,
                poolingStrategy: .mean
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
        let handler = EmbedTextHandler(embedder: embedder)
        
        // Create and execute command
        let command = EmbedTextCommand(
            text: "Understanding context through embeddings",
            metadata: ["source": "example"]
        )
        
        let embedding = try await handler.handle(command)
        print("Command generated embedding with \(embedding.dimensions) dimensions")
    }
    
    /// Example 3: Batch processing with commands
    public static func batchCommandExample() async throws {
        // Create components
        let embedder = MockTextEmbedder()
        let batchHandler = EmbedBatchHandler(embedder: embedder)
        
        // Create batch command
        let batchCommand = EmbedBatchCommand(
            texts: [
                "First document about technology",
                "Second document about science", 
                "Third document about innovation"
            ]
        )
        
        let embeddings = try await batchHandler.handle(batchCommand)
        print("Batch processed \(embeddings.count) embeddings")
    }
    
    /// Example 4: Streaming embeddings
    public static func streamingExample() async throws {
        let embedder = MockTextEmbedder()
        let streamHandler = EmbedStreamHandler<AsyncStream<String>>(embedder: embedder)
        
        // Create async sequence of texts
        let texts = AsyncStream<String> { continuation in
            Task {
                for i in 1...10 {
                    continuation.yield("Document \(i): This is sample text for streaming")
                    try await Task.sleep(nanoseconds: 100_000_000) // 0.1 second
                }
                continuation.finish()
            }
        }
        
        // Create stream command
        let streamCommand = EmbedStreamCommand(
            texts: texts,
            maxConcurrency: 3
        )
        
        // Process stream
        let resultStream = try await streamHandler.handle(streamCommand)
        
        var processedCount = 0
        for await result in resultStream {
            switch result {
            case .success(let embedding):
                processedCount += 1
                print("Processed embedding \(processedCount) with \(embedding.dimensions) dimensions")
            case .failure(let error):
                print("Error processing embedding: \(error)")
            }
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

// Mock embedder implementation for examples
actor MockTextEmbedder: TextEmbedder {
    let configuration = EmbeddingConfiguration()
    let dimensions: Int
    let modelIdentifier = "mock-embedder"
    private(set) var isReady = false
    
    init(dimensions: Int = 768) {
        self.dimensions = dimensions
    }
    
    func embed(_ text: String) async throws -> EmbeddingVector {
        guard isReady else { throw EmbeddingError.modelNotLoaded }
        
        // Generate deterministic mock embedding based on text
        var values = [Float](repeating: 0, count: dimensions)
        let hash = text.hashValue
        
        for i in 0..<dimensions {
            values[i] = Float((hash &+ i)) / Float(Int32.max)
        }
        
        // Normalize
        let norm = sqrt(values.reduce(0) { $0 + $1 * $1 })
        if norm > 0 {
            values = values.map { $0 / norm }
        }
        
        return EmbeddingVector(values)
    }
    
    func loadModel() async throws {
        isReady = true
    }
    
    func unloadModel() async throws {
        isReady = false
    }
}
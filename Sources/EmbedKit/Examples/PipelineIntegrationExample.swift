import Foundation
import PipelineKit

/// Example demonstrating EmbedKit and VectorStoreKit integration through PipelineKit
public struct PipelineIntegrationExample {
    
    public static func runExample() async throws {
        print("=== EmbedKit Pipeline Integration Example ===\n")
        
        // Initialize embedder
        let embedder = try await createEmbedder()
        
        // Example 1: Document Indexing Pipeline
        print("1. Document Indexing Pipeline:")
        try await demonstrateDocumentIndexing(embedder: embedder)
        
        // Example 2: Search Pipeline
        print("\n2. Search Pipeline:")
        try await demonstrateSearch(embedder: embedder)
        
        // Example 3: Batch Processing Pipeline
        print("\n3. Batch Processing Pipeline:")
        try await demonstrateBatchProcessing(embedder: embedder)
        
        // Example 4: Streaming Pipeline
        print("\n4. Streaming Pipeline:")
        try await demonstrateStreaming(embedder: embedder)
        
        print("\n=== Example Complete ===")
    }
    
    // MARK: - Helper Functions
    
    private static func createEmbedder() async throws -> TextEmbedder {
        // In a real application, this would load a proper model
        // For demo, using mock embedder
        return MockTextEmbedder(dimensions: 384)
    }
    
    // MARK: - Example 1: Document Indexing
    
    private static func demonstrateDocumentIndexing(embedder: TextEmbedder) async throws {
        // Create handler for document indexing
        let handler = IndexDocumentHandler(embedder: embedder)
        
        // Create pipeline with indexing configuration
        let telemetry = TelemetrySystem()
        let middlewares = EmbeddingPipelineFactory.createIndexingConfiguration(
            embedder: embedder,
            telemetry: telemetry
        )
        
        // Build the pipeline
        var builder = EmbeddingPipelineBuilder(handler: handler)
        for middleware in middlewares {
            builder = builder.addMiddleware(middleware)
        }
        let pipeline = try await builder.build()
        
        // The handler implementation
        struct IndexDocumentHandler: CommandHandler {
            let embedder: TextEmbedder
            
            func handle(_ command: IndexDocumentCommand) async throws -> IndexResult {
                // Embed the document
                let embedding = try await embedder.embed(command.document.text)
                
                // In a real app, this would store in vector database
                print("   - Indexed document '\(command.document.id)' with embedding dimension \(embedding.dimensions)")
                
                return IndexResult(
                    documentId: command.document.id,
                    embedding: embedding,
                    duration: 0.1
                )
            }
        }
        
        // Create and index a document
        let document = EmbeddingDocument(
            id: "doc-001",
            text: "EmbedKit provides high-performance text embeddings for iOS and macOS applications.",
            metadata: ["category": "documentation", "version": "1.0"]
        )
        
        let command = IndexDocumentCommand(document: document)
        let metadata = DefaultCommandMetadata()
        let result = try await pipeline.execute(command, metadata: metadata)
        
        print("   - Successfully indexed in \(String(format: "%.3f", result.duration))s")
    }
    
    // MARK: - Example 2: Search
    
    private static func demonstrateSearch(embedder: TextEmbedder) async throws {
        // Create handler for search operations
        let handler = SearchDocumentsHandler(embedder: embedder)
        
        // Get search configuration from factory
        let telemetry = TelemetrySystem()
        let middlewares = EmbeddingPipelineFactory.createSearchConfiguration(
            embedder: embedder,
            telemetry: telemetry
        )
        
        // Build the pipeline
        var builder = EmbeddingPipelineBuilder(handler: handler)
        for middleware in middlewares {
            builder = builder.addMiddleware(middleware)
        }
        let pipeline = try await builder.build()
        
        // The handler implementation
        struct SearchDocumentsHandler: CommandHandler {
            let embedder: TextEmbedder
            
            func handle(_ command: SearchDocumentsCommand) async throws -> [SearchResult] {
                // Embed the query
                let queryEmbedding = try await embedder.embed(command.query)
                
                // In a real app, this would search the vector database
                // For demo, create mock results
                let mockResults: [SearchResult] = [
                    SearchResult(
                        document: EmbeddingDocument(
                            id: "doc-001",
                            text: "EmbedKit provides high-performance text embeddings.",
                            metadata: ["category": "documentation"]
                        ),
                        embedding: queryEmbedding,
                        score: 0.95,
                        rank: 1
                    ),
                    SearchResult(
                        document: EmbeddingDocument(
                            id: "doc-002",
                            text: "Vector search enables semantic similarity matching.",
                            metadata: ["category": "tutorial"]
                        ),
                        embedding: queryEmbedding,
                        score: 0.87,
                        rank: 2
                    )
                ]
                
                print("   - Found \(mockResults.count) results for query: '\(command.query)'")
                for result in mockResults {
                    print("     • \(result.document.id): score=\(String(format: "%.3f", result.score))")
                }
                
                return mockResults
            }
        }
        
        // Perform search
        let searchCommand = SearchDocumentsCommand(
            query: "high-performance embeddings",
            k: 5
        )
        
        let metadata = DefaultCommandMetadata()
        let results = try await pipeline.execute(searchCommand, metadata: metadata)
        print("   - Search completed, top result score: \(String(format: "%.3f", results.first?.score ?? 0))")
    }
    
    // MARK: - Example 3: Batch Processing
    
    private static func demonstrateBatchProcessing(embedder: TextEmbedder) async throws {
        // Create handler for batch operations
        let handler = BatchIndexDocumentsHandler(embedder: embedder)
        
        // Get batch configuration from factory
        let telemetry = TelemetrySystem()
        let middlewares = EmbeddingPipelineFactory.createBatchConfiguration(
            embedder: embedder,
            maxConcurrency: 5,
            telemetry: telemetry
        )
        
        // Build the pipeline
        var builder = EmbeddingPipelineBuilder(handler: handler)
        for middleware in middlewares {
            builder = builder.addMiddleware(middleware)
        }
        let pipeline = try await builder.build()
        
        // The handler implementation
        struct BatchIndexDocumentsHandler: CommandHandler {
            let embedder: TextEmbedder
            
            func handle(_ command: BatchIndexDocumentsCommand) async throws -> BatchIndexResult {
                let startTime = Date()
                var successful: [IndexResult] = []
                var failed: [(document: EmbeddingDocument, error: String)] = []
                
                // Process documents with concurrency control
                await withTaskGroup(of: (IndexResult?, EmbeddingDocument, String?).self) { group in
                    for document in command.documents {
                        group.addTask {
                            do {
                                let embedding = try await embedder.embed(document.text)
                                let result = IndexResult(
                                    documentId: document.id,
                                    embedding: embedding,
                                    duration: 0.05
                                )
                                return (result, document, nil)
                            } catch {
                                return (nil, document, error.localizedDescription)
                            }
                        }
                    }
                    
                    for await (result, document, error) in group {
                        if let result = result {
                            successful.append(result)
                        } else if let error = error {
                            failed.append((document, error))
                        }
                    }
                }
                
                let totalDuration = Date().timeIntervalSince(startTime)
                
                print("   - Batch processed: \(successful.count) successful, \(failed.count) failed")
                print("   - Total time: \(String(format: "%.3f", totalDuration))s")
                print("   - Average time per document: \(String(format: "%.3f", totalDuration / Double(command.documents.count)))s")
                
                return BatchIndexResult(
                    successful: successful,
                    failed: failed,
                    totalDuration: totalDuration
                )
            }
        }
        
        // Create batch of documents
        let documents = (1...10).map { i in
            EmbeddingDocument(
                id: "batch-doc-\(i)",
                text: "This is document number \(i) in our batch processing example.",
                metadata: ["batch": "example", "index": "\(i)"]
            )
        }
        
        let batchCommand = BatchIndexDocumentsCommand(
            documents: documents,
            configuration: BatchIndexConfiguration(batchSize: 5, parallelism: 3)
        )
        
        let metadata = DefaultCommandMetadata()
        let result = try await pipeline.execute(batchCommand, metadata: metadata)
        print("   - Success rate: \(String(format: "%.1f%%", result.successRate * 100))")
    }
    
    // MARK: - Example 4: Streaming
    
    private static func demonstrateStreaming(embedder: TextEmbedder) async throws {
        // Create handler for streaming operations
        let handler = StreamIndexDocumentsHandler(embedder: embedder)
        
        // Get streaming configuration from factory
        let telemetry = TelemetrySystem()
        let middlewares = EmbeddingPipelineFactory.createStreamingConfiguration(
            embedder: embedder,
            bufferSize: 50,
            telemetry: telemetry
        )
        
        // Build the pipeline
        var builder = EmbeddingPipelineBuilder(handler: handler)
        for middleware in middlewares {
            builder = builder.addMiddleware(middleware)
        }
        let pipeline = try await builder.build()
        
        // The handler implementation
        struct StreamIndexDocumentsHandler: CommandHandler {
            let embedder: TextEmbedder
            
            func handle(_ command: StreamIndexDocumentsCommand) async throws -> AsyncThrowingStream<IndexResult, Error> {
                AsyncThrowingStream<IndexResult, Error> { continuation in
                    Task {
                        var count = 0
                        
                        do {
                            // Cast to the proper type since documentSource is AsyncDocumentSource
                            for try await document in command.documentSource {
                                if let embeddingDocument = document as? EmbeddingDocument {
                                    let embedding = try await embedder.embed(embeddingDocument.text)
                                    let result = IndexResult(
                                        documentId: embeddingDocument.id,
                                        embedding: embedding,
                                        duration: 0.05
                                    )
                                    
                                    continuation.yield(result)
                                    count += 1
                                    
                                    // Simulate backpressure handling
                                    if count % 10 == 0 {
                                        try await Task.sleep(nanoseconds: 100_000_000) // 0.1s
                                    }
                                }
                            }
                            continuation.finish()
                        } catch {
                            continuation.finish(throwing: error)
                        }
                    }
                }
            }
        }
        
        // Create streaming source
        let documents = (1...25).map { i in
            EmbeddingDocument(
                id: "stream-doc-\(i)",
                text: "Streaming document \(i) demonstrates async processing.",
                metadata: ["stream": "true", "sequence": "\(i)"]
            )
        }
        
        let documentSource = ArrayDocumentSource(documents)
        let streamCommand = StreamIndexDocumentsCommand(documentSource: documentSource)
        
        print("   - Starting document stream...")
        
        let metadata = DefaultCommandMetadata()
        let stream = try await pipeline.execute(streamCommand, metadata: metadata)
        var processedCount = 0
        
        for try await _ in stream {
            processedCount += 1
            if processedCount % 5 == 0 {
                print("   - Processed \(processedCount) documents...")
            }
        }
        
        print("   - Stream completed: \(processedCount) documents processed")
    }
}
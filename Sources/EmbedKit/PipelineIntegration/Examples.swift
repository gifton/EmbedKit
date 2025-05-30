import Foundation
import PipelineKit

// MARK: - Pipeline Integration Examples

/// Examples demonstrating how to use EmbedKit through PipelineKit
public struct PipelineIntegrationExamples {
    
    // MARK: - Basic Usage Example
    
    /// Basic example of embedding text through the pipeline
    public static func basicEmbeddingExample() async throws {
        print("=== Basic Embedding Example ===\n")
        
        // Create model manager and embedder
        let modelManager = EmbeddingModelManager()
        let embedder = try await modelManager.loadModel(
            identifier: "all-MiniLM-L6-v2",
            configuration: EmbeddingConfiguration()
        )
        
        // Create pipeline with default configuration
        let pipeline = try await EmbeddingPipeline(
            embedder: embedder,
            modelManager: modelManager
        )
        
        // Embed a single text
        let text = "The quick brown fox jumps over the lazy dog."
        let result = try await pipeline.embed(text)
        
        print("✅ Text embedded successfully")
        print("📊 Model: \(result.modelIdentifier)")
        print("📏 Dimensions: \(result.embedding.dimensions)")
        print("⏱️ Duration: \(String(format: "%.3f", result.duration))s")
        print("💾 From cache: \(result.fromCache)")
        
        // Embed the same text again (should be cached)
        let cachedResult = try await pipeline.embed(text)
        print("\n✅ Second embedding (cached): \(cachedResult.fromCache)")
    }
    
    // MARK: - Batch Processing Example
    
    /// Example of batch processing with progress monitoring
    public static func batchProcessingExample() async throws {
        print("\n=== Batch Processing Example ===\n")
        
        // Create components
        let modelManager = EmbeddingModelManager()
        let embedder = try await modelManager.loadModel(
            identifier: "all-MiniLM-L6-v2",
            configuration: EmbeddingConfiguration()
        )
        
        // Create pipeline optimized for batch processing
        let pipeline = try await EmbeddingPipelineFactory.highPerformance(
            embedder: embedder,
            modelManager: modelManager
        )
        
        // Prepare batch of texts
        let texts = [
            "Machine learning is transforming the world.",
            "Natural language processing enables computers to understand text.",
            "Deep learning models can generate human-like text.",
            "Embeddings represent text as numerical vectors.",
            "Vector databases store and search embeddings efficiently.",
            "Similarity search finds related documents quickly.",
            "Semantic search understands meaning beyond keywords.",
            "AI assistants help users complete complex tasks.",
            "Large language models process billions of parameters.",
            "Transfer learning adapts models to new domains."
        ]
        
        print("📦 Processing batch of \(texts.count) texts...")
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let batchResult = try await pipeline.embedBatch(texts, batchSize: 5)
        let totalTime = CFAbsoluteTimeGetCurrent() - startTime
        
        print("✅ Batch processing complete")
        print("📊 Total embeddings: \(batchResult.embeddings.count)")
        print("⏱️ Total time: \(String(format: "%.3f", totalTime))s")
        print("⚡ Average time per text: \(String(format: "%.3f", batchResult.averageDuration))s")
        print("💾 Cache hit rate: \(String(format: "%.1f%%", batchResult.cacheHitRate * 100))")
        
        // Show similarity between first and last embedding
        if let first = batchResult.embeddings.first,
           let last = batchResult.embeddings.last {
            let similarity = first.cosineSimilarity(with: last)
            print("🔗 Similarity (first vs last): \(String(format: "%.3f", similarity))")
        }
    }
    
    // MARK: - Streaming Example
    
    /// Example of streaming embeddings for large datasets
    public static func streamingExample() async throws {
        print("\n=== Streaming Example ===\n")
        
        // Create components
        let modelManager = EmbeddingModelManager()
        let embedder = try await modelManager.loadModel(
            identifier: "all-MiniLM-L6-v2",
            configuration: EmbeddingConfiguration()
        )
        
        let pipeline = try await EmbeddingPipelineFactory.balanced(
            embedder: embedder,
            modelManager: modelManager
        )
        
        // Create a mock document source
        let documents = (1...100).map { i in
            "Document \(i): This is a sample document with content about topic \(i % 10). " +
            "It contains information that needs to be embedded for semantic search."
        }
        
        let textSource = ArrayTextSource(documents)
        
        print("🌊 Starting streaming embeddings for \(documents.count) documents...")
        print("⚡ Max concurrency: 10")
        
        var processedCount = 0
        var totalDimensions = 0
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let stream = try await pipeline.streamEmbeddings(
            from: textSource,
            maxConcurrency: 10,
            bufferSize: 1000
        )
        
        for try await result in stream {
            processedCount += 1
            totalDimensions = result.embedding.dimensions
            
            // Print progress every 10 documents
            if processedCount % 10 == 0 {
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let rate = Double(processedCount) / elapsed
                print("📈 Progress: \(processedCount)/\(documents.count) " +
                      "(\(String(format: "%.1f", rate)) docs/sec)")
            }
        }
        
        let totalTime = CFAbsoluteTimeGetCurrent() - startTime
        print("\n✅ Streaming complete")
        print("📊 Total processed: \(processedCount)")
        print("📏 Embedding dimensions: \(totalDimensions)")
        print("⏱️ Total time: \(String(format: "%.3f", totalTime))s")
        print("⚡ Throughput: \(String(format: "%.1f", Double(processedCount) / totalTime)) docs/sec")
    }
    
    // MARK: - Model Management Example
    
    /// Example of model loading, swapping, and management
    public static func modelManagementExample() async throws {
        print("\n=== Model Management Example ===\n")
        
        let modelManager = EmbeddingModelManager()
        let initialEmbedder = try await modelManager.loadModel(
            identifier: "all-MiniLM-L6-v2",
            configuration: EmbeddingConfiguration()
        )
        
        let pipeline = try await EmbeddingPipelineFactory.development(
            embedder: initialEmbedder,
            modelManager: modelManager
        )
        
        // Load initial model
        print("🤖 Loading initial model...")
        let loadResult = try await pipeline.loadModel(
            "all-MiniLM-L6-v2",
            preload: true,
            useGPU: true
        )
        
        print("✅ Model loaded: \(loadResult.modelIdentifier)")
        print("💾 Model size: \(loadResult.modelSize / 1024 / 1024)MB")
        print("⏱️ Load time: \(String(format: "%.3f", loadResult.loadDuration))s")
        
        // Embed some text with the first model
        let testText = "This is a test sentence for model comparison."
        let result1 = try await pipeline.embed(testText)
        print("\n📊 Embedding with model 1:")
        print("   Dimensions: \(result1.embedding.dimensions)")
        
        // Swap to a different model
        print("\n🔄 Swapping to a different model...")
        let swapResult = try await pipeline.swapModel(
            to: "all-mpnet-base-v2",
            unloadCurrent: true,
            warmupAfterSwap: true
        )
        
        print("✅ Model swapped")
        print("   Previous: \(swapResult.previousModel ?? "none")")
        print("   New: \(swapResult.newModel)")
        print("   Swap time: \(String(format: "%.3f", swapResult.swapDuration))s")
        if let warmupTime = swapResult.warmupDuration {
            print("   Warmup time: \(String(format: "%.3f", warmupTime))s")
        }
        
        // Embed with the new model
        let result2 = try await pipeline.embed(testText, useCache: false)
        print("\n📊 Embedding with model 2:")
        print("   Dimensions: \(result2.embedding.dimensions)")
        
        // Compare embeddings (different models will produce different dimensions)
        print("\n🔍 Model comparison:")
        print("   Model 1 dimensions: \(result1.embedding.dimensions)")
        print("   Model 2 dimensions: \(result2.embedding.dimensions)")
        
        // Unload model and clear cache
        print("\n🧹 Cleaning up...")
        let unloadResult = try await pipeline.unloadModel(clearCache: true)
        print("✅ Model unloaded")
        print("   Freed memory: \(unloadResult.freedMemory / 1024 / 1024)MB")
        print("   Cache cleared: \(unloadResult.cacheCleared)")
    }
    
    // MARK: - Error Handling Example
    
    /// Example demonstrating error handling and retry mechanisms
    public static func errorHandlingExample() async throws {
        print("\n=== Error Handling Example ===\n")
        
        let modelManager = EmbeddingModelManager()
        let embedder = try await modelManager.loadModel(
            identifier: "all-MiniLM-L6-v2",
            configuration: EmbeddingConfiguration()
        )
        
        // Create pipeline with rate limiting enabled
        let pipeline = try await EmbeddingPipelineFactory.development(
            embedder: embedder,
            modelManager: modelManager
        )
        
        // Test validation errors
        print("🧪 Testing validation errors...")
        
        do {
            // Empty text should fail validation
            _ = try await pipeline.embed("")
            print("❌ Should have failed validation")
        } catch {
            print("✅ Validation error caught: \(error.localizedDescription)")
        }
        
        // Test with very long text
        let longText = String(repeating: "a", count: 15_000)
        do {
            _ = try await pipeline.embed(longText)
            print("❌ Should have failed validation")
        } catch {
            print("✅ Length validation error caught: \(error.localizedDescription)")
        }
        
        // Test batch validation
        print("\n🧪 Testing batch validation...")
        
        let invalidBatch = ["Valid text", "", "Another valid text"]
        do {
            _ = try await pipeline.embedBatch(invalidBatch)
            print("❌ Should have failed batch validation")
        } catch {
            print("✅ Batch validation error caught: \(error.localizedDescription)")
        }
        
        // Test model not found
        print("\n🧪 Testing model errors...")
        
        do {
            _ = try await pipeline.loadModel("non-existent-model")
            print("❌ Should have failed to load model")
        } catch {
            print("✅ Model load error caught: \(error.localizedDescription)")
        }
    }
    
    // MARK: - Telemetry and Monitoring Example
    
    /// Example showing telemetry and monitoring capabilities
    public static func telemetryExample() async throws {
        print("\n=== Telemetry and Monitoring Example ===\n")
        
        let modelManager = EmbeddingModelManager()
        let embedder = try await modelManager.loadModel(
            identifier: "all-MiniLM-L6-v2",
            configuration: EmbeddingConfiguration()
        )
        
        // Create pipeline with full monitoring
        let pipeline = try await EmbeddingPipelineFactory.development(
            embedder: embedder,
            modelManager: modelManager
        )
        
        // Reset telemetry to start fresh
        await pipeline.resetTelemetry()
        
        // Perform various operations
        print("📊 Performing operations for telemetry...")
        
        // Single embeddings
        for i in 1...5 {
            _ = try await pipeline.embed("Test text \(i)")
        }
        
        // Batch embedding
        let batchTexts = (1...20).map { "Batch text \(i)" }
        _ = try await pipeline.embedBatch(batchTexts, batchSize: 10)
        
        // Some cache hits
        for i in 1...3 {
            _ = try await pipeline.embed("Test text \(i)") // Should hit cache
        }
        
        // Get statistics
        let stats = await pipeline.getStatistics()
        
        print("\n📈 Pipeline Statistics:")
        print("   Cache hits: \(stats.cacheStatistics.hits)")
        print("   Cache misses: \(stats.cacheStatistics.misses)")
        print("   Cache size: \(stats.cacheStatistics.currentSize)")
        print("   Hit rate: \(String(format: "%.1f%%", stats.cacheStatistics.hitRate * 100))")
        print("   Memory usage: \(String(format: "%.1f", stats.systemMetrics.memoryUsage))MB")
        print("   Model ready: \(stats.isReady)")
        print("   Current model: \(stats.currentModel)")
        
        // Export telemetry data
        if let telemetryData = await pipeline.getTelemetryData() {
            print("\n📊 Telemetry data exported: \(telemetryData.count) bytes")
            
            // Parse and display some metrics
            if let export = try? JSONDecoder().decode(MetricsExport.self, from: telemetryData) {
                print("\n📈 Recorded Metrics:")
                print("   Total events: \(export.events.count)")
                print("   Counters: \(export.counters.keys.joined(separator: ", "))")
                
                if let commandsStarted = export.counters["commands.started"] {
                    print("   Commands started: \(commandsStarted)")
                }
                if let commandsSucceeded = export.counters["commands.succeeded"] {
                    print("   Commands succeeded: \(commandsSucceeded)")
                }
            }
        }
    }
    
    // MARK: - Cache Management Example
    
    /// Example demonstrating cache preloading and management
    public static func cacheManagementExample() async throws {
        print("\n=== Cache Management Example ===\n")
        
        let modelManager = EmbeddingModelManager()
        let embedder = try await modelManager.loadModel(
            identifier: "all-MiniLM-L6-v2",
            configuration: EmbeddingConfiguration()
        )
        
        let pipeline = try await EmbeddingPipelineFactory.balanced(
            embedder: embedder,
            modelManager: modelManager
        )
        
        // Clear cache to start fresh
        print("🧹 Clearing cache...")
        let clearResult = try await pipeline.clearCache()
        print("✅ Cleared \(clearResult.entriesCleared) entries")
        
        // Preload frequently used texts
        print("\n📦 Preloading cache with common queries...")
        
        let commonQueries = [
            "What is machine learning?",
            "How does natural language processing work?",
            "Explain deep learning",
            "What are embeddings?",
            "How do vector databases work?",
            "What is semantic search?",
            "Explain transformer models",
            "What is BERT?",
            "How does GPT work?",
            "What is attention mechanism?"
        ]
        
        let preloadResult = try await pipeline.preloadCache(texts: commonQueries)
        print("✅ Preloaded \(preloadResult.textsProcessed) texts")
        print("⏱️ Total time: \(String(format: "%.3f", preloadResult.duration))s")
        print("⚡ Average per text: \(String(format: "%.3f", preloadResult.averageEmbeddingTime))s")
        
        // Test cache hits
        print("\n🧪 Testing cache performance...")
        
        var hits = 0
        var misses = 0
        
        // These should all be cache hits
        for query in commonQueries.prefix(5) {
            let result = try await pipeline.embed(query)
            if result.fromCache {
                hits += 1
            } else {
                misses += 1
            }
        }
        
        // These should be cache misses
        let newQueries = [
            "What is quantum computing?",
            "Explain blockchain technology",
            "How does cryptocurrency work?"
        ]
        
        for query in newQueries {
            let result = try await pipeline.embed(query)
            if result.fromCache {
                hits += 1
            } else {
                misses += 1
            }
        }
        
        print("✅ Cache test results:")
        print("   Hits: \(hits)")
        print("   Misses: \(misses)")
        print("   Hit rate: \(String(format: "%.1f%%", Double(hits) / Double(hits + misses) * 100))")
        
        // Get final cache statistics
        let cacheStats = await pipeline.getCacheStatistics()
        print("\n📊 Final cache statistics:")
        print("   Total entries: \(cacheStats.currentSize)")
        print("   Max size: \(cacheStats.maxSize)")
        print("   Total hits: \(cacheStats.hits)")
        print("   Total misses: \(cacheStats.misses)")
        print("   Overall hit rate: \(String(format: "%.1f%%", cacheStats.hitRate * 100))")
    }
    
    // MARK: - Run All Examples
    
    /// Run all examples in sequence
    public static func runAllExamples() async throws {
        print("🚀 Running EmbedKit Pipeline Integration Examples\n")
        
        do {
            try await basicEmbeddingExample()
            try await batchProcessingExample()
            try await streamingExample()
            try await modelManagementExample()
            try await errorHandlingExample()
            try await telemetryExample()
            try await cacheManagementExample()
            
            print("\n✅ All examples completed successfully!")
        } catch {
            print("\n❌ Example failed: \(error)")
            throw error
        }
    }
}
import Foundation

/// Example implementation showing expressive logging integration
public actor ExpressiveLoggingExample {
    private let logger = EmbedKitLogger.embeddings()
    private let cache = LRUCache<String, [Float]>(maxSize: 100)
    
    /// Example embedding operation with rich logging
    public func embedTextWithLogging(_ text: String) async throws -> [Float] {
        let _ = LogContext(
            requestId: UUID().uuidString,
            operation: "text_embedding",
            metadata: ["text_length": "\(text.count)"]
        )
        
        // Log operation start
        logger.start("Text embedding", details: "Length: \(text.count) characters")
        
        // Check cache with logging
        let cacheKey = "embed_\(text.hashValue)"
        if let cached = await cache.get(cacheKey) {
            logger.cache("Cache HIT", hitRate: 0.85, size: await cache.count)
            return cached
        }
        
        logger.cache("Cache MISS", size: await cache.count)
        
        // Simulate embedding with progress
        logger.processing("text tokenization", progress: 0.2)
        try await Task.sleep(nanoseconds: 100_000_000) // 0.1s
        
        logger.processing("neural network inference", progress: 0.6)
        try await Task.sleep(nanoseconds: 200_000_000) // 0.2s
        
        logger.processing("post-processing", progress: 0.9)
        
        // Generate mock embedding
        let embedding = (0..<768).map { _ in Float.random(in: -1...1) }
        
        // Store in cache
        await cache.set(cacheKey, value: embedding)
        
        // Log completion with performance metrics
        logger.complete("Text embedding", result: "768-dimensional vector generated")
        logger.performance("Embedding generation", duration: 0.3, throughput: 3.33)
        
        return embedding
    }
    
    /// Example batch processing with logging
    public func processBatchWithLogging(_ texts: [String]) async throws {
        logger.start("Batch processing", details: "\(texts.count) documents")
        
        let startTime = CFAbsoluteTimeGetCurrent()
        var processedCount = 0
        
        for (index, text) in texts.enumerated() {
            logger.processing("Document \(index + 1)/\(texts.count)", 
                            progress: Double(index) / Double(texts.count))
            
            do {
                _ = try await embedTextWithLogging(text)
                processedCount += 1
            } catch {
                logger.error("Failed to process document \(index + 1)", error: error)
            }
        }
        
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        let throughput = Double(processedCount) / duration
        
        logger.complete("Batch processing", 
                       result: "\(processedCount)/\(texts.count) successful")
        logger.performance("Batch throughput", duration: duration, throughput: throughput)
    }
    
    /// Example model loading with logging
    public func loadModelWithLogging(modelId: String, version: String) async throws {
        logger.model("Loading model", modelId: modelId, version: version)
        logger.thinking("model compatibility check")
        
        // Simulate compatibility check
        try await Task.sleep(nanoseconds: 50_000_000)
        logger.success("Model compatibility verified")
        
        // Simulate download with progress
        for progress in stride(from: 0.0, through: 1.0, by: 0.1) {
            logger.processing("Downloading model", progress: progress)
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        
        logger.success("Model downloaded")
        
        // Simulate loading into memory
        logger.memory("Loading model into memory", bytes: 512_000_000) // 512MB
        try await Task.sleep(nanoseconds: 200_000_000)
        
        // Simulate optimization
        logger.thinking("model optimization")
        try await Task.sleep(nanoseconds: 300_000_000)
        
        logger.complete("Model loading", result: "Ready for inference")
        logger.model("Model active", modelId: modelId, version: version)
    }
    
    /// Example security operation with logging
    public func verifyModelSignatureWithLogging(modelId: String) async throws -> Bool {
        logger.security("Starting signature verification", status: "pending")
        
        // Simulate verification steps
        logger.processing("Extracting public key", progress: 0.25)
        try await Task.sleep(nanoseconds: 100_000_000)
        
        logger.processing("Computing file hash", progress: 0.5)
        try await Task.sleep(nanoseconds: 150_000_000)
        
        logger.processing("Verifying signature", progress: 0.75)
        try await Task.sleep(nanoseconds: 100_000_000)
        
        let isValid = Bool.random() // Simulate verification result
        
        if isValid {
            logger.security("Signature verification passed", status: "trusted")
            logger.success("Model \(modelId) signature verified")
        } else {
            logger.security("Signature verification failed", status: "untrusted")
            logger.warning("Model \(modelId) signature invalid")
        }
        
        return isValid
    }
    
    /// Example error handling with logging
    public func handleErrorsWithLogging() async {
        enum DemoError: Error {
            case simulatedFailure
        }
        
        do {
            // Simulate operation that might fail
            if Bool.random() {
                throw DemoError.simulatedFailure
            }
        } catch {
            logger.error("Operation failed", error: error, context: "error_handling_demo")
            
            // Log recovery attempt
            logger.info("Attempting automatic recovery")
            logger.thinking("analyzing error pattern")
            
            // Simulate recovery
            logger.success("Automatic recovery successful")
        }
    }
    
    /// Example memory monitoring with logging
    public func monitorMemoryWithLogging() async {
        var currentMemory: Int64 = 1_024_000_000 // 1GB
        let peakMemory: Int64 = 2_048_000_000 // 2GB
        
        logger.memory("Current memory usage", bytes: currentMemory, peak: peakMemory)
        
        // Simulate dynamic memory check
        if Bool.random() {
            currentMemory = 1_600_000_000 // Simulate high memory
        }
        
        if currentMemory > 1_500_000_000 {
            logger.warning("High memory usage detected")
            logger.memory("Triggering garbage collection", bytes: currentMemory)
            
            // Simulate cleanup
            let freedMemory: Int64 = 500_000_000
            logger.success("Freed \(freedMemory / 1_000_000)MB of memory")
            logger.memory("Memory after cleanup", bytes: currentMemory - freedMemory)
        }
    }
}

// MARK: - Usage Example

public func demonstrateExpressiveLogging() async throws {
    let example = ExpressiveLoggingExample()
    
    print("\n🎨 === Expressive Logging Demo ===\n")
    
    // Single embedding
    print("1️⃣ Single Embedding:")
    _ = try await example.embedTextWithLogging("Hello, expressive logging!")
    
    print("\n2️⃣ Batch Processing:")
    let texts = [
        "First document",
        "Second document", 
        "Third document with more content",
        "Fourth document"
    ]
    try await example.processBatchWithLogging(texts)
    
    print("\n3️⃣ Model Loading:")
    try await example.loadModelWithLogging(modelId: "text-embedding-3", version: "1.2.0")
    
    print("\n4️⃣ Security Verification:")
    _ = try await example.verifyModelSignatureWithLogging(modelId: "text-embedding-3")
    
    print("\n5️⃣ Error Handling:")
    await example.handleErrorsWithLogging()
    
    print("\n6️⃣ Memory Monitoring:")
    await example.monitorMemoryWithLogging()
    
    print("\n✨ === Demo Complete ===\n")
}
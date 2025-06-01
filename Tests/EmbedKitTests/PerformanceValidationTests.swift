import Testing
import Foundation
@testable import EmbedKit

/// Performance validation tests with real numbers
@Suite("Performance Validation Tests")
struct PerformanceValidationTests {
    let logger = EmbedKitLogger.benchmarks()
    
    // MARK: - Embedding Performance Tests
    
    @Test("Single embedding performance baseline")
    func testSingleEmbeddingPerformance() async throws {
        logger.start("Single embedding performance test")
        
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        
        let testTexts = [
            "Short text",
            "Medium length text with several words and phrases",
            "Very long text that contains many words and should test the embedding system's ability to handle variable input lengths efficiently without compromising performance"
        ]
        
        var totalDuration: TimeInterval = 0
        let iterations = 100
        
        for text in testTexts {
            let startTime = CFAbsoluteTimeGetCurrent()
            
            for _ in 0..<iterations {
                _ = try await embedder.embed(text)
            }
            
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            totalDuration += duration
            
            let avgLatency = duration / Double(iterations) * 1000 // ms
            logger.performance("Single embedding (\(text.count) chars)", 
                             duration: duration, 
                             throughput: Double(iterations) / duration)
            logger.info("Average latency: \(String(format: "%.2f", avgLatency))ms")
        }
        
        let overallThroughput = Double(testTexts.count * iterations) / totalDuration
        logger.success("Overall throughput: \(String(format: "%.1f", overallThroughput)) embeddings/sec")
        
        // Performance assertions
        #expect(overallThroughput > 100) // Should handle >100 embeddings/sec
    }
    
    @Test("Batch embedding performance and speedup")
    func testBatchEmbeddingPerformance() async throws {
        logger.start("Batch embedding performance test")
        
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        
        let batchSizes = [1, 8, 16, 32, 64]
        let texts = (1...64).map { "Test document number \($0) with some content" }
        
        var results: [(size: Int, throughput: Double, latency: Double)] = []
        
        for batchSize in batchSizes {
            let batches = texts.chunked(into: batchSize)
            
            let startTime = CFAbsoluteTimeGetCurrent()
            
            for batch in batches {
                _ = try await embedder.embed(batch: batch)
            }
            
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            let throughput = Double(texts.count) / duration
            let avgLatency = duration / Double(batches.count) * 1000
            
            results.append((size: batchSize, throughput: throughput, latency: avgLatency))
            
            logger.performance("Batch size \(batchSize)", 
                             duration: duration, 
                             throughput: throughput)
        }
        
        // Find optimal batch size
        let optimal = results.max(by: { $0.throughput < $1.throughput })!
        logger.success("Optimal batch size: \(optimal.size) with \(String(format: "%.1f", optimal.throughput)) embeddings/sec")
        
        // Calculate speedup
        let singleThroughput = results.first(where: { $0.size == 1 })!.throughput
        let speedup = optimal.throughput / singleThroughput
        logger.success("Batch speedup: \(String(format: "%.2fx", speedup))")
        
        #expect(speedup > 2.0) // Batch should be at least 2x faster
    }
    
    // MARK: - Metal Acceleration Tests
    
    @Test("Metal acceleration performance impact")
    func testMetalAccelerationPerformance() async throws {
        guard MetalAccelerator.shared != nil else {
            logger.warning("Metal not available, skipping test")
            return
        }
        
        logger.start("Metal acceleration performance test")
        
        // Create embedders with and without Metal
        let embedderWithMetal = MockTextEmbedder(dimensions: 768, useMetalAcceleration: true)
        let embedderWithoutMetal = MockTextEmbedder(dimensions: 768, useMetalAcceleration: false)
        
        try await embedderWithMetal.loadModel()
        try await embedderWithoutMetal.loadModel()
        
        let texts = (1...100).map { "Performance test text \($0)" }
        
        // Test without Metal
        let cpuStart = CFAbsoluteTimeGetCurrent()
        let cpuEmbeddings = try await embedderWithoutMetal.embed(batch: texts)
        let cpuDuration = CFAbsoluteTimeGetCurrent() - cpuStart
        
        // Test with Metal
        let metalStart = CFAbsoluteTimeGetCurrent()
        let metalEmbeddings = try await embedderWithMetal.embed(batch: texts)
        let metalDuration = CFAbsoluteTimeGetCurrent() - metalStart
        
        let metalSpeedup = cpuDuration / metalDuration
        
        logger.performance("CPU processing", duration: cpuDuration, throughput: Double(texts.count) / cpuDuration)
        logger.performance("Metal processing", duration: metalDuration, throughput: Double(texts.count) / metalDuration)
        logger.success("Metal speedup: \(String(format: "%.2fx", metalSpeedup))")
        
        // Verify results are similar
        for i in 0..<min(10, texts.count) {
            let similarity = cpuEmbeddings[i].cosineSimilarity(to: metalEmbeddings[i])
            #expect(similarity > 0.99) // Should be very similar
        }
        
        #expect(metalSpeedup > 1.2) // Metal should provide at least 20% speedup
    }
    
    // MARK: - Cache Performance Tests
    
    @Test("Cache performance impact")
    func testCachePerformance() async throws {
        logger.start("Cache performance test")
        
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        
        let cache = EmbeddingCache(maxEntries: 1000)
        let texts = (1...100).map { "Cacheable text \($0)" }
        
        // First pass - populate cache
        logger.info("Populating cache...")
        let populateStart = CFAbsoluteTimeGetCurrent()
        
        for text in texts {
            let embedding = try await embedder.embed(text)
            await cache.set(text: text, modelIdentifier: "test", embedding: embedding)
        }
        
        let populateDuration = CFAbsoluteTimeGetCurrent() - populateStart
        
        // Second pass - cache hits
        logger.info("Testing cache hits...")
        var cacheHits = 0
        let hitStart = CFAbsoluteTimeGetCurrent()
        
        for text in texts {
            if let _ = await cache.get(text: text, modelIdentifier: "test") {
                cacheHits += 1
            } else {
                _ = try await embedder.embed(text)
            }
        }
        
        let hitDuration = CFAbsoluteTimeGetCurrent() - hitStart
        let cacheSpeedup = populateDuration / hitDuration
        
        logger.performance("Cache miss pass", duration: populateDuration, throughput: Double(texts.count) / populateDuration)
        logger.performance("Cache hit pass", duration: hitDuration, throughput: Double(texts.count) / hitDuration)
        logger.success("Cache speedup: \(String(format: "%.1fx", cacheSpeedup))")
        logger.success("Cache hit rate: \(String(format: "%.1f%%", Double(cacheHits) / Double(texts.count) * 100))")
        
        #expect(cacheHits == texts.count)
        #expect(cacheSpeedup > 10.0) // Cache should be much faster
    }
    
    // MARK: - Memory Usage Tests
    
    @Test("Memory usage validation")
    func testMemoryUsage() async throws {
        logger.start("Memory usage validation test")
        
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        
        // Get baseline memory
        let baselineMemory = getCurrentMemoryUsage()
        logger.memory("Baseline", bytes: Int64(baselineMemory * 1024 * 1024))
        
        // Generate many embeddings
        let texts = (1...1000).map { "Memory test text \($0)" }
        var embeddings: [EmbeddingVector] = []
        
        for text in texts {
            let embedding = try await embedder.embed(text)
            embeddings.append(embedding)
        }
        
        let afterEmbeddingsMemory = getCurrentMemoryUsage()
        let embeddingMemoryIncrease = afterEmbeddingsMemory - baselineMemory
        
        logger.memory("After 1000 embeddings", 
                     bytes: Int64(afterEmbeddingsMemory * 1024 * 1024),
                     peak: Int64(afterEmbeddingsMemory * 1024 * 1024))
        logger.info("Memory increase: \(String(format: "%.1f", embeddingMemoryIncrease))MB")
        
        // Calculate memory per embedding
        let memoryPerEmbedding = embeddingMemoryIncrease / Double(embeddings.count) * 1024 // KB
        logger.info("Memory per embedding: \(String(format: "%.2f", memoryPerEmbedding))KB")
        
        // Expected: 768 floats * 4 bytes = ~3KB per embedding + overhead
        #expect(memoryPerEmbedding < 10.0) // Should be less than 10KB per embedding
        
        // Clear embeddings
        embeddings.removeAll()
        
        // Check for memory leaks
        let afterClearMemory = getCurrentMemoryUsage()
        let memoryLeak = afterClearMemory - baselineMemory
        
        logger.memory("After cleanup", bytes: Int64(afterClearMemory * 1024 * 1024))
        logger.info("Potential memory leak: \(String(format: "%.1f", memoryLeak))MB")
        
        #expect(memoryLeak < 10.0) // Should have minimal memory leak
    }
    
    // MARK: - Streaming Performance Tests
    
    @Test("Streaming performance and backpressure")
    func testStreamingPerformance() async throws {
        logger.start("Streaming performance test")
        
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        
        let streamingEmbedder = StreamingEmbedder(
            embedder: embedder,
            configuration: StreamingEmbedder.StreamingConfiguration(
                maxConcurrency: 5,
                inputBufferSize: 100,
                batchSize: 16
            )
        )
        
        // Create a fast text source
        let textCount = 500
        let texts = (1...textCount).map { "Streaming text \($0)" }
        let source = MockTextSource(texts: texts, delay: 0.001) // 1ms between texts
        
        let startTime = CFAbsoluteTimeGetCurrent()
        var processedCount = 0
        
        let resultStream = await streamingEmbedder.embedTextStream(source)
        
        for try await result in resultStream {
            processedCount += 1
            
            if processedCount % 50 == 0 {
                let progress = Double(processedCount) / Double(textCount)
                logger.processing("Streaming progress", progress: progress)
            }
        }
        
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        let throughput = Double(processedCount) / duration
        
        logger.performance("Streaming", duration: duration, throughput: throughput)
        logger.success("Processed \(processedCount) texts via streaming")
        
        #expect(processedCount == textCount)
        #expect(throughput > 50) // Should handle >50 texts/sec with streaming
    }
    
    // MARK: - Concurrent Operations Tests
    
    @Test("Concurrent embedding operations")
    func testConcurrentOperations() async throws {
        logger.start("Concurrent operations test")
        
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        
        let concurrencyLevels = [1, 2, 4, 8, 16]
        let textsPerTask = 50
        
        for concurrency in concurrencyLevels {
            let startTime = CFAbsoluteTimeGetCurrent()
            
            try await withThrowingTaskGroup(of: Int.self) { group in
                for i in 0..<concurrency {
                    group.addTask {
                        var count = 0
                        for j in 0..<textsPerTask {
                            let text = "Concurrent text \(i)-\(j)"
                            _ = try await embedder.embed(text)
                            count += 1
                        }
                        return count
                    }
                }
                
                var totalProcessed = 0
                for try await count in group {
                    totalProcessed += count
                }
                
                let duration = CFAbsoluteTimeGetCurrent() - startTime
                let throughput = Double(totalProcessed) / duration
                
                logger.performance("Concurrency \(concurrency)", 
                                 duration: duration, 
                                 throughput: throughput)
            }
        }
        
        logger.success("Concurrent operations test completed")
    }
}

// MARK: - Helper Functions

private func getCurrentMemoryUsage() -> Double {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
    
    let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
            task_info(mach_task_self_,
                     task_flavor_t(MACH_TASK_BASIC_INFO),
                     $0,
                     &count)
        }
    }
    
    if kerr == KERN_SUCCESS {
        return Double(info.resident_size) / (1024 * 1024) // MB
    }
    
    return 0.0
}

// Mock text source for streaming tests
struct MockTextSource: AsyncSequence {
    typealias Element = String
    
    let texts: [String]
    let delay: TimeInterval
    
    struct AsyncIterator: AsyncIteratorProtocol {
        var texts: [String]
        let delay: TimeInterval
        var index = 0
        
        mutating func next() async throws -> String? {
            guard index < texts.count else { return nil }
            
            if delay > 0 {
                try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            }
            
            let text = texts[index]
            index += 1
            return text
        }
    }
    
    func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(texts: texts, delay: delay)
    }
}
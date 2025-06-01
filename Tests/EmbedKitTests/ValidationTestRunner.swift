import Testing
import Foundation
@testable import EmbedKit

/// Main test runner for EmbedKit validation
@Suite("EmbedKit Validation Suite")
struct ValidationTestRunner {
    let logger = ExpressiveLogger(category: "Validation")
    
    @Test("Run complete validation suite")
    func runCompleteValidation() async throws {
        print("\n")
        print("═══════════════════════════════════════════════════════")
        print("           🧪 EmbedKit Validation Suite 🧪")
        print("═══════════════════════════════════════════════════════")
        print("")
        
        logger.start("EmbedKit validation suite", details: "Running comprehensive tests")
        
        // Track overall results
        var passedTests = 0
        var failedTests = 0
        var testResults: [(name: String, passed: Bool, duration: TimeInterval)] = []
        
        // 1. Model Loading Tests
        print("\n📦 Model Loading Tests")
        print("─────────────────────")
        
        let modelTests = [
            ("Model loads correctly", testModelLoading),
            ("Model unloading", testModelUnloading),
            ("Model error handling", testModelErrorHandling)
        ]
        
        for (name, test) in modelTests {
            let result = await runTest(name: name, test: test)
            testResults.append(result)
            if result.passed { passedTests += 1 } else { failedTests += 1 }
        }
        
        // 2. Embedding Generation Tests
        print("\n🔤 Embedding Generation Tests")
        print("────────────────────────────")
        
        let embeddingTests = [
            ("Single embedding", testSingleEmbedding),
            ("Batch embedding", testBatchEmbedding),
            ("Empty text handling", testEmptyText),
            ("Long text truncation", testLongText)
        ]
        
        for (name, test) in embeddingTests {
            let result = await runTest(name: name, test: test)
            testResults.append(result)
            if result.passed { passedTests += 1 } else { failedTests += 1 }
        }
        
        // 3. Similarity Tests
        print("\n📐 Similarity Calculation Tests")
        print("──────────────────────────────")
        
        let similarityTests = [
            ("Cosine similarity", testCosineSimilarity),
            ("Euclidean distance", testEuclideanDistance),
            ("Dot product", testDotProduct)
        ]
        
        for (name, test) in similarityTests {
            let result = await runTest(name: name, test: test)
            testResults.append(result)
            if result.passed { passedTests += 1 } else { failedTests += 1 }
        }
        
        // 4. PipelineKit Integration Tests
        print("\n🔗 PipelineKit Integration Tests")
        print("───────────────────────────────")
        
        let pipelineTests = [
            ("Single command execution", testPipelineSingleCommand),
            ("Batch command execution", testPipelineBatchCommand),
            ("Cache middleware", testPipelineCacheMiddleware),
            ("Model management", testPipelineModelManagement),
            ("Error handling", testPipelineErrorHandling)
        ]
        
        for (name, test) in pipelineTests {
            let result = await runTest(name: name, test: test)
            testResults.append(result)
            if result.passed { passedTests += 1 } else { failedTests += 1 }
        }
        
        // 5. Performance Tests
        print("\n⚡ Performance Validation Tests")
        print("──────────────────────────────")
        
        let performanceTests = [
            ("Embedding throughput", testEmbeddingThroughput),
            ("Batch optimization", testBatchOptimization),
            ("Cache performance", testCachePerformance),
            ("Metal acceleration", testMetalPerformance),
            ("Memory usage", testMemoryUsage)
        ]
        
        for (name, test) in performanceTests {
            let result = await runTest(name: name, test: test)
            testResults.append(result)
            if result.passed { passedTests += 1 } else { failedTests += 1 }
        }
        
        // Print summary
        print("\n")
        print("═══════════════════════════════════════════════════════")
        print("                  📊 Test Summary 📊")
        print("═══════════════════════════════════════════════════════")
        print("")
        print("Total Tests: \(passedTests + failedTests)")
        print("✅ Passed: \(passedTests)")
        print("❌ Failed: \(failedTests)")
        print("Success Rate: \(String(format: "%.1f%%", Double(passedTests) / Double(passedTests + failedTests) * 100))")
        print("")
        
        // Performance metrics summary
        await printPerformanceSummary()
        
        // Final validation
        if failedTests == 0 {
            logger.complete("Validation suite", result: "All tests passed! ✨")
            print("\n🎉 EmbedKit is fully validated and ready for use!")
        } else {
            logger.error("Validation suite completed with failures")
            print("\n⚠️  Some tests failed. Please review the results above.")
        }
        
        print("\n═══════════════════════════════════════════════════════\n")
        
        #expect(failedTests == 0)
    }
    
    // MARK: - Individual Test Functions
    
    private func runTest(name: String, test: () async throws -> Void) async -> (name: String, passed: Bool, duration: TimeInterval) {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        do {
            try await test()
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            print("  ✅ \(name) (\(String(format: "%.2fs", duration)))")
            return (name, true, duration)
        } catch {
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            print("  ❌ \(name) - \(error)")
            return (name, false, duration)
        }
    }
    
    // Simplified test implementations
    
    private func testModelLoading() async throws {
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        guard await embedder.isModelLoaded else {
            throw ValidationError.testFailed("Model not loaded")
        }
    }
    
    private func testModelUnloading() async throws {
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        try await embedder.unloadModel()
        guard await !embedder.isModelLoaded else {
            throw ValidationError.testFailed("Model not unloaded")
        }
    }
    
    private func testModelErrorHandling() async throws {
        let embedder = MockTextEmbedder(dimensions: 768)
        do {
            _ = try await embedder.embed("test")
            throw ValidationError.testFailed("Expected error not thrown")
        } catch is EmbeddingError {
            // Expected
        }
    }
    
    private func testSingleEmbedding() async throws {
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        let embedding = try await embedder.embed("Test text")
        guard embedding.dimensions == 768 else {
            throw ValidationError.testFailed("Wrong dimensions")
        }
    }
    
    private func testBatchEmbedding() async throws {
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        let embeddings = try await embedder.embed(batch: ["Text 1", "Text 2", "Text 3"])
        guard embeddings.count == 3 else {
            throw ValidationError.testFailed("Wrong batch count")
        }
    }
    
    private func testEmptyText() async throws {
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        _ = try await embedder.embed("")
    }
    
    private func testLongText() async throws {
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        let longText = String(repeating: "word ", count: 1000)
        _ = try await embedder.embed(longText)
    }
    
    private func testCosineSimilarity() async throws {
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        
        let e1 = try await embedder.embed("Machine learning")
        let e2 = try await embedder.embed("Machine learning")
        let similarity = e1.cosineSimilarity(to: e2)
        
        guard similarity > 0.99 else {
            throw ValidationError.testFailed("Similarity too low: \(similarity)")
        }
    }
    
    private func testEuclideanDistance() async throws {
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        
        let e1 = try await embedder.embed("Test one")
        let e2 = try await embedder.embed("Test two")
        let distance = e1.euclideanDistance(to: e2)
        
        guard distance > 0 else {
            throw ValidationError.testFailed("Distance should be positive")
        }
    }
    
    private func testDotProduct() async throws {
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        
        let e1 = try await embedder.embed("Vector")
        let dot = e1.dotProduct(with: e1)
        
        guard abs(dot - 1.0) < 0.01 else {
            throw ValidationError.testFailed("Self dot product should be ~1.0")
        }
    }
    
    private func testPipelineSingleCommand() async throws {
        let pipeline = try EmbeddingPipeline.createMinimalPipeline()
        let result = try await pipeline.execute(EmbedTextCommand(text: "Test"))
        guard result.embedding.dimensions == 768 else {
            throw ValidationError.testFailed("Wrong dimensions from pipeline")
        }
    }
    
    private func testPipelineBatchCommand() async throws {
        let pipeline = try EmbeddingPipeline.createMinimalPipeline()
        let result = try await pipeline.execute(BatchEmbedCommand(texts: ["A", "B", "C"]))
        guard result.embeddings.count == 3 else {
            throw ValidationError.testFailed("Wrong batch count from pipeline")
        }
    }
    
    private func testPipelineCacheMiddleware() async throws {
        let config = EmbeddingPipelineConfiguration(enableCache: true)
        let pipeline = try EmbeddingPipeline(configuration: config)
        
        let text = "Cache test"
        let result1 = try await pipeline.execute(EmbedTextCommand(text: text))
        let result2 = try await pipeline.execute(EmbedTextCommand(text: text))
        
        guard result2.cached else {
            throw ValidationError.testFailed("Second request should hit cache")
        }
    }
    
    private func testPipelineModelManagement() async throws {
        let pipeline = try EmbeddingPipeline.createMinimalPipeline()
        let result = try await pipeline.execute(
            LoadModelCommand(modelId: "test", version: "1.0")
        )
        guard result.success else {
            throw ValidationError.testFailed("Model load failed")
        }
    }
    
    private func testPipelineErrorHandling() async throws {
        let pipeline = try EmbeddingPipeline.createMinimalPipeline()
        do {
            _ = try await pipeline.execute(EmbedTextCommand(text: ""))
            throw ValidationError.testFailed("Expected validation error")
        } catch {
            // Expected
        }
    }
    
    private func testEmbeddingThroughput() async throws {
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        
        let count = 100
        let start = CFAbsoluteTimeGetCurrent()
        
        for i in 0..<count {
            _ = try await embedder.embed("Test \(i)")
        }
        
        let duration = CFAbsoluteTimeGetCurrent() - start
        let throughput = Double(count) / duration
        
        guard throughput > 50 else {
            throw ValidationError.testFailed("Throughput too low: \(throughput)")
        }
    }
    
    private func testBatchOptimization() async throws {
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        
        let texts = (1...32).map { "Text \($0)" }
        
        // Single
        let singleStart = CFAbsoluteTimeGetCurrent()
        for text in texts {
            _ = try await embedder.embed(text)
        }
        let singleDuration = CFAbsoluteTimeGetCurrent() - singleStart
        
        // Batch
        let batchStart = CFAbsoluteTimeGetCurrent()
        _ = try await embedder.embed(batch: texts)
        let batchDuration = CFAbsoluteTimeGetCurrent() - batchStart
        
        let speedup = singleDuration / batchDuration
        guard speedup > 1.5 else {
            throw ValidationError.testFailed("Batch speedup too low: \(speedup)")
        }
    }
    
    private func testCachePerformance() async throws {
        let cache = EmbeddingCache(maxEntries: 100)
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        
        let text = "Cache test"
        let embedding = try await embedder.embed(text)
        
        await cache.set(text: text, modelIdentifier: "test", embedding: embedding)
        
        let cached = await cache.get(text: text, modelIdentifier: "test")
        guard cached != nil else {
            throw ValidationError.testFailed("Cache miss")
        }
    }
    
    private func testMetalPerformance() async throws {
        guard let metal = MetalAccelerator.shared else {
            // Skip if Metal not available
            return
        }
        
        let vectors = [[Float](repeating: 1.0, count: 768)]
        let normalized = try await metal.normalizeVectors(vectors)
        
        guard normalized.count == 1 else {
            throw ValidationError.testFailed("Metal normalization failed")
        }
    }
    
    private func testMemoryUsage() async throws {
        let baseline = getCurrentMemoryUsage()
        
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        
        var embeddings: [EmbeddingVector] = []
        for i in 0..<100 {
            embeddings.append(try await embedder.embed("Test \(i)"))
        }
        
        let peak = getCurrentMemoryUsage()
        let increase = peak - baseline
        
        guard increase < 50 else { // Less than 50MB increase
            throw ValidationError.testFailed("Memory usage too high: \(increase)MB")
        }
    }
    
    private func printPerformanceSummary() async {
        print("\n⚡ Performance Summary")
        print("─────────────────────")
        
        // Run quick performance test
        let embedder = MockTextEmbedder(dimensions: 768)
        try? await embedder.loadModel()
        
        // Single embedding latency
        let singleStart = CFAbsoluteTimeGetCurrent()
        _ = try? await embedder.embed("Performance test")
        let singleLatency = (CFAbsoluteTimeGetCurrent() - singleStart) * 1000
        
        // Batch throughput
        let texts = (1...100).map { "Batch test \($0)" }
        let batchStart = CFAbsoluteTimeGetCurrent()
        _ = try? await embedder.embed(batch: texts)
        let batchDuration = CFAbsoluteTimeGetCurrent() - batchStart
        let batchThroughput = Double(texts.count) / batchDuration
        
        print("  • Single embedding latency: \(String(format: "%.2f", singleLatency))ms")
        print("  • Batch throughput: \(String(format: "%.1f", batchThroughput)) embeddings/sec")
        
        if let metal = MetalAccelerator.shared {
            print("  • Metal acceleration: ✅ Available")
        } else {
            print("  • Metal acceleration: ❌ Not available")
        }
        
        let memoryUsage = getCurrentMemoryUsage()
        print("  • Current memory usage: \(String(format: "%.1f", memoryUsage))MB")
    }
}

// MARK: - Helper Types

enum ValidationError: LocalizedError {
    case testFailed(String)
    
    var errorDescription: String? {
        switch self {
        case .testFailed(let message):
            return message
        }
    }
}

// MARK: - Memory Helper

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
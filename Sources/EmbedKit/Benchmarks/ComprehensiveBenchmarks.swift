import Foundation
import OSLog

/// Comprehensive benchmark suite for EmbedKit performance testing
public actor ComprehensiveBenchmarkSuite {
    private let logger = Logger(subsystem: "EmbedKit", category: "ComprehensiveBenchmarks")
    
    public init() {}
    
    /// Run the complete benchmark suite
    public func runFullSuite() async throws -> BenchmarkSuiteResults {
        logger.info("Starting comprehensive benchmark suite")
        
        let startTime = Date()
        
        // Core functionality benchmarks
        let embeddingBenchmarks = try await runEmbeddingBenchmarks()
        let metalBenchmarks = try await runMetalBenchmarks()
        let cacheBenchmarks = try await runCacheBenchmarks()
        let streamingBenchmarks = try await runStreamingBenchmarks()
        let modelManagementBenchmarks = try await runModelManagementBenchmarks()
        let memoryBenchmarks = try await runMemoryBenchmarks()
        let concurrencyBenchmarks = try await runConcurrencyBenchmarks()
        
        let totalTime = Date().timeIntervalSince(startTime)
        
        let results = BenchmarkSuiteResults(
            embedding: embeddingBenchmarks,
            metal: metalBenchmarks,
            cache: cacheBenchmarks,
            streaming: streamingBenchmarks,
            modelManagement: modelManagementBenchmarks,
            memory: memoryBenchmarks,
            concurrency: concurrencyBenchmarks,
            totalTime: totalTime,
            timestamp: Date()
        )
        
        logger.info("Benchmark suite completed in \(totalTime)s")
        return results
    }
    
    /// Benchmark core embedding operations
    private func runEmbeddingBenchmarks() async throws -> EmbeddingBenchmarkResults {
        logger.info("Running embedding benchmarks")
        
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        
        // Single embedding benchmark
        let singleText = "The quick brown fox jumps over the lazy dog"
        let singleStartTime = CFAbsoluteTimeGetCurrent()
        
        for _ in 0..<100 {
            _ = try await embedder.embed(singleText)
        }
        
        let singleTime = CFAbsoluteTimeGetCurrent() - singleStartTime
        let singleThroughput = 100.0 / singleTime
        
        // Batch embedding benchmark
        let batchTexts = Array(repeating: singleText, count: 32)
        let batchStartTime = CFAbsoluteTimeGetCurrent()
        
        for _ in 0..<10 {
            _ = try await embedder.embed(batch: batchTexts)
        }
        
        let batchTime = CFAbsoluteTimeGetCurrent() - batchStartTime
        let batchThroughput = (10.0 * 32.0) / batchTime
        
        // Variable length benchmark
        let variableLengthTexts = [
            "Short",
            "Medium length text with several words",
            "Very long text that contains many words and should test the embedding system's ability to handle variable input lengths efficiently without compromising performance metrics"
        ]
        
        let variableStartTime = CFAbsoluteTimeGetCurrent()
        
        for _ in 0..<50 {
            for text in variableLengthTexts {
                _ = try await embedder.embed(text)
            }
        }
        
        let variableTime = CFAbsoluteTimeGetCurrent() - variableStartTime
        let variableThroughput = (50.0 * 3.0) / variableTime
        
        return EmbeddingBenchmarkResults(
            singleEmbeddingLatency: singleTime / 100.0,
            singleEmbeddingThroughput: singleThroughput,
            batchEmbeddingLatency: batchTime / 10.0,
            batchEmbeddingThroughput: batchThroughput,
            variableLengthThroughput: variableThroughput,
            batchSpeedup: batchThroughput / singleThroughput
        )
    }
    
    /// Benchmark Metal acceleration vs CPU
    private func runMetalBenchmarks() async throws -> MetalBenchmarkResults {
        logger.info("Running Metal benchmarks")
        
        guard let metalAccelerator = MetalAccelerator.shared else {
            logger.warning("Metal not available, skipping Metal benchmarks")
            return MetalBenchmarkResults(
                isMetalAvailable: false,
                normalizationSpeedup: 1.0,
                poolingSpeedup: 1.0,
                cosineSimilaritySpeedup: 1.0,
                metalMemoryUsage: 0
            )
        }
        
        // Generate test data
        let dimensions = 768
        let vectorCount = 100
        var testVectors: [[Float]] = []
        
        for _ in 0..<vectorCount {
            let vector = (0..<dimensions).map { _ in Float.random(in: -1...1) }
            testVectors.append(vector)
        }
        
        // Benchmark normalization
        let metalNormStart = CFAbsoluteTimeGetCurrent()
        _ = try await metalAccelerator.normalizeVectors(testVectors)
        let metalNormTime = CFAbsoluteTimeGetCurrent() - metalNormStart
        
        // CPU normalization for comparison
        let cpuNormStart = CFAbsoluteTimeGetCurrent()
        for vector in testVectors {
            let norm = sqrt(vector.reduce(0) { $0 + $1 * $1 })
            _ = norm > 0 ? vector.map { $0 / norm } : vector
        }
        let cpuNormTime = CFAbsoluteTimeGetCurrent() - cpuNormStart
        
        // Benchmark pooling
        let tokenEmbeddings = Array(testVectors.prefix(10))
        
        let metalPoolStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<10 {
            _ = try await metalAccelerator.poolEmbeddings(tokenEmbeddings, strategy: .mean)
        }
        let metalPoolTime = CFAbsoluteTimeGetCurrent() - metalPoolStart
        
        // CPU pooling for comparison
        let cpuPoolStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<10 {
            let result = (0..<dimensions).map { dim in
                tokenEmbeddings.map { $0[dim] }.reduce(0, +) / Float(tokenEmbeddings.count)
            }
            _ = result
        }
        let cpuPoolTime = CFAbsoluteTimeGetCurrent() - cpuPoolStart
        
        // Benchmark cosine similarity
        let queries = Array(testVectors.prefix(20))
        let keys = Array(testVectors.suffix(20))
        
        let metalSimStart = CFAbsoluteTimeGetCurrent()
        _ = try await metalAccelerator.cosineSimilarityMatrix(queries: queries, keys: keys)
        let metalSimTime = CFAbsoluteTimeGetCurrent() - metalSimStart
        
        // CPU cosine similarity for comparison  
        let cpuSimStart = CFAbsoluteTimeGetCurrent()
        for query in queries {
            for key in keys {
                let dotProduct = zip(query, key).map(*).reduce(0, +)
                let queryNorm = sqrt(query.map { $0 * $0 }.reduce(0, +))
                let keyNorm = sqrt(key.map { $0 * $0 }.reduce(0, +))
                _ = dotProduct / (queryNorm * keyNorm)
            }
        }
        let cpuSimTime = CFAbsoluteTimeGetCurrent() - cpuSimStart
        
        return MetalBenchmarkResults(
            isMetalAvailable: true,
            normalizationSpeedup: cpuNormTime / metalNormTime,
            poolingSpeedup: cpuPoolTime / metalPoolTime,
            cosineSimilaritySpeedup: cpuSimTime / metalSimTime,
            metalMemoryUsage: 0 // Would need Metal performance counters
        )
    }
    
    /// Benchmark caching performance
    private func runCacheBenchmarks() async throws -> CacheBenchmarkResults {
        logger.info("Running cache benchmarks")
        
        let cache = LRUCache<String, [Float]>(maxSize: 100)
        
        // Generate test data
        let testKeys = (1...200).map { "test_key_\($0)" }
        let testValues = (1...200).map { _ in (1...768).map { _ in Float.random(in: -1...1) } }
        
        // Benchmark cache writes
        let writeStartTime = CFAbsoluteTimeGetCurrent()
        for (key, value) in zip(testKeys, testValues) {
            await cache.set(key, value: value)
        }
        let writeTime = CFAbsoluteTimeGetCurrent() - writeStartTime
        
        // Benchmark cache hits
        let hitStartTime = CFAbsoluteTimeGetCurrent()
        var hitCount = 0
        for key in testKeys.suffix(100) { // These should be hits
            if await cache.get(key) != nil {
                hitCount += 1
            }
        }
        let hitTime = CFAbsoluteTimeGetCurrent() - hitStartTime
        
        // Benchmark cache misses
        let missStartTime = CFAbsoluteTimeGetCurrent()
        var missCount = 0
        for i in 1000...1100 { // These should be misses
            if await cache.get("missing_key_\(i)") == nil {
                missCount += 1
            }
        }
        let missTime = CFAbsoluteTimeGetCurrent() - missStartTime
        
        return CacheBenchmarkResults(
            writeLatency: writeTime / Double(testKeys.count),
            hitLatency: hitTime / Double(hitCount),
            missLatency: missTime / Double(missCount),
            hitRate: Double(hitCount) / 100.0,
            evictionOverhead: writeTime / Double(testKeys.count) // Simplified metric
        )
    }
    
    /// Benchmark streaming performance
    private func runStreamingBenchmarks() async throws -> StreamingBenchmarkResults {
        logger.info("Running streaming benchmarks")
        
        let embedder = MockTextEmbedder(dimensions: 384)
        try await embedder.loadModel()
        
        let streamingEmbedder = StreamingEmbedder(
            embedder: embedder,
            configuration: StreamingEmbedder.StreamingConfiguration(
                maxConcurrency: 5,
                inputBufferSize: 100,
                batchSize: 16
            )
        )
        
        // Create test data stream
        let textCount = 1000
        let testTexts = (1...textCount).map { "Test document \($0) with some content" }
        let mockSource = MockTextSource(texts: testTexts, delay: 0.001) // 1ms delay
        
        // Benchmark streaming throughput
        let streamStartTime = CFAbsoluteTimeGetCurrent()
        let resultStream = await streamingEmbedder.embedTextStream(mockSource)
        
        var processedCount = 0
        for try await _ in resultStream {
            processedCount += 1
        }
        
        let streamTime = CFAbsoluteTimeGetCurrent() - streamStartTime
        let streamThroughput = Double(processedCount) / streamTime
        
        // Benchmark batch streaming
        let batchStartTime = CFAbsoluteTimeGetCurrent()
        let batchSource = MockTextSource(texts: Array(testTexts.prefix(100)), delay: 0.001)
        let batchStream = await streamingEmbedder.embedBatchStream(batchSource)
        
        var batchCount = 0
        for try await batch in batchStream {
            batchCount += batch.count
        }
        
        let batchTime = CFAbsoluteTimeGetCurrent() - batchStartTime
        let batchThroughput = Double(batchCount) / batchTime
        
        return StreamingBenchmarkResults(
            streamingThroughput: streamThroughput,
            batchStreamingThroughput: batchThroughput,
            backpressureLatency: 0.0, // Would need specific backpressure test
            memoryConcurrency: 5.0,
            streamingSpeedup: batchThroughput / streamThroughput
        )
    }
    
    /// Benchmark model management operations
    private func runModelManagementBenchmarks() async throws -> ModelManagementBenchmarkResults {
        logger.info("Running model management benchmarks")
        
        let registry = ModelVersionRegistry()
        
        // Create test model versions
        let modelCount = 50
        let versionsPerModel = 5
        
        // Benchmark registration
        let regStartTime = CFAbsoluteTimeGetCurrent()
        
        for modelIndex in 1...modelCount {
            for versionIndex in 1...versionsPerModel {
                let version = ModelVersion(
                    identifier: "test_model_\(modelIndex)",
                    version: "1.\(versionIndex)",
                    buildNumber: versionIndex
                )
                
                // Create a temporary file for testing
                let tempURL = URL(fileURLWithPath: NSTemporaryDirectory())
                    .appendingPathComponent("test_model_\(modelIndex)_\(versionIndex).mlmodel")
                
                try "dummy model data".write(to: tempURL, atomically: true, encoding: .utf8)
                
                try await registry.register(version: version, modelURL: tempURL)
            }
        }
        
        let regTime = CFAbsoluteTimeGetCurrent() - regStartTime
        
        // Benchmark version lookup
        let lookupStartTime = CFAbsoluteTimeGetCurrent()
        
        for modelIndex in 1...modelCount {
            _ = await registry.getVersions(for: "test_model_\(modelIndex)")
        }
        
        let lookupTime = CFAbsoluteTimeGetCurrent() - lookupStartTime
        
        // Benchmark active version setting
        let activeStartTime = CFAbsoluteTimeGetCurrent()
        
        for modelIndex in 1...min(10, modelCount) {
            let version = ModelVersion(
                identifier: "test_model_\(modelIndex)",
                version: "1.3",
                buildNumber: 3
            )
            try await registry.setActiveVersion(version)
        }
        
        let activeTime = CFAbsoluteTimeGetCurrent() - activeStartTime
        
        return ModelManagementBenchmarkResults(
            registrationLatency: regTime / Double(modelCount * versionsPerModel),
            versionLookupLatency: lookupTime / Double(modelCount),
            activeVersionSetLatency: activeTime / 10.0,
            registryMemoryUsage: 0.0 // Would need memory profiling
        )
    }
    
    /// Benchmark memory usage patterns
    private func runMemoryBenchmarks() async throws -> MemoryBenchmarkResults {
        logger.info("Running memory benchmarks")
        
        let initialMemory = getCurrentMemoryUsage()
        
        // Test memory usage with large embeddings
        let embedder = MockTextEmbedder(dimensions: 2048) // Larger dimensions
        try await embedder.loadModel()
        
        let afterLoadMemory = getCurrentMemoryUsage()
        
        // Generate many embeddings to test memory growth
        let largeText = String(repeating: "This is a test sentence. ", count: 100)
        var embeddings: [[Float]] = []
        
        for _ in 0..<100 {
            let embedding = try await embedder.embed(largeText)
            embeddings.append(Array(0..<embedding.dimensions).map { embedding[$0] })
        }
        
        let afterEmbeddingsMemory = getCurrentMemoryUsage()
        
        // Test memory with cache
        let cache = LRUCache<String, [Float]>(maxSize: 1000)
        
        for (index, embedding) in embeddings.enumerated() {
            await cache.set("embedding_\(index)", value: embedding)
        }
        
        let afterCacheMemory = getCurrentMemoryUsage()
        
        // Cleanup and measure
        embeddings.removeAll()
        
        let afterCleanupMemory = getCurrentMemoryUsage()
        
        return MemoryBenchmarkResults(
            baselineMemory: initialMemory,
            modelLoadMemoryIncrease: afterLoadMemory - initialMemory,
            embeddingMemoryIncrease: afterEmbeddingsMemory - afterLoadMemory,
            cacheMemoryIncrease: afterCacheMemory - afterEmbeddingsMemory,
            memoryLeakage: afterCleanupMemory - afterCacheMemory,
            peakMemoryUsage: max(afterLoadMemory, afterEmbeddingsMemory, afterCacheMemory)
        )
    }
    
    /// Benchmark concurrency performance
    private func runConcurrencyBenchmarks() async throws -> ConcurrencyBenchmarkResults {
        logger.info("Running concurrency benchmarks")
        
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        
        let testText = "Concurrent embedding test text with reasonable length"
        let operationCount = 100
        
        // Benchmark sequential operations
        let sequentialStartTime = CFAbsoluteTimeGetCurrent()
        
        for _ in 0..<operationCount {
            _ = try await embedder.embed(testText)
        }
        
        let sequentialTime = CFAbsoluteTimeGetCurrent() - sequentialStartTime
        
        // Benchmark concurrent operations
        let concurrentStartTime = CFAbsoluteTimeGetCurrent()
        
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<operationCount {
                group.addTask {
                    _ = try await embedder.embed(testText)
                }
            }
            
            for try await _ in group {}
        }
        
        let concurrentTime = CFAbsoluteTimeGetCurrent() - concurrentStartTime
        
        // Test different concurrency levels
        var concurrencyResults: [Int: Double] = [:]
        
        for concurrency in [1, 2, 4, 8, 16] {
            let startTime = CFAbsoluteTimeGetCurrent()
            
            try await withThrowingTaskGroup(of: Void.self) { group in
                let opsPerTask = operationCount / concurrency
                
                for _ in 0..<concurrency {
                    group.addTask {
                        for _ in 0..<opsPerTask {
                            _ = try await embedder.embed(testText)
                        }
                    }
                }
                
                for try await _ in group {}
            }
            
            let time = CFAbsoluteTimeGetCurrent() - startTime
            concurrencyResults[concurrency] = time
        }
        
        return ConcurrencyBenchmarkResults(
            sequentialTime: sequentialTime,
            concurrentTime: concurrentTime,
            concurrencySpeedup: sequentialTime / concurrentTime,
            optimalConcurrency: concurrencyResults.min(by: { $0.value < $1.value })?.key ?? 1,
            concurrencyScaling: concurrencyResults
        )
    }
    
    // MARK: - Helper Methods
    
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
}

// MARK: - Result Types

public struct BenchmarkSuiteResults: Sendable, Codable {
    public let embedding: EmbeddingBenchmarkResults
    public let metal: MetalBenchmarkResults  
    public let cache: CacheBenchmarkResults
    public let streaming: StreamingBenchmarkResults
    public let modelManagement: ModelManagementBenchmarkResults
    public let memory: MemoryBenchmarkResults
    public let concurrency: ConcurrencyBenchmarkResults
    public let totalTime: TimeInterval
    public let timestamp: Date
    
    public init(embedding: EmbeddingBenchmarkResults, metal: MetalBenchmarkResults, cache: CacheBenchmarkResults, streaming: StreamingBenchmarkResults, modelManagement: ModelManagementBenchmarkResults, memory: MemoryBenchmarkResults, concurrency: ConcurrencyBenchmarkResults, totalTime: TimeInterval, timestamp: Date) {
        self.embedding = embedding
        self.metal = metal
        self.cache = cache
        self.streaming = streaming
        self.modelManagement = modelManagement
        self.memory = memory
        self.concurrency = concurrency
        self.totalTime = totalTime
        self.timestamp = timestamp
    }
}

public struct EmbeddingBenchmarkResults: Sendable, Codable {
    public let singleEmbeddingLatency: TimeInterval
    public let singleEmbeddingThroughput: Double
    public let batchEmbeddingLatency: TimeInterval
    public let batchEmbeddingThroughput: Double
    public let variableLengthThroughput: Double
    public let batchSpeedup: Double
    
    public init(singleEmbeddingLatency: TimeInterval, singleEmbeddingThroughput: Double, batchEmbeddingLatency: TimeInterval, batchEmbeddingThroughput: Double, variableLengthThroughput: Double, batchSpeedup: Double) {
        self.singleEmbeddingLatency = singleEmbeddingLatency
        self.singleEmbeddingThroughput = singleEmbeddingThroughput
        self.batchEmbeddingLatency = batchEmbeddingLatency
        self.batchEmbeddingThroughput = batchEmbeddingThroughput
        self.variableLengthThroughput = variableLengthThroughput
        self.batchSpeedup = batchSpeedup
    }
}

public struct MetalBenchmarkResults: Sendable, Codable {
    public let isMetalAvailable: Bool
    public let normalizationSpeedup: Double
    public let poolingSpeedup: Double
    public let cosineSimilaritySpeedup: Double
    public let metalMemoryUsage: Double
    
    public init(isMetalAvailable: Bool, normalizationSpeedup: Double, poolingSpeedup: Double, cosineSimilaritySpeedup: Double, metalMemoryUsage: Double) {
        self.isMetalAvailable = isMetalAvailable
        self.normalizationSpeedup = normalizationSpeedup
        self.poolingSpeedup = poolingSpeedup
        self.cosineSimilaritySpeedup = cosineSimilaritySpeedup
        self.metalMemoryUsage = metalMemoryUsage
    }
}

public struct CacheBenchmarkResults: Sendable, Codable {
    public let writeLatency: TimeInterval
    public let hitLatency: TimeInterval
    public let missLatency: TimeInterval
    public let hitRate: Double
    public let evictionOverhead: TimeInterval
    
    public init(writeLatency: TimeInterval, hitLatency: TimeInterval, missLatency: TimeInterval, hitRate: Double, evictionOverhead: TimeInterval) {
        self.writeLatency = writeLatency
        self.hitLatency = hitLatency
        self.missLatency = missLatency
        self.hitRate = hitRate
        self.evictionOverhead = evictionOverhead
    }
}

public struct StreamingBenchmarkResults: Sendable, Codable {
    public let streamingThroughput: Double
    public let batchStreamingThroughput: Double
    public let backpressureLatency: TimeInterval
    public let memoryConcurrency: Double
    public let streamingSpeedup: Double
    
    public init(streamingThroughput: Double, batchStreamingThroughput: Double, backpressureLatency: TimeInterval, memoryConcurrency: Double, streamingSpeedup: Double) {
        self.streamingThroughput = streamingThroughput
        self.batchStreamingThroughput = batchStreamingThroughput
        self.backpressureLatency = backpressureLatency
        self.memoryConcurrency = memoryConcurrency
        self.streamingSpeedup = streamingSpeedup
    }
}

public struct ModelManagementBenchmarkResults: Sendable, Codable {
    public let registrationLatency: TimeInterval
    public let versionLookupLatency: TimeInterval
    public let activeVersionSetLatency: TimeInterval
    public let registryMemoryUsage: Double
    
    public init(registrationLatency: TimeInterval, versionLookupLatency: TimeInterval, activeVersionSetLatency: TimeInterval, registryMemoryUsage: Double) {
        self.registrationLatency = registrationLatency
        self.versionLookupLatency = versionLookupLatency
        self.activeVersionSetLatency = activeVersionSetLatency
        self.registryMemoryUsage = registryMemoryUsage
    }
}

public struct MemoryBenchmarkResults: Sendable, Codable {
    public let baselineMemory: Double
    public let modelLoadMemoryIncrease: Double
    public let embeddingMemoryIncrease: Double
    public let cacheMemoryIncrease: Double
    public let memoryLeakage: Double
    public let peakMemoryUsage: Double
    
    public init(baselineMemory: Double, modelLoadMemoryIncrease: Double, embeddingMemoryIncrease: Double, cacheMemoryIncrease: Double, memoryLeakage: Double, peakMemoryUsage: Double) {
        self.baselineMemory = baselineMemory
        self.modelLoadMemoryIncrease = modelLoadMemoryIncrease
        self.embeddingMemoryIncrease = embeddingMemoryIncrease
        self.cacheMemoryIncrease = cacheMemoryIncrease
        self.memoryLeakage = memoryLeakage
        self.peakMemoryUsage = peakMemoryUsage
    }
}

public struct ConcurrencyBenchmarkResults: Sendable, Codable {
    public let sequentialTime: TimeInterval
    public let concurrentTime: TimeInterval
    public let concurrencySpeedup: Double
    public let optimalConcurrency: Int
    public let concurrencyScaling: [Int: Double]
    
    public init(sequentialTime: TimeInterval, concurrentTime: TimeInterval, concurrencySpeedup: Double, optimalConcurrency: Int, concurrencyScaling: [Int: Double]) {
        self.sequentialTime = sequentialTime
        self.concurrentTime = concurrentTime
        self.concurrencySpeedup = concurrencySpeedup
        self.optimalConcurrency = optimalConcurrency
        self.concurrencyScaling = concurrencyScaling
    }
}

// MARK: - Mock Text Embedder for Testing

/// Simple mock embedder for testing and benchmarking purposes
///
/// This mock embedder provides:
/// - Deterministic embeddings for consistent testing
/// - Realistic simulation of model loading and processing times
/// - Proper error handling with contextual information
/// - Batch processing optimization simulation
public actor MockTextEmbedder: TextEmbedder {
    public let modelIdentifier = ModelIdentifier(family: "mock", variant: "test", version: "v1")
    public let dimensions: Int
    public var isReady: Bool = false
    public let configuration: Configuration
    
    // Simple cache for deterministic behavior
    private var embeddingCache: [String: EmbeddingVector] = [:]
    
    public init(dimensions: Int = 768) {
        self.dimensions = dimensions
        self.configuration = Configuration()
    }
    
    public func loadModel() async throws {
        // Simulate model loading
        try await Task.sleep(nanoseconds: 100_000_000) // 0.1 seconds
        isReady = true
    }
    
    public func unloadModel() async throws {
        isReady = false
    }
    
    public func embed(_ text: String) async throws -> EmbeddingVector {
        guard isReady else {
            throw ContextualEmbeddingError.modelNotLoaded(
                context: ErrorContext.modelLoading(modelIdentifier)
            )
        }
        
        // Check cache first for deterministic behavior
        if let cached = embeddingCache[text] {
            return cached
        }
        
        // Simulate realistic embedding computation time
        try await Task.sleep(nanoseconds: UInt64(text.count * 10_000)) // 10µs per character
        
        // Generate deterministic embedding based on text hash
        let embedding = generateDeterministicEmbedding(for: text)
        
        // Cache the result
        if embeddingCache.count < 1000 { // Simple cache size limit
            embeddingCache[text] = embedding
        }
        
        return embedding
    }
    
    public func embed(batch texts: [String]) async throws -> [EmbeddingVector] {
        guard isReady else {
            throw ContextualEmbeddingError.modelNotLoaded(
                context: ErrorContext.modelLoading(modelIdentifier)
            )
        }
        
        // Simulate batch processing optimization (30% faster than individual calls)
        let totalCharacterCount = texts.reduce(0) { $0 + $1.count }
        let batchOptimizedTime = UInt64(Double(totalCharacterCount * 10_000) * 0.7)
        try await Task.sleep(nanoseconds: batchOptimizedTime)
        
        // Process batch with potential cache hits
        var results: [EmbeddingVector] = []
        results.reserveCapacity(texts.count)
        
        for text in texts {
            if let cached = embeddingCache[text] {
                results.append(cached)
            } else {
                let embedding = generateDeterministicEmbedding(for: text)
                results.append(embedding)
                
                // Cache if space available
                if embeddingCache.count < 1000 {
                    embeddingCache[text] = embedding
                }
            }
        }
        
        return results
    }
    
    // MARK: - Private Helper Methods
    
    /// Generate a deterministic embedding based on text content
    /// This ensures consistent results for testing while still being realistic
    private func generateDeterministicEmbedding(for text: String) -> EmbeddingVector {
        // Use text hash as seed for consistent pseudo-random generation
        var hasher = Hasher()
        hasher.combine(text)
        let hashValue = hasher.finalize()
        
        // Convert to UInt64 safely
        let seed = UInt64(bitPattern: Int64(hashValue))
        
        // Create deterministic random generator
        var generator = SeededRandomNumberGenerator(seed: seed)
        
        // Generate values that sum to approximately 1.0 for realistic embeddings
        var values: [Float] = []
        values.reserveCapacity(dimensions)
        
        for _ in 0..<dimensions {
            let value = Float.random(in: -0.1...0.1, using: &generator)
            values.append(value)
        }
        
        // Normalize to unit vector for realistic embedding behavior
        let magnitude = sqrt(values.reduce(0) { $0 + $1 * $1 })
        if magnitude > 0 {
            for i in 0..<values.count {
                values[i] /= magnitude
            }
        }
        
        return EmbeddingVector(values)
    }
}

/// Seeded random number generator for deterministic pseudo-random values
private struct SeededRandomNumberGenerator: RandomNumberGenerator {
    private var state: UInt64
    
    init(seed: UInt64) {
        self.state = seed == 0 ? 1 : seed
    }
    
    mutating func next() -> UInt64 {
        // Linear congruential generator (simple but sufficient for testing)
        state = state &* 1103515245 &+ 12345
        return state
    }
}
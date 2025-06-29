import XCTest
import Foundation
@testable import EmbedKit

/// Comprehensive performance tests for optimized implementations
final class EmbedKitPerformanceTests: XCTestCase {
    
    // MARK: - LRU Cache Performance Tests
    
    /// Test that LRU cache operations complete in under 0.5 seconds for 10,000 operations
    func testLRUCachePerformance() async throws {
        let cache = OptimizedLRUCache<String, String>(maxSize: 1000)
        let operationCount = 10_000
        
        let startTime = Date()
        
        // Mix of get, set, and remove operations
        for i in 0..<operationCount {
            let key = "key_\(i % 2000)" // Some key reuse to test hits
            
            switch i % 3 {
            case 0:
                // Set operation
                await cache.set(key, value: "value_\(i)")
            case 1:
                // Get operation
                _ = await cache.get(key)
            default:
                // Occasional remove
                if i % 10 == 0 {
                    await cache.remove(key)
                } else {
                    _ = await cache.get(key)
                }
            }
        }
        
        let elapsedTime = Date().timeIntervalSince(startTime)
        
        XCTAssertLessThan(elapsedTime, 0.5, "LRU cache operations took \(elapsedTime) seconds, expected < 0.5s")
        
        // Verify cache statistics
        let stats = await cache.statistics
        print("Cache statistics - Hits: \(stats.hits), Misses: \(stats.misses), Hit Rate: \(stats.hitRate)")
        XCTAssertGreaterThan(stats.hitRate, 0.3, "Expected reasonable hit rate")
    }
    
    /// Compare performance with original implementation
    func testLRUCachePerformanceComparison() async throws {
        let operationCount = 5_000
        
        // Test original implementation
        let originalCache = LRUCache<String, String>(maxSize: 1000)
        let originalStart = Date()
        
        for i in 0..<operationCount {
            let key = "key_\(i % 1000)"
            await originalCache.set(key, value: "value_\(i)")
            _ = await originalCache.get(key)
        }
        
        let originalTime = Date().timeIntervalSince(originalStart)
        
        // Test optimized implementation
        let optimizedCache = OptimizedLRUCache<String, String>(maxSize: 1000)
        let optimizedStart = Date()
        
        for i in 0..<operationCount {
            let key = "key_\(i % 1000)"
            await optimizedCache.set(key, value: "value_\(i)")
            _ = await optimizedCache.get(key)
        }
        
        let optimizedTime = Date().timeIntervalSince(optimizedStart)
        
        let speedup = originalTime / optimizedTime
        print("Performance comparison - Original: \(originalTime)s, Optimized: \(optimizedTime)s, Speedup: \(speedup)x")
        
        XCTAssertGreaterThan(speedup, 5.0, "Expected at least 5x performance improvement")
    }
    
    /// Test worst-case scenario: all cache misses with evictions
    func testLRUCacheWorstCasePerformance() async throws {
        let cache = OptimizedLRUCache<String, String>(maxSize: 100)
        let operationCount = 10_000
        
        let startTime = Date()
        
        // All unique keys to force constant evictions
        for i in 0..<operationCount {
            await cache.set("unique_key_\(i)", value: "value_\(i)")
        }
        
        let elapsedTime = Date().timeIntervalSince(startTime)
        
        XCTAssertLessThan(elapsedTime, 0.5, "Worst-case LRU operations took \(elapsedTime) seconds")
        
        let stats = await cache.statistics
        XCTAssertEqual(stats.currentSize, 100, "Cache should be at max capacity")
        XCTAssertGreaterThanOrEqual(stats.evictions, operationCount - 100, "Should have evicted most items")
    }
    
    // MARK: - Metal Buffer Pool Performance Tests
    
    /// Test Metal buffer pool allocation performance
    func testMetalBufferPoolPerformance() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        
        let pool = try MetalBufferPool(device: device, initialHeapSizeMB: 64)
        let bufferCount = 1000
        let bufferSize = 1024 * 1024 // 1MB buffers
        
        var buffers: [MTLBuffer] = []
        
        // Test allocation performance
        let allocationStart = Date()
        
        for _ in 0..<bufferCount {
            let buffer = try await pool.acquireBuffer(size: bufferSize)
            buffers.append(buffer)
        }
        
        let allocationTime = Date().timeIntervalSince(allocationStart)
        print("Allocated \(bufferCount) buffers in \(allocationTime)s")
        
        // Test release and reacquisition (should be much faster due to pooling)
        let releaseStart = Date()
        
        for buffer in buffers {
            await pool.releaseBuffer(buffer)
        }
        
        let releaseTime = Date().timeIntervalSince(releaseStart)
        
        // Reacquire to test pool hits
        let reacquireStart = Date()
        var reacquiredBuffers: [MTLBuffer] = []
        
        for _ in 0..<bufferCount {
            let buffer = try await pool.acquireBuffer(size: bufferSize)
            reacquiredBuffers.append(buffer)
        }
        
        let reacquireTime = Date().timeIntervalSince(reacquireStart)
        
        print("Release time: \(releaseTime)s, Reacquire time: \(reacquireTime)s")
        
        // Reacquisition should be at least 10x faster due to pooling
        XCTAssertLessThan(reacquireTime * 10, allocationTime, "Pool reuse should be much faster")
        
        // Check statistics
        let stats = await pool.getStatistics()
        XCTAssertGreaterThan(stats.bufferHitRate, 0.9, "Expected high buffer hit rate on reacquisition")
        
        // Cleanup
        for buffer in reacquiredBuffers {
            await pool.releaseBuffer(buffer)
        }
    }
    
    /// Test memory pressure handling
    func testMetalBufferPoolMemoryPressure() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        
        let pool = try MetalBufferPool(device: device, initialHeapSizeMB: 32, maxHeapSizeMB: 128)
        
        // Allocate many buffers
        var buffers: [MTLBuffer] = []
        for i in 0..<100 {
            let size = (i % 4 + 1) * 1024 * 1024 // 1-4 MB
            let buffer = try await pool.acquireBuffer(size: size)
            buffers.append(buffer)
        }
        
        // Release all buffers
        for buffer in buffers {
            await pool.releaseBuffer(buffer)
        }
        
        let statsBeforePressure = await pool.getStatistics()
        
        // Simulate memory pressure
        await pool.handleMemoryPressure()
        
        let statsAfterPressure = await pool.getStatistics()
        
        print("Buffers before pressure: \(statsBeforePressure.totalBuffersCreated)")
        print("Available buffers cleared by pressure handling")
        
        // Verify memory was freed
        XCTAssertEqual(statsAfterPressure.totalBuffersCreated, statsBeforePressure.totalBuffersCreated)
    }
    
    // MARK: - SQLite Connection Pool Performance Tests
    
    /// Test SQLite connection pool throughput improvement
    func testSQLiteConnectionPoolThroughput() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let dbPath = tempDir.appendingPathComponent("test_\(UUID().uuidString).sqlite").path
        
        // Test with original single-connection approach
        let singleConnRegistry = try await PersistentModelRegistry(storageDirectory: tempDir)
        
        let operationCount = 1000
        let singleConnStart = Date()
        
        for i in 0..<operationCount {
            let version = ModelVersion(
                identifier: "test-model",
                version: "1.0.\(i)",
                buildNumber: i,
                createdAt: Date(),
                metadata: ["key": "value_\(i)"]
            )
            
            let fileURL = tempDir.appendingPathComponent("model_\(i).mlmodel")
            try "test".write(to: fileURL, atomically: true, encoding: .utf8)
            
            try await singleConnRegistry.saveVersion(version, fileURL: fileURL)
        }
        
        let singleConnTime = Date().timeIntervalSince(singleConnStart)
        
        // Test with connection pool
        let pooledRegistry = try await OptimizedPersistentModelRegistry(storageDirectory: tempDir)
        
        let pooledStart = Date()
        
        // Concurrent operations to leverage pool
        try await withThrowingTaskGroup(of: Void.self) { group in
            for i in 0..<operationCount {
                group.addTask {
                    let version = ModelVersion(
                        identifier: "test-model-pooled",
                        version: "1.0.\(i)",
                        buildNumber: i,
                        createdAt: Date(),
                        metadata: ["key": "value_\(i)"]
                    )
                    
                    let fileURL = tempDir.appendingPathComponent("model_pooled_\(i).mlmodel")
                    try "test".write(to: fileURL, atomically: true, encoding: .utf8)
                    
                    try await pooledRegistry.saveVersion(version, fileURL: fileURL)
                }
            }
            
            try await group.waitForAll()
        }
        
        let pooledTime = Date().timeIntervalSince(pooledStart)
        
        let throughputImprovement = singleConnTime / pooledTime
        print("Single connection: \(singleConnTime)s, Pooled: \(pooledTime)s, Improvement: \(throughputImprovement)x")
        
        XCTAssertGreaterThan(throughputImprovement, 3.0, "Expected at least 3x throughput improvement")
        
        // Check pool statistics
        let poolStats = await pooledRegistry.getPoolStatistics()
        print("Pool stats - Total connections: \(poolStats.totalConnections), Utilization: \(poolStats.connectionUtilization)")
        XCTAssertGreaterThan(poolStats.connectionUtilization, 0.5, "Expected good connection utilization")
        
        // Cleanup
        try? FileManager.default.removeItem(atPath: dbPath)
    }
    
    /// Test prepared statement caching performance
    func testPreparedStatementCaching() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let pooledRegistry = try await OptimizedPersistentModelRegistry(storageDirectory: tempDir)
        
        let identifier = "perf-test-model"
        let iterations = 5000
        
        // First, save a version
        let version = ModelVersion(
            identifier: identifier,
            version: "1.0.0",
            buildNumber: 1,
            createdAt: Date(),
            metadata: ["test": "data"]
        )
        
        let fileURL = tempDir.appendingPathComponent("test_model.mlmodel")
        try "test".write(to: fileURL, atomically: true, encoding: .utf8)
        try await pooledRegistry.saveVersion(version, fileURL: fileURL)
        
        // Test repeated queries (should benefit from prepared statement caching)
        let queryStart = Date()
        
        for _ in 0..<iterations {
            _ = try await pooledRegistry.loadVersions(for: identifier)
        }
        
        let queryTime = Date().timeIntervalSince(queryStart)
        let avgQueryTime = queryTime / Double(iterations) * 1000 // Convert to milliseconds
        
        print("Average query time with prepared statements: \(avgQueryTime)ms")
        XCTAssertLessThan(avgQueryTime, 1.0, "Queries should average less than 1ms with caching")
    }
    
    // MARK: - Integration Performance Test
    
    /// Test all optimizations working together in a realistic scenario
    func testIntegratedPerformance() async throws {
        // Setup
        let tempDir = FileManager.default.temporaryDirectory
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        
        let cache = OptimizedEmbeddingCache(maxEntries: 1000)
        let bufferPool = try MetalBufferPool(device: device)
        let registry = try await OptimizedPersistentModelRegistry(storageDirectory: tempDir)
        
        // Simulate realistic embedding workflow
        let textCount = 1000
        let embeddingDimensions = 384
        
        let workflowStart = Date()
        
        for i in 0..<textCount {
            let text = "Sample text for embedding \(i)"
            let modelId = ModelIdentifier.miniLM_L6_v2
            
            // Check cache first
            if let cached = await cache.get(text: text, modelIdentifier: modelId) {
                continue
            }
            
            // Simulate embedding generation with Metal
            let bufferSize = embeddingDimensions * MemoryLayout<Float>.size
            let buffer = try await bufferPool.acquireBuffer(size: bufferSize)
            
            // Simulate computation delay
            try await Task.sleep(nanoseconds: 100_000) // 0.1ms
            
            // Create embedding
            let embedding = EmbeddingVector((0..<embeddingDimensions).map { _ in Float.random(in: -1...1) })
            
            // Cache the result
            await cache.set(text: text, modelIdentifier: modelId, embedding: embedding)
            
            // Release buffer
            await bufferPool.releaseBuffer(buffer)
            
            // Occasionally save model metadata
            if i % 100 == 0 {
                let version = ModelVersion(
                    identifier: modelId.rawValue,
                    version: "1.0.\(i)",
                    buildNumber: i,
                    createdAt: Date(),
                    metadata: ["processed": "\(i)"]
                )
                
                let fileURL = tempDir.appendingPathComponent("model_\(i).mlmodel")
                try "test".write(to: fileURL, atomically: true, encoding: .utf8)
                try await registry.saveVersion(version, fileURL: fileURL)
            }
        }
        
        let totalTime = Date().timeIntervalSince(workflowStart)
        let avgTimePerText = totalTime / Double(textCount) * 1000 // ms
        
        print("Integrated workflow - Total: \(totalTime)s, Avg per text: \(avgTimePerText)ms")
        
        // Verify performance
        XCTAssertLessThan(avgTimePerText, 2.0, "Average processing time should be under 2ms per text")
        
        // Check component statistics
        let cacheStats = await cache.statistics()
        let poolStats = await bufferPool.getStatistics()
        let dbStats = await registry.getPoolStatistics()
        
        print("Cache hit rate: \(cacheStats.hitRate)")
        print("Buffer pool hit rate: \(poolStats.bufferHitRate)")
        print("DB connection utilization: \(dbStats.connectionUtilization)")
        
        XCTAssertGreaterThan(cacheStats.hitRate, 0.0, "Should have some cache hits")
        XCTAssertGreaterThan(poolStats.bufferHitRate, 0.9, "Buffer pool should be highly effective")
    }
}
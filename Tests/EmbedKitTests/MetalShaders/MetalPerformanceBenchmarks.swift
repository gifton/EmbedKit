import XCTest
import Metal
@testable import EmbedKit

/// Performance benchmarks for Metal shader operations
///
/// These benchmarks measure:
/// - Throughput (elements/second, GFLOPS)
/// - Latency (milliseconds)
/// - Scaling characteristics
/// - Comparison between precompiled and string-compiled libraries
///
/// Benchmarks are designed to be informative rather than strict gates.
/// They establish baselines and detect performance regressions.
///
final class MetalPerformanceBenchmarks: XCTestCase {
    var accelerator: MetalAccelerator!

    override func setUp() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }

        accelerator = try XCTUnwrap(
            MetalAccelerator.shared,
            "MetalAccelerator.shared should be available"
        )

        try await accelerator.setupPipelines()

        // Print device info for context
        if let info = MetalTestUtilities.deviceInfo() {
            print("\n=== Performance Benchmark Configuration ===")
            print(info)
            print("===========================================\n")
        }
    }

    // MARK: - Library Loading Performance

    func testMetallibLoadingPerformance() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }

        print("\n=== Metallib Loading Performance ===")

        // Clear cache to measure cold start
        await MetalLibraryLoader.clearCache()

        // Benchmark cold start
        let coldStats = try await MetalTestUtilities.measure(
            warmup: 0,  // No warmup for cold start
            iterations: 10
        ) {
            try await MetalLibraryLoader.loadLibrary(device: device)
            await MetalLibraryLoader.clearCache()  // Clear for next iteration
        }

        print("Cold Start (with cache clear):")
        print(coldStats.description)

        // Benchmark warm start (cached)
        let warmStats = try await MetalTestUtilities.measure(
            warmup: 1,
            iterations: 10
        ) {
            // Don't clear cache - measure cached performance
            _ = try await MetalLibraryLoader.loadLibrary(device: device)
        }

        print("\nWarm Start (cached):")
        print(warmStats.description)

        // Assertions (informative, not strict)
        XCTAssertLessThan(coldStats.median, 100.0,
                         "Metallib cold start should be <100ms")
        XCTAssertLessThan(warmStats.median, 1.0,
                         "Metallib warm start should be <1ms (cached)")

        print("====================================\n")
    }

    // MARK: - L2 Normalization Benchmarks

    func testL2NormalizationThroughput() async throws {
        print("\n=== L2 Normalization Throughput ===")

        let dimensions = 384  // MiniLM-L12-v2 dimension
        let batchSizes = [1, 16, 64, 256, 1000]

        print("Configuration: \(dimensions) dimensions")
        print("Batch Size | Latency (ms) | Throughput (M elem/s)")
        print("-----------|--------------|---------------------")

        for batchSize in batchSizes {
            let vectors = MetalTestUtilities.randomBatch(
                batchSize: batchSize,
                dimensions: dimensions
            )

            let stats = try await MetalTestUtilities.measure(iterations: 10) {
                let batch = try VectorBatch(vectors: vectors)
                _ = try await self.accelerator.normalizeVectors(batch)
            }

            let elementsProcessed = batchSize * dimensions
            let throughput = Double(elementsProcessed) / (stats.median / 1000.0) / 1_000_000

            print(String(format: "%10d | %12.3f | %19.2f",
                        batchSize, stats.median, throughput))

            // Sanity check: should process at least 5M elements/sec on any GPU
            if batchSize >= 64 {
                XCTAssertGreaterThan(throughput, 5.0,
                                   "Throughput for batch \(batchSize) should be >5M elem/s")
            }
        }

        print("===================================\n")
    }

    func testL2NormalizationScaling() async throws {
        print("\n=== L2 Normalization Scaling (Fixed Batch=100) ===")

        let batchSize = 100
        let dimensions = [128, 256, 384, 512, 768, 1024]

        print("Dimensions | Latency (ms) | Throughput (M elem/s)")
        print("-----------|--------------|---------------------")

        for dim in dimensions {
            let vectors = MetalTestUtilities.randomBatch(
                batchSize: batchSize,
                dimensions: dim
            )

            let stats = try await MetalTestUtilities.measure(iterations: 10) {
                let batch = try VectorBatch(vectors: vectors)
                _ = try await self.accelerator.normalizeVectors(batch)
            }

            let elementsProcessed = batchSize * dim
            let throughput = Double(elementsProcessed) / (stats.median / 1000.0) / 1_000_000

            print(String(format: "%10d | %12.3f | %19.2f",
                        dim, stats.median, throughput))
        }

        print("=================================================\n")
    }

    // MARK: - Pooling Benchmarks

    func testMeanPoolingPerformance() async throws {
        print("\n=== Mean Pooling Performance ===")

        let dimensions = 384
        let sequenceLengths = [16, 32, 64, 128, 256, 512]

        print("Sequence Len | Latency (ms)")
        print("-------------|-------------")

        for seqLen in sequenceLengths {
            let embeddings = MetalTestUtilities.randomBatch(
                batchSize: seqLen,
                dimensions: dimensions
            )

            let stats = try await MetalTestUtilities.measure(iterations: 10) {
                let batch = try VectorBatch(vectors: embeddings)
                _ = try await self.accelerator.poolEmbeddings(
                    batch,
                    strategy: .mean,
                    attentionMask: nil
                )
            }

            print(String(format: "%12d | %11.3f", seqLen, stats.median))

            // Should be fast for typical sequences
            if seqLen <= 512 {
                XCTAssertLessThan(stats.median, 10.0,
                                "Mean pooling should be <10ms for seq=\(seqLen)")
            }
        }

        print("================================\n")
    }

    func testMaxPoolingPerformance() async throws {
        print("\n=== Max Pooling Performance ===")

        let dimensions = 384
        let sequenceLengths = [16, 32, 64, 128, 256, 512]

        print("Sequence Len | Latency (ms)")
        print("-------------|-------------")

        for seqLen in sequenceLengths {
            let embeddingsArray = MetalTestUtilities.randomBatch(
                batchSize: seqLen,
                dimensions: dimensions
            )
            let embeddings = try VectorBatch(vectors: embeddingsArray)

            let stats = try await MetalTestUtilities.measure(iterations: 10) {
                _ = try await self.accelerator.poolEmbeddings(
                    embeddings,
                    strategy: .max,
                    attentionMask: nil
                )
            }

            print(String(format: "%12d | %11.3f", seqLen, stats.median))

            if seqLen <= 512 {
                XCTAssertLessThan(stats.median, 10.0,
                                "Max pooling should be <10ms for seq=\(seqLen)")
            }
        }

        print("===============================\n")
    }

    func testAttentionPoolingPerformance() async throws {
        print("\n=== Attention-Weighted Pooling Performance ===")

        let dimensions = 384
        let sequenceLengths = [16, 32, 64, 128, 256, 512]

        print("Sequence Len | Latency (ms)")
        print("-------------|-------------")

        for seqLen in sequenceLengths {
            let embeddingsArray = MetalTestUtilities.randomBatch(
                batchSize: seqLen,
                dimensions: dimensions
            )
            let embeddings = try VectorBatch(vectors: embeddingsArray)

            // Generate softmax-like weights
            let weights: [Float] = (0..<seqLen).map { i in
                let x = Float(i) / Float(seqLen)
                return exp(-x * x)
            }
            let weightSum = weights.reduce(0, +)
            let normalizedWeights = weights.map { $0 / weightSum }

            let stats = try await MetalTestUtilities.measure(iterations: 10) {
                _ = try await self.accelerator.attentionWeightedPooling(
                    embeddings,
                    attentionWeights: normalizedWeights
                )
            }

            print(String(format: "%12d | %11.3f", seqLen, stats.median))

            if seqLen <= 512 {
                XCTAssertLessThan(stats.median, 10.0,
                                "Attention pooling should be <10ms for seq=\(seqLen)")
            }
        }

        print("==============================================\n")
    }

    // MARK: - Cosine Similarity Benchmarks

    func testCosineSimilarityPairPerformance() async throws {
        print("\n=== Cosine Similarity (Pair) Performance ===")

        let dimensions = [128, 256, 384, 512, 768, 1024]

        print("Dimensions | Latency (ms)")
        print("-----------|-------------")

        for dim in dimensions {
            let v1 = MetalTestUtilities.randomVector(dimensions: dim)
            let v2 = MetalTestUtilities.randomVector(dimensions: dim)

            let stats = try await MetalTestUtilities.measure(iterations: 10) {
                _ = try await self.accelerator.cosineSimilarity(v1, v2)
            }

            print(String(format: "%10d | %11.3f", dim, stats.median))

            // Should be very fast for single pair
            XCTAssertLessThan(stats.median, 5.0,
                            "Pair similarity should be <5ms for \(dim) dimensions")
        }

        print("============================================\n")
    }

    func testCosineSimilarityBatchPerformance() async throws {
        print("\n=== Cosine Similarity (Batch) Performance ===")

        let dimensions = 384
        let batchSizes = [10, 50, 100, 500, 1000]

        print("Batch Size | Latency (ms) | Throughput (pairs/s)")
        print("-----------|--------------|---------------------")

        for batchSize in batchSizes {
            let vectorsA = MetalTestUtilities.randomBatch(
                batchSize: batchSize,
                dimensions: dimensions
            )
            let vectorsB = MetalTestUtilities.randomBatch(
                batchSize: batchSize,
                dimensions: dimensions
            )

            let vectorPairs = zip(vectorsA, vectorsB).map { ($0, $1) }

            let stats = try await MetalTestUtilities.measure(iterations: 10) {
                _ = try await self.accelerator.cosineSimilarityBatch(vectorPairs)
            }

            let throughput = Double(batchSize) / (stats.median / 1000.0)

            print(String(format: "%10d | %12.3f | %19.0f",
                        batchSize, stats.median, throughput))

            // Should handle batches efficiently
            if batchSize >= 100 {
                XCTAssertGreaterThan(throughput, 1000.0,
                                   "Batch throughput should be >1000 pairs/s for \(batchSize) pairs")
            }
        }

        print("=============================================\n")
    }

    func testCosineSimilarityMatrixScaling() async throws {
        print("\n=== Cosine Similarity Matrix Scaling ===")

        let dimensions = 384
        let sizes = [8, 16, 32, 64, 128]  // Query/Key counts

        print("Size (QxK) | Latency (ms) | GFLOPS | Comparisons/s")
        print("-----------|--------------|--------|---------------")

        for size in sizes {
            let queriesArray = MetalTestUtilities.randomBatch(
                batchSize: size,
                dimensions: dimensions
            )
            let queries = try VectorBatch(vectors: queriesArray)
            let keys = queries  // Self-similarity for simplicity

            let stats = try await MetalTestUtilities.measure(iterations: 5) {
                _ = try await self.accelerator.cosineSimilarityMatrix(
                    queries: queries,
                    keys: keys
                )
            }

            // Each similarity requires: dot product (D muls, D adds) + 2 norms (2*D muls, 2*D adds) = 4*D ops
            // Total for matrix: Q*K*4*D operations
            let operations = size * size * dimensions * 4
            let gflops = Double(operations) / (stats.median / 1000.0) / 1e9

            // Comparisons per second
            let comparisons = size * size
            let compsPerSec = Double(comparisons) / (stats.median / 1000.0)

            print(String(format: "%10dx%-3d | %12.3f | %6.2f | %14.0f",
                        size, size, stats.median, gflops, compsPerSec))

            // Use comparisons/sec for robustness at tiny sizes where overhead dominates
            // Threshold keeps the test meaningful but resilient across devices
            XCTAssertGreaterThan(compsPerSec, 100_000,
                               "Comparisons/s for \(size)x\(size) should be >100k")
        }

        print("========================================\n")
    }

    // MARK: - Batch Size Scaling

    func testBatchSizeScaling() async throws {
        print("\n=== Batch Size Scaling (Normalization) ===")

        let dimensions = 384
        let batchSizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

        print("Batch Size | Latency (ms) | Latency/Item (μs)")
        print("-----------|--------------|-------------------")

        for batchSize in batchSizes {
            let vectors = MetalTestUtilities.randomBatch(
                batchSize: batchSize,
                dimensions: dimensions
            )

            let stats = try await MetalTestUtilities.measure(iterations: 10) {
                let batch = try VectorBatch(vectors: vectors)
                _ = try await self.accelerator.normalizeVectors(batch)
            }

            let latencyPerItem = (stats.median / Double(batchSize)) * 1000  // Convert to microseconds

            print(String(format: "%10d | %12.3f | %17.2f",
                        batchSize, stats.median, latencyPerItem))
        }

        print("==========================================\n")
    }

    // MARK: - End-to-End Pipeline Performance

    func testEndToEndEmbeddingPipeline() async throws {
        print("\n=== End-to-End Embedding Pipeline ===")
        print("Simulates: Normalize → Pool → Similarity")

        let batchSizes = [1, 10, 50, 100]
        let dimensions = 384
        let seqLen = 32

        print("Batch Size | Total (ms) | Normalize | Pool | Similarity")
        print("-----------|------------|-----------|------|------------")

        for batchSize in batchSizes {
            // Generate test data
            var allEmbeddings: [[[Float]]] = []
            for _ in 0..<batchSize {
                let embeddings = MetalTestUtilities.randomBatch(
                    batchSize: seqLen,
                    dimensions: dimensions
                )
                allEmbeddings.append(embeddings)
            }

            // Measure full pipeline
            let totalStats = try await MetalTestUtilities.measure(iterations: 5) {
                for embeddingsArray in allEmbeddings {
                    let embeddings = try VectorBatch(vectors: embeddingsArray)

                    // Step 1: Normalize
                    let normalized = try await self.accelerator.normalizeVectors(embeddings)

                    // Step 2: Pool
                    let pooled = try await self.accelerator.poolEmbeddings(
                        normalized,
                        strategy: .mean,
                        attentionMask: nil
                    )

                    // Step 3: Self-similarity (just for demonstration)
                    _ = try await self.accelerator.cosineSimilarity(pooled, pooled)
                }
            }

            // Also measure individual steps
            let normalizeStats = try await MetalTestUtilities.measure(iterations: 5) {
                for embeddingsArray in allEmbeddings {
                    let embeddings = try VectorBatch(vectors: embeddingsArray)
                    _ = try await self.accelerator.normalizeVectors(embeddings)
                }
            }

            let poolStats = try await MetalTestUtilities.measure(iterations: 5) {
                for embeddingsArray in allEmbeddings {
                    let embeddings = try VectorBatch(vectors: embeddingsArray)
                    let normalized = try await self.accelerator.normalizeVectors(embeddings)
                    _ = try await self.accelerator.poolEmbeddings(
                        normalized,
                        strategy: .mean,
                        attentionMask: nil
                    )
                }
            }

            let simStats = try await MetalTestUtilities.measure(iterations: 5) {
                let v = MetalTestUtilities.randomVector(dimensions: dimensions)
                _ = try await self.accelerator.cosineSimilarity(v, v)
            }

            print(String(format: "%10d | %10.2f | %9.2f | %4.2f | %10.3f",
                        batchSize,
                        totalStats.median,
                        normalizeStats.median,
                        poolStats.median - normalizeStats.median,
                        simStats.median))
        }

        print("=====================================\n")
    }

    // MARK: - Memory Bandwidth Utilization

    func testMemoryBandwidthUtilization() async throws {
        print("\n=== Estimated Memory Bandwidth Utilization ===")

        let dimensions = 384
        let batchSize = 1000

        let vectorsArray = MetalTestUtilities.randomBatch(
            batchSize: batchSize,
            dimensions: dimensions
        )
        let vectors = try VectorBatch(vectors: vectorsArray)

        let stats = try await MetalTestUtilities.measure(iterations: 10) {
            _ = try await self.accelerator.normalizeVectors(vectors)
        }

        // Memory traffic: read entire input + write entire output = 2 * batch * dim * 4 bytes
        let bytesTransferred = 2 * batchSize * dimensions * 4
        let bandwidth = Double(bytesTransferred) / (stats.median / 1000.0) / (1024 * 1024 * 1024)  // GB/s

        print("Batch Size: \(batchSize)")
        print("Dimensions: \(dimensions)")
        print("Data Transferred: \(bytesTransferred / (1024 * 1024)) MB")
        print("Median Latency: \(String(format: "%.3f", stats.median)) ms")
        print("Effective Bandwidth: \(String(format: "%.2f", bandwidth)) GB/s")

        #if arch(arm64)
        print("\nNote: Apple Silicon unified memory bandwidth typically 50-400 GB/s")
        print("GPU operations are often compute-bound rather than memory-bound")
        #endif

        print("==============================================\n")
    }

    // MARK: - Performance Summary

    func testPerformanceSummary() async throws {
        print("\n===============================================")
        print("=== PERFORMANCE BENCHMARK SUMMARY ===")
        print("===============================================")

        let dimensions = 384

        // L2 Normalization
        let normVectorsArray = MetalTestUtilities.randomBatch(batchSize: 100, dimensions: dimensions)
        let normVectors = try VectorBatch(vectors: normVectorsArray)
        let normStats = try await MetalTestUtilities.measure(iterations: 10) {
            _ = try await self.accelerator.normalizeVectors(normVectors)
        }

        // Mean Pooling
        let poolEmbeddingsArray = MetalTestUtilities.randomBatch(batchSize: 32, dimensions: dimensions)
        let poolEmbeddings = try VectorBatch(vectors: poolEmbeddingsArray)
        let poolStats = try await MetalTestUtilities.measure(iterations: 10) {
            _ = try await self.accelerator.poolEmbeddings(poolEmbeddings, strategy: .mean, attentionMask: nil)
        }

        // Cosine Similarity
        let v1 = MetalTestUtilities.randomVector(dimensions: dimensions)
        let v2 = MetalTestUtilities.randomVector(dimensions: dimensions)
        let simStats = try await MetalTestUtilities.measure(iterations: 10) {
            _ = try await self.accelerator.cosineSimilarity(v1, v2)
        }

        // Similarity Matrix
        let queriesArray = MetalTestUtilities.randomBatch(batchSize: 32, dimensions: dimensions)
        let queries = try VectorBatch(vectors: queriesArray)
        let keys = queries
        let matrixStats = try await MetalTestUtilities.measure(iterations: 5) {
            _ = try await self.accelerator.cosineSimilarityMatrix(queries: queries, keys: keys)
        }

        print("\nOperation                  | Median Latency")
        print("---------------------------|---------------")
        print(String(format: "L2 Normalize (100x384)     | %8.3f ms", normStats.median))
        print(String(format: "Mean Pool (32x384)         | %8.3f ms", poolStats.median))
        print(String(format: "Cosine Sim (single pair)   | %8.3f ms", simStats.median))
        print(String(format: "Cosine Matrix (32x32)      | %8.3f ms", matrixStats.median))

        print("\n===============================================\n")
    }
}

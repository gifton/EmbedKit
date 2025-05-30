import Foundation
import OSLog

/// Utility for benchmarking embedding operations
public struct EmbeddingBenchmark {
    private let logger = Logger(subsystem: "EmbedKit", category: "Benchmark")
    
    public init() {}
    
    /// Benchmark result containing performance metrics
    public struct Result: CustomStringConvertible {
        public let name: String
        public let totalDuration: TimeInterval
        public let operationCount: Int
        public let operationsPerSecond: Double
        public let averageDuration: TimeInterval
        public let minDuration: TimeInterval
        public let maxDuration: TimeInterval
        public let percentile95: TimeInterval
        
        public var description: String {
            """
            Benchmark: \(name)
            Total Duration: \(String(format: "%.3f", totalDuration))s
            Operations: \(operationCount)
            Ops/sec: \(String(format: "%.1f", operationsPerSecond))
            Average: \(String(format: "%.3f", averageDuration * 1000))ms
            Min: \(String(format: "%.3f", minDuration * 1000))ms
            Max: \(String(format: "%.3f", maxDuration * 1000))ms
            P95: \(String(format: "%.3f", percentile95 * 1000))ms
            """
        }
    }
    
    /// Run a benchmark on single text embedding
    public func benchmarkSingleEmbedding(
        embedder: any TextEmbedder,
        texts: [String],
        name: String = "Single Embedding"
    ) async throws -> Result {
        var durations: [TimeInterval] = []
        durations.reserveCapacity(texts.count)
        
        let totalStart = Date()
        
        for text in texts {
            let start = Date()
            _ = try await embedder.embed(text)
            let duration = Date().timeIntervalSince(start)
            durations.append(duration)
        }
        
        let totalDuration = Date().timeIntervalSince(totalStart)
        
        return calculateResult(
            name: name,
            durations: durations,
            totalDuration: totalDuration
        )
    }
    
    /// Run a benchmark on batch embedding
    public func benchmarkBatchEmbedding(
        embedder: any TextEmbedder,
        textBatches: [[String]],
        name: String = "Batch Embedding"
    ) async throws -> Result {
        var durations: [TimeInterval] = []
        durations.reserveCapacity(textBatches.count)
        
        let totalStart = Date()
        var totalTexts = 0
        
        for batch in textBatches {
            totalTexts += batch.count
            let start = Date()
            _ = try await embedder.embed(batch: batch)
            let duration = Date().timeIntervalSince(start)
            durations.append(duration)
        }
        
        let totalDuration = Date().timeIntervalSince(totalStart)
        
        return calculateResult(
            name: name,
            durations: durations,
            totalDuration: totalDuration,
            operationCount: totalTexts
        )
    }
    
    /// Compare performance with and without features
    public func comparePerformance(
        embedderWithFeatures: any TextEmbedder,
        embedderWithoutFeatures: any TextEmbedder,
        texts: [String],
        batchSize: Int = 32
    ) async throws {
        logger.info("Starting performance comparison")
        
        // Single embedding comparison
        let singleWithout = try await benchmarkSingleEmbedding(
            embedder: embedderWithoutFeatures,
            texts: Array(texts.prefix(100)),
            name: "Single (No Optimization)"
        )
        
        let singleWith = try await benchmarkSingleEmbedding(
            embedder: embedderWithFeatures,
            texts: Array(texts.prefix(100)),
            name: "Single (With Optimization)"
        )
        
        // Batch embedding comparison
        let batches = texts.chunked(into: batchSize)
        
        let batchWithout = try await benchmarkBatchEmbedding(
            embedder: embedderWithoutFeatures,
            textBatches: Array(batches.prefix(10)),
            name: "Batch (No Optimization)"
        )
        
        let batchWith = try await benchmarkBatchEmbedding(
            embedder: embedderWithFeatures,
            textBatches: Array(batches.prefix(10)),
            name: "Batch (With Optimization)"
        )
        
        // Print results
        print("\n=== Performance Comparison ===\n")
        print(singleWithout)
        print("\n" + singleWith.description)
        print("\nSingle Speedup: \(String(format: "%.2fx", singleWithout.averageDuration / singleWith.averageDuration))")
        
        print("\n" + batchWithout.description)
        print("\n" + batchWith.description)
        print("\nBatch Speedup: \(String(format: "%.2fx", batchWithout.averageDuration / batchWith.averageDuration))")
        
        // Cache hit rate (if available)
        if let _ = embedderWithFeatures as? CoreMLTextEmbedder {
            // Run again to test cache
            let cachedResult = try await benchmarkSingleEmbedding(
                embedder: embedderWithFeatures,
                texts: Array(texts.prefix(100)),
                name: "Single (From Cache)"
            )
            print("\n" + cachedResult.description)
            print("\nCache Speedup: \(String(format: "%.2fx", singleWith.averageDuration / cachedResult.averageDuration))")
        }
    }
    
    private func calculateResult(
        name: String,
        durations: [TimeInterval],
        totalDuration: TimeInterval,
        operationCount: Int? = nil
    ) -> Result {
        let count = operationCount ?? durations.count
        let sorted = durations.sorted()
        
        return Result(
            name: name,
            totalDuration: totalDuration,
            operationCount: count,
            operationsPerSecond: Double(count) / totalDuration,
            averageDuration: durations.reduce(0, +) / Double(durations.count),
            minDuration: sorted.first ?? 0,
            maxDuration: sorted.last ?? 0,
            percentile95: sorted[Int(Double(sorted.count) * 0.95)]
        )
    }
}

// Array extension for chunking
extension Array {
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }
}

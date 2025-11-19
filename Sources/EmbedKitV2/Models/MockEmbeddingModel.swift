// EmbedKitV2 - Mock Embedding Model (Week 1)

import Foundation
import Logging

actor MockEmbeddingModel: EmbeddingModel {
    nonisolated let id = ModelID(provider: "mock", name: "test", version: "1.0")
    nonisolated let dimensions = 384
    nonisolated let device: ComputeDevice = .cpu
    private let configuration: EmbeddingConfiguration
    private let logger = Logger(label: "EmbedKitV2.MockEmbeddingModel")

    private var metricsData = MetricsData()

    init(configuration: EmbeddingConfiguration) {
        self.configuration = configuration
    }

    func embed(_ text: String) async throws -> Embedding {
        let start = CFAbsoluteTimeGetCurrent()

        // Deterministic fake embedding based on text content
        let seed: Int = text.utf8.reduce(0) { $0 &+ Int($1) }
        let vector: [Float] = (0..<dimensions).map { i in
            let v = sin(Double(seed + i)) * 0.5 + 0.5
            return Float(v)
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        metricsData.record(tokenCount: text.count, time: elapsed)

        return Embedding(
            vector: configuration.normalizeOutput ? normalize(vector) : vector,
            metadata: EmbeddingMetadata(
                modelID: id,
                tokenCount: text.count,
                processingTime: elapsed,
                normalized: configuration.normalizeOutput,
                poolingStrategy: configuration.poolingStrategy,
                truncated: false,
                custom: [:]
            )
        )
    }

    func embedBatch(_ texts: [String], options: BatchOptions) async throws -> [Embedding] {
        var results: [Embedding] = []
        results.reserveCapacity(texts.count)
        for text in texts {
            results.append(try await embed(text))
        }
        return results
    }

    func warmup() async throws { logger.debug("warmup()") }
    func release() async throws {}

    var metrics: ModelMetrics { metricsData.snapshot(memoryUsage: currentMemoryUsage()) }

    func resetMetrics() async throws { metricsData = MetricsData() }

    private func normalize(_ v: [Float]) -> [Float] {
        let sumSquares = v.reduce(0) { $0 + $1 * $1 }
        let mag = sqrt(max(1e-12, sumSquares))
        return v.map { $0 / mag }
    }
}

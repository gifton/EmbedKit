// EmbedKit - Apple NLContextualEmbedding Adapter

import Foundation
import Logging

import NaturalLanguage

/// Adapter around Apple's NLContextualEmbedding that conforms to the EmbedKit EmbeddingModel API.
/// Provides simple, dependency‑light sentence embeddings with mean pooling over token vectors.
public actor AppleNLContextualModel: EmbeddingModel {
    // MARK: - Identity
    public nonisolated let id: ModelID
    public nonisolated let dimensions: Int
    public nonisolated let device: ComputeDevice = .cpu

    // MARK: - State
    private let logger = Logger(label: "EmbedKit.AppleNLContextualModel")
    private let language: String
    private var configuration: EmbeddingConfiguration
    private var metricsData = MetricsData()
    private let profiler: Profiler?

    private let embedding: NLContextualEmbedding

    // MARK: - Init
    public init(
        language: String = NLLanguage.english.rawValue,
        configuration: EmbeddingConfiguration = EmbeddingConfiguration(),
        id: ModelID? = nil,
        dimensions: Int = 0,
        profiler: Profiler? = nil
    ) throws {
        self.language = language
        self.configuration = configuration
        self.id = id ?? ModelID(provider: "apple", name: "nl-contextual", version: "system")
        self.dimensions = dimensions // 0 means unknown until first call
        self.profiler = profiler

        let lang = NLLanguage(rawValue: language)
        guard let emb = NLContextualEmbedding(language: lang) else {
            throw EmbedKitError.modelLoadFailed("NLContextualEmbedding unavailable for language \(language)")
        }
        self.embedding = emb
    }

    // MARK: - EmbeddingModel
    public func embed(_ text: String) async throws -> Embedding {
        let t0 = CFAbsoluteTimeGetCurrent()

        try await ensureAssets()
        guard let result = try? embedding.embeddingResult(for: text, language: NLLanguage(rawValue: language)) else {
            throw EmbedKitError.inferenceFailed("Failed NLContextual embedding")
        }

        var pooled: [Double]? = nil
        var count = 0
        result.enumerateTokenVectors(in: text.startIndex..<text.endIndex) { (vec, _) in
            if pooled == nil { pooled = [Double](repeating: 0, count: vec.count) }
            guard let _ = pooled, pooled!.count == vec.count else { return false }
            for i in 0..<vec.count { pooled![i] += vec[i] }
            count += 1
            return true
        }
        guard var sum = pooled, count > 0 else {
            throw EmbedKitError.inferenceFailed("No token vectors from NLContextualEmbedding")
        }
        let denom = Double(count)
        for i in 0..<sum.count { sum[i] /= denom }
        var out = sum.map { Float($0) }
        if configuration.normalizeOutput {
            let mag = max(1e-12, sqrt(out.reduce(0) { $0 + $1 * $1 }))
            let inv = Float(1.0 / mag)
            for i in 0..<out.count { out[i] *= inv }
        }

        let dt = CFAbsoluteTimeGetCurrent() - t0
        metricsData.record(tokenCount: count, time: dt)
        profiler?.recordStage(model: id, items: 1, tokenization: 0, inference: dt, pooling: 0, context: ["path": "nl-single"])

        return Embedding(
            vector: out,
            metadata: EmbeddingMetadata(
                modelID: id,
                tokenCount: count,
                processingTime: dt,
                normalized: configuration.normalizeOutput,
                poolingStrategy: .mean,
                truncated: false,
                custom: [:]
            )
        )
    }

    public func embedBatch(_ texts: [String], options: BatchOptions) async throws -> [Embedding] {
        // Parallelize embedding across texts; ignore padding/batching knobs
        var concurrency = options.tokenizationConcurrency ?? 0
        if concurrency <= 0 { concurrency = min(8, max(1, ProcessInfo.processInfo.activeProcessorCount)) }

        var results = Array<Embedding?>(repeating: nil, count: texts.count)
        try await withThrowingTaskGroup(of: (Int, Embedding).self) { group in
            let chunk = max(1, texts.count / concurrency)
            for (idx, text) in texts.enumerated() {
                group.addTask {
                    let emb = try await self.embed(text)
                    return (idx, emb)
                }
                if (idx + 1) % chunk == 0 { /* allow backpressure */ }
            }
            for try await (idx, emb) in group { results[idx] = emb }
        }
        return results.compactMap { $0 }
    }

    public func warmup() async throws {
        try await ensureAssets()
    }

    public func release() async throws {
        // No-op for NLContextualEmbedding
    }

    public var metrics: ModelMetrics {
        get async { metricsData.snapshot(memoryUsage: 0, cacheHitRate: 0.0) }
    }

    public func resetMetrics() async throws {
        metricsData = MetricsData()
    }

    // MARK: - Helpers
    private func ensureAssets() async throws {
        if !embedding.hasAvailableAssets {
            _ = try await embedding.requestAssets()
            logger.info("Downloaded NLContextualEmbedding assets", metadata: ["language": .string(language)])
        }
    }
}

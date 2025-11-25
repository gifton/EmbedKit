// Test target: EmbedKitTests (migrated)
import Testing
@testable import EmbedKit

@Suite("Stage Metrics Reset")
struct StageMetricsResetTests {
    @Test
    func resetClearsStageAveragesAndCacheStats() async throws {
        let backend = NoOpBackend()
        let tokenizer = SimpleTokenizer()
        var cfg = EmbeddingConfiguration()
        cfg.includeSpecialTokens = false

        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            id: ModelID(provider: "test", name: "resetmetrics", version: "1.0"),
            dimensions: 4,
            device: .cpu
        )

        // Run a few embeds to populate metrics and cache stats
        _ = try await model.embed("a b c")
        _ = try await model.embed("a b c") // cache should hit for tokenization

        let sm1 = await model.stageMetricsSnapshot
        #expect(sm1.samples >= 1)
        let m1 = await model.metrics
        #expect(m1.cacheHitRate >= 0) // non-negative

        // Reset and validate cleared state
        try await model.resetMetrics()
        let sm2 = await model.stageMetricsSnapshot
        #expect(sm2.samples == 0)
        let m2 = await model.metrics
        #expect(m2.cacheHitRate == 0)

        // Subsequent embed repopulates
        _ = try await model.embed("a b c")
        let sm3 = await model.stageMetricsSnapshot
        #expect(sm3.samples >= 1)
    }
}

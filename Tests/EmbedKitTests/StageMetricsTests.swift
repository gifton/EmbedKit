import Testing
@testable import EmbedKit

@Suite("Stage Metrics")
struct StageMetricsTestsSuite {
@Test
func stageMetrics_updatesAfterEmbed() async throws {
    let backend = NoOpBackend()
    let tokenizer = SimpleTokenizer()
    let cfg = EmbeddingConfiguration(
        maxTokens: 8,
        paddingStrategy: .none,
        includeSpecialTokens: false
    )
    let model = AppleEmbeddingModel(
        backend: backend,
        tokenizer: tokenizer,
        configuration: cfg,
        id: ModelID(provider: "test", name: "stagemetrics", version: "1.0"),
        dimensions: 4,
        device: .cpu
    )

    _ = try await model.embed("a b c d")
    let sm = await model.stageMetricsSnapshot
    #expect(sm.samples >= 1)
    #expect(sm.tokenizationAverage >= 0)
    #expect(sm.inferenceAverage >= 0)
    #expect(sm.poolingAverage >= 0)
}
}

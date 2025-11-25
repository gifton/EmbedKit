import Testing
@testable import EmbedKit

@Suite("Week1 Integration")
struct Week1IntegrationTestsSuite {
@Test
func week1_mockModelEmbeddingAndMetrics() async throws {
    let manager = ModelManager()
    let model = try await manager.loadMockModel()
    // Single embed
    let emb = try await model.embed("hello v2")
    #expect(emb.dimensions == 384)
    #expect(!emb.vector.isEmpty)
    // Metrics should reflect at least one request
    let m = await model.metrics
    #expect(m.totalRequests >= 1)
    #expect(m.totalTokensProcessed > 0)
}
}


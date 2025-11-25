import Testing
@testable import EmbedKit

actor NoOpBackend: CoreMLProcessingBackend {
    private(set) var isLoaded: Bool = false
    var memoryUsage: Int64 { 0 }
    func load() async throws { isLoaded = true }
    func unload() async throws { isLoaded = false }
    func process(_ input: CoreMLInput) async throws -> CoreMLOutput {
        // Return [tokens, dim=4] zeros
        let dim = 4
        return CoreMLOutput(values: Array(repeating: 0, count: input.tokenIDs.count * dim), shape: [input.tokenIDs.count, dim])
    }
    func processBatch(_ inputs: [CoreMLInput]) async throws -> [CoreMLOutput] {
        let dim = 4
        return inputs.map { inp in
            CoreMLOutput(values: Array(repeating: 0, count: inp.tokenIDs.count * dim), shape: [inp.tokenIDs.count, dim])
        }
    }
}

@Suite("Truncation Flag")
struct TruncationFlagTestsSuite {
@Test
func truncationFlag_embed_maxPadding() async throws {
    let backend = NoOpBackend()
    let vocab = Vocabulary(tokens: ["[PAD]","a","b","c","d","e"]) // PAD for batch padding
    let tokenizer = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
    var cfg = EmbeddingConfiguration()
    cfg.includeSpecialTokens = false
    cfg.maxTokens = 4
    cfg.paddingStrategy = .max

    let model = AppleEmbeddingModel(
        backend: backend,
        tokenizer: tokenizer,
        configuration: cfg,
        id: ModelID(provider: "test", name: "trunc", version: "1.0"),
        dimensions: 4,
        device: .cpu
    )

    // 6 tokens -> should be truncated (originalLen 6 > 4)
    let emb = try await model.embed("a b c d e f")
    #expect(emb.metadata.truncated == true)
}

@Test
func truncationFlag_embedBatch_batchPadding() async throws {
    let backend = NoOpBackend()
    let vocab = Vocabulary(tokens: ["[PAD]","a","b","c","d","e"]) // PAD for batch padding
    let tokenizer = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
    var cfg = EmbeddingConfiguration()
    cfg.includeSpecialTokens = false
    cfg.maxTokens = 4
    cfg.paddingStrategy = .batch

    let model = AppleEmbeddingModel(
        backend: backend,
        tokenizer: tokenizer,
        configuration: cfg,
        id: ModelID(provider: "test", name: "trunc-batch", version: "1.0"),
        dimensions: 4,
        device: .cpu
    )

    // First truncated, others not
    let texts = ["a b c d e", "a b", "a b c"]
    let embs = try await model.embedBatch(texts, options: .init())
    #expect(embs.count == 3)
    #expect(embs[0].metadata.truncated == true)
    #expect(embs[1].metadata.truncated == false)
    #expect(embs[2].metadata.truncated == false)
}
}

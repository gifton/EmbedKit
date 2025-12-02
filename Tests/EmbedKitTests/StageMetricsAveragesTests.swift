import Testing
@testable import EmbedKit

@Suite("Stage Metrics Averages")
struct StageMetricsAveragesTests {
    actor NoOpBackend: CoreMLProcessingBackend {
        private(set) var isLoaded: Bool = false
        var memoryUsage: Int64 { 0 }
        func load() async throws { isLoaded = true }
        func unload() async throws { isLoaded = false }
        func process(_ input: CoreMLInput) async throws -> CoreMLOutput {
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

    struct SlowTokenizer: Tokenizer {
        let base: WordPieceTokenizer
        let delayMicros: UInt64
        init(vocab: Vocabulary, delayMicros: UInt64 = 5_000) {
            self.base = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
            self.delayMicros = delayMicros
        }
        var vocabularySize: Int { base.vocabularySize }
        var specialTokens: SpecialTokens { base.specialTokens }
        func encode(_ text: String, config: TokenizerConfig) async throws -> TokenizedText {
            try await Task.sleep(nanoseconds: delayMicros * 1_000) // ~5ms
            return try await base.encode(text, config: config)
        }
        func decode(_ ids: [Int]) async throws -> String { try await base.decode(ids) }
    }

    @Test
    func tokenizationAverage_reflectsTotalTokenizationTimePerItem() async throws {
        let backend = NoOpBackend()
        let vocab = Vocabulary(tokens: ["[PAD]","[CLS]","[SEP]","[UNK]","a","b","c","d","e","f"])
        let tokenizer = SlowTokenizer(vocab: vocab, delayMicros: 5_000) // ~5ms per encode
        let cfg = EmbeddingConfiguration(
            maxTokens: 32,
            paddingStrategy: .none,
            includeSpecialTokens: false,
            poolingStrategy: .mean,
            normalizeOutput: false
        )
        let model = AppleEmbeddingModel(backend: backend, tokenizer: tokenizer, configuration: cfg, id: ModelID(provider: "test", name: "sm", version: "1.0"), dimensions: 4, device: .cpu)

        // Use unique texts to avoid cache hits (cache would skip the tokenizer delay)
        let texts = ["a b c", "a b d", "a b e", "a b f", "c d e", "d e f"]
        // Force sequential tokenization to get predictable timing
        let opts = BatchOptions(
            tokenizationConcurrency: 1
        )
        _ = try await model.embedBatch(texts, options: opts)
        let sm = await model.stageMetricsSnapshot
        // Expect ~5ms per item average for tokenization; allow generous bounds for system variance
        #expect(sm.tokenizationAverage >= 0.003)
        #expect(sm.tokenizationAverage < 1.0)
    }
}


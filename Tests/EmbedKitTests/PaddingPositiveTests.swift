// Test target: EmbedKitTests (migrated)
import Testing
@testable import EmbedKit

@Suite("Padding Positive Paths")
struct PaddingPositiveTests {
    @Test
    func paddingMax_withPAD_succeeds() async throws {
        let vocab = Vocabulary(tokens: ["[PAD]","[CLS]","[SEP]","[UNK]","[MASK]","hello"])
        let tok = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
        var cfg = TokenizerConfig()
        cfg.addSpecialTokens = true
        cfg.padding = .max
        cfg.maxLength = 6
        let out = try await tok.encode("hello", config: cfg)
        #expect(out.ids.count == 6)
        #expect(out.ids[0] == (vocab["[CLS]"] ?? -1))
        #expect(out.ids[1] == (vocab["hello"] ?? -1))
        #expect(out.ids[2] == (vocab["[SEP]"] ?? -1))
        #expect(out.ids[3] == (vocab["[PAD]"] ?? -1))
        #expect(out.attentionMask == [1,1,1,0,0,0])
    }

    @Test
    func batchPadding_withPAD_succeeds() async throws {
        actor RecBackend: CoreMLProcessingBackend {
            private(set) var isLoaded: Bool = false
            var memoryUsage: Int64 { 0 }
            private(set) var lengths: [Int] = []
            func load() async throws { isLoaded = true }
            func unload() async throws { isLoaded = false }
            func process(_ input: CoreMLInput) async throws -> CoreMLOutput { CoreMLOutput(values: Array(repeating: 0, count: input.tokenIDs.count * 4), shape: [input.tokenIDs.count, 4]) }
            func processBatch(_ inputs: [CoreMLInput]) async throws -> [CoreMLOutput] {
                lengths = inputs.map { $0.tokenIDs.count }
                return inputs.map { inp in CoreMLOutput(values: Array(repeating: 0, count: inp.tokenIDs.count * 4), shape: [inp.tokenIDs.count, 4]) }
            }
        }

        let backend = RecBackend()
        let vocab = Vocabulary(tokens: ["[PAD]","[CLS]","[SEP]","[UNK]","a","b","c"])
        let tok = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
        var cfg = EmbeddingConfiguration()
        cfg.includeSpecialTokens = false
        cfg.paddingStrategy = .batch
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tok,
            configuration: cfg,
            id: ModelID(provider: "test", name: "padok", version: "1.0"),
            dimensions: 4,
            device: .cpu
        )
        let texts = ["a b c", "a"]
        _ = try await model.embedBatch(texts, options: .init())
        let lengths = await backend.lengths
        #expect(!lengths.isEmpty)
        #expect(lengths.dropFirst().allSatisfy { $0 == lengths.first })
    }
}

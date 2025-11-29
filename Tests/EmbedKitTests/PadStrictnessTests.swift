import Testing
@testable import EmbedKit

@Suite("PAD Strictness")
struct PadStrictnessTestsSuite {
@Test
func paddingMax_requiresPadToken() async {
    // Tokenizer without PAD token
    let vocab = Vocabulary(tokens: ["[CLS]","[SEP]","[UNK]","hello"]) // no [PAD]
    let tok = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
    let cfg = TokenizerConfig(
        maxLength: 8,
        padding: .max,
        addSpecialTokens: true
    )
    do {
        _ = try await tok.encode("hello", config: cfg)
        #expect(Bool(false), "Expected invalidConfiguration when PAD missing for padding .max")
    } catch {
        guard let ek = error as? EmbedKitError else { return #expect(Bool(false), "Unexpected error: \(error)") }
        switch ek {
        case .invalidConfiguration:
            #expect(true)
        default:
            #expect(Bool(false), "Unexpected error: \(ek)")
        }
    }
}

@Test
func batchPadding_requiresPadToken() async {
    // Model using tokenizer without PAD; should throw on batch padding
    let backend = NoOpBackend()
    let vocab = Vocabulary(tokens: ["[CLS]","[SEP]","[UNK]","hello"]) // no [PAD]
    let tok = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
    let cfg = EmbeddingConfiguration(
        paddingStrategy: .batch,
        includeSpecialTokens: true
    )
    let model = AppleEmbeddingModel(
        backend: backend,
        tokenizer: tok,
        configuration: cfg,
        id: ModelID(provider: "test", name: "padstrict", version: "1.0"),
        dimensions: 4,
        device: .cpu
    )
    do {
        _ = try await model.embedBatch(["hello", "hello world"], options: BatchOptions())
        #expect(Bool(false), "Expected invalidConfiguration when PAD missing for batch padding")
    } catch {
        guard let ek = error as? EmbedKitError else { return #expect(Bool(false), "Unexpected error: \(error)") }
        switch ek {
        case .invalidConfiguration:
            #expect(true)
        default:
            #expect(Bool(false), "Unexpected error: \(ek)")
        }
    }
}
}

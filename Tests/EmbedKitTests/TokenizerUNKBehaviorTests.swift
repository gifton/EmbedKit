// Test target: EmbedKitTests (migrated)
import Testing
@testable import EmbedKit

@Suite("Tokenizer UNK Behavior")
struct TokenizerUNKBehaviorTests {
    @Test
    func longWord_overMaxInputChars_mapsToUNK() async throws {
        let vocab = Vocabulary(tokens: ["[PAD]","[CLS]","[SEP]","[UNK]"])
        let tok = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
        var cfg = TokenizerConfig(); cfg.addSpecialTokens = true
        let long = String(repeating: "a", count: 210)
        let out = try await tok.encode(long, config: cfg)
        #expect(out.ids.count == 3)
        #expect(out.ids[1] == (vocab["[UNK]"] ?? -1))
    }

    @Test
    func unicodeSymbols_emojis_mapToUNK() async throws {
        let vocab = Vocabulary(tokens: ["[PAD]","[CLS]","[SEP]","[UNK]","hello"]) // no emoji
        let tok = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
        var cfg = TokenizerConfig(); cfg.addSpecialTokens = true
        let out = try await tok.encode("hello 😀", config: cfg)
        #expect(out.ids.count == 4)
        #expect(out.ids[1] == (vocab["hello"] ?? -1))
        #expect(out.ids[2] == (vocab["[UNK]"] ?? -1))
    }
}

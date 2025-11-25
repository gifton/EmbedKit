// Test target: EmbedKitTests (migrated)
import Testing
@testable import EmbedKit

@Suite("Tokenizer Subword Decode")
struct TokenizerSubwordDecodeTests {
    @Test
    func mergesSubwordsAndSkipsSpecials_roundTrip() async throws {
        // Vocab with subword piece for "embedding" -> "embed" + "##ding"
        let tokens = [
            "[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]",
            "hello", "world", "embed", "##ding"
        ]
        let vocab = Vocabulary(tokens: tokens)
        let tok = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)

        var cfg = TokenizerConfig()
        cfg.addSpecialTokens = true
        let out = try await tok.encode("Hello world embedding", config: cfg)

        // Expect tokens with specials and subword split
        #expect(out.tokens == ["[CLS]", "hello", "world", "embed", "##ding", "[SEP]"])

        let decoded = try await tok.decode(out.ids)
        // Specials removed, subwords merged, lowercased as per encode behavior
        #expect(decoded == "hello world embedding")
    }

    @Test
    func lowercasing_affectsIDsButDecodeIsStable() async throws {
        let tokens = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]", "hello"]
        let vocab = Vocabulary(tokens: tokens)

        var cfg = TokenizerConfig(); cfg.addSpecialTokens = true

        // Lowercasing ON: "HELLO" should map to "hello"
        let tLower = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
        let outLower = try await tLower.encode("HELLO", config: cfg)
        #expect(outLower.ids.count == 3)
        #expect(outLower.ids[1] == (vocab["hello"] ?? -1))
        let decLower = try await tLower.decode(outLower.ids)
        #expect(decLower == "hello")

        // Lowercasing OFF: with only lowercase in vocab, "HELLO" becomes UNK
        let tNoLower = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: false)
        let outNoLower = try await tNoLower.encode("HELLO", config: cfg)
        #expect(outNoLower.ids.count == 3)
        #expect(outNoLower.ids[1] == (vocab["[UNK]"] ?? -1))
        let decNoLower = try await tNoLower.decode(outNoLower.ids)
        // Decoding UNK results in literal [UNK]
        #expect(decNoLower == "[UNK]")
    }
}

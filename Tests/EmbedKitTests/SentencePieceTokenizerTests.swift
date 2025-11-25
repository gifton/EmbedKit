import Testing
@testable import EmbedKit

@Suite("SentencePiece Tokenizer")
struct SentencePieceTokenizerTests {
    private func makeVocab() -> Vocabulary {
        // Order defines ids
        // Include "▁w" for character fallback tests (word-initial marker + character)
        let tokens = [
            "[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]",
            "▁hello", "▁world", "▁h", "▁w", "e", "l", "o", "w", "r", "d", "u"
        ]
        return Vocabulary(tokens: tokens)
    }

    @Test
    func basic_wholeWordMatches_withSpecials() async throws {
        let vocab = makeVocab()
        let tok = SentencePieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
        var cfg = TokenizerConfig()
        cfg.addSpecialTokens = true
        let out = try await tok.encode("Hello world", config: cfg)
        #expect(out.tokens == ["[CLS]", "▁hello", "▁world", "[SEP]"])
    }

    @Test
    func fallback_toCharPieces() async throws {
        let vocab = makeVocab()
        let tok = SentencePieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
        var cfg = TokenizerConfig()
        cfg.addSpecialTokens = false
        // "wurld" not present; expect fallback to character pieces: "▁w","u","r","l","d"
        let out = try await tok.encode("wurld", config: cfg)
        #expect(out.tokens.first?.hasPrefix("▁") == true)
        #expect(out.tokens.count >= 4) // At least "▁w", "u", "r", "l", "d" or subsets
        #expect(out.attentionMask.count == out.ids.count)
    }

    @Test
    func padding_max_and_truncation() async throws {
        let vocab = makeVocab()
        let tok = SentencePieceTokenizer(vocabulary: vocab)
        var cfg = TokenizerConfig()
        cfg.addSpecialTokens = true
        cfg.maxLength = 8
        cfg.padding = .max
        let out = try await tok.encode("hello", config: cfg)
        #expect(out.tokens.count == 8)
        // Change to truncation
        cfg.padding = .none
        cfg.maxLength = 3
        cfg.truncation = .end
        let out2 = try await tok.encode("hello", config: cfg)
        #expect(out2.tokens.count == 3)
    }
}


import Testing
@testable import EmbedKit

@Suite("BPE Tokenizer")
struct BPETokenizerTests {
    private func makeVocabAndMerges() -> (Vocabulary, [String: String]) {
        // Tokens are indexed by array order
        let tokens = [
            "[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]",
            "h", "e", "l", "o", "w", "r", "d",
            "he", "ll", "llo"
        ]
        let vocab = Vocabulary(tokens: tokens)
        // Merges: h+e -> he, l+l -> ll, ll+o -> llo
        let merges = [
            "h e": "he",
            "l l": "ll",
            "ll o": "llo"
        ]
        return (vocab, merges)
    }

    @Test
    func basicMerges_withSpecials() async throws {
        let (vocab, merges) = makeVocabAndMerges()
        let tok = BPETokenizer(vocabulary: vocab, merges: merges, unkToken: "[UNK]", lowercase: true)
        let cfg = TokenizerConfig(addSpecialTokens: true)
        let out = try await tok.encode("Hello world", config: cfg)
        // Expect: [CLS], he, llo, w, o, r, l, d, [SEP]
        let expected = ["[CLS]", "he", "llo", "w", "o", "r", "l", "d", "[SEP]"]
        #expect(out.tokens == expected)
        // Spot-check IDs for merged tokens
        #expect(out.ids[1] == vocab["he"]) // he
        #expect(out.ids[2] == vocab["llo"]) // llo
    }

    @Test
    func truncation_end() async throws {
        let (vocab, merges) = makeVocabAndMerges()
        let tok = BPETokenizer(vocabulary: vocab, merges: merges)
        let cfg = TokenizerConfig(
            maxLength: 5,
            truncation: .end,
            addSpecialTokens: true
        )
        let out = try await tok.encode("hello world", config: cfg)
        #expect(out.tokens.count == 5)
        #expect(out.attentionMask.count == 5)
    }

    @Test
    func padding_max() async throws {
        let (vocab, merges) = makeVocabAndMerges()
        let tok = BPETokenizer(vocabulary: vocab, merges: merges)
        let cfg = TokenizerConfig(
            maxLength: 10,
            padding: .max,
            addSpecialTokens: true
        )
        let out = try await tok.encode("hello", config: cfg)
        #expect(out.tokens.count == 10)
        // Padding zeros after actual tokens in mask
        let ones = out.attentionMask.prefix { $0 == 1 }.count
        #expect(ones < 10)
    }
}


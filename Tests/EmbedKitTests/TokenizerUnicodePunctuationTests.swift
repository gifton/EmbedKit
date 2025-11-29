// Test target: EmbedKitTests (migrated)
import Testing
@testable import EmbedKit

@Suite("Unicode & Punctuation")
struct TokenizerUnicodePunctuationTests {
    private func makeVocab() -> Vocabulary {
        // Minimal vocab with standard specials and a couple of words
        let tokens = [
            "[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]",
            "hello", "world"
        ]
        return Vocabulary(tokens: tokens)
    }

    private func ids(_ pieces: [String], vocab: Vocabulary) -> [Int] {
        let unk = vocab["[UNK]"] ?? 3
        return pieces.map { vocab[$0] ?? unk }
    }

    @Test
    func asciiPunctuationProducesUNK() async throws {
        let vocab = makeVocab()
        let tok = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
        let cfg = TokenizerConfig(
            addSpecialTokens: true
        )
        let out = try await tok.encode("Hello, world!", config: cfg)
        // Expect: [CLS] hello [UNK] world [UNK] [SEP]
        let expectedTokens = ["[CLS]", "hello", "[UNK]", "world", "[UNK]", "[SEP]"]
        #expect(out.ids == ids(expectedTokens, vocab: vocab))
        #expect(out.attentionMask == Array(repeating: 1, count: expectedTokens.count))
    }

    @Test
    func unicodePunctuationAndEmoji() async throws {
        let vocab = makeVocab()
        let tok = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
        let cfg = TokenizerConfig(
            addSpecialTokens: true
        )        // Em dash and ellipsis (— …) with emoji
        let out = try await tok.encode("Hello — world… 😀", config: cfg)
        // Tokens split as: hello, —, world, …, 😀 → unknown except known words
        let expectedTokens = ["[CLS]", "hello", "[UNK]", "world", "[UNK]", "[UNK]", "[SEP]"]
        #expect(out.ids == ids(expectedTokens, vocab: vocab))
        #expect(out.attentionMask == Array(repeating: 1, count: expectedTokens.count))
    }

    @Test
    func nonLatinScriptsFallbackToUNK() async throws {
        let vocab = makeVocab()
        let tok = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
        let cfg = TokenizerConfig(
            addSpecialTokens: true
        )        // Chinese and Arabic examples
        let out = try await tok.encode("你好，世界 — مرحبا بالعالم", config: cfg)
        // All unknowns except specials because vocab doesn't include these scripts
        // We only assert the count and that all non-special are UNK
        #expect(out.ids.first == vocab["[CLS]"])
        #expect(out.ids.last == vocab["[SEP]"])
        // All interior tokens should be UNK
        let interior = out.ids.dropFirst().dropLast()
        let unk = vocab["[UNK]"] ?? 3
        #expect(interior.allSatisfy { $0 == unk })
    }
}

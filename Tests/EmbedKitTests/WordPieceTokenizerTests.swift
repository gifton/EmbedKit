import Testing
@testable import EmbedKit

@Suite("WordPiece Tokenizer")
struct WordPieceTokenizerTests {
@Test
func wordPieceTokenizer_basicEncodingWithSpecialTokens() async throws {
    
    func makeVocab() -> Vocabulary {
        // Fixed order so IDs are deterministic
        let tokens = [
            "[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]",
            "hello", "world", "embed", "##ding"
        ]
        return Vocabulary(tokens: tokens)
    }
    let vocab = makeVocab()
    let tok = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
    var cfg = TokenizerConfig()
    cfg.addSpecialTokens = true

    let out = try await tok.encode("Hello world embedding", config: cfg)
    #expect(out.tokens == ["[CLS]", "hello", "world", "embed", "##ding", "[SEP]"])
    #expect(out.ids == [1, 5, 6, 7, 8, 2])
    #expect(out.attentionMask == [1, 1, 1, 1, 1, 1])
}

@Test
func wordPieceTokenizer_truncationEnd() async throws {
    func makeVocab() -> Vocabulary { Vocabulary(tokens: ["[PAD]","[CLS]","[SEP]","[UNK]","[MASK]","hello","world","embed","##ding"]) }
    let vocab = makeVocab()
    let tok = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
    var cfg = TokenizerConfig()
    cfg.addSpecialTokens = true
    cfg.maxLength = 5
    cfg.truncation = .end
    let out = try await tok.encode("Hello world embedding", config: cfg)
    #expect(out.tokens == ["[CLS]", "hello", "world", "embed", "##ding"]) 
    #expect(out.ids == [1, 5, 6, 7, 8])
    #expect(out.attentionMask == [1, 1, 1, 1, 1])
}

@Test
func wordPieceTokenizer_paddingMax() async throws {
    func makeVocab() -> Vocabulary { Vocabulary(tokens: ["[PAD]","[CLS]","[SEP]","[UNK]","[MASK]","hello","world","embed","##ding"]) }
    let vocab = makeVocab()
    let tok = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
    var cfg = TokenizerConfig()
    cfg.addSpecialTokens = true
    cfg.maxLength = 8
    cfg.padding = .max
    let out = try await tok.encode("Hello world embedding", config: cfg)
    #expect(out.tokens == ["[CLS]", "hello", "world", "embed", "##ding", "[SEP]", "[PAD]", "[PAD]"])
    #expect(out.ids == [1, 5, 6, 7, 8, 2, 0, 0])
    #expect(out.attentionMask == [1, 1, 1, 1, 1, 1, 0, 0])
}

@Test
func wordPieceTokenizer_unknownFallsBack() async throws {
    let vocab = Vocabulary(tokens: ["[PAD]","[CLS]","[SEP]","[UNK]","[MASK]"])
    let tok = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
    var cfg = TokenizerConfig()
    cfg.addSpecialTokens = true
    let out = try await tok.encode("xyz", config: cfg)
    #expect(out.tokens == ["[CLS]", "[UNK]", "[SEP]"])
    #expect(out.ids == [1, 3, 2])
    #expect(out.attentionMask == [1, 1, 1])
}
}

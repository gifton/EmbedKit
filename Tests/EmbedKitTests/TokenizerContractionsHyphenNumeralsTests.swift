import Testing
@testable import EmbedKit

@Suite("Tokenizer Contractions/Hyphen/Numerals")
struct TokenizerContractionsHyphenNumeralsTests {
@Test
func contractions_splitIntoUNKsWhenNotInVocab() async throws {
    let tokens = ["[PAD]","[CLS]","[SEP]","[UNK]","stop"]
    let vocab = Vocabulary(tokens: tokens)
    let tok = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
    let cfg = TokenizerConfig(
        addSpecialTokens: true
    )
    let out = try await tok.encode("Don't stop", config: cfg)
    // Expect: [CLS], UNK (don), UNK ('), UNK (t), stop, [SEP]
    #expect(out.ids.count == 6)
    #expect(out.ids[1] == vocab["[UNK]"])
    #expect(out.ids[2] == vocab["[UNK]"])
    #expect(out.ids[3] == vocab["[UNK]"])
    #expect(out.ids[4] == vocab["stop"]) 
}

@Test
func hyphenation_splitsAndRecognizesKnownWords() async throws {
    let tokens = ["[PAD]","[CLS]","[SEP]","[UNK]","hello","world"]
    let vocab = Vocabulary(tokens: tokens)
    let tok = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
    let cfg = TokenizerConfig(
        addSpecialTokens: true
    )
    let out = try await tok.encode("Hello-world", config: cfg)
    // Expect: [CLS], hello, UNK(-), world, [SEP]
    #expect(out.ids.count == 5)
    #expect(out.ids[1] == vocab["hello"]) 
    #expect(out.ids[2] == vocab["[UNK]"]) 
    #expect(out.ids[3] == vocab["world"]) 
}

@Test
func numerals_splitAroundPunctuation() async throws {
    let tokens = ["[PAD]","[CLS]","[SEP]","[UNK]","ver"]
    let vocab = Vocabulary(tokens: tokens)
    let tok = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
    let cfg = TokenizerConfig(
        addSpecialTokens: true
    )
    let out = try await tok.encode("ver 2.0", config: cfg)
    // Expect: [CLS], ver, UNK(2), UNK(.), UNK(0), [SEP]
    #expect(out.ids.count == 6)
    #expect(out.ids[1] == vocab["ver"]) 
    #expect(out.ids[2] == vocab["[UNK]"]) 
    #expect(out.ids[3] == vocab["[UNK]"]) 
    #expect(out.ids[4] == vocab["[UNK]"]) 
}
}

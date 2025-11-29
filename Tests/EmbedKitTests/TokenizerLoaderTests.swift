import Testing
@testable import EmbedKit
import Foundation

@Suite("Tokenizer Loader Tests")
struct TokenizerLoaderTests {
    private func writeTempFile(contents: String, name: String) throws -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        let url = dir.appendingPathComponent(name)
        try contents.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    @Test
    func bpe_loadsFromFiles() async throws {
        // Vocab
        let tokens = [
            "[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]",
            "h", "e", "l", "o", "w", "r", "d",
            "he", "ll", "llo"
        ]
        let vocabURL = try writeTempFile(contents: tokens.joined(separator: "\n"), name: "bpe_vocab.txt")
        // Merges
        let merges = ["h e", "l l", "ll o"].joined(separator: "\n")
        let mergesURL = try writeTempFile(contents: merges, name: "bpe_merges.txt")

        let tok = try BPETokenizer.load(vocabURL: vocabURL, mergesURL: mergesURL)
        let cfg = TokenizerConfig(
            addSpecialTokens: true
        )
        let out = try await tok.encode("hello", config: cfg)
        // Expect [CLS], he, llo, [SEP]
        #expect(out.tokens.first == "[CLS]")
        #expect(out.tokens.contains("he"))
        #expect(out.tokens.contains("llo"))
    }

    @Test
    func sentencepiece_loadsFromVocab() async throws {
        // sp.vocab-like content: token score
        let spLines = [
            "[PAD] 0", "[CLS] 0", "[SEP] 0", "[UNK] 0", "[MASK] 0",
            "▁hello 0", "▁world 0", "▁h 0", "e 0", "l 0", "o 0", "w 0", "r 0", "d 0"
        ].joined(separator: "\n")
        let url = try writeTempFile(contents: spLines, name: "sp_vocab.txt")
        let tok = try SentencePieceTokenizer.load(spVocabURL: url)
        let cfg = TokenizerConfig(
            addSpecialTokens: true
        )
        let out = try await tok.encode("hello world", config: cfg)
        #expect(out.tokens == ["[CLS]", "▁hello", "▁world", "[SEP]"])
    }
}


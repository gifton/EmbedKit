// Test target: EmbedKitTests (migrated)
// NOTE: BertTokenizer was removed during refactor - this test is now obsolete
import Testing
@testable import EmbedKit

@Suite("Tokenizer Parity")
struct TokenizerParityTests {
    @Test
    func wordpiece_tokenizer_basic() async throws {
        // Basic WordPieceTokenizer test (BertTokenizer was removed)
        let tokens = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]", "hello", "world", "embed", "##ding"]
        let tokenizer = WordPieceTokenizer(vocabulary: Vocabulary(tokens: tokens), unkToken: "[UNK]", lowercase: true)

        var cfg = TokenizerConfig()
        cfg.addSpecialTokens = true
        cfg.maxLength = 0 // no truncation

        let result = try await tokenizer.encode("hello world", config: cfg)
        #expect(result.ids.count > 0, "Should produce tokens")
    }
}

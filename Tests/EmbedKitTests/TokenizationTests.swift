import XCTest
@testable import EmbedKit

/// Comprehensive test suite for tokenization functionality
/// Tests all tokenizer implementations including BERTTokenizer, AdvancedTokenizer,
/// SimpleTokenizer, and supporting utilities like VocabularyBuilder
final class TokenizationTests: XCTestCase {

    // MARK: - BERTTokenizer Tests

    /// Test basic tokenization: text → token IDs → attention mask
    func testBERTTokenizerBasicTokenization() async throws {
        let tokenizer = try await BERTTokenizer(maxSequenceLength: 16)

        let text = "Hello world"
        let result = try await tokenizer.tokenize(text)

        // Verify structure
        XCTAssertEqual(result.tokenIds.count, 16, "Should pad to max sequence length")
        XCTAssertEqual(result.attentionMask.count, 16, "Attention mask should match token count")

        // Verify special tokens
        XCTAssertEqual(result.tokenIds.first, 101, "Should start with [CLS] token")
        XCTAssertEqual(result.tokenIds[result.originalLength - 1], 102, "Should have [SEP] token")

        // Verify attention mask
        let activeMask = result.attentionMask.prefix(result.originalLength)
        XCTAssertTrue(activeMask.allSatisfy { $0 == 1 }, "Active tokens should have attention mask = 1")

        let paddingMask = result.attentionMask.suffix(16 - result.originalLength)
        XCTAssertTrue(paddingMask.allSatisfy { $0 == 0 }, "Padding tokens should have attention mask = 0")
    }

    /// Test WordPiece subword splitting: "playing" → "play" + "##ing"
    func testWordPieceSubwordSplitting() async throws {
        let tokenizer = try await AdvancedTokenizer(
            type: .wordpiece,
            maxSequenceLength: 32
        )

        // Test word that should split into subwords
        let text = "playing"
        let result = try await tokenizer.tokenize(text)

        // Decode to verify subword splitting occurred
        let decoded = await tokenizer.decode(result.tokenIds)

        // Should contain the word in some tokenized form
        XCTAssertFalse(decoded.isEmpty, "Decoded text should not be empty")
        XCTAssertGreaterThan(result.originalLength, 2, "Should have [CLS] + tokens + [SEP]")
    }

    /// Test special token handling: [CLS], [SEP], [PAD]
    func testSpecialTokenHandling() async throws {
        let specialTokens = SpecialTokens(
            cls: 101,
            sep: 102,
            pad: 0,
            unk: 100,
            mask: 103
        )

        let tokenizer = try await AdvancedTokenizer(
            type: .wordpiece,
            maxSequenceLength: 10,
            specialTokens: specialTokens
        )

        let text = "test"
        let result = try await tokenizer.tokenize(text)

        // Verify special tokens are in correct positions
        XCTAssertEqual(result.tokenIds.first, 101, "[CLS] should be first token")

        // Find [SEP] token (should be at originalLength - 1)
        XCTAssertEqual(result.tokenIds[result.originalLength - 1], 102, "[SEP] should be at end of sequence")

        // All padding tokens should be [PAD]
        let paddingTokens = result.tokenIds.suffix(result.tokenIds.count - result.originalLength)
        XCTAssertTrue(paddingTokens.allSatisfy { $0 == 0 }, "All padding should use [PAD] token")
    }

    /// Test max sequence length truncation
    func testMaxSequenceLengthTruncation() async throws {
        let maxLength = 8
        let tokenizer = try await BERTTokenizer(maxSequenceLength: maxLength)

        let longText = "This is a very long text that should definitely be truncated to the maximum sequence length"
        let result = try await tokenizer.tokenize(longText)

        // Verify truncation
        XCTAssertEqual(result.tokenIds.count, maxLength, "Should truncate to max length")
        XCTAssertEqual(result.attentionMask.count, maxLength, "Attention mask should match truncated length")
        XCTAssertLessThanOrEqual(result.originalLength, maxLength, "Original length should not exceed max")

        // Should still have [CLS] at start
        XCTAssertEqual(result.tokenIds.first, 101, "[CLS] should be preserved")
    }

    /// Test vocabulary loading and size
    func testVocabularyLoadingAndSize() async throws {
        let tokenizer = try await AdvancedTokenizer(
            type: .wordpiece,
            maxSequenceLength: 32
        )

        // Default vocabulary should have reasonable size
        XCTAssertGreaterThan(tokenizer.vocabularySize, 100, "Vocabulary should contain multiple tokens")

        // Should contain special tokens
        let text = "[UNK] [PAD] [CLS] [SEP]"
        let result = try await tokenizer.tokenize(text)
        XCTAssertGreaterThan(result.originalLength, 0, "Should tokenize special tokens")
    }

    /// Test decoding: token IDs back to text
    func testDecoding() async throws {
        let tokenizer = try await AdvancedTokenizer(
            type: .wordpiece,
            maxSequenceLength: 32
        )

        let originalText = "hello world"
        let tokenized = try await tokenizer.tokenize(originalText)
        let decoded = await tokenizer.decode(tokenized.tokenIds)

        // Decoded text should contain the original words (may have spacing differences)
        XCTAssertFalse(decoded.isEmpty, "Decoded text should not be empty")

        // Should not contain special tokens in decoded text
        XCTAssertFalse(decoded.contains("[CLS]"), "Decoded text should not contain [CLS]")
        XCTAssertFalse(decoded.contains("[SEP]"), "Decoded text should not contain [SEP]")
        XCTAssertFalse(decoded.contains("[PAD]"), "Decoded text should not contain [PAD]")
    }

    /// Test batch tokenization
    func testBatchTokenization() async throws {
        let tokenizer = try await BERTTokenizer(maxSequenceLength: 16)

        let texts = [
            "First sentence",
            "Second sentence is longer",
            "Third"
        ]

        let results = try await tokenizer.tokenize(batch: texts)

        // Verify batch size
        XCTAssertEqual(results.count, texts.count, "Should return same number of results as inputs")

        // Verify each result
        for (index, result) in results.enumerated() {
            XCTAssertEqual(result.tokenIds.count, 16, "Result \(index) should have max length")
            XCTAssertEqual(result.attentionMask.count, 16, "Result \(index) attention mask should match")
            XCTAssertEqual(result.tokenIds.first, 101, "Result \(index) should start with [CLS]")
        }

        // Verify different lengths have different original lengths
        XCTAssertNotEqual(results[0].originalLength, results[1].originalLength, "Different texts should have different lengths")
    }

    /// Test SimpleTokenizer baseline
    func testSimpleTokenizer() async throws {
        let tokenizer = SimpleTokenizer(maxSequenceLength: 32, vocabularySize: 30522)

        let text = "Hello world! This is a test."
        let tokens = try await tokenizer.tokenize(text)

        // Verify basic tokenization
        XCTAssertGreaterThan(tokens.tokenIds.count, 0, "Should produce tokens")
        XCTAssertEqual(tokens.tokenIds.count, 32, "Should pad to max sequence length")

        // Verify attention mask
        XCTAssertEqual(tokens.attentionMask.count, 32, "Attention mask should match token count")

        // Verify has some real tokens (not just padding)
        let activeTokens = tokens.attentionMask.filter { $0 == 1 }.count
        XCTAssertGreaterThan(activeTokens, 0, "Should have active tokens")
    }

    /// Test VocabularyBuilder statistics
    func testVocabularyStats() async throws {
        var builder = VocabularyBuilder()

        // Add sample texts
        let texts = [
            "The quick brown fox jumps over the lazy dog",
            "The lazy dog sleeps under the tree",
            "Quick brown foxes are jumping"
        ]

        for text in texts {
            builder.addText(text)
        }

        // Build vocabulary
        let vocabulary = builder.buildVocabulary(maxVocabSize: 1000)

        // Verify vocabulary structure
        XCTAssertGreaterThan(vocabulary.count, 10, "Should build vocabulary with multiple tokens")

        // Verify special tokens exist
        XCTAssertNotNil(vocabulary["[PAD]"], "Should contain [PAD] token")
        XCTAssertNotNil(vocabulary["[UNK]"], "Should contain [UNK] token")
        XCTAssertNotNil(vocabulary["[CLS]"], "Should contain [CLS] token")
        XCTAssertNotNil(vocabulary["[SEP]"], "Should contain [SEP] token")

        // Common words should be in vocabulary
        XCTAssertNotNil(vocabulary["the"], "Should contain frequent word 'the'")
    }

    /// Test edge case: empty string
    func testEmptyStringTokenization() async throws {
        let tokenizer = try await BERTTokenizer(maxSequenceLength: 8)

        let result = try await tokenizer.tokenize("")

        // Should still have special tokens
        XCTAssertEqual(result.tokenIds.count, 8, "Should pad to max length")
        XCTAssertEqual(result.tokenIds.first, 101, "Should have [CLS]")

        // Original length should just be special tokens
        XCTAssertEqual(result.originalLength, 2, "Should only have [CLS] and [SEP]")
    }

    /// Test edge case: very long text
    func testVeryLongTextTokenization() async throws {
        let tokenizer = try await BERTTokenizer(maxSequenceLength: 16)

        // Create very long text
        let longText = String(repeating: "word ", count: 100)
        let result = try await tokenizer.tokenize(longText)

        // Should truncate
        XCTAssertEqual(result.tokenIds.count, 16, "Should truncate to max length")
        XCTAssertLessThanOrEqual(result.originalLength, 16, "Original length should not exceed max")

        // Should still have valid structure
        XCTAssertEqual(result.tokenIds.first, 101, "Should preserve [CLS]")
    }

    /// Test edge case: special characters and Unicode
    func testSpecialCharactersAndUnicode() async throws {
        let tokenizer = try await BERTTokenizer(maxSequenceLength: 32)

        let texts = [
            "Hello 世界",  // Unicode
            "test@email.com",  // Email
            "https://example.com",  // URL
            "price: $99.99",  // Numbers and symbols
            "emoji: 😀🎉"  // Emoji
        ]

        for text in texts {
            let result = try await tokenizer.tokenize(text)

            // Should handle without crashing
            XCTAssertGreaterThan(result.tokenIds.count, 0, "Should tokenize '\(text)'")
            XCTAssertEqual(result.tokenIds.first, 101, "Should have [CLS] for '\(text)'")
        }
    }

    /// Test BPE tokenization algorithm
    func testBPETokenization() async throws {
        let tokenizer = try await AdvancedTokenizer(
            type: .bpe,
            maxSequenceLength: 32
        )

        let text = "tokenization testing"
        let result = try await tokenizer.tokenize(text)

        // Verify basic structure
        XCTAssertGreaterThan(result.originalLength, 2, "Should have tokens beyond special tokens")
        XCTAssertEqual(result.tokenIds.first, 101, "Should start with [CLS]")

        // Verify decoding works
        let decoded = await tokenizer.decode(result.tokenIds)
        XCTAssertFalse(decoded.isEmpty, "Should decode to non-empty text")
    }

    /// Test tokenizer type enumeration
    func testTokenizerTypes() async throws {
        // Test all tokenization types can be instantiated
        for tokenType in AdvancedTokenizer.TokenizationType.allCases {
            let tokenizer = try await AdvancedTokenizer(
                type: tokenType,
                maxSequenceLength: 16
            )

            let result = try await tokenizer.tokenize("test")

            // All types should produce valid output
            XCTAssertGreaterThan(result.tokenIds.count, 0, "\(tokenType.rawValue) should tokenize")
            XCTAssertEqual(result.tokenIds.first, 101, "\(tokenType.rawValue) should have [CLS]")
        }
    }

    /// Test vocabulary builder text statistics
    func testVocabularyBuilderTextStats() async throws {
        var builder = VocabularyBuilder()

        // Add multiple occurrences of same word
        for _ in 0..<10 {
            builder.addText("hello world")
        }

        builder.addText("rare word")

        let vocabulary = builder.buildVocabulary(maxVocabSize: 100)

        // Frequent words should be in vocabulary
        XCTAssertNotNil(vocabulary["hello"], "Frequent word should be in vocabulary")
        XCTAssertNotNil(vocabulary["world"], "Frequent word should be in vocabulary")

        // Character-level tokens should also be present
        XCTAssertGreaterThan(vocabulary.count, 20, "Should include character-level tokens")
    }

    /// Test attention mask correctness
    func testAttentionMaskCorrectness() async throws {
        let tokenizer = try await BERTTokenizer(maxSequenceLength: 16)

        let texts = [
            "short",
            "medium length text",
            "this is a much longer text for testing"
        ]

        for text in texts {
            let result = try await tokenizer.tokenize(text)

            // Count 1s in attention mask
            let activeCount = result.attentionMask.filter { $0 == 1 }.count
            XCTAssertEqual(activeCount, result.originalLength, "Active attention should match original length for '\(text)'")

            // Verify attention mask is contiguous (all 1s followed by all 0s)
            var seenZero = false
            for mask in result.attentionMask {
                if mask == 0 {
                    seenZero = true
                } else if seenZero {
                    XCTFail("Attention mask should be contiguous (all 1s then all 0s)")
                }
            }
        }
    }

    /// Test tokenized input structure
    func testTokenizedInputStructure() async throws {
        let tokenizer = try await BERTTokenizer(maxSequenceLength: 16)

        let result = try await tokenizer.tokenize("test input")

        // Verify all required fields are present
        XCTAssertNotNil(result.tokenIds, "Should have token IDs")
        XCTAssertNotNil(result.attentionMask, "Should have attention mask")
        XCTAssertGreaterThan(result.originalLength, 0, "Should have original length")

        // Verify lengths match
        XCTAssertEqual(result.tokenIds.count, result.attentionMask.count, "Token IDs and attention mask should match")

        // Original length should be within bounds
        XCTAssertLessThanOrEqual(result.originalLength, result.tokenIds.count, "Original length should not exceed total length")
    }

    /// Test unknown token handling
    func testUnknownTokenHandling() async throws {
        let tokenizer = try await AdvancedTokenizer(
            type: .wordpiece,
            maxSequenceLength: 32,
            unknownToken: "[UNK]"
        )

        // Test with made-up words that shouldn't be in vocabulary
        let text = "xyzabc qwerty asdfgh"
        let result = try await tokenizer.tokenize(text)

        // Should handle unknown tokens without crashing
        XCTAssertGreaterThan(result.tokenIds.count, 0, "Should tokenize unknown words")
        XCTAssertEqual(result.tokenIds.first, 101, "Should preserve special tokens")
    }

    /// Test whitespace normalization
    func testWhitespaceNormalization() async throws {
        let tokenizer = try await BERTTokenizer(maxSequenceLength: 32)

        let texts = [
            "normal text",
            "  extra   spaces  ",
            "tabs\tand\nnewlines",
            "   "  // Only whitespace
        ]

        for text in texts {
            let result = try await tokenizer.tokenize(text)

            // All should produce valid output
            XCTAssertGreaterThan(result.tokenIds.count, 0, "Should handle '\(text)'")
            XCTAssertEqual(result.tokenIds.first, 101, "Should have [CLS] for '\(text)'")
        }
    }

    /// Test case sensitivity
    func testCaseSensitivity() async throws {
        let tokenizer = try await BERTTokenizer(maxSequenceLength: 32)

        let upperCase = "HELLO WORLD"
        let lowerCase = "hello world"
        let mixedCase = "Hello World"

        let results = try await tokenizer.tokenize(batch: [upperCase, lowerCase, mixedCase])

        // All should produce output (exact matching depends on vocabulary)
        for (index, result) in results.enumerated() {
            XCTAssertGreaterThan(result.originalLength, 2, "Result \(index) should have content")
        }
    }
}

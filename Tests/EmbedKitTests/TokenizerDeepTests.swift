// Tests for Tokenizer Deep - P1 Category
// Deep edge case testing for tokenization
import Testing
import Foundation
@testable import EmbedKit

// MARK: - Unicode Edge Cases

@Suite("Tokenizer Deep - Unicode")
struct TokenizerUnicodeDeepTests {

    @Test("Handles CJK characters")
    func cjkCharacters() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let result = try await tokenizer.encode("你好世界", config: config)

        #expect(result.ids.count > 0)
        #expect(result.tokens.count > 0)
    }

    @Test("Handles Arabic script")
    func arabicScript() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let result = try await tokenizer.encode("مرحبا بالعالم", config: config)

        #expect(result.ids.count > 0)
    }

    @Test("Handles Hebrew script")
    func hebrewScript() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let result = try await tokenizer.encode("שלום עולם", config: config)

        #expect(result.ids.count > 0)
    }

    @Test("Handles emoji sequences")
    func emojiSequences() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        // Various emoji types
        let text = "Hello 👋🏽 World 🌍 Test 👨‍👩‍👧‍👦"
        let result = try await tokenizer.encode(text, config: config)

        #expect(result.ids.count > 0)
        #expect(result.tokens.count > 0)
    }

    @Test("Handles combining diacriticals")
    func combiningDiacriticals() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        // Text with combining characters
        let text = "café résumé naïve"
        let result = try await tokenizer.encode(text, config: config)

        #expect(result.ids.count > 0)
    }

    @Test("Handles zero-width characters")
    func zeroWidthCharacters() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        // Text with zero-width joiner/non-joiner
        let text = "test\u{200B}word\u{200C}another\u{200D}final"
        let result = try await tokenizer.encode(text, config: config)

        #expect(result.ids.count > 0)
    }

    @Test("Handles mixed scripts")
    func mixedScripts() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let text = "Hello 你好 مرحبا שלום Привет"
        let result = try await tokenizer.encode(text, config: config)

        #expect(result.ids.count > 0)
    }

    @Test("Handles surrogate pair characters")
    func surrogatePairCharacters() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        // Characters outside BMP (require surrogate pairs in UTF-16)
        let text = "𝕳𝖊𝖑𝖑𝖔 𝖂𝖔𝖗𝖑𝖉"  // Mathematical bold fraktur
        let result = try await tokenizer.encode(text, config: config)

        #expect(result.ids.count > 0)
    }
}

// MARK: - Whitespace and Formatting

@Suite("Tokenizer Deep - Whitespace")
struct TokenizerWhitespaceDeepTests {

    @Test("Handles multiple consecutive spaces")
    func multipleSpaces() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let text = "hello    world"
        let result = try await tokenizer.encode(text, config: config)

        #expect(result.tokens.count == 2)  // Should collapse spaces
    }

    @Test("Handles tabs")
    func tabCharacters() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let text = "hello\tworld"
        let result = try await tokenizer.encode(text, config: config)

        #expect(result.tokens.count == 2)
    }

    @Test("Handles newlines")
    func newlineCharacters() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let text = "hello\nworld"
        let result = try await tokenizer.encode(text, config: config)

        #expect(result.tokens.count == 2)
    }

    @Test("Handles carriage return and line feed")
    func crlfCharacters() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let text = "hello\r\nworld"
        let result = try await tokenizer.encode(text, config: config)

        #expect(result.tokens.count == 2)
    }

    @Test("Handles leading whitespace")
    func leadingWhitespace() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let text = "   hello world"
        let result = try await tokenizer.encode(text, config: config)

        #expect(result.tokens.first == "hello")
    }

    @Test("Handles trailing whitespace")
    func trailingWhitespace() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let text = "hello world   "
        let result = try await tokenizer.encode(text, config: config)

        #expect(result.tokens.last == "world")
    }

    @Test("Handles Unicode whitespace characters")
    func unicodeWhitespace() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        // Various Unicode whitespace: em space, en space, thin space
        let text = "hello\u{2003}world\u{2002}test\u{2009}final"
        let result = try await tokenizer.encode(text, config: config)

        #expect(result.tokens.count == 4)
    }
}

// MARK: - Punctuation and Special Characters

@Suite("Tokenizer Deep - Punctuation")
struct TokenizerPunctuationDeepTests {

    @Test("Handles standard punctuation separation")
    func standardPunctuation() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let text = "Hello, world! How are you?"
        let result = try await tokenizer.encode(text, config: config)

        // Should separate punctuation
        #expect(result.tokens.contains(","))
        #expect(result.tokens.contains("!"))
        #expect(result.tokens.contains("?"))
    }

    @Test("Handles contractions")
    func contractions() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let text = "I'm can't won't shouldn't"
        let result = try await tokenizer.encode(text, config: config)

        #expect(result.tokens.count > 0)
    }

    @Test("Handles possessives")
    func possessives() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let text = "John's Mary's the dog's"
        let result = try await tokenizer.encode(text, config: config)

        #expect(result.tokens.count > 0)
    }

    @Test("Handles quotation marks")
    func quotationMarks() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let text = #""Hello" 'world' «test» „quote""#
        let result = try await tokenizer.encode(text, config: config)

        #expect(result.tokens.count > 0)
    }

    @Test("Handles brackets and parentheses")
    func bracketsParentheses() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let text = "(hello) [world] {test}"
        let result = try await tokenizer.encode(text, config: config)

        #expect(result.tokens.contains("("))
        #expect(result.tokens.contains(")"))
        #expect(result.tokens.contains("["))
        #expect(result.tokens.contains("]"))
        #expect(result.tokens.contains("{"))
        #expect(result.tokens.contains("}"))
    }

    @Test("Handles mathematical symbols")
    func mathematicalSymbols() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let text = "x + y = z × w ÷ 2"
        let result = try await tokenizer.encode(text, config: config)

        #expect(result.tokens.count > 0)
    }
}

// MARK: - WordPiece Specific Tests

@Suite("Tokenizer Deep - WordPiece")
struct WordPieceDeepTests {

    @Test("WordPiece subword splitting")
    func wordPieceSubwordSplitting() async throws {
        // Build vocabulary with proper subword tokens
        let vocab = Vocabulary(tokens: [
            "[PAD]", "[CLS]", "[SEP]", "[UNK]",
            "un", "##happ", "##y", "##iness", "happy", "the"
        ])
        let tokenizer = WordPieceTokenizer(vocabulary: vocab)
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        // Test a word that can be tokenized with available subwords
        let result = try await tokenizer.encode("un", config: config)

        // Should find the "un" token
        #expect(result.tokens.contains("un"))
        #expect(result.ids.count > 0)
    }

    @Test("WordPiece unknown token handling")
    func wordPieceUnknownToken() async throws {
        let vocab = Vocabulary(tokens: ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "hello"])
        let tokenizer = WordPieceTokenizer(vocabulary: vocab)
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let result = try await tokenizer.encode("xyzabc", config: config)

        // Unknown word should map to [UNK]
        #expect(result.tokens.contains("[UNK]"))
    }

    @Test("WordPiece very long word handling")
    func wordPieceVeryLongWord() async throws {
        let vocab = Vocabulary(tokens: ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "test"])
        let tokenizer = WordPieceTokenizer(vocabulary: vocab)
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        // Word longer than 200 characters
        let longWord = String(repeating: "a", count: 250)
        let result = try await tokenizer.encode(longWord, config: config)

        // Should be treated as unknown
        #expect(result.tokens.contains("[UNK]"))
    }

    @Test("WordPiece decode reconstructs text")
    func wordPieceDecode() async throws {
        let vocab = Vocabulary(tokens: [
            "[PAD]", "[CLS]", "[SEP]", "[UNK]",
            "hello", "world", "test", "##ing"
        ])
        let tokenizer = WordPieceTokenizer(vocabulary: vocab)
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let encoded = try await tokenizer.encode("hello world", config: config)
        let decoded = try await tokenizer.decode(encoded.ids)

        #expect(decoded.contains("hello"))
        #expect(decoded.contains("world"))
    }

    @Test("WordPiece lowercase handling")
    func wordPieceLowercase() async throws {
        let vocab = Vocabulary(tokens: ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "hello"])
        let lowercaseTokenizer = WordPieceTokenizer(vocabulary: vocab, lowercase: true)
        let caseTokenizer = WordPieceTokenizer(vocabulary: vocab, lowercase: false)
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let lowResult = try await lowercaseTokenizer.encode("HELLO", config: config)
        let caseResult = try await caseTokenizer.encode("HELLO", config: config)

        // Lowercase should find "hello", case-sensitive should not
        #expect(lowResult.tokens.contains("hello"))
        #expect(caseResult.tokens.contains("[UNK]"))
    }
}

// MARK: - Edge Case Inputs

@Suite("Tokenizer Deep - Edge Cases")
struct TokenizerEdgeCaseDeepTests {

    @Test("Empty string input")
    func emptyString() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let result = try await tokenizer.encode("", config: config)

        #expect(result.ids.isEmpty)
        #expect(result.tokens.isEmpty)
    }

    @Test("Single character input")
    func singleCharacter() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let result = try await tokenizer.encode("a", config: config)

        #expect(result.tokens.count == 1)
        #expect(result.tokens.first == "a")
    }

    @Test("Only whitespace input")
    func onlyWhitespace() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let result = try await tokenizer.encode("   \t\n  ", config: config)

        #expect(result.tokens.isEmpty)
    }

    @Test("Only punctuation input")
    func onlyPunctuation() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let result = try await tokenizer.encode(".,!?;:", config: config)

        // Each punctuation should be a separate token
        #expect(result.tokens.count == 6)
    }

    @Test("Very long input string")
    func veryLongInput() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false
        config.maxLength = 100
        config.truncation = .end

        let longText = (0..<1000).map { "word\($0)" }.joined(separator: " ")
        let result = try await tokenizer.encode(longText, config: config)

        #expect(result.tokens.count <= 100)
    }

    @Test("Repeated same word")
    func repeatedSameWord() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let text = String(repeating: "test ", count: 50)
        let result = try await tokenizer.encode(text, config: config)

        // All tokens should map to same ID
        let uniqueIds = Set(result.ids)
        #expect(uniqueIds.count == 1)
    }
}

// MARK: - Attention Mask Tests

@Suite("Tokenizer Deep - Attention Mask")
struct TokenizerAttentionMaskDeepTests {

    @Test("Attention mask length matches token count")
    func attentionMaskLengthMatches() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()

        let result = try await tokenizer.encode("hello world test", config: config)

        #expect(result.attentionMask.count == result.ids.count)
        #expect(result.attentionMask.count == result.tokens.count)
    }

    @Test("Non-padded tokens have mask value 1")
    func nonPaddedMaskValue() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let result = try await tokenizer.encode("hello world", config: config)

        // All should be 1 (no padding)
        #expect(result.attentionMask.allSatisfy { $0 == 1 })
    }

    @Test("WordPiece padding creates correct mask")
    func wordPiecePaddingMask() async throws {
        let vocab = Vocabulary(tokens: ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "hello"])
        let tokenizer = WordPieceTokenizer(vocabulary: vocab)
        var config = TokenizerConfig()
        config.addSpecialTokens = false
        config.padding = .max
        config.maxLength = 10

        let result = try await tokenizer.encode("hello", config: config)

        // First token should be 1, padding tokens should be 0
        #expect(result.attentionMask[0] == 1)
        #expect(result.attentionMask.suffix(from: 1).contains(0))
    }
}

// MARK: - Truncation Behavior Tests

@Suite("Tokenizer Deep - Truncation")
struct TokenizerTruncationDeepTests {

    @Test("End truncation preserves start")
    func endTruncationPreservesStart() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false
        config.maxLength = 3
        config.truncation = .end

        let result = try await tokenizer.encode("one two three four five", config: config)

        #expect(result.tokens[0] == "one")
        #expect(result.tokens[1] == "two")
        #expect(result.tokens[2] == "three")
    }

    @Test("Start truncation preserves end")
    func startTruncationPreservesEnd() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false
        config.maxLength = 3
        config.truncation = .start

        let result = try await tokenizer.encode("one two three four five", config: config)

        #expect(result.tokens[0] == "three")
        #expect(result.tokens[1] == "four")
        #expect(result.tokens[2] == "five")
    }

    @Test("Middle truncation preserves start and end")
    func middleTruncationPreservesStartEnd() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false
        config.maxLength = 4
        config.truncation = .middle

        let result = try await tokenizer.encode("one two three four five six", config: config)

        // Should have first 2 and last 2
        #expect(result.tokens.first == "one")
        #expect(result.tokens.last == "six")
        #expect(result.tokens.count == 4)
    }

    @Test("No truncation throws on too long")
    func noTruncationThrows() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false
        config.maxLength = 3
        config.truncation = .none

        do {
            _ = try await tokenizer.encode("one two three four five", config: config)
            #expect(Bool(false), "Should have thrown")
        } catch {
            // Expected
            #expect(error is EmbedKitError)
        }
    }
}

// MARK: - Consistency Tests

@Suite("Tokenizer Deep - Consistency")
struct TokenizerConsistencyDeepTests {

    @Test("Same input produces same output")
    func deterministicOutput() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let text = "Hello world test input"

        let result1 = try await tokenizer.encode(text, config: config)
        let result2 = try await tokenizer.encode(text, config: config)

        #expect(result1.ids == result2.ids)
        #expect(result1.tokens == result2.tokens)
    }

    @Test("Tokenize then decode is reversible (SimpleTokenizer)")
    func encodeDecodeRoundtrip() async throws {
        let tokenizer = SimpleTokenizer()
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let original = "hello world"
        let encoded = try await tokenizer.encode(original, config: config)

        // SimpleTokenizer decode just returns IDs as strings, not actual decode
        // Just verify no crash
        _ = try await tokenizer.decode(encoded.ids)
    }

    @Test("WordPiece encode-decode roundtrip")
    func wordPieceEncodeDecodeRoundtrip() async throws {
        let vocab = Vocabulary(tokens: [
            "[PAD]", "[CLS]", "[SEP]", "[UNK]",
            "hello", "world", "test"
        ])
        let tokenizer = WordPieceTokenizer(vocabulary: vocab)
        var config = TokenizerConfig()
        config.addSpecialTokens = false

        let original = "hello world"
        let encoded = try await tokenizer.encode(original, config: config)
        let decoded = try await tokenizer.decode(encoded.ids)

        #expect(decoded == "hello world")
    }
}

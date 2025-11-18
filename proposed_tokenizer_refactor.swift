// Proposed flexible tokenization strategy

public enum TokenizationStrategy {
    case wordPiece      // BERT-style with ## prefix
    case bpe            // GPT-style byte pair encoding
    case sentencePiece  // T5/LLaMA style with ▁
    case characterLevel // Simple character tokenization
    case wordLevel      // Simple word tokenization
}

public struct VocabularyBuilder {
    private var tokenFrequency: [String: Int] = [:]
    private let specialTokens: SpecialTokens
    private let strategy: TokenizationStrategy

    public init(
        specialTokens: SpecialTokens = SpecialTokens(),
        strategy: TokenizationStrategy = .wordPiece  // Default for backward compat
    ) {
        self.specialTokens = specialTokens
        self.strategy = strategy
    }

    public mutating func addText(_ text: String) {
        let words = text.lowercased().split(separator: " ")

        for word in words {
            let wordStr = String(word)

            switch strategy {
            case .wordPiece:
                addWordPieceTokens(wordStr)
            case .bpe:
                addBPETokens(wordStr)
            case .sentencePiece:
                addSentencePieceTokens(wordStr)
            case .characterLevel:
                addCharacterTokens(wordStr)
            case .wordLevel:
                tokenFrequency[wordStr, default: 0] += 1
            }
        }
    }

    private mutating func addWordPieceTokens(_ word: String) {
        // Current BERT-style implementation
        tokenFrequency[word, default: 0] += 1

        // Characters
        for char in word {
            tokenFrequency[String(char), default: 0] += 1
        }

        // Subwords with ## prefix
        if word.count > 2 {
            for length in 2...min(3, word.count - 1) {
                let prefix = String(word.prefix(length))
                tokenFrequency[prefix + "##", default: 0] += 1

                let suffix = String(word.suffix(length))
                tokenFrequency["##" + suffix, default: 0] += 1
            }
        }
    }

    private mutating func addBPETokens(_ word: String) {
        // Simplified BPE: word + character pairs
        tokenFrequency[word, default: 0] += 1

        // Character pairs (bigrams)
        let chars = Array(word)
        for i in 0..<chars.count-1 {
            let bigram = String(chars[i...i+1])
            tokenFrequency[bigram, default: 0] += 1
        }

        // Individual characters as fallback
        for char in word {
            tokenFrequency[String(char), default: 0] += 1
        }
    }

    private mutating func addSentencePieceTokens(_ word: String) {
        // Add with underscore prefix for word boundary
        tokenFrequency["▁" + word, default: 0] += 1

        // Subwords without special markers
        if word.count > 3 {
            for i in 2..<word.count-1 {
                let subword = String(word.prefix(i))
                tokenFrequency[subword, default: 0] += 1
            }
        }

        // Characters as fallback
        for char in word {
            tokenFrequency[String(char), default: 0] += 1
        }
    }

    private mutating func addCharacterTokens(_ word: String) {
        // Only character-level tokens
        for char in word {
            tokenFrequency[String(char), default: 0] += 1
        }
    }
}

// Test should be strategy-aware
func testVocabularyBuilderTextStats() async throws {
    // Test different strategies
    let testCases: [(TokenizationStrategy, Int)] = [
        (.wordPiece, 20),      // Expects many subword tokens
        (.bpe, 15),            // Expects fewer tokens
        (.sentencePiece, 18),  // Moderate number
        (.characterLevel, 8),  // Only unique characters
        (.wordLevel, 3)        // Only whole words
    ]

    for (strategy, expectedMinTokens) in testCases {
        var builder = VocabularyBuilder(strategy: strategy)

        for _ in 0..<10 {
            builder.addText("hello world")
        }
        builder.addText("rare word")

        let vocabulary = builder.buildVocabulary(maxVocabSize: 100)

        XCTAssertGreaterThan(
            vocabulary.count,
            expectedMinTokens,
            "Strategy \(strategy) should generate > \(expectedMinTokens) tokens"
        )
    }
}
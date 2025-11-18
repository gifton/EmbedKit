import Foundation
import OSLog

/// Advanced tokenizer supporting BPE and WordPiece algorithms
public actor AdvancedTokenizer: Tokenizer {
    private let logger = Logger(subsystem: "EmbedKit", category: "AdvancedTokenizer")

    public let maxSequenceLength: Int
    nonisolated(unsafe) private var _vocabularySize: Int = 0
    nonisolated public var vocabularySize: Int { _vocabularySize }
    public let specialTokens: SpecialTokens

    private let tokenizationType: TokenizationType
    private var vocabulary: [String: Int] = [:]
    private var inverseVocabulary: [Int: String] = [:]
    private var mergeRules: [(String, String)] = []
    private let unknownToken: String

    public enum TokenizationType: String, CaseIterable, Sendable {
        case bpe = "BPE"
        case wordpiece = "WordPiece"
        case sentencepiece = "SentencePiece"
    }

    public init(
        type: TokenizationType,
        vocabularyPath: String? = nil,
        maxSequenceLength: Int = 512,
        specialTokens: SpecialTokens = SpecialTokens(),
        unknownToken: String = "[UNK]"
    ) async throws {
        self.tokenizationType = type
        self.maxSequenceLength = maxSequenceLength
        self.specialTokens = specialTokens
        self.unknownToken = unknownToken
        // Initialize with a basic vocabulary if no path provided
        if let vocabPath = vocabularyPath {
            try await loadVocabulary(from: vocabPath)
        } else {
            await initializeDefaultVocabulary()
        }

        self._vocabularySize = vocabulary.count
    }

    public func tokenize(_ text: String) async throws -> TokenizedInput {
        let preprocessed = preprocess(text)
        let tokens: [String]

        switch tokenizationType {
        case .bpe:
            tokens = try await tokenizeBPE(preprocessed)
        case .wordpiece:
            tokens = try await tokenizeWordPiece(preprocessed)
        case .sentencepiece:
            tokens = try await tokenizeSentencePiece(preprocessed)
        }

        // Convert to token IDs and add special tokens
        var tokenIds = [specialTokens.cls]
        for token in tokens {
            if tokenIds.count >= maxSequenceLength - 1 {
                break
            }
            let tokenId = vocabulary[token] ?? vocabulary[unknownToken] ?? specialTokens.unk
            tokenIds.append(tokenId)
        }
        tokenIds.append(specialTokens.sep)

        // Pad to max length
        let originalLength = tokenIds.count
        while tokenIds.count < maxSequenceLength {
            tokenIds.append(specialTokens.pad)
        }

        // Create attention mask
        let attentionMask = Array(repeating: 1, count: originalLength) +
                           Array(repeating: 0, count: maxSequenceLength - originalLength)

        return TokenizedInput(
            tokenIds: Array(tokenIds.prefix(maxSequenceLength)),
            attentionMask: Array(attentionMask.prefix(maxSequenceLength)),
            tokenTypeIds: nil,
            originalLength: originalLength
        )
    }

    public func tokenize(batch texts: [String]) async throws -> [TokenizedInput] {
        var results: [TokenizedInput] = []
        results.reserveCapacity(texts.count)

        for text in texts {
            let tokenized = try await tokenize(text)
            results.append(tokenized)
        }

        return results
    }

    // MARK: - Private Methods

    private func preprocess(_ text: String) -> String {
        // Basic preprocessing: lowercase, normalize whitespace
        return text.lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
    }

    private func tokenizeBPE(_ text: String) async throws -> [String] {
        // Split into initial tokens (characters or words)
        var tokens = text.map { String($0) }

        // Apply merge rules iteratively
        for (first, second) in mergeRules {
            var mergedTokens: [String] = []
            var i = 0

            while i < tokens.count {
                if i < tokens.count - 1 && tokens[i] == first && tokens[i + 1] == second {
                    // Merge these tokens
                    mergedTokens.append(first + second)
                    i += 2
                } else {
                    mergedTokens.append(tokens[i])
                    i += 1
                }
            }

            tokens = mergedTokens
        }

        return tokens
    }

    private func tokenizeWordPiece(_ text: String) async throws -> [String] {
        let words = text.split(separator: " ").map(String.init)
        var tokens: [String] = []

        for word in words {
            let wordTokens = try await tokenizeWordPieceWord(word)
            tokens.append(contentsOf: wordTokens)
        }

        return tokens
    }

    private func tokenizeWordPieceWord(_ word: String) async throws -> [String] {
        if vocabulary[word] != nil {
            return [word]
        }

        var tokens: [String] = []
        var start = 0
        let chars = Array(word)

        while start < chars.count {
            var end = chars.count
            var subToken: String?

            // Try to find the longest subword that exists in vocabulary
            while start < end {
                let candidate = String(chars[start..<end])
                let prefixedCandidate = start > 0 ? "##" + candidate : candidate

                if vocabulary[prefixedCandidate] != nil {
                    subToken = prefixedCandidate
                    break
                }
                end -= 1
            }

            if let token = subToken {
                tokens.append(token)
                start = end
            } else {
                // Cannot tokenize - use unknown token
                tokens.append(unknownToken)
                start += 1
            }
        }

        return tokens
    }

    private func tokenizeSentencePiece(_ text: String) async throws -> [String] {
        // MARK: - Future Enhancement: SentencePiece Integration
        //
        // SentencePiece is a state-of-the-art tokenization algorithm used by many
        // modern language models (T5, mBART, XLM-R, etc.). Full integration would require:
        //
        // 1. Adding the SentencePiece Swift library as a dependency
        // 2. Loading pre-trained SentencePiece models (.model files)
        // 3. Implementing proper subword tokenization with:
        //    - Unigram language model-based tokenization
        //    - Proper handling of unknown tokens
        //    - Support for both encoding and decoding
        //
        // Current implementation falls back to BPE which provides similar
        // subword tokenization capabilities.
        //
        // To implement SentencePiece support:
        // - Add package dependency: https://github.com/google/sentencepiece
        // - Load model: SentencePieceProcessor.load(modelPath)
        // - Tokenize: processor.encode(text)

        logger.info("SentencePiece tokenization requested, falling back to BPE")
        return try await tokenizeBPE(text)
    }

    private func loadVocabulary(from path: String) async throws {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)

        if path.hasSuffix(".json") {
            try await loadJSONVocabulary(data)
        } else {
            try await loadTextVocabulary(data)
        }

        self._vocabularySize = vocabulary.count
        logger.info("Loaded vocabulary with \(self.vocabulary.count) tokens")
    }

    private func loadJSONVocabulary(_ data: Data) async throws {
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]

        if let vocab = json?["vocab"] as? [String: Int] {
            self.vocabulary = vocab
        } else if let vocabArray = json as? [String: Int] {
            self.vocabulary = vocabArray
        } else {
            throw TokenizationError.invalidVocabularyFormat
        }

        // Load merge rules if available
        if let merges = json?["merges"] as? [String] {
            self.mergeRules = merges.compactMap { merge in
                let parts = merge.split(separator: " ", maxSplits: 1)
                guard parts.count == 2 else { return nil }
                return (String(parts[0]), String(parts[1]))
            }
        }

        buildInverseVocabulary()
    }

    private func loadTextVocabulary(_ data: Data) async throws {
        let content = String(data: data, encoding: .utf8) ?? ""
        let lines = content.components(separatedBy: .newlines)

        for (index, line) in lines.enumerated() {
            let token = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if !token.isEmpty {
                vocabulary[token] = index
            }
        }

        buildInverseVocabulary()
    }

    private func initializeDefaultVocabulary() async {
        // Initialize with special tokens
        vocabulary["[PAD]"] = specialTokens.pad
        vocabulary["[UNK]"] = specialTokens.unk
        vocabulary["[CLS]"] = specialTokens.cls
        vocabulary["[SEP]"] = specialTokens.sep
        if let mask = specialTokens.mask {
            vocabulary["[MASK]"] = mask
        }

        // Add basic ASCII characters and common subwords
        var tokenId = 1000

        // Add individual characters
        for i in 32...126 { // Printable ASCII
            let char = String(Character(UnicodeScalar(i)!))
            vocabulary[char] = tokenId
            tokenId += 1
        }

        // Add common English words and subwords
        let commonTokens = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have",
            "i", "it", "for", "not", "on", "with", "he", "as", "you",
            "do", "at", "this", "but", "his", "by", "from", "they",
            "we", "say", "her", "she", "or", "an", "will", "my",
            "one", "all", "would", "there", "their", "what", "so",
            "##ed", "##ing", "##er", "##ly", "##s", "##ion", "##tion"
        ]

        for token in commonTokens {
            vocabulary[token] = tokenId
            tokenId += 1
        }

        buildInverseVocabulary()
    }

    private func buildInverseVocabulary() {
        inverseVocabulary = Dictionary(uniqueKeysWithValues: vocabulary.map { ($1, $0) })
    }

    /// Decode token IDs back to text
    public func decode(_ tokenIds: [Int]) async -> String {
        let tokens = tokenIds.compactMap { inverseVocabulary[$0] }
        return tokens.joined(separator: " ")
            .replacingOccurrences(of: " ##", with: "")
            .replacingOccurrences(of: "[PAD]", with: "")
            .replacingOccurrences(of: "[CLS]", with: "")
            .replacingOccurrences(of: "[SEP]", with: "")
            .trimmingCharacters(in: .whitespaces)
    }
}

/// Errors specific to tokenization
public enum TokenizationError: LocalizedError {
    case invalidVocabularyFormat
    case vocabularyNotFound(String)
    case tokenizationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidVocabularyFormat:
            return "Invalid vocabulary file format"
        case .vocabularyNotFound(let path):
            return "Vocabulary file not found at path: \(path)"
        case .tokenizationFailed(let details):
            return "Tokenization failed: \(details)"
        }
    }
}

/// Vocabulary builder for creating custom vocabularies
public struct VocabularyBuilder {
    private var tokenFrequency: [String: Int] = [:]
    private let specialTokens: SpecialTokens

    public init(specialTokens: SpecialTokens = SpecialTokens()) {
        self.specialTokens = specialTokens
    }

    /// Add text to build vocabulary statistics
    public mutating func addText(_ text: String) {
        let words = text.lowercased().split(separator: " ")
        for word in words {
            let wordStr = String(word)
            tokenFrequency[wordStr, default: 0] += 1

            // Also add character-level tokens
            for char in wordStr {
                tokenFrequency[String(char), default: 0] += 1
            }

            // Add subword tokens (WordPiece-style)
            // Generate prefixes and suffixes for words longer than 2 characters
            if wordStr.count > 2 {
                // Add prefix tokens (first 2-3 chars)
                for length in 2...min(3, wordStr.count - 1) {
                    let prefix = String(wordStr.prefix(length))
                    tokenFrequency[prefix + "##", default: 0] += 1
                }

                // Add suffix tokens (last 2-3 chars with ## prefix)
                for length in 2...min(3, wordStr.count - 1) {
                    let suffix = String(wordStr.suffix(length))
                    tokenFrequency["##" + suffix, default: 0] += 1
                }
            }
        }
    }

    /// Build vocabulary from collected statistics
    public func buildVocabulary(maxVocabSize: Int = 30000) -> [String: Int] {
        var vocabulary: [String: Int] = [:]
        var tokenId = 0

        // Add special tokens first
        vocabulary["[PAD]"] = specialTokens.pad
        vocabulary["[UNK]"] = specialTokens.unk
        vocabulary["[CLS]"] = specialTokens.cls
        vocabulary["[SEP]"] = specialTokens.sep
        if let mask = specialTokens.mask {
            vocabulary["[MASK]"] = mask
        }
        tokenId = 1000

        // Sort tokens by frequency and add to vocabulary
        let sortedTokens = tokenFrequency.sorted { $0.value > $1.value }

        for (token, _) in sortedTokens.prefix(maxVocabSize - vocabulary.count) {
            vocabulary[token] = tokenId
            tokenId += 1
        }

        return vocabulary
    }

    /// Save vocabulary to JSON file
    public func saveVocabulary(_ vocabulary: [String: Int], to path: String) throws {
        let url = URL(fileURLWithPath: path)
        let data = try JSONSerialization.data(withJSONObject: vocabulary, options: .prettyPrinted)
        try data.write(to: url)
    }
}

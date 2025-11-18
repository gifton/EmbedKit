import Foundation
import OSLog

/// BERT WordPiece tokenizer implementation
///
/// This tokenizer implements the WordPiece algorithm used by BERT models,
/// which splits unknown words into subword units to handle out-of-vocabulary words.
public actor BERTTokenizer: Tokenizer {
    private let logger = Logger(subsystem: "EmbedKit", category: "BERTTokenizer")

    public let maxSequenceLength: Int
    nonisolated(unsafe) private var _vocabularySize: Int = 0
    public nonisolated var vocabularySize: Int {
        _vocabularySize
    }
    nonisolated(unsafe) public private(set) var specialTokens: SpecialTokens

    private var vocabulary: [String: Int] = [:]
    private var inverseVocabulary: [Int: String] = [:]
    private let doLowerCase: Bool
    private let unknownToken: String
    private let maxInputCharsPerWord: Int

    /// Special BERT tokens
    private let clsToken = "[CLS]"
    private let sepToken = "[SEP]"
    private let padToken = "[PAD]"
    private let unkToken = "[UNK]"
    private let maskToken = "[MASK]"

    public init(
        vocabularyPath: String? = nil,
        maxSequenceLength: Int = 512,
        doLowerCase: Bool = true,
        unknownToken: String = "[UNK]",
        maxInputCharsPerWord: Int = 200,
        specialTokens: SpecialTokens? = nil
    ) async throws {
        self.maxSequenceLength = maxSequenceLength
        self.doLowerCase = doLowerCase
        self.unknownToken = unknownToken
        self.maxInputCharsPerWord = maxInputCharsPerWord

        // Initialize special tokens early with defaults or provided values
        self.specialTokens = specialTokens ?? SpecialTokens(
            cls: 101,   // [CLS]
            sep: 102,   // [SEP]
            pad: 0,     // [PAD]
            unk: 100,   // [UNK]
            mask: 103   // [MASK]
        )

        // Load vocabulary
        if let vocabPath = vocabularyPath {
            try await loadVocabulary(from: vocabPath)
        } else {
            await loadDefaultBERTVocabulary()
        }

        // Update special tokens if we can extract them from vocabulary (unless custom ones were provided)
        if specialTokens == nil {
            if let extractedTokens = extractSpecialTokensFromVocabulary() {
                self.specialTokens = extractedTokens
            }
        }

        self._vocabularySize = vocabulary.count
    }

    public func tokenize(_ text: String) async throws -> TokenizedInput {
        // Basic text cleaning
        let cleanedText = cleanText(text)

        // Split by whitespace and punctuation
        let tokens = tokenizeBasic(cleanedText)

        // Apply WordPiece tokenization
        var wordpieceTokens: [String] = []
        wordpieceTokens.append(clsToken) // Add [CLS] token

        for token in tokens {
            let subTokens = try await wordpieceTokenize(token)
            wordpieceTokens.append(contentsOf: subTokens)

            // Check if we're exceeding max length (leaving room for [SEP])
            if wordpieceTokens.count >= maxSequenceLength - 1 {
                break
            }
        }

        wordpieceTokens.append(sepToken) // Add [SEP] token

        // Convert tokens to IDs
        let tokenIds = wordpieceTokens.map { token in
            vocabulary[token] ?? vocabulary[unkToken] ?? specialTokens.unk
        }

        // Create attention mask (1 for real tokens, 0 for padding)
        let originalLength = tokenIds.count
        var paddedTokenIds = tokenIds
        var attentionMask = Array(repeating: 1, count: originalLength)

        // Pad to max sequence length
        while paddedTokenIds.count < maxSequenceLength {
            paddedTokenIds.append(specialTokens.pad)
            attentionMask.append(0)
        }

        // Truncate if necessary
        if paddedTokenIds.count > maxSequenceLength {
            paddedTokenIds = Array(paddedTokenIds.prefix(maxSequenceLength))
            attentionMask = Array(attentionMask.prefix(maxSequenceLength))
        }

        // Create token_type_ids (all zeros for single sequence)
        let tokenTypeIds = Array(repeating: 0, count: paddedTokenIds.count)

        return TokenizedInput(
            tokenIds: paddedTokenIds,
            attentionMask: attentionMask,
            tokenTypeIds: tokenTypeIds,
            originalLength: min(originalLength, maxSequenceLength)
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

    /// Clean text by normalizing whitespace and handling special characters
    private func cleanText(_ text: String) -> String {
        var cleaned = text

        // Normalize whitespace
        cleaned = cleaned.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
        cleaned = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)

        // Convert to lowercase if configured
        if doLowerCase {
            cleaned = cleaned.lowercased()
        }

        // Remove control characters and non-printable characters
        cleaned = cleaned.filter { char in
            let scalar = char.unicodeScalars.first!
            return scalar.value >= 32 && scalar.value != 127
        }

        return cleaned
    }

    /// Basic tokenization that splits on whitespace and punctuation
    private func tokenizeBasic(_ text: String) -> [String] {
        var tokens: [String] = []
        var currentToken = ""

        for char in text {
            if char.isWhitespace {
                if !currentToken.isEmpty {
                    tokens.append(currentToken)
                    currentToken = ""
                }
            } else if isPunctuation(char) {
                if !currentToken.isEmpty {
                    tokens.append(currentToken)
                    currentToken = ""
                }
                tokens.append(String(char))
            } else {
                currentToken.append(char)
            }
        }

        if !currentToken.isEmpty {
            tokens.append(currentToken)
        }

        return tokens
    }

    /// Check if a character is punctuation
    private func isPunctuation(_ char: Character) -> Bool {
        let punctuationSet = CharacterSet.punctuationCharacters
        if let scalar = char.unicodeScalars.first {
            return punctuationSet.contains(scalar)
        }
        return false
    }

    /// Apply WordPiece tokenization to a single word
    private func wordpieceTokenize(_ word: String) async throws -> [String] {
        if word.count > maxInputCharsPerWord {
            return [unkToken]
        }

        var outputTokens: [String] = []
        var isBad = false
        var start = 0
        let chars = Array(word)

        while start < chars.count {
            var end = chars.count
            var curSubstr: String? = nil

            while start < end {
                var substr = String(chars[start..<end])

                // Add "##" prefix for subwords (not at the beginning)
                if start > 0 {
                    substr = "##" + substr
                }

                if vocabulary[substr] != nil {
                    curSubstr = substr
                    break
                }

                end -= 1
            }

            if curSubstr == nil {
                isBad = true
                break
            }

            outputTokens.append(curSubstr!)
            start = end
        }

        if isBad {
            return [unkToken]
        } else {
            return outputTokens
        }
    }

    /// Load vocabulary from file
    private func loadVocabulary(from path: String) async throws {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)

        if path.hasSuffix(".json") {
            // Load from JSON format (common for BERT)
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Int] {
                self.vocabulary = json
            } else {
                throw TokenizationError.invalidVocabularyFormat
            }
        } else {
            // Load from text format (one token per line)
            let content = String(data: data, encoding: .utf8) ?? ""
            let lines = content.components(separatedBy: .newlines)

            for (index, line) in lines.enumerated() {
                let token = line.trimmingCharacters(in: .whitespacesAndNewlines)
                if !token.isEmpty {
                    vocabulary[token] = index
                }
            }
        }

        // Build inverse vocabulary
        buildInverseVocabulary()

        logger.info("Loaded BERT vocabulary with \(self.vocabulary.count) tokens")
    }

    /// Load default BERT vocabulary
    private func loadDefaultBERTVocabulary() async {
        // Initialize with BERT special tokens
        vocabulary[padToken] = 0
        vocabulary[unkToken] = 100
        vocabulary[clsToken] = 101
        vocabulary[sepToken] = 102
        vocabulary[maskToken] = 103

        // Add common BERT tokens
        // In production, this should load from a proper vocab.txt file
        var tokenId = 104

        // Add some essential tokens for basic functionality
        let essentialTokens = [
            "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            ":", ";", "<", "=", ">", "?", "@",
            "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~"
        ]

        for token in essentialTokens {
            vocabulary[token] = tokenId
            tokenId += 1
        }

        // Add lowercase letters
        for i in 97...122 { // a-z
            let char = String(Character(UnicodeScalar(i)!))
            vocabulary[char] = tokenId
            tokenId += 1
        }

        // Add uppercase letters
        for i in 65...90 { // A-Z
            let char = String(Character(UnicodeScalar(i)!))
            vocabulary[char] = tokenId
            tokenId += 1
        }

        // Add common words and subwords
        let commonWords = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "##s", "##ed", "##ing", "##ly", "##er", "##est", "##ion", "##tion",
            "##ment", "##ness", "##ity", "##ous", "##ful", "##less", "##able"
        ]

        for word in commonWords {
            vocabulary[word] = tokenId
            tokenId += 1
        }

        buildInverseVocabulary()

        logger.info("Initialized default BERT vocabulary with \(self.vocabulary.count) tokens")
    }

    /// Build inverse vocabulary for decoding
    private func buildInverseVocabulary() {
        inverseVocabulary = Dictionary(uniqueKeysWithValues: vocabulary.map { ($1, $0) })
    }

    /// Extract special tokens from loaded vocabulary
    private func extractSpecialTokensFromVocabulary() -> SpecialTokens? {
        guard !vocabulary.isEmpty else { return nil }

        // Look for standard BERT special tokens
        guard let cls = vocabulary["[CLS]"],
              let sep = vocabulary["[SEP]"],
              let pad = vocabulary["[PAD]"],
              let unk = vocabulary["[UNK]"] else {
            logger.warning("Could not find all required special tokens in vocabulary")
            return nil
        }

        let mask = vocabulary["[MASK]"]

        logger.debug("Extracted special tokens from vocabulary: CLS=\(cls), SEP=\(sep), PAD=\(pad), UNK=\(unk), MASK=\(mask ?? -1)")

        return SpecialTokens(
            cls: cls,
            sep: sep,
            pad: pad,
            unk: unk,
            mask: mask
        )
    }

    /// Decode token IDs back to text
    public func decode(_ tokenIds: [Int]) async -> String {
        var tokens: [String] = []

        for tokenId in tokenIds {
            if let token = inverseVocabulary[tokenId] {
                // Skip special tokens
                if token == padToken || token == clsToken || token == sepToken {
                    continue
                }
                tokens.append(token)
            }
        }

        // Join tokens and handle WordPiece markers
        var text = ""
        for (index, token) in tokens.enumerated() {
            if token.hasPrefix("##") {
                // Remove ## and append directly
                text += token.dropFirst(2)
            } else if index > 0 {
                // Add space before regular tokens
                text += " " + token
            } else {
                text += token
            }
        }

        return text.trimmingCharacters(in: .whitespaces)
    }
}

/// Extension to support vocabulary statistics
public extension BERTTokenizer {
    /// Get vocabulary statistics
    func getVocabularyStats() async -> VocabularyStats {
        let specialTokenCount = 5 // [PAD], [UNK], [CLS], [SEP], [MASK]
        let subwordCount = vocabulary.keys.filter { $0.hasPrefix("##") }.count
        let wholeWordCount = vocabulary.count - specialTokenCount - subwordCount

        return VocabularyStats(
            totalSize: vocabulary.count,
            specialTokens: specialTokenCount,
            wholeWords: wholeWordCount,
            subwords: subwordCount
        )
    }

    /// Check if a token exists in vocabulary
    func hasToken(_ token: String) async -> Bool {
        vocabulary[token] != nil
    }

    /// Get token ID for a specific token
    func getTokenId(_ token: String) async -> Int? {
        vocabulary[token]
    }
}

/// Vocabulary statistics
public struct VocabularyStats: Sendable {
    public let totalSize: Int
    public let specialTokens: Int
    public let wholeWords: Int
    public let subwords: Int

    public init(totalSize: Int, specialTokens: Int, wholeWords: Int, subwords: Int) {
        self.totalSize = totalSize
        self.specialTokens = specialTokens
        self.wholeWords = wholeWords
        self.subwords = subwords
    }
}

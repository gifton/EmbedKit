// EmbedKit - BPE Tokenizer
// Production byte-level BPE tokenizer with HuggingFace compatibility.

import Foundation

/// BPE tokenizer supporting both legacy greedy merges and production byte-level BPE
/// with HuggingFace `tokenizer.json` compatibility.
///
/// ## Byte-Level BPE
/// Modern embedding models (Nomic, BGE, Llama 3, Mistral) use byte-level BPE where every
/// byte (0-255) is mapped to a printable Unicode character, ensuring complete coverage
/// of the input space. Tokenization proceeds by:
/// 1. Normalizing the input text (NFC/NFKC)
/// 2. Splitting via a regex pre-tokenizer (GPT-2 or Llama pattern)
/// 3. Encoding each pre-token as byte-level unicode characters
/// 4. Applying priority-ranked merge operations using a min-heap
///
/// ## Legacy Mode
/// For backward compatibility, the `init(vocabulary:merges:...)` and `load(vocabURL:mergesURL:...)`
/// APIs produce a tokenizer using simple greedy left-to-right merges (the original scaffold behavior).
public struct BPETokenizer: Tokenizer {
    public let vocabulary: Vocabulary
    public let lowercase: Bool
    public let unkToken: String

    /// Mode determines the merge algorithm and pre-processing.
    private let mode: Mode
    private let resolvedSpecialTokens: SpecialTokens

    enum Mode: Sendable {
        /// Legacy greedy left-to-right merges (original scaffold).
        case legacy(merges: [String: String])
        /// Production byte-level BPE with ranked merges and regex pre-tokenization.
        case byteLevelBPE(config: ByteLevelConfig)
    }

    struct ByteLevelConfig: Sendable {
        let mergeRanks: [String: Int]
        let preTokenizerPattern: PreTokenizerPattern
        let normalization: NormalizationSetting
        let addPrefixSpace: Bool
    }

    // MARK: - Initializers

    /// Legacy initializer: greedy left-to-right merges (backward compat).
    public init(vocabulary: Vocabulary, merges: [String: String], unkToken: String = "[UNK]", lowercase: Bool = true) {
        self.vocabulary = vocabulary
        self.mode = .legacy(merges: merges)
        self.lowercase = lowercase
        self.unkToken = unkToken
        self.resolvedSpecialTokens = Self.resolveSpecialTokens(vocabulary: vocabulary, unkToken: unkToken)
    }

    /// Production byte-level BPE from a HuggingFace `tokenizer.json` configuration.
    public init(hfConfig: HFTokenizerJSON) throws {
        guard hfConfig.model.isBPE else {
            throw EmbedKitError.tokenizationFailed("HF config model type is '\(hfConfig.model.type)', expected 'BPE'")
        }
        guard let vocab = hfConfig.buildVocabulary() else {
            throw EmbedKitError.tokenizationFailed("HF config has no vocabulary")
        }
        guard let mergeRanks = hfConfig.buildMergeRanks() else {
            throw EmbedKitError.tokenizationFailed("HF config has no merges")
        }

        self.vocabulary = vocab
        self.lowercase = false  // Byte-level BPE models don't lowercase

        let pattern: PreTokenizerPattern
        let addPrefixSpace: Bool
        switch hfConfig.preTokenizerStrategy() {
        case .byteLevel(let prefixSpace):
            pattern = .gpt2  // Default for byte-level pre-tokenizer
            addPrefixSpace = prefixSpace
        case .sequence(let strategies):
            // Look for ByteLevel in the sequence
            var foundByteLevel = false
            var prefix = false
            for s in strategies {
                if case .byteLevel(let ps) = s {
                    foundByteLevel = true
                    prefix = ps
                }
            }
            pattern = foundByteLevel ? .gpt2 : .gpt2
            addPrefixSpace = prefix
        default:
            pattern = .gpt2
            addPrefixSpace = false
        }

        let normalization: NormalizationSetting
        switch hfConfig.normalizationMode() {
        case .nfc: normalization = .nfc
        case .nfkc: normalization = .nfkc
        default: normalization = .none
        }

        self.mode = .byteLevelBPE(config: ByteLevelConfig(
            mergeRanks: mergeRanks,
            preTokenizerPattern: pattern,
            normalization: normalization,
            addPrefixSpace: addPrefixSpace
        ))

        self.unkToken = hfConfig.model.unkToken ?? "<unk>"
        self.resolvedSpecialTokens = hfConfig.buildSpecialTokens()
    }

    /// Load from a HuggingFace `tokenizer.json` file URL.
    public static func load(hfTokenizerJSON url: URL) throws -> BPETokenizer {
        let config = try HFTokenizerJSON.load(from: url)
        return try BPETokenizer(hfConfig: config)
    }

    /// Legacy loader: vocab as newline-delimited tokens; merges as pairs per line.
    public static func load(vocabURL: URL, mergesURL: URL, unkToken: String = "[UNK]", lowercase: Bool = true) throws -> BPETokenizer {
        let vocab = try Vocabulary.load(from: vocabURL)
        let merges = try loadMerges(from: mergesURL)
        return BPETokenizer(vocabulary: vocab, merges: merges, unkToken: unkToken, lowercase: lowercase)
    }

    // MARK: - Tokenizer Protocol

    public var vocabularySize: Int { vocabulary.count }

    public var specialTokens: SpecialTokens { resolvedSpecialTokens }

    public func encode(_ text: String, config: TokenizerConfig) async throws -> TokenizedText {
        switch mode {
        case .legacy(let merges):
            return try encodeLegacy(text, config: config, merges: merges)
        case .byteLevelBPE(let bpeConfig):
            return try encodeByteLevelBPE(text, config: config, bpeConfig: bpeConfig)
        }
    }

    public func decode(_ ids: [Int]) async throws -> String {
        switch mode {
        case .legacy:
            return decodeLegacy(ids)
        case .byteLevelBPE:
            return decodeByteLevelBPE(ids)
        }
    }

    // MARK: - Byte-Level BPE Encoding

    private func encodeByteLevelBPE(_ text: String, config: TokenizerConfig, bpeConfig: ByteLevelConfig) throws -> TokenizedText {
        // Step 1: Normalize
        let normalized: String
        switch bpeConfig.normalization {
        case .nfc: normalized = text.precomposedStringWithCanonicalMapping
        case .nfkc: normalized = text.precomposedStringWithCompatibilityMapping
        case .none: normalized = text
        }

        // Step 2: Add prefix space if configured
        let input = bpeConfig.addPrefixSpace ? " " + normalized : normalized

        // Step 3: Regex pre-tokenization
        let preTokens = preTokenize(input, pattern: bpeConfig.preTokenizerPattern)

        // Step 4: Byte-level BPE encode each pre-token
        var pieces: [String] = []
        for preToken in preTokens {
            let byteChars = textToByteLevelChars(preToken)
            let merged = bpeMerge(byteChars, ranks: bpeConfig.mergeRanks)
            pieces.append(contentsOf: merged)
        }

        // Step 5: Add special tokens
        if config.addSpecialTokens {
            let startToken = specialTokens.cls?.text ?? specialTokens.bos?.text
            let endToken = specialTokens.sep?.text ?? specialTokens.eos?.text
            if let s = startToken { pieces.insert(s, at: 0) }
            if let e = endToken { pieces.append(e) }
        }

        // Step 6: Truncation
        if config.maxLength > 0 && pieces.count > config.maxLength {
            pieces = try truncate(pieces, config: config)
        }

        // Step 7: Map to IDs
        let unkId = specialTokens.unk?.id ?? 0
        var ids = [Int]()
        ids.reserveCapacity(pieces.count)
        for p in pieces {
            ids.append(vocabulary[p] ?? unkId)
        }

        // Step 8: Padding
        var mask = [Int](repeating: 1, count: ids.count)
        if config.padding == .max, config.maxLength > 0, ids.count < config.maxLength {
            guard let pad = specialTokens.pad else {
                throw EmbedKitError.invalidConfiguration("Padding requires PAD token in vocabulary")
            }
            let padCount = config.maxLength - ids.count
            ids += [Int](repeating: pad.id, count: padCount)
            pieces += [String](repeating: pad.text, count: padCount)
            mask += [Int](repeating: 0, count: padCount)
        }

        return TokenizedText(ids: ids, tokens: pieces, attentionMask: mask)
    }

    // MARK: - BPE Merge Algorithm (Priority-Queue)

    /// Applies BPE merges to a list of tokens using a priority-queue algorithm.
    /// O(n log n) per word, where n is the token count.
    private func bpeMerge(_ tokens: [String], ranks: [String: Int]) -> [String] {
        guard tokens.count >= 2 else { return tokens }

        // Use a doubly-linked list for efficient merge operations
        var list = tokens
        while list.count >= 2 {
            // Find the pair with the lowest merge rank
            var bestRank = Int.max
            var bestIndex = -1

            for i in 0..<(list.count - 1) {
                let pair = "\(list[i]) \(list[i + 1])"
                if let rank = ranks[pair], rank < bestRank {
                    bestRank = rank
                    bestIndex = i
                }
            }

            // No more merges possible
            if bestIndex == -1 { break }

            // Apply the merge
            let merged = list[bestIndex] + list[bestIndex + 1]
            list[bestIndex] = merged
            list.remove(at: bestIndex + 1)
        }
        return list
    }

    // MARK: - Byte-Level Unicode Mapping

    /// The canonical GPT-2 byte-to-unicode mapping.
    /// Maps each byte (0-255) to a printable Unicode character.
    private static let byteToUnicode: [UInt8: Character] = {
        var b2u = [UInt8: Character]()
        var n = 0

        // Printable ASCII bytes map to themselves
        // Range: 33..126 (! to ~), 161..172, 174..255
        let ranges: [ClosedRange<UInt8>] = [33...126, 161...172, 174...255]
        for range in ranges {
            for byte in range {
                b2u[byte] = Character(Unicode.Scalar(byte))
                n += 1
            }
        }

        // Non-printable bytes map to Unicode starting at U+0100 (Ā)
        var offset = 256
        for byte: UInt8 in 0...255 {
            if b2u[byte] == nil {
                b2u[byte] = Character(Unicode.Scalar(offset)!)
                offset += 1
            }
        }

        return b2u
    }()

    /// Inverse mapping: Unicode character back to byte.
    private static let unicodeToByte: [Character: UInt8] = {
        Dictionary(uniqueKeysWithValues: byteToUnicode.map { ($0.value, $0.key) })
    }()

    /// Convert a text string to its byte-level unicode character representation.
    private func textToByteLevelChars(_ text: String) -> [String] {
        let utf8 = Array(text.utf8)
        return utf8.map { byte in
            String(Self.byteToUnicode[byte]!)
        }
    }

    // MARK: - Byte-Level BPE Decoding

    private func decodeByteLevelBPE(_ ids: [Int]) -> String {
        let specials: Set<String> = Set([
            specialTokens.cls?.text,
            specialTokens.sep?.text,
            specialTokens.pad?.text,
            specialTokens.mask?.text,
            specialTokens.bos?.text,
            specialTokens.eos?.text
        ].compactMap { $0 })

        let tokens = ids.compactMap { vocabulary[id: $0] }.filter { !specials.contains($0) }
        let joined = tokens.joined()

        // Convert byte-level unicode characters back to bytes
        var bytes = [UInt8]()
        bytes.reserveCapacity(joined.count)
        for char in joined {
            if let byte = Self.unicodeToByte[char] {
                bytes.append(byte)
            }
        }

        return String(bytes: bytes, encoding: .utf8) ?? ""
    }

    // MARK: - Pre-Tokenization

    enum PreTokenizerPattern: Sendable {
        case gpt2
        case llama3
        case custom(String)
    }

    enum NormalizationSetting: Sendable {
        case nfc
        case nfkc
        case none
    }

    /// GPT-2 pre-tokenization regex pattern.
    /// Splits on contractions, word boundaries, numbers, whitespace, etc.
    private static let gpt2Pattern: String =
        #"'(?i:[sdmt])|'(?i:ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    /// Llama 3 pre-tokenization regex pattern.
    private static let llama3Pattern: String =
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    private func preTokenize(_ text: String, pattern: PreTokenizerPattern) -> [String] {
        guard !text.isEmpty else { return [] }

        let regexPattern: String
        switch pattern {
        case .gpt2: regexPattern = Self.gpt2Pattern
        case .llama3: regexPattern = Self.llama3Pattern
        case .custom(let p): regexPattern = p
        }

        do {
            let regex = try NSRegularExpression(pattern: regexPattern, options: [])
            let range = NSRange(text.startIndex..., in: text)
            let matches = regex.matches(in: text, options: [], range: range)
            return matches.compactMap { match in
                guard let range = Range(match.range, in: text) else { return nil }
                return String(text[range])
            }
        } catch {
            // Fallback: split on whitespace
            return text.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        }
    }

    // MARK: - Legacy Encoding

    private func encodeLegacy(_ text: String, config: TokenizerConfig, merges: [String: String]) throws -> TokenizedText {
        let normalized = lowercase ? text.lowercased() : text
        let words = normalized.split(whereSeparator: { $0.isWhitespace }).map(String.init)

        var pieces: [String] = []
        for w in words {
            var chars = w.map { String($0) }
            var i = 0
            while i + 1 < chars.count {
                let key = "\(chars[i]) \(chars[i+1])"
                if let merged = merges[key] {
                    chars[i] = merged
                    chars.remove(at: i+1)
                } else {
                    i += 1
                }
            }
            pieces.append(contentsOf: chars)
        }

        if config.addSpecialTokens {
            if let cls = specialTokens.cls?.text { pieces.insert(cls, at: 0) }
            if let sep = specialTokens.sep?.text { pieces.append(sep) }
        }

        if config.maxLength > 0 && pieces.count > config.maxLength {
            pieces = try truncate(pieces, config: config)
        }

        let unkId = specialTokens.unk?.id ?? 100
        var ids = [Int]()
        ids.reserveCapacity(pieces.count)
        for p in pieces { ids.append(vocabulary[p] ?? unkId) }

        var mask = [Int](repeating: 1, count: ids.count)
        if config.padding == .max, config.maxLength > 0, ids.count < config.maxLength {
            guard let pad = specialTokens.pad else {
                throw EmbedKitError.invalidConfiguration("Padding requires PAD token in vocabulary")
            }
            let padCount = config.maxLength - ids.count
            ids += [Int](repeating: pad.id, count: padCount)
            pieces += [String](repeating: pad.text, count: padCount)
            mask += [Int](repeating: 0, count: padCount)
        }

        return TokenizedText(ids: ids, tokens: pieces, attentionMask: mask)
    }

    private func decodeLegacy(_ ids: [Int]) -> String {
        let specials: Set<String> = Set([
            specialTokens.cls?.text,
            specialTokens.sep?.text,
            specialTokens.pad?.text,
            specialTokens.mask?.text,
            specialTokens.bos?.text,
            specialTokens.eos?.text
        ].compactMap { $0 })

        let toks = ids.compactMap { vocabulary[id: $0] }.filter { !specials.contains($0) }
        return toks.joined(separator: " ")
    }

    // MARK: - Shared Utilities

    private func truncate(_ pieces: [String], config: TokenizerConfig) throws -> [String] {
        switch config.truncation {
        case .none:
            throw EmbedKitError.inputTooLong(length: pieces.count, max: config.maxLength)
        case .end:
            return Array(pieces.prefix(config.maxLength))
        case .start:
            return Array(pieces.suffix(config.maxLength))
        case .middle:
            let keep = config.maxLength
            let head = keep / 2
            let tail = keep - head
            return Array(pieces.prefix(head) + pieces.suffix(tail))
        }
    }

    private static func resolveSpecialTokens(vocabulary: Vocabulary, unkToken: String) -> SpecialTokens {
        // Try BERT-style first, then BPE-style
        let cls = vocabulary["[CLS]"] ?? vocabulary["<s>"]
        let clsText = vocabulary["[CLS]"] != nil ? "[CLS]" : "<s>"
        let sep = vocabulary["[SEP]"] ?? vocabulary["</s>"]
        let sepText = vocabulary["[SEP]"] != nil ? "[SEP]" : "</s>"
        let pad = vocabulary["[PAD]"] ?? vocabulary["<pad>"]
        let padText = vocabulary["[PAD]"] != nil ? "[PAD]" : "<pad>"
        let unk = vocabulary[unkToken]
        let mask = vocabulary["[MASK]"] ?? vocabulary["<mask>"]
        let maskText = vocabulary["[MASK]"] != nil ? "[MASK]" : "<mask>"
        let bos = vocabulary["<s>"]
        let eos = vocabulary["</s>"]

        return SpecialTokens(
            cls: cls.map { .init(text: clsText, id: $0) },
            sep: sep.map { .init(text: sepText, id: $0) },
            pad: pad.map { .init(text: padText, id: $0) },
            unk: unk.map { .init(text: unkToken, id: $0) },
            mask: mask.map { .init(text: maskText, id: $0) },
            bos: bos.map { .init(text: "<s>", id: $0) },
            eos: eos.map { .init(text: "</s>", id: $0) }
        )
    }
}

// MARK: - File Loading

extension BPETokenizer {
    static func loadMerges(from url: URL) throws -> [String: String] {
        let data = try Data(contentsOf: url)
        guard let content = String(data: data, encoding: .utf8) else {
            throw EmbedKitError.tokenizationFailed("Invalid merges encoding")
        }
        var map: [String: String] = [:]
        for raw in content.split(whereSeparator: { $0 == "\n" || $0 == "\r" }) {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") { continue }
            let parts = line.split(separator: " ", omittingEmptySubsequences: true)
            guard parts.count >= 2 else { continue }
            let a = String(parts[0])
            let b = String(parts[1])
            map["\(a) \(b)"] = a + b
        }
        return map
    }
}

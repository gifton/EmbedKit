// EmbedKit - WordPiece Tokenizer

import Foundation

/// Greedy WordPiece tokenizer with configurable lowercasing and strict PAD requirements for padding.
/// - Note: Punctuation and symbols (ASCII and Unicode) are split into standalone tokens. Unknown tokens
///   map to `unkToken`.
public struct WordPieceTokenizer: Tokenizer {
    public let vocabulary: Vocabulary
    public let unkToken: String
    public let lowercase: Bool

    public init(vocabulary: Vocabulary, unkToken: String = "[UNK]", lowercase: Bool = true) {
        self.vocabulary = vocabulary
        self.unkToken = unkToken
        self.lowercase = lowercase
    }

    public var vocabularySize: Int { vocabulary.count }

    public var specialTokens: SpecialTokens {
        let cls = vocabulary["[CLS]"]
        let sep = vocabulary["[SEP]"]
        let pad = vocabulary["[PAD]"]
        let unk = vocabulary[unkToken]
        let mask = vocabulary["[MASK]"]
        return SpecialTokens(
            cls: cls.map { .init(text: "[CLS]", id: $0) },
            sep: sep.map { .init(text: "[SEP]", id: $0) },
            pad: pad.map { .init(text: "[PAD]", id: $0) },
            unk: unk.map { .init(text: unkToken, id: $0) },
            mask: mask.map { .init(text: "[MASK]", id: $0) }
        )
    }

    public func encode(_ text: String, config: TokenizerConfig) async throws -> TokenizedText {
        // Normalize
        let normalized = lowercase ? text.lowercased() : text

        // Basic tokenization (whitespace + punctuation separation)
        let words = basicTokenize(normalized)

        // WordPiece tokenization (greedy longest-match-first)
        let unkId = specialTokens.unk?.id ?? 100
        var pieces: [String] = []
        pieces.reserveCapacity(words.count * 2)
        for word in words {
            let subs = wordpiece(word)
            if subs.isEmpty { pieces.append(unkToken) }
            else { pieces.append(contentsOf: subs) }
        }

        // Add special tokens if requested
        if config.addSpecialTokens {
            if let cls = specialTokens.cls?.text { pieces.insert(cls, at: 0) }
            if let sep = specialTokens.sep?.text { pieces.append(sep) }
        }

        // Truncation
        if config.maxLength > 0 && pieces.count > config.maxLength {
            switch config.truncation {
            case .none:
                throw EmbedKitError.inputTooLong(length: pieces.count, max: config.maxLength)
            case .end:
                pieces = Array(pieces.prefix(config.maxLength))
            case .start:
                pieces = Array(pieces.suffix(config.maxLength))
            case .middle:
                let keep = config.maxLength
                let head = keep / 2
                let tail = keep - head
                pieces = Array(pieces.prefix(head) + pieces.suffix(tail))
            }
        }

        // Map to IDs
        var ids: [Int] = ids(for: pieces, unkId: unkId)

        // Padding: only for .max; .batch is handled at batch assembly time
        var mask = Array(repeating: 1, count: ids.count)
        if config.padding == .max, config.maxLength > 0, ids.count < config.maxLength {
            guard let pad = specialTokens.pad else {
                throw EmbedKitError.invalidConfiguration("Padding requires PAD token in vocabulary")
            }
            let padId = pad.id
            let padCount = config.maxLength - ids.count
            ids += Array(repeating: padId, count: padCount)
            pieces += Array(repeating: pad.text, count: padCount)
            mask += Array(repeating: 0, count: padCount)
        }

        return TokenizedText(ids: ids, tokens: pieces, attentionMask: mask)
    }

    public func decode(_ ids: [Int]) async throws -> String {
        // Merge subword tokens (##) into whole words; skip special tokens.
        let specials: Set<String> = [
            specialTokens.cls?.text,
            specialTokens.sep?.text,
            specialTokens.pad?.text,
            specialTokens.mask?.text,
            specialTokens.bos?.text,
            specialTokens.eos?.text
        ].compactMap { $0 }.reduce(into: Set<String>()) { $0.insert($1) }

        let pieces = ids.compactMap { vocabulary[id: $0] }
        var words: [String] = []
        for p in pieces where !specials.contains(p) {
            if p.hasPrefix("##") {
                let sub = String(p.dropFirst(2))
                if words.isEmpty { words.append(sub) }
                else { words[words.count - 1] += sub }
            } else {
                words.append(p)
            }
        }
        return words.joined(separator: " ")
    }

    // MARK: - Private

    // Basic tokenizer: split on whitespace and detach punctuation/symbols (ASCII + Unicode)
    private func basicTokenize(_ text: String) -> [String] {
        var tokens: [String] = []
        tokens.reserveCapacity(text.count / 4)
        var current = ""
        @inline(__always) func flush() { if !current.isEmpty { tokens.append(current); current.removeAll(keepingCapacity: true) } }
        @inline(__always) func isPunctOrSymbol(_ ch: Character) -> Bool {
            for s in ch.unicodeScalars {
                if CharacterSet.punctuationCharacters.contains(s) { return true }
                if CharacterSet.symbols.contains(s) { return true }
            }
            return false
        }
        for ch in text {
            if ch.isWhitespace { flush() }
            else if isPunctOrSymbol(ch) { flush(); tokens.append(String(ch)) }
            else { current.append(ch) }
        }
        flush()
        return tokens
    }

    // Greedy WordPiece for a single word
    private func wordpiece(_ word: String) -> [String] {
        // Reject overly long words per BERT ref implementation
        if word.count > 200 { return [unkToken] }
        var start = word.startIndex
        var result: [String] = []

        while start < word.endIndex {
            var end = word.endIndex
            var curMatch: String? = nil
            while start < end {
                var sub = String(word[start..<end])
                if start > word.startIndex { sub = "##" + sub }
                if let _ = vocabulary[sub] { curMatch = sub; break }
                end = word.index(before: end)
            }
            if let match = curMatch {
                result.append(match)
                start = end
            } else {
                return [unkToken]
            }
        }
        return result
    }

    @inline(__always)
    private func ids(for pieces: [String], unkId: Int) -> [Int] {
        var out: [Int] = []
        out.reserveCapacity(pieces.count)
        for p in pieces { out.append(vocabulary[p] ?? unkId) }
        return out
    }
}

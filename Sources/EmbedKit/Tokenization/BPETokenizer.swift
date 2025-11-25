// EmbedKit - Simple BPE Tokenizer

import Foundation

/// Minimal BPE tokenizer scaffold with configurable merges and strict PAD requirement for padding.
/// This is not a full byte-level BPE; it supports greedy left-to-right merges based on a provided table.
public struct BPETokenizer: Tokenizer {
    public let vocabulary: Vocabulary
    /// Merges map from "a b" -> "ab"
    private let merges: [String: String]
    public let lowercase: Bool
    public let unkToken: String

    public init(vocabulary: Vocabulary, merges: [String: String], unkToken: String = "[UNK]", lowercase: Bool = true) {
        self.vocabulary = vocabulary
        self.merges = merges
        self.lowercase = lowercase
        self.unkToken = unkToken
    }

    // Convenience loader: vocab as newline-delimited tokens; merges as pairs per line: "a b"
    public static func load(vocabURL: URL, mergesURL: URL, unkToken: String = "[UNK]", lowercase: Bool = true) throws -> BPETokenizer {
        let vocab = try Vocabulary.load(from: vocabURL)
        let merges = try loadMerges(from: mergesURL)
        return BPETokenizer(vocabulary: vocab, merges: merges, unkToken: unkToken, lowercase: lowercase)
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
        let normalized = lowercase ? text.lowercased() : text
        // Split on whitespace for words
        let words = normalized.split(whereSeparator: { $0.isWhitespace }).map(String.init)

        var pieces: [String] = []
        for w in words {
            var chars = w.map { String($0) }
            // Greedy left-to-right pair merges until no changes
            var i = 0
            while i + 1 < chars.count {
                let key = "\(chars[i]) \(chars[i+1])"
                if let merged = merges[key] {
                    // Replace pair with merged token
                    chars[i] = merged
                    chars.remove(at: i+1)
                    // Do not advance i, try to merge again with next neighbor
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

        // Map to IDs with UNK fallback
        let unkId = specialTokens.unk?.id ?? 100
        var ids: [Int] = []
        ids.reserveCapacity(pieces.count)
        for p in pieces { ids.append(vocabulary[p] ?? unkId) }

        // Padding: only for .max; .batch is handled by the model's batch assembly
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
        // For this scaffold, simply join known tokens and skip special tokens
        let specials: Set<String> = [
            specialTokens.cls?.text,
            specialTokens.sep?.text,
            specialTokens.pad?.text,
            specialTokens.mask?.text,
            specialTokens.bos?.text,
            specialTokens.eos?.text
        ].compactMap { $0 }.reduce(into: Set<String>()) { $0.insert($1) }

        let toks = ids.compactMap { vocabulary[id: $0] }.filter { !specials.contains($0) }
        return toks.joined(separator: " ")
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
            if line.isEmpty { continue }
            if line.hasPrefix("#") { continue }
            let parts = line.split(separator: " ", omittingEmptySubsequences: true)
            guard parts.count >= 2 else { continue }
            let a = String(parts[0])
            let b = String(parts[1])
            map["\(a) \(b)"] = a + b
        }
        return map
    }
}

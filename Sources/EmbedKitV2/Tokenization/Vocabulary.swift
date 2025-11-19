// EmbedKitV2 - Vocabulary (Scaffold)

import Foundation

public struct Vocabulary: Sendable {
    private let tokenToID: [String: Int]
    private let idToToken: [Int: String]

    public var count: Int { tokenToID.count }

    public init(tokens: [String]) {
        var t2i: [String: Int] = [:]
        var i2t: [Int: String] = [:]
        t2i.reserveCapacity(tokens.count)
        i2t.reserveCapacity(tokens.count)
        for (i, tok) in tokens.enumerated() {
            t2i[tok] = i
            i2t[i] = tok
        }
        self.tokenToID = t2i
        self.idToToken = i2t
    }

    public subscript(_ token: String) -> Int? { tokenToID[token] }
    public subscript(id id: Int) -> String? { idToToken[id] }

    // Optional convenience loader; simple line-by-line reader.
    public static func load(from url: URL) throws -> Vocabulary {
        let data = try Data(contentsOf: url)
        guard let content = String(data: data, encoding: .utf8) else {
            throw EmbedKitError.tokenizationFailed("Invalid vocabulary encoding")
        }
        let tokens = content.split(whereSeparator: { $0 == "\n" || $0 == "\r" }).map(String.init)
        return Vocabulary(tokens: tokens)
    }
}

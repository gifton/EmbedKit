// EmbedKit - Convenience API
// High-level, user-friendly interfaces for common operations

import Foundation

// MARK: - ModelManager Convenience Extensions

public extension ModelManager {

    /// Simple one-line embedding using Apple's system embedding model.
    ///
    /// Uses NLContextualEmbedding to generate semantically meaningful embeddings.
    /// The model is cached after first use for efficiency across multiple calls.
    ///
    /// - Parameter text: The text to embed.
    /// - Returns: The raw embedding vector.
    /// - Throws: `EmbedKitError` if the system model is unavailable or embedding fails.
    ///
    /// Example:
    /// ```swift
    /// let manager = ModelManager()
    /// let vector = try await manager.quickEmbed("Hello, world!")
    /// print("Embedding dimensions: \(vector.count)")
    /// ```
    ///
    /// - Note: Requires iOS 17+/macOS 14+ for NLContextualEmbedding support.
    func quickEmbed(_ text: String) async throws -> [Float] {
        let model = try await getOrCreateSystemModel()
        let embedding = try await model.embed(text)
        return embedding.vector
    }

    /// Semantic search: finds the most similar documents to a query.
    ///
    /// Uses Apple's NLContextualEmbedding to generate semantically meaningful embeddings
    /// for the query and all documents, then returns the top-K documents ranked by
    /// cosine similarity.
    ///
    /// - Parameters:
    ///   - query: The search query.
    ///   - documents: Array of documents to search through.
    ///   - topK: Maximum number of results to return (default: 10).
    /// - Returns: Array of (index, score) tuples sorted by descending similarity.
    /// - Throws: `EmbedKitError` if the system model is unavailable or embedding fails.
    ///
    /// Example:
    /// ```swift
    /// let manager = ModelManager()
    /// let results = try await manager.semanticSearch(
    ///     query: "machine learning",
    ///     in: ["AI is transforming industries", "The weather is nice", "Deep learning basics"],
    ///     topK: 2
    /// )
    /// // results: [(index: 0, score: 0.85), (index: 2, score: 0.72)]
    /// ```
    ///
    /// - Note: Requires iOS 17+/macOS 14+ for NLContextualEmbedding support.
    func semanticSearch(
        query: String,
        in documents: [String],
        topK: Int = 10
    ) async throws -> [(index: Int, score: Float)] {
        guard !documents.isEmpty else { return [] }

        let model = try await getOrCreateSystemModel()

        // Generate query embedding
        let queryEmbedding = try await model.embed(query)

        // Generate document embeddings in batch for efficiency
        let docEmbeddings = try await model.embedBatch(documents, options: BatchOptions())

        // Compute similarities and rank
        let similarities = docEmbeddings.enumerated().map { index, docEmbed in
            (index: index, score: queryEmbedding.similarity(to: docEmbed))
        }

        return similarities
            .sorted { $0.score > $1.score }
            .prefix(topK)
            .map { $0 }
    }

    /// Find the single most similar document to a query.
    ///
    /// Convenience wrapper around `semanticSearch` that returns only the best match.
    ///
    /// - Parameters:
    ///   - query: The search query.
    ///   - documents: Array of documents to search through.
    /// - Returns: Tuple of (index, document, score) for the best match, or nil if no documents.
    /// - Throws: `EmbedKitError` if the system model is unavailable or embedding fails.
    ///
    /// - Note: Requires iOS 17+/macOS 14+ for NLContextualEmbedding support.
    func findMostSimilar(
        query: String,
        in documents: [String]
    ) async throws -> (index: Int, document: String, score: Float)? {
        let results = try await semanticSearch(query: query, in: documents, topK: 1)
        guard let best = results.first else { return nil }
        return (index: best.index, document: documents[best.index], score: best.score)
    }

    /// Compute pairwise similarity matrix for a set of documents.
    ///
    /// Uses Apple's NLContextualEmbedding to generate semantically meaningful embeddings,
    /// then computes cosine similarity between all pairs.
    ///
    /// Returns an NxN matrix where element [i][j] is the similarity between documents i and j.
    ///
    /// - Parameter documents: Array of documents.
    /// - Returns: 2D similarity matrix (row-major). Diagonal elements are 1.0 (self-similarity).
    /// - Throws: `EmbedKitError` if the system model is unavailable or embedding fails.
    ///
    /// - Note: Requires iOS 17+/macOS 14+ for NLContextualEmbedding support.
    func similarityMatrix(_ documents: [String]) async throws -> [[Float]] {
        guard !documents.isEmpty else { return [] }

        let model = try await getOrCreateSystemModel()
        let embeddings = try await model.embedBatch(documents, options: BatchOptions())

        var matrix: [[Float]] = []
        for i in 0..<embeddings.count {
            var row: [Float] = []
            for j in 0..<embeddings.count {
                row.append(embeddings[i].similarity(to: embeddings[j]))
            }
            matrix.append(row)
        }
        return matrix
    }
}

// MARK: - Async Sequence Support

public extension ModelManager {

    /// Embed texts from an async sequence, yielding embeddings as they're computed.
    ///
    /// Uses Apple's NLContextualEmbedding to generate semantically meaningful embeddings.
    /// Useful for processing streaming data or large datasets without loading all into memory.
    ///
    /// - Parameter texts: An async sequence of texts to embed.
    /// - Returns: An async throwing stream of embeddings.
    ///
    /// Example:
    /// ```swift
    /// let manager = ModelManager()
    /// let lines = FileHandle.standardInput.bytes.lines
    ///
    /// for try await embedding in manager.embedSequence(lines.map { String($0) }) {
    ///     print("Embedded: \(embedding.dimensions) dimensions")
    /// }
    /// ```
    ///
    /// - Note: Requires iOS 17+/macOS 14+ for NLContextualEmbedding support.
    func embedSequence<S: AsyncSequence>(
        _ texts: S
    ) -> AsyncThrowingStream<Embedding, Error> where S.Element == String {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let model = try await getOrCreateSystemModel()
                    for try await text in texts {
                        let embedding = try await model.embed(text)
                        continuation.yield(embedding)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Embed texts from a regular sequence using an async stream.
    ///
    /// Uses Apple's NLContextualEmbedding to generate semantically meaningful embeddings.
    ///
    /// - Parameter texts: A sequence of texts to embed.
    /// - Returns: An async throwing stream of embeddings.
    ///
    /// - Note: Requires iOS 17+/macOS 14+ for NLContextualEmbedding support.
    func embedSequence<S: Sequence>(
        _ texts: S
    ) -> AsyncThrowingStream<Embedding, Error> where S.Element == String {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let model = try await getOrCreateSystemModel()
                    for text in texts {
                        let embedding = try await model.embed(text)
                        continuation.yield(embedding)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}

// MARK: - Embedding Array Extensions

public extension Array where Element == Embedding {

    /// Compute average embedding from an array of embeddings.
    ///
    /// Useful for combining multiple embeddings into a single representative vector.
    ///
    /// - Parameter normalize: Whether to normalize the result (default: true).
    /// - Returns: The averaged embedding, or nil if the array is empty.
    func averaged(normalize: Bool = true) -> Embedding? {
        guard let first = self.first else { return nil }

        let dim = first.dimensions
        var sum = [Float](repeating: 0, count: dim)

        for emb in self {
            guard emb.dimensions == dim else { continue }
            for i in 0..<dim {
                sum[i] += emb.vector[i]
            }
        }

        let count = Float(self.count)
        var avg = sum.map { $0 / count }

        if normalize {
            let mag = sqrt(avg.reduce(0) { $0 + $1 * $1 })
            if mag > 0 {
                avg = avg.map { $0 / mag }
            }
        }

        return Embedding(
            vector: avg,
            metadata: EmbeddingMetadata(
                modelID: first.metadata.modelID,
                tokenCount: self.reduce(0) { $0 + $1.metadata.tokenCount },
                processingTime: self.reduce(0) { $0 + $1.metadata.processingTime },
                normalized: normalize,
                poolingStrategy: .mean,
                custom: ["averaged_count": "\(self.count)"]
            )
        )
    }

    /// Find the embedding most similar to a target.
    ///
    /// - Parameter target: The target embedding to compare against.
    /// - Returns: Tuple of (index, embedding, score) for the best match.
    func mostSimilar(to target: Embedding) -> (index: Int, embedding: Embedding, score: Float)? {
        guard !self.isEmpty else { return nil }

        var bestIndex = 0
        var bestScore: Float = -Float.greatestFiniteMagnitude

        for (i, emb) in self.enumerated() {
            let score = target.similarity(to: emb)
            if score > bestScore {
                bestScore = score
                bestIndex = i
            }
        }

        return (index: bestIndex, embedding: self[bestIndex], score: bestScore)
    }
}


// EmbedKit - Streaming Processor
// Process large documents by chunking and streaming embeddings

import Foundation

// MARK: - Chunking Strategy

/// Strategy for splitting text into chunks.
public enum ChunkingStrategy: Sendable {
    /// Split by approximate character count, respecting word boundaries.
    case characters(size: Int)

    /// Split by approximate token count (estimated from characters).
    /// Uses a heuristic of ~4 characters per token for English text.
    case tokens(count: Int)

    /// Split by sentences, grouping up to N sentences per chunk.
    case sentences(count: Int)

    /// Split by paragraphs (double newline separated), grouping up to N paragraphs.
    case paragraphs(count: Int)

    /// Custom splitting using a provided closure.
    case custom(@Sendable (String) -> [String])
}

// MARK: - Chunk

/// A piece of text extracted from a larger document.
public struct Chunk: Sendable {
    /// The text content of this chunk.
    public let text: String

    /// Character offset from the start of the original document.
    public let startOffset: Int

    /// Character offset of the end (exclusive) in the original document.
    public let endOffset: Int

    /// Index of this chunk (0-based).
    public let index: Int

    /// Total number of chunks in the document.
    public let totalChunks: Int

    /// Number of characters overlapping with the previous chunk.
    public let overlapWithPrevious: Int

    /// Number of characters overlapping with the next chunk.
    public let overlapWithNext: Int

    public init(
        text: String,
        startOffset: Int,
        endOffset: Int,
        index: Int,
        totalChunks: Int,
        overlapWithPrevious: Int = 0,
        overlapWithNext: Int = 0
    ) {
        self.text = text
        self.startOffset = startOffset
        self.endOffset = endOffset
        self.index = index
        self.totalChunks = totalChunks
        self.overlapWithPrevious = overlapWithPrevious
        self.overlapWithNext = overlapWithNext
    }

    /// Whether this is the first chunk.
    public var isFirst: Bool { index == 0 }

    /// Whether this is the last chunk.
    public var isLast: Bool { index == totalChunks - 1 }
}

// MARK: - Chunk Embedding

/// An embedding for a specific chunk of a document.
public struct ChunkEmbedding: Sendable {
    /// The chunk this embedding represents.
    public let chunk: Chunk

    /// The computed embedding.
    public let embedding: Embedding

    public init(chunk: Chunk, embedding: Embedding) {
        self.chunk = chunk
        self.embedding = embedding
    }
}

// MARK: - Aggregation Strategy

/// Strategy for combining multiple chunk embeddings into a single embedding.
public enum AggregationStrategy: Sendable {
    /// Average all chunk embeddings (uniform weights).
    case mean

    /// Weighted average, with weights proportional to chunk length.
    case weightedByLength

    /// Use only the first chunk's embedding.
    case first

    /// Use only the last chunk's embedding.
    case last

    /// Concatenate all embeddings (warning: changes dimension).
    case concatenate

    /// Custom aggregation function.
    case custom(@Sendable ([ChunkEmbedding]) -> [Float])
}

// MARK: - Streaming Configuration

/// Configuration for streaming document processing.
public struct StreamingConfig: Sendable {
    /// How to split the document into chunks.
    public var chunkingStrategy: ChunkingStrategy = .tokens(count: 256)

    /// Number of characters/tokens to overlap between adjacent chunks.
    /// Helps preserve context at chunk boundaries.
    public var overlap: Int = 50

    /// Minimum chunk size (chunks smaller than this are merged with neighbors).
    public var minChunkSize: Int = 50

    /// Maximum number of chunks to process concurrently.
    public var concurrency: Int = 4

    /// Whether to stop processing on first error.
    public var stopOnError: Bool = true

    /// Options for the underlying batch embedding calls.
    public var batchOptions: BatchOptions = BatchOptions()

    public init() {}
}

// MARK: - Streaming Result

/// Result of processing a document with streaming.
public struct StreamingResult: Sendable {
    /// All chunk embeddings in order.
    public let chunkEmbeddings: [ChunkEmbedding]

    /// Total processing time.
    public let processingTime: TimeInterval

    /// Number of chunks processed.
    public var chunkCount: Int { chunkEmbeddings.count }

    /// Aggregate the chunk embeddings using the specified strategy.
    public func aggregate(strategy: AggregationStrategy, dimensions: Int) -> [Float] {
        guard !chunkEmbeddings.isEmpty else { return [] }

        switch strategy {
        case .mean:
            return aggregateMean()
        case .weightedByLength:
            return aggregateWeightedByLength()
        case .first:
            return chunkEmbeddings.first?.embedding.vector ?? []
        case .last:
            return chunkEmbeddings.last?.embedding.vector ?? []
        case .concatenate:
            return chunkEmbeddings.flatMap { $0.embedding.vector }
        case .custom(let fn):
            return fn(chunkEmbeddings)
        }
    }

    private func aggregateMean() -> [Float] {
        guard let first = chunkEmbeddings.first else { return [] }
        let dim = first.embedding.vector.count
        var result = [Float](repeating: 0, count: dim)
        for ce in chunkEmbeddings {
            for (i, v) in ce.embedding.vector.enumerated() where i < dim {
                result[i] += v
            }
        }
        let n = Float(chunkEmbeddings.count)
        return result.map { $0 / n }
    }

    private func aggregateWeightedByLength() -> [Float] {
        guard let first = chunkEmbeddings.first else { return [] }
        let dim = first.embedding.vector.count
        var result = [Float](repeating: 0, count: dim)
        var totalWeight: Float = 0

        for ce in chunkEmbeddings {
            let weight = Float(ce.chunk.text.count)
            totalWeight += weight
            for (i, v) in ce.embedding.vector.enumerated() where i < dim {
                result[i] += v * weight
            }
        }

        guard totalWeight > 0 else { return result }
        return result.map { $0 / totalWeight }
    }
}

// MARK: - Streaming Processor

/// Processes large documents by splitting them into chunks and streaming embeddings.
///
/// `StreamingProcessor` is useful for documents that exceed the model's context window.
/// It splits text into overlapping chunks, embeds each chunk, and can aggregate
/// the results into a single document embedding.
///
/// ## Example Usage
/// ```swift
/// let processor = StreamingProcessor(model: model)
///
/// // Stream embeddings as they're computed
/// for try await chunkEmbedding in processor.embedStream("Very long document...") {
///     print("Chunk \(chunkEmbedding.chunk.index): embedded")
/// }
///
/// // Or get all results at once
/// let result = try await processor.embedDocument("Long document...")
/// let docEmbedding = result.aggregate(strategy: .weightedByLength, dimensions: 384)
/// ```
public struct StreamingProcessor: Sendable {
    private let model: any EmbeddingModel
    private let config: StreamingConfig

    /// Creates a new streaming processor.
    ///
    /// - Parameters:
    ///   - model: The embedding model to use.
    ///   - config: Configuration for chunking and processing.
    public init(model: any EmbeddingModel, config: StreamingConfig = StreamingConfig()) {
        self.model = model
        self.config = config
    }

    // MARK: - Public API

    /// Stream embeddings for a document, yielding each chunk embedding as it's computed.
    ///
    /// - Parameter text: The document text to process.
    /// - Returns: An async stream of chunk embeddings.
    public func embedStream(_ text: String) -> AsyncThrowingStream<ChunkEmbedding, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let chunks = createChunks(from: text)
                    for chunk in chunks {
                        let embedding = try await model.embed(chunk.text)
                        let chunkEmbedding = ChunkEmbedding(chunk: chunk, embedding: embedding)
                        continuation.yield(chunkEmbedding)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Stream embeddings with concurrent processing for better throughput.
    ///
    /// - Parameter text: The document text to process.
    /// - Returns: An async stream of chunk embeddings (may arrive out of order).
    public func embedStreamConcurrent(_ text: String) -> AsyncThrowingStream<ChunkEmbedding, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let chunks = createChunks(from: text)
                    try await withThrowingTaskGroup(of: ChunkEmbedding.self) { group in
                        var pending = 0
                        var nextIndex = 0

                        // Seed initial concurrent tasks
                        while nextIndex < chunks.count && pending < config.concurrency {
                            let chunk = chunks[nextIndex]
                            nextIndex += 1
                            pending += 1
                            group.addTask {
                                let embedding = try await self.model.embed(chunk.text)
                                return ChunkEmbedding(chunk: chunk, embedding: embedding)
                            }
                        }

                        // Process results and add new tasks
                        for try await result in group {
                            continuation.yield(result)
                            pending -= 1

                            if nextIndex < chunks.count {
                                let chunk = chunks[nextIndex]
                                nextIndex += 1
                                pending += 1
                                group.addTask {
                                    let embedding = try await self.model.embed(chunk.text)
                                    return ChunkEmbedding(chunk: chunk, embedding: embedding)
                                }
                            }
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Process an entire document and return all chunk embeddings.
    ///
    /// - Parameter text: The document text to process.
    /// - Returns: A result containing all chunk embeddings and metadata.
    public func embedDocument(_ text: String) async throws -> StreamingResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        let chunks = createChunks(from: text)

        // Use batch embedding for efficiency
        let texts = chunks.map { $0.text }
        let embeddings = try await model.embedBatch(texts, options: config.batchOptions)

        let chunkEmbeddings = zip(chunks, embeddings).map { chunk, embedding in
            ChunkEmbedding(chunk: chunk, embedding: embedding)
        }

        return StreamingResult(
            chunkEmbeddings: chunkEmbeddings,
            processingTime: CFAbsoluteTimeGetCurrent() - startTime
        )
    }

    /// Process a document and return a single aggregated embedding.
    ///
    /// - Parameters:
    ///   - text: The document text to process.
    ///   - aggregation: How to combine chunk embeddings.
    /// - Returns: The aggregated embedding vector.
    public func embedDocumentAggregated(
        _ text: String,
        aggregation: AggregationStrategy = .weightedByLength
    ) async throws -> Embedding {
        let result = try await embedDocument(text)
        let aggregated = result.aggregate(strategy: aggregation, dimensions: model.dimensions)

        // Normalize the aggregated vector
        let magnitude = sqrt(aggregated.reduce(0) { $0 + $1 * $1 })
        let normalized = magnitude > 0 ? aggregated.map { $0 / magnitude } : aggregated

        return Embedding(
            vector: normalized,
            metadata: EmbeddingMetadata(
                modelID: model.id,
                tokenCount: result.chunkEmbeddings.reduce(0) { $0 + $1.embedding.metadata.tokenCount },
                processingTime: result.processingTime,
                normalized: true,
                poolingStrategy: .mean,
                truncated: false,
                custom: ["chunks": "\(result.chunkCount)"]
            )
        )
    }

    /// Get chunks without embedding them (useful for previewing chunking).
    ///
    /// - Parameter text: The document text to chunk.
    /// - Returns: The chunks that would be created.
    public func previewChunks(_ text: String) -> [Chunk] {
        createChunks(from: text)
    }

    // MARK: - Chunking Implementation

    private func createChunks(from text: String) -> [Chunk] {
        let rawChunks: [String]

        switch config.chunkingStrategy {
        case .characters(let size):
            rawChunks = chunkByCharacters(text, size: size)
        case .tokens(let count):
            // Estimate ~4 chars per token for English
            rawChunks = chunkByCharacters(text, size: count * 4)
        case .sentences(let count):
            rawChunks = chunkBySentences(text, count: count)
        case .paragraphs(let count):
            rawChunks = chunkByParagraphs(text, count: count)
        case .custom(let fn):
            rawChunks = fn(text)
        }

        // Build chunk objects with offset information
        return buildChunksWithMetadata(rawChunks: rawChunks, originalText: text)
    }

    private func chunkByCharacters(_ text: String, size: Int) -> [String] {
        guard !text.isEmpty && size > 0 else { return [] }

        var chunks: [String] = []
        let overlap = min(config.overlap, size / 2) // Overlap can't exceed half chunk size
        var startIndex = text.startIndex

        while startIndex < text.endIndex {
            // Calculate end index for this chunk
            var endIndex = text.index(startIndex, offsetBy: size, limitedBy: text.endIndex) ?? text.endIndex

            // Try to break at a word boundary (space, newline)
            if endIndex < text.endIndex {
                let searchRange = text.index(endIndex, offsetBy: -min(50, size/4), limitedBy: startIndex) ?? startIndex
                if let spaceIndex = text[searchRange..<endIndex].lastIndex(where: { $0.isWhitespace }) {
                    endIndex = text.index(after: spaceIndex)
                }
            }

            let chunk = String(text[startIndex..<endIndex])
            if chunk.count >= config.minChunkSize || chunks.isEmpty {
                chunks.append(chunk)
            } else if let last = chunks.popLast() {
                // Merge small final chunk with previous
                chunks.append(last + chunk)
            }

            // Move start with overlap
            if endIndex >= text.endIndex {
                break
            }

            let advanceBy = max(1, text.distance(from: startIndex, to: endIndex) - overlap)
            startIndex = text.index(startIndex, offsetBy: advanceBy, limitedBy: text.endIndex) ?? text.endIndex
        }

        return chunks
    }

    private func chunkBySentences(_ text: String, count: Int) -> [String] {
        guard !text.isEmpty && count > 0 else { return [] }

        // Simple sentence splitting using CharacterSet for sentence terminators
        var sentences: [String] = []
        var current = ""

        for char in text {
            current.append(char)
            if char == "." || char == "!" || char == "?" {
                // Check if followed by whitespace or end (lookahead simulation)
                let trimmed = current.trimmingCharacters(in: .whitespaces)
                if !trimmed.isEmpty {
                    sentences.append(trimmed)
                }
                current = ""
            }
        }
        // Handle remaining text without terminal punctuation
        let remaining = current.trimmingCharacters(in: .whitespaces)
        if !remaining.isEmpty {
            sentences.append(remaining)
        }

        guard !sentences.isEmpty else { return [text] }

        var chunks: [String] = []
        var currentChunk: [String] = []
        let overlapSentences = max(1, config.overlap / 50) // Roughly estimate sentences for overlap

        for sentence in sentences {
            currentChunk.append(sentence)

            if currentChunk.count >= count {
                chunks.append(currentChunk.joined(separator: " "))

                // Keep overlap sentences for next chunk
                let keepCount = min(overlapSentences, currentChunk.count - 1)
                currentChunk = Array(currentChunk.suffix(keepCount))
            }
        }

        // Handle remaining sentences
        if !currentChunk.isEmpty {
            let remainingText = currentChunk.joined(separator: " ")
            if remainingText.count >= config.minChunkSize || chunks.isEmpty {
                chunks.append(remainingText)
            } else if let last = chunks.popLast() {
                chunks.append(last + " " + remainingText)
            }
        }

        return chunks
    }

    private func chunkByParagraphs(_ text: String, count: Int) -> [String] {
        guard !text.isEmpty && count > 0 else { return [] }

        let paragraphs = text.components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        guard !paragraphs.isEmpty else { return [text] }

        var chunks: [String] = []
        var currentChunk: [String] = []

        for paragraph in paragraphs {
            currentChunk.append(paragraph)

            if currentChunk.count >= count {
                chunks.append(currentChunk.joined(separator: "\n\n"))
                // Keep last paragraph for overlap
                currentChunk = config.overlap > 0 ? [currentChunk.last!] : []
            }
        }

        if !currentChunk.isEmpty {
            let remaining = currentChunk.joined(separator: "\n\n")
            if remaining.count >= config.minChunkSize || chunks.isEmpty {
                chunks.append(remaining)
            } else if let last = chunks.popLast() {
                chunks.append(last + "\n\n" + remaining)
            }
        }

        return chunks
    }

    private func buildChunksWithMetadata(rawChunks: [String], originalText: String) -> [Chunk] {
        guard !rawChunks.isEmpty else { return [] }

        var chunks: [Chunk] = []
        var currentOffset = 0
        let total = rawChunks.count

        for (index, text) in rawChunks.enumerated() {
            // Find actual position in original text
            if let range = originalText.range(of: text, range: originalText.index(originalText.startIndex, offsetBy: max(0, currentOffset - config.overlap))..<originalText.endIndex) {
                let startOffset = originalText.distance(from: originalText.startIndex, to: range.lowerBound)
                let endOffset = originalText.distance(from: originalText.startIndex, to: range.upperBound)

                // Calculate overlaps
                let overlapPrev = index > 0 ? max(0, chunks[index - 1].endOffset - startOffset) : 0
                let overlapNext = index < total - 1 ? config.overlap : 0

                chunks.append(Chunk(
                    text: text,
                    startOffset: startOffset,
                    endOffset: endOffset,
                    index: index,
                    totalChunks: total,
                    overlapWithPrevious: overlapPrev,
                    overlapWithNext: overlapNext
                ))

                currentOffset = endOffset
            } else {
                // Fallback if exact match not found
                chunks.append(Chunk(
                    text: text,
                    startOffset: currentOffset,
                    endOffset: currentOffset + text.count,
                    index: index,
                    totalChunks: total,
                    overlapWithPrevious: 0,
                    overlapWithNext: 0
                ))
                currentOffset += text.count
            }
        }

        return chunks
    }
}

// MARK: - Convenience Extensions

extension StreamingProcessor {
    /// Create a processor optimized for short documents (fewer, larger chunks).
    public static func forShortDocuments(model: any EmbeddingModel) -> StreamingProcessor {
        var config = StreamingConfig()
        config.chunkingStrategy = .tokens(count: 400)
        config.overlap = 20
        config.minChunkSize = 100
        return StreamingProcessor(model: model, config: config)
    }

    /// Create a processor optimized for long documents (more, smaller chunks with overlap).
    public static func forLongDocuments(model: any EmbeddingModel) -> StreamingProcessor {
        var config = StreamingConfig()
        config.chunkingStrategy = .tokens(count: 200)
        config.overlap = 50
        config.minChunkSize = 50
        config.concurrency = 8
        return StreamingProcessor(model: model, config: config)
    }

    /// Create a processor for sentence-level granularity.
    public static func forSentences(model: any EmbeddingModel, sentencesPerChunk: Int = 3) -> StreamingProcessor {
        var config = StreamingConfig()
        config.chunkingStrategy = .sentences(count: sentencesPerChunk)
        config.overlap = 30
        config.minChunkSize = 20
        return StreamingProcessor(model: model, config: config)
    }
}

// EmbedKit - Core Protocols

import Foundation

// MARK: - EmbeddingModel Protocol

/// Fundamental abstraction for embedding models used by EmbedKit.
/// Conforming types must be actors to ensure thread safety.
public protocol EmbeddingModel: Actor {
    nonisolated var id: ModelID { get }
    nonisolated var dimensions: Int { get }
    nonisolated var device: ComputeDevice { get }

    /// Compute an embedding for a single string.
    func embed(_ text: String) async throws -> Embedding
    /// Compute embeddings for a batch of strings. Implementations may perform micro‑batching under the hood.
    /// - SeeAlso: `EmbeddingConfiguration.paddingStrategy`, `BatchOptions.bucketSize`, and `BatchOptions.maxBatchTokens`.
    func embedBatch(_ texts: [String], options: BatchOptions) async throws -> [Embedding]

    func warmup() async throws
    func release() async throws

    var metrics: ModelMetrics { get async }

    /// Reset any internal metrics collected so far (useful for benchmark warmup).
    func resetMetrics() async throws
}

// MARK: - Tokenizer Protocol

/// Tokenization interface used to convert text to model-ready inputs.
public protocol Tokenizer: Sendable {
    func encode(_ text: String, config: TokenizerConfig) async throws -> TokenizedText
    func decode(_ ids: [Int]) async throws -> String
    var vocabularySize: Int { get }
    var specialTokens: SpecialTokens { get }

    /// Encode multiple texts in parallel.
    ///
    /// Default implementation uses Swift concurrency for parallel processing.
    /// Override for custom batch optimization.
    ///
    /// - Parameters:
    ///   - texts: Array of texts to tokenize
    ///   - config: Tokenization configuration
    ///   - maxConcurrency: Maximum parallel tokenization tasks (default: ProcessInfo.activeProcessorCount)
    /// - Returns: Array of tokenized texts in the same order as input
    func encodeBatch(
        _ texts: [String],
        config: TokenizerConfig,
        maxConcurrency: Int?
    ) async throws -> [TokenizedText]
}

// MARK: - Tokenizer Default Batch Implementation

public extension Tokenizer {
    /// Default parallel batch tokenization implementation.
    ///
    /// Uses structured concurrency with controlled parallelism for efficient
    /// tokenization of large text batches.
    func encodeBatch(
        _ texts: [String],
        config: TokenizerConfig,
        maxConcurrency: Int? = nil
    ) async throws -> [TokenizedText] {
        guard !texts.isEmpty else { return [] }

        // Single text - no parallelism needed
        if texts.count == 1 {
            return [try await encode(texts[0], config: config)]
        }

        // Determine concurrency level
        let concurrency = maxConcurrency ?? min(ProcessInfo.processInfo.activeProcessorCount, texts.count)

        // For small batches, process directly without chunking overhead
        if texts.count <= concurrency {
            return try await withThrowingTaskGroup(of: (Int, TokenizedText).self) { group in
                for (index, text) in texts.enumerated() {
                    group.addTask {
                        let result = try await self.encode(text, config: config)
                        return (index, result)
                    }
                }

                var results = [TokenizedText?](repeating: nil, count: texts.count)
                for try await (index, tokenized) in group {
                    results[index] = tokenized
                }
                return results.compactMap { $0 }
            }
        }

        // For larger batches, chunk to control memory and task overhead
        let chunkSize = max(1, (texts.count + concurrency - 1) / concurrency)
        var allResults = [TokenizedText?](repeating: nil, count: texts.count)

        try await withThrowingTaskGroup(of: [(Int, TokenizedText)].self) { group in
            var startIndex = 0

            while startIndex < texts.count {
                let endIndex = min(startIndex + chunkSize, texts.count)
                let chunkStart = startIndex

                group.addTask {
                    var chunkResults: [(Int, TokenizedText)] = []
                    chunkResults.reserveCapacity(endIndex - chunkStart)

                    for i in chunkStart..<endIndex {
                        let result = try await self.encode(texts[i], config: config)
                        chunkResults.append((i, result))
                    }
                    return chunkResults
                }

                startIndex = endIndex
            }

            for try await chunkResults in group {
                for (index, tokenized) in chunkResults {
                    allResults[index] = tokenized
                }
            }
        }

        return allResults.compactMap { $0 }
    }
}

// MARK: - Model Backend Protocol

/// Abstraction of the underlying compute backend (CoreML/ONNX/etc.)
public protocol ModelBackend: Actor {
    associatedtype Input
    associatedtype Output

    func process(_ input: Input) async throws -> Output
    func processBatch(_ inputs: [Input]) async throws -> [Output]

    func load() async throws
    func unload() async throws

    var isLoaded: Bool { get }
    var memoryUsage: Int64 { get }
}

// MARK: - CoreML Processing Backend (type-erased interface for AppleEmbeddingModel)

/// Narrow, concrete interface for backends that accept CoreMLInput and produce CoreMLOutput.
/// This avoids exposing associated types at the call site and enables test doubles.
public protocol CoreMLProcessingBackend: Actor {
    func process(_ input: CoreMLInput) async throws -> CoreMLOutput
    func processBatch(_ inputs: [CoreMLInput]) async throws -> [CoreMLOutput]
    func load() async throws
    func unload() async throws
    var isLoaded: Bool { get }
    var memoryUsage: Int64 { get }
}

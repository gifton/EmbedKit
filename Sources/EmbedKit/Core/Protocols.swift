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

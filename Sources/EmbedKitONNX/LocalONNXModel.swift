// EmbedKitONNX - Local ONNX Model
// User-facing API for ONNX embedding models

import Foundation
import EmbedKit

// MARK: - Local ONNX Model

/// Embedding model that uses ONNX Runtime for inference.
///
/// Provides the same API as `LocalCoreMLModel` but for ONNX model files.
/// This allows users to use models from HuggingFace and other sources
/// without converting to CoreML format.
///
/// Example:
/// ```swift
/// import EmbedKit
/// import EmbedKitONNX
///
/// // Load an ONNX model
/// let model = LocalONNXModel(
///     modelURL: URL(fileURLWithPath: "all-MiniLM-L6-v2.onnx"),
///     tokenizer: wordPieceTokenizer,
///     dimensions: 384
/// )
///
/// // Use same API as CoreML models
/// let embedding = try await model.embed("Hello world")
/// let batch = try await model.embedBatch(["Hello", "World"], options: .init())
/// ```
public actor LocalONNXModel: EmbeddingModel {

    // MARK: - EmbeddingModel Properties

    public nonisolated let id: ModelID
    public nonisolated let dimensions: Int
    public nonisolated let device: ComputeDevice

    // MARK: - Internal State

    private let backend: ONNXBackend
    private let tokenizer: any Tokenizer
    private let configuration: EmbeddingConfiguration

    // Simple metrics tracking
    private var totalEmbeddings: Int = 0
    private var totalTokens: Int = 0
    private var totalTime: TimeInterval = 0
    private var latencies: [TimeInterval] = []

    // MARK: - Initialization

    /// Create a new ONNX embedding model.
    ///
    /// - Parameters:
    ///   - modelURL: Path to the .onnx model file
    ///   - tokenizer: Tokenizer for text processing
    ///   - configuration: Embedding configuration options
    ///   - backendConfig: ONNX backend configuration
    ///   - id: Optional model identifier (derived from filename if nil)
    ///   - dimensions: Output embedding dimensions
    public init(
        modelURL: URL,
        tokenizer: any Tokenizer,
        configuration: EmbeddingConfiguration = EmbeddingConfiguration(),
        backendConfig: ONNXBackendConfiguration = .default,
        id: ModelID? = nil,
        dimensions: Int = 384
    ) {
        self.backend = ONNXBackend(modelPath: modelURL, config: backendConfig)
        self.tokenizer = tokenizer
        self.configuration = configuration
        self.dimensions = dimensions

        // Derive model ID from filename if not provided
        if let providedID = id {
            self.id = providedID
        } else {
            let name = modelURL.deletingPathExtension().lastPathComponent
            self.id = ModelID(provider: "onnx", name: name, version: "1.0")
        }

        // ONNX backend uses CoreML provider if available, otherwise CPU
        self.device = backendConfig.useCoreMLProvider ? .auto : .cpu
    }

    // MARK: - EmbeddingModel Protocol

    /// Generate an embedding for a single text.
    public func embed(_ text: String) async throws -> Embedding {
        let startTime = ContinuousClock.now

        // Ensure model is loaded
        let loaded = await backend.isLoaded
        if !loaded {
            try await backend.load()
        }

        // Tokenize
        var config = TokenizerConfig()
        config.maxLength = configuration.maxTokens
        config.truncation = configuration.truncationStrategy
        config.padding = configuration.paddingStrategy
        config.addSpecialTokens = configuration.includeSpecialTokens

        let tokenized = try await tokenizer.encode(text, config: config)

        // Create input
        let input = ONNXInput(
            tokenIDs: tokenized.ids,
            attentionMask: tokenized.attentionMask
        )

        // Run inference
        let output = try await backend.process(input)

        // Pool the output
        let pooled = pool(output: output, attentionMask: tokenized.attentionMask)

        // Normalize if configured
        let normalized: [Float]
        if configuration.normalizeOutput {
            normalized = normalize(pooled)
        } else {
            normalized = pooled
        }

        let totalDuration = ContinuousClock.now - startTime
        let durationSeconds = totalDuration.asSeconds

        // Update metrics
        totalEmbeddings += 1
        totalTokens += tokenized.ids.count
        totalTime += durationSeconds
        latencies.append(durationSeconds)
        if latencies.count > 512 {
            latencies.removeFirst(latencies.count - 512)
        }

        // Create embedding
        let truncated = tokenized.ids.count >= configuration.maxTokens

        return Embedding(
            vector: normalized,
            metadata: EmbeddingMetadata(
                modelID: id,
                tokenCount: tokenized.ids.count,
                processingTime: durationSeconds,
                normalized: configuration.normalizeOutput,
                poolingStrategy: configuration.poolingStrategy,
                truncated: truncated
            )
        )
    }

    /// Generate embeddings for a batch of texts.
    public func embedBatch(_ texts: [String], options: BatchOptions) async throws -> [Embedding] {
        guard !texts.isEmpty else { return [] }

        // Ensure model is loaded
        let loaded = await backend.isLoaded
        if !loaded {
            try await backend.load()
        }

        var embeddings: [Embedding] = []
        embeddings.reserveCapacity(texts.count)

        // Process in batches based on options
        let batchSize = min(options.maxBatchSize, texts.count)
        var index = 0

        while index < texts.count {
            let endIndex = min(index + batchSize, texts.count)
            let batch = Array(texts[index..<endIndex])

            // Process batch (currently sequential, could be optimized)
            for text in batch {
                let embedding = try await embed(text)
                embeddings.append(embedding)
            }

            index = endIndex
        }

        return embeddings
    }

    /// Warm up the model by running a test inference.
    public func warmup() async throws {
        let loaded = await backend.isLoaded
        if !loaded {
            try await backend.load()
        }

        // Run a minimal inference to warm caches
        let testInput = ONNXInput(
            tokenIDs: [101, 102],  // [CLS] [SEP]
            attentionMask: [1, 1]
        )
        _ = try await backend.process(testInput)
    }

    /// Release model resources.
    public func release() async throws {
        try await backend.unload()
    }

    /// Current model metrics.
    public var metrics: ModelMetrics {
        let avg = totalEmbeddings > 0 ? totalTime / Double(totalEmbeddings) : 0
        let throughput = totalTime > 0 ? Double(totalTokens) / totalTime : 0

        return ModelMetrics(
            totalRequests: totalEmbeddings,
            totalTokensProcessed: totalTokens,
            averageLatency: avg,
            p50Latency: percentile(latencies, 50),
            p95Latency: percentile(latencies, 95),
            p99Latency: percentile(latencies, 99),
            throughput: throughput,
            cacheHitRate: 0,
            memoryUsage: 0,
            lastUsed: Date(),
            latencyHistogram: latencies,
            tokenHistogram: []
        )
    }

    /// Reset collected metrics.
    public func resetMetrics() async throws {
        totalEmbeddings = 0
        totalTokens = 0
        totalTime = 0
        latencies = []
    }

    /// Stage-level metrics snapshot.
    public var stageMetricsSnapshot: StageMetrics {
        StageMetrics(
            tokenizationAverage: 0,
            inferenceAverage: 0,
            poolingAverage: 0,
            samples: totalEmbeddings,
            averageBatchSize: 1.0
        )
    }

    // MARK: - Private Helpers

    private func percentile(_ values: [TimeInterval], _ p: Int) -> TimeInterval {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let rank = max(0, min(sorted.count - 1, (sorted.count - 1) * p / 100))
        return sorted[rank]
    }

    private func pool(output: ONNXOutput, attentionMask: [Int]) -> [Float] {
        let shape = output.shape
        let values = output.values

        // Determine dimensions from shape
        // Common shapes: [1, seq_len, hidden_dim] or [seq_len, hidden_dim]
        let seqLen: Int
        let hiddenDim: Int

        if shape.count == 3 {
            seqLen = shape[1]
            hiddenDim = shape[2]
        } else if shape.count == 2 {
            seqLen = shape[0]
            hiddenDim = shape[1]
        } else {
            // Unknown shape, return as-is if matches dimensions
            if values.count == dimensions {
                return values
            }
            // Otherwise return zeros
            return [Float](repeating: 0, count: dimensions)
        }

        // Apply pooling strategy
        switch configuration.poolingStrategy {
        case .mean:
            return meanPool(values: values, seqLen: seqLen, hiddenDim: hiddenDim, mask: attentionMask)
        case .max:
            return maxPool(values: values, seqLen: seqLen, hiddenDim: hiddenDim)
        case .cls:
            return clsPool(values: values, hiddenDim: hiddenDim)
        case .attention:
            // Attention pooling without weights falls back to mean pooling
            return meanPool(values: values, seqLen: seqLen, hiddenDim: hiddenDim, mask: attentionMask)
        }
    }

    private func meanPool(values: [Float], seqLen: Int, hiddenDim: Int, mask: [Int]) -> [Float] {
        var result = [Float](repeating: 0, count: hiddenDim)
        var count: Float = 0

        for i in 0..<min(seqLen, mask.count) {
            if mask[i] == 1 {
                for j in 0..<hiddenDim {
                    let idx = i * hiddenDim + j
                    if idx < values.count {
                        result[j] += values[idx]
                    }
                }
                count += 1
            }
        }

        if count > 0 {
            for j in 0..<hiddenDim {
                result[j] /= count
            }
        }

        return result
    }

    private func maxPool(values: [Float], seqLen: Int, hiddenDim: Int) -> [Float] {
        var result = [Float](repeating: -.greatestFiniteMagnitude, count: hiddenDim)

        for i in 0..<seqLen {
            for j in 0..<hiddenDim {
                let idx = i * hiddenDim + j
                if idx < values.count {
                    result[j] = max(result[j], values[idx])
                }
            }
        }

        // Replace -inf with 0 if no values
        for j in 0..<hiddenDim {
            if result[j] == -.greatestFiniteMagnitude {
                result[j] = 0
            }
        }

        return result
    }

    private func clsPool(values: [Float], hiddenDim: Int) -> [Float] {
        // Return first token's hidden state
        if values.count >= hiddenDim {
            return Array(values.prefix(hiddenDim))
        }
        return [Float](repeating: 0, count: hiddenDim)
    }

    private func normalize(_ vector: [Float]) -> [Float] {
        var sumSquared: Float = 0
        for v in vector {
            sumSquared += v * v
        }

        let magnitude = sqrt(sumSquared)
        guard magnitude > 1e-12 else {
            return vector
        }

        return vector.map { $0 / magnitude }
    }
}

// MARK: - Duration Extension

private extension Duration {
    var asSeconds: TimeInterval {
        let (seconds, attoseconds) = components
        return Double(seconds) + Double(attoseconds) / 1e18
    }
}

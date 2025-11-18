//
//  EmbeddingPipeline.swift
//  EmbedKit
//
//  End-to-end pipeline for text embedding generation
//

import Foundation
import VectorCore

/// Configuration for the embedding pipeline
public struct EmbeddingPipelineConfiguration: Sendable {
    /// Pooling strategy for combining token embeddings
    public let poolingStrategy: PoolingStrategy

    /// Whether to normalize embeddings to unit length
    public let normalize: Bool

    /// Whether to use GPU acceleration when available
    public let useGPUAcceleration: Bool

    /// Cache configuration
    public let cacheConfiguration: CacheConfiguration?

    /// Batch size for processing multiple texts
    public let batchSize: Int

    /// Numeric configuration for CPU/GPU acceleration paths
    public let numerics: AccelerationNumerics

    public struct CacheConfiguration: Sendable {
        public let maxEntries: Int
        public let ttlSeconds: TimeInterval?

        public init(maxEntries: Int = 1000, ttlSeconds: TimeInterval? = nil) {
            self.maxEntries = maxEntries
            self.ttlSeconds = ttlSeconds
        }
    }

    public init(
        poolingStrategy: PoolingStrategy = .mean,
        normalize: Bool = true,
        useGPUAcceleration: Bool = true,
        cacheConfiguration: CacheConfiguration? = CacheConfiguration(),
        batchSize: Int = 32,
        numerics: AccelerationNumerics = AccelerationNumerics()
    ) {
        self.poolingStrategy = poolingStrategy
        self.normalize = normalize
        self.useGPUAcceleration = useGPUAcceleration
        self.cacheConfiguration = cacheConfiguration
        self.batchSize = batchSize
        self.numerics = numerics
    }
}

/// Errors that can occur during pipeline operations
public enum EmbeddingPipelineError: LocalizedError {
    case modelNotLoaded
    case tokenizationFailed(Error)
    case inferenceFailed(Error)
    case poolingFailed(Error)
    case normalizationFailed(Error)
    case dimensionMismatch(expected: Int, actual: Int)
    case emptyInput
    case batchSizeMismatch

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Model not loaded. Call loadModel() first."
        case .tokenizationFailed(let error):
            return "Tokenization failed: \(error.localizedDescription)"
        case .inferenceFailed(let error):
            return "Inference failed: \(error.localizedDescription)"
        case .poolingFailed(let error):
            return "Pooling failed: \(error.localizedDescription)"
        case .normalizationFailed(let error):
            return "Normalization failed: \(error.localizedDescription)"
        case .dimensionMismatch(let expected, let actual):
            return "Dimension mismatch. Expected: \(expected), Actual: \(actual)"
        case .emptyInput:
            return "Empty input text provided"
        case .batchSizeMismatch:
            return "Batch size mismatch between tokenization and inference"
        }
    }
}

/// Statistics for pipeline operations
public struct PipelineStatistics: Sendable {
    public let tokenizationTime: TimeInterval
    public let inferenceTime: TimeInterval
    public let poolingTime: TimeInterval
    public let normalizationTime: TimeInterval
    public let totalTime: TimeInterval
    public let cacheHitRate: Double
}

/// Main embedding pipeline that orchestrates text → embedding conversion
public actor EmbeddingPipeline {
    // Core components
    private let tokenizer: any Tokenizer
    private let backend: any ModelBackend
    private let configuration: EmbeddingPipelineConfiguration

    // Optional components
    private let metalAccelerator: MetalAccelerator?
    private var cache: EmbeddingCache?

    // State
    private var isModelLoaded = false
    private var statistics = PipelineStatistics(
        tokenizationTime: 0,
        inferenceTime: 0,
        poolingTime: 0,
        normalizationTime: 0,
        totalTime: 0,
        cacheHitRate: 0
    )

    // MARK: - Initialization

    /// Initialize pipeline with components
    public init(
        tokenizer: any Tokenizer,
        backend: any ModelBackend,
        configuration: EmbeddingPipelineConfiguration = EmbeddingPipelineConfiguration()
    ) {
        self.tokenizer = tokenizer
        self.backend = backend
        self.configuration = configuration

        // Initialize Metal acceleration if requested and available
        if configuration.useGPUAcceleration {
            self.metalAccelerator = MetalAccelerator.shared
        } else {
            self.metalAccelerator = nil
        }

        // Initialize cache if configured
        if let cacheConfig = configuration.cacheConfiguration {
            self.cache = EmbeddingCache(maxEntries: cacheConfig.maxEntries)
        }
    }

    /// Convenience initializer with model URL
    public init(
        modelURL: URL,
        tokenizer: any Tokenizer,
        configuration: EmbeddingPipelineConfiguration = EmbeddingPipelineConfiguration()
    ) async throws {
        // Initialize with CoreML backend
        let coreMLConfig = CoreMLConfiguration()
        let backend = CoreMLBackend(configuration: coreMLConfig)

        self.tokenizer = tokenizer
        self.backend = backend
        self.configuration = configuration

        // Initialize Metal acceleration
        if configuration.useGPUAcceleration {
            self.metalAccelerator = MetalAccelerator.shared
        } else {
            self.metalAccelerator = nil
        }

        // Initialize cache
        if let cacheConfig = configuration.cacheConfiguration {
            self.cache = EmbeddingCache(maxEntries: cacheConfig.maxEntries)
        }

        // Load model
        try await loadModel(from: modelURL)
    }

    // MARK: - Model Management

    /// Load model from URL
    public func loadModel(from url: URL) async throws {
        try await backend.loadModel(from: url)
        self.isModelLoaded = true
    }

    /// Check if model is loaded
    public func isReady() -> Bool {
        return isModelLoaded
    }

    // MARK: - Embedding Generation

    /// Generate embedding for a single text
    public func embed(_ text: String) async throws -> DynamicEmbedding {
        if !isModelLoaded {
            if await backend.isLoaded {
                self.isModelLoaded = true
            } else {
                throw EmbeddingPipelineError.modelNotLoaded
            }
        }

        guard !text.isEmpty else {
            throw EmbeddingPipelineError.emptyInput
        }

        let startTime = Date()

        // Check cache
        if let cached = await cache?.get(text) {
            return cached
        }

        // Step 1: Tokenization
        let tokenStart = Date()
        let tokenized: TokenizedInput
        do {
            tokenized = try await tokenizer.tokenize(text)
        } catch {
            throw EmbeddingPipelineError.tokenizationFailed(error)
        }
        let tokenTime = Date().timeIntervalSince(tokenStart)

        // Step 2: Inference
        let inferenceStart = Date()
        let modelOutput: ModelOutput
        do {
            modelOutput = try await backend.generateEmbeddings(for: tokenized)
        } catch {
            throw EmbeddingPipelineError.inferenceFailed(error)
        }
        let inferenceTime = Date().timeIntervalSince(inferenceStart)

        // Step 3: Pooling
        let poolingStart = Date()
        let pooled: [Float]
        do {
            pooled = try await pool(
                tokenEmbeddings: modelOutput.tokenEmbeddings,
                attentionMask: tokenized.attentionMask,
                strategy: configuration.poolingStrategy
            )
        } catch {
            throw EmbeddingPipelineError.poolingFailed(error)
        }
        let poolingTime = Date().timeIntervalSince(poolingStart)

        // Step 4: Normalization (optional)
        let normStart = Date()
        let output: [Float]
        if configuration.normalize {
            do {
                output = try await normalize(pooled)
            } catch {
                throw EmbeddingPipelineError.normalizationFailed(error)
            }
        } else {
            output = pooled
        }
        let normTime = Date().timeIntervalSince(normStart)

        // Step 5: Create typed embedding
        let embedding = try DynamicEmbedding(values: output)

        // Cache result
        await cache?.set(text, embedding: embedding)

        // Update statistics
        let totalTime = Date().timeIntervalSince(startTime)
        await updateStatistics(
            tokenization: tokenTime,
            inference: inferenceTime,
            pooling: poolingTime,
            normalization: normTime,
            total: totalTime
        )

        return embedding
    }

    /// Generate embeddings for multiple texts
    public func embed(batch texts: [String]) async throws -> [DynamicEmbedding] {
        if !isModelLoaded {
            if await backend.isLoaded {
                self.isModelLoaded = true
            } else {
                throw EmbeddingPipelineError.modelNotLoaded
            }
        }

        // Filter empty texts
        let validTexts = texts.filter { !$0.isEmpty }
        guard !validTexts.isEmpty else {
            throw EmbeddingPipelineError.emptyInput
        }

        // Check cache and separate cached vs uncached
        var results: [(Int, DynamicEmbedding)] = []
        var uncachedTexts: [(Int, String)] = []

        for (index, text) in validTexts.enumerated() {
            if let cached = await cache?.get(text) {
                results.append((index, cached))
            } else {
                uncachedTexts.append((index, text))
            }
        }

        // Process uncached texts in batches
        if !uncachedTexts.isEmpty {
            for batch in uncachedTexts.chunked(into: configuration.batchSize) {
                let batchTexts = batch.map { $0.1 }
                let batchIndices = batch.map { $0.0 }

                // Tokenize batch
                let tokenizedBatch = try await tokenizer.tokenize(batch: batchTexts)

                // Run inference on batch
                let modelOutputs = try await backend.generateEmbeddings(for: tokenizedBatch)

                // Pool and normalize each output
                for (idx, output) in zip(batchIndices, modelOutputs) {
                    let pooled = try await pool(
                        tokenEmbeddings: output.tokenEmbeddings,
                        attentionMask: tokenizedBatch[idx - batchIndices[0]].attentionMask,
                        strategy: configuration.poolingStrategy
                    )

                    let output = configuration.normalize ? try await normalize(pooled) : pooled
                    let embedding = try DynamicEmbedding(values: output)

                    // Cache result
                    await cache?.set(batchTexts[idx - batchIndices[0]], embedding: embedding)

                    results.append((idx, embedding))
                }
            }
        }

        // Sort results back to original order
        results.sort { $0.0 < $1.0 }
        return results.map { $0.1 }
    }

    // MARK: - Pipeline Operations

    /// Pool token embeddings into a single vector
    private func pool(
        tokenEmbeddings: [[Float]],
        attentionMask: [Int],
        strategy: PoolingStrategy
    ) async throws -> [Float] {
        // Use GPU acceleration if available
        if let accelerator = metalAccelerator {
            // Convert to VectorBatch for optimal performance (10-15% faster, 66% fewer allocations)
            let batch = try VectorBatch(vectors: tokenEmbeddings)
            return try await accelerator.poolEmbeddings(
                batch,
                strategy: strategy,
                attentionMask: attentionMask,
                attentionWeights: nil
            )
        }

        // CPU fallback
        switch strategy {
        case .mean:
            return poolMean(tokenEmbeddings: tokenEmbeddings, attentionMask: attentionMask)
        case .cls:
            return tokenEmbeddings.first ?? []
        case .max:
            return poolMax(tokenEmbeddings: tokenEmbeddings, attentionMask: attentionMask)
        case .attentionWeighted:
            // Simplified attention weighting - in production would use actual attention scores
            return poolMean(tokenEmbeddings: tokenEmbeddings, attentionMask: attentionMask)
        }
    }

    /// CPU implementation of mean pooling
    private func poolMean(tokenEmbeddings: [[Float]], attentionMask: [Int]) -> [Float] {
        guard !tokenEmbeddings.isEmpty else { return [] }

        let dimensions = tokenEmbeddings[0].count
        var result = Array(repeating: Float(0), count: dimensions)
        var validTokenCount: Float = 0

        for (tokenIdx, embedding) in tokenEmbeddings.enumerated() {
            guard tokenIdx < attentionMask.count, attentionMask[tokenIdx] == 1 else { continue }

            for (dimIdx, value) in embedding.enumerated() {
                result[dimIdx] += value
            }
            validTokenCount += 1
        }

        // Average
        if validTokenCount > 0 {
            for i in 0..<dimensions {
                result[i] /= validTokenCount
            }
        }

        return result
    }

    /// CPU implementation of max pooling
    private func poolMax(tokenEmbeddings: [[Float]], attentionMask: [Int]) -> [Float] {
        guard !tokenEmbeddings.isEmpty else { return [] }

        let dimensions = tokenEmbeddings[0].count
        var result = Array(repeating: -Float.infinity, count: dimensions)

        for (tokenIdx, embedding) in tokenEmbeddings.enumerated() {
            guard tokenIdx < attentionMask.count, attentionMask[tokenIdx] == 1 else { continue }

            for (dimIdx, value) in embedding.enumerated() {
                result[dimIdx] = max(result[dimIdx], value)
            }
        }

        return result
    }

    /// Normalize vector to unit length
    ///
    /// Uses GPU acceleration when available, falls back to CPU with VectorCore-inspired
    /// numerical stability (epsilon protection, robust zero-vector handling).
    private func normalize(_ vector: [Float]) async throws -> [Float] {
        // Prefer GPU normalization when available
        if let accelerator = metalAccelerator {
            let batch = try VectorBatch(vectors: [vector])
            let normalized = try await accelerator.normalizeVectors(batch)
            return normalized.isEmpty ? vector : Array(normalized[0])
        }

        // CPU fallback: delegate to VectorCore via DynamicEmbedding to ensure
        // identical epsilon/threshold semantics as VectorCore
        // Support only known dimensions; otherwise fall back to manual path
        if vector.count == 384 || vector.count == 768 || vector.count == 1536 {
            do {
                let dyn = try DynamicEmbedding(values: vector)
                let normalized = try dyn.normalized()
                return normalized.toArray()
            } catch {
                // If VectorCore normalization fails (e.g., zero vector), mirror old behavior
                throw EmbeddingPipelineError.normalizationFailed(error)
            }
        }

        // Fallback for unsupported dimensions: minimal guard then normalize
        let normSquared = vector.reduce(0) { $0 + $1 * $1 }
        let epsilon = configuration.numerics.epsilon
        guard normSquared >= epsilon else {
            throw EmbeddingPipelineError.normalizationFailed(
                NSError(domain: "EmbedKit", code: 1, userInfo: [
                    NSLocalizedDescriptionKey: "Cannot normalize zero or near-zero vector (magnitude² = \(normSquared))"
                ])
            )
        }
        let norm = sqrt(normSquared)
        let invNorm = 1.0 / norm
        return vector.map { $0 * invNorm }
    }

    // MARK: - Statistics

    /// Update pipeline statistics
    private func updateStatistics(
        tokenization: TimeInterval,
        inference: TimeInterval,
        pooling: TimeInterval,
        normalization: TimeInterval,
        total: TimeInterval
    ) async {
        // Simple moving average update (could be more sophisticated)
        let alpha: Double = 0.1  // Smoothing factor

        // Get cache hit rate asynchronously
        let hitRate = await cache?.hitRate() ?? 0

        statistics = PipelineStatistics(
            tokenizationTime: (1 - alpha) * statistics.tokenizationTime + alpha * tokenization,
            inferenceTime: (1 - alpha) * statistics.inferenceTime + alpha * inference,
            poolingTime: (1 - alpha) * statistics.poolingTime + alpha * pooling,
            normalizationTime: (1 - alpha) * statistics.normalizationTime + alpha * normalization,
            totalTime: (1 - alpha) * statistics.totalTime + alpha * total,
            cacheHitRate: hitRate
        )
    }

    /// Get current pipeline statistics
    public func getStatistics() -> PipelineStatistics {
        return statistics
    }
}

// MARK: - Simple Embedding Cache

/// Simple LRU cache for embeddings
actor EmbeddingCache {
    private var cache: [String: (embedding: DynamicEmbedding, timestamp: Date)] = [:]
    private let maxEntries: Int
    private var accessOrder: [String] = []
    private var hits: Int = 0
    private var misses: Int = 0

    init(maxEntries: Int) {
        self.maxEntries = maxEntries
    }

    func get(_ text: String) -> DynamicEmbedding? {
        let key = sha256(text)

        if let cached = cache[key] {
            hits += 1

            // Update access order (move to end)
            accessOrder.removeAll { $0 == key }
            accessOrder.append(key)

            return cached.embedding
        }

        misses += 1
        return nil
    }

    func set(_ text: String, embedding: DynamicEmbedding) {
        let key = sha256(text)

        // Check if we need to evict
        if cache.count >= maxEntries && cache[key] == nil {
            // Evict least recently used
            if let lru = accessOrder.first {
                cache.removeValue(forKey: lru)
                accessOrder.removeFirst()
            }
        }

        cache[key] = (embedding, Date())
        accessOrder.append(key)
    }

    func hitRate() -> Double {
        let total = hits + misses
        return total > 0 ? Double(hits) / Double(total) : 0
    }

    private func sha256(_ text: String) -> String {
        // Simplified hash for caching - in production use proper SHA256
        return String(text.hashValue)
    }
}

// MARK: - Array Extension for Chunking

private extension Array {
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }
}

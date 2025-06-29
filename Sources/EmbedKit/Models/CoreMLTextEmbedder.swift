import Foundation
import CoreML
import Accelerate
import OSLog

/// Core ML-based text embedder implementation
///
/// Production-ready implementation using Apple's CoreML framework for on-device inference.
/// Optimized for Apple Silicon with automatic GPU/Neural Engine acceleration.
///
/// Architecture decisions:
/// - Actor-based for thread safety with expensive model resources
/// - Integrated caching to avoid redundant computations
/// - Metal acceleration for vector operations when available
/// - Memory-aware caching that responds to system pressure
///
/// Performance characteristics:
/// - First inference: ~50-200ms (model loading)
/// - Subsequent inferences: ~5-20ms (cached model)
/// - Memory usage: 100-500MB depending on model size
public actor CoreMLTextEmbedder: TextEmbedder {
    private let logger = Logger(subsystem: "EmbedKit", category: "CoreMLTextEmbedder")
    
    public let configuration: Configuration
    public let modelIdentifier: ModelIdentifier
    
    private let backend: CoreMLBackend
    private var tokenizer: any Tokenizer
    nonisolated(unsafe) private var _dimensions: Int?
    private let metalAccelerator: MetalAccelerator?
    private let cache: EmbeddingCache?
    private let memoryAwareCache: MemoryAwareCache?
    
    nonisolated public var dimensions: Int {
        // Return cached dimensions or 0 if unknown
        // This is updated after model loading
        _dimensions ?? 0
    }
    
    public var isReady: Bool {
        get async {
            await backend.isLoaded
        }
    }
    
    /// Initialize a CoreML text embedder
    ///
    /// - Parameters:
    ///   - modelIdentifier: Type-safe model identifier
    ///   - configuration: Unified configuration settings
    ///   - tokenizer: Custom tokenizer or nil for default
    ///   - enableCaching: Whether to cache embeddings (recommended for production)
    ///
    /// Design notes:
    /// - Lazy model loading prevents memory waste if embedder is never used
    /// - Optional Metal acceleration automatically detected and enabled
    /// - Cache is memory-aware and will auto-evict under pressure
    public init(
        modelIdentifier: ModelIdentifier,
        configuration: Configuration,
        tokenizer: (any Tokenizer)? = nil,
        enableCaching: Bool = true
    ) {
        self.modelIdentifier = modelIdentifier
        self.configuration = configuration
        self.backend = CoreMLBackend(identifier: modelIdentifier.rawValue)
        
        // Use provided tokenizer or create simple one initially with placeholder vocab size
        // This should be updated after model loading when vocab size is known
        self.tokenizer = tokenizer ?? SimpleTokenizer(
            maxSequenceLength: configuration.model.maxSequenceLength,
            vocabularySize: 30522 // Default BERT vocab size, will be updated if model provides different value
        )
        
        // Setup Metal acceleration if requested and available
        if configuration.performance.useMetalAcceleration {
            self.metalAccelerator = MetalAccelerator.shared
        } else {
            self.metalAccelerator = nil
        }
        
        // Setup caching if enabled
        if enableCaching && configuration.cache.maxCacheSize > 0 {
            self.cache = EmbeddingCache()
            self.memoryAwareCache = MemoryAwareCache(embeddingCache: cache!)
        } else {
            self.cache = nil
            self.memoryAwareCache = nil
        }
    }
    
    public func loadModel() async throws {
        let context = ErrorContext.modelLoading(
            modelIdentifier,
            location: SourceLocation()
        )
        
        do {
            // Check for custom model URL first
            if let customURL = configuration.model.loadingOptions.customModelURL {
                try await backend.loadModel(from: customURL)
            } else {
                // Try bundled model
                guard let modelURL = Bundle.main.url(forResource: modelIdentifier.rawValue, withExtension: "mlmodelc") else {
                    // Try in Documents directory
                    let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                    let modelPath = documentsPath.appendingPathComponent("\(modelIdentifier.rawValue).mlmodelc")
                    
                    if FileManager.default.fileExists(atPath: modelPath.path) {
                        try await backend.loadModel(from: modelPath)
                    } else {
                        throw ContextualEmbeddingError.resourceUnavailable(
                            context: context,
                            resource: .model
                        )
                    }
                    return
                }
                
                try await backend.loadModel(from: modelURL)
            }
        } catch let contextualError as ContextualError {
            throw contextualError
        } catch {
            throw ContextualEmbeddingError.modelNotLoaded(
                context: context,
                underlyingError: error
            )
        }
        
        // Update dimensions from model
        if let outputDims = await backend.outputDimensions() {
            self._dimensions = outputDims
        } else if let metadata = await backend.metadata, metadata.embeddingDimensions > 0 {
            self._dimensions = metadata.embeddingDimensions
        }
        
        // Log if dimensions are still unknown
        if self._dimensions == nil {
            logger.warning("Could not determine embedding dimensions from loaded model")
        }
        
        // Update tokenizer configuration if we have metadata
        if let metadata = await backend.metadata {
            // If using SimpleTokenizer and metadata has better info, update it
            if tokenizer is SimpleTokenizer && metadata.vocabularySize > 0 {
                let updatedTokenizer = SimpleTokenizer(
                    maxSequenceLength: metadata.maxSequenceLength > 0 ? metadata.maxSequenceLength : configuration.model.maxSequenceLength,
                    vocabularySize: metadata.vocabularySize
                )
                self.tokenizer = updatedTokenizer
                logger.info("Updated SimpleTokenizer with metadata: vocab=\(metadata.vocabularySize), maxLength=\(metadata.maxSequenceLength)")
            }
        }
    }
    
    public func unloadModel() async throws {
        try await backend.unloadModel()
        self._dimensions = nil
    }
    
    /// Update the tokenizer (useful for loading specialized tokenizers after init)
    public func updateTokenizer(_ newTokenizer: any Tokenizer) async {
        self.tokenizer = newTokenizer
        logger.info("Updated tokenizer to \(type(of: newTokenizer))")
    }
    
    public func embed(_ text: String) async throws -> EmbeddingVector {
        // Check cache first
        if let cached = await cache?.get(text: text, modelIdentifier: modelIdentifier) {
            logger.trace("Using cached embedding")
            return cached
        }
        
        let context = ErrorContext.embedding(
            modelIdentifier: modelIdentifier,
            inputSize: text.count,
            location: SourceLocation()
        )
        
        // Ensure model is loaded
        guard await backend.isLoaded else {
            throw ContextualEmbeddingError.modelNotLoaded(
                context: context
            )
        }
        
        // Tokenize text
        let tokenized = try await tokenizer.tokenize(text)
        
        // Generate embeddings
        let output = try await backend.generateEmbeddings(for: tokenized)
        
        // Pool token embeddings into a single vector
        let pooled: [Float]
        if let accelerator = metalAccelerator {
            // Use Metal acceleration if available
            pooled = try await accelerator.poolEmbeddings(
                output.tokenEmbeddings,
                strategy: configuration.model.poolingStrategy,
                attentionMask: tokenized.attentionMask,
                attentionWeights: output.attentionWeights?.first
            )
        } else {
            pooled = try pool(
                tokenEmbeddings: output.tokenEmbeddings, 
                strategy: configuration.model.poolingStrategy,
                attentionWeights: output.attentionWeights?.first
            )
        }
        
        // Normalize if requested
        let final: [Float]
        if configuration.model.normalizeEmbeddings {
            if let accelerator = metalAccelerator {
                let normalizedBatch = try await accelerator.normalizeVectors([pooled])
                final = normalizedBatch[0]
            } else {
                final = normalize(pooled)
            }
        } else {
            final = pooled
        }
        
        let embedding = EmbeddingVector(final)
        
        // Cache the result
        await cache?.set(text: text, modelIdentifier: modelIdentifier, embedding: embedding)
        
        return embedding
    }
    
    public func embed(batch texts: [String]) async throws -> [EmbeddingVector] {
        let context = ErrorContext.batchEmbedding(
            modelIdentifier: modelIdentifier,
            batchSize: texts.count,
            location: SourceLocation()
        )
        
        // Ensure model is loaded
        guard await backend.isLoaded else {
            throw ContextualEmbeddingError.modelNotLoaded(
                context: context
            )
        }
        
        var results: [EmbeddingVector] = []
        results.reserveCapacity(texts.count)
        var textsToProcess: [(index: Int, text: String)] = []
        
        // Check cache for each text
        for (index, text) in texts.enumerated() {
            if let cached = await cache?.get(text: text, modelIdentifier: modelIdentifier) {
                results.append(cached)
            } else {
                textsToProcess.append((index, text))
            }
        }
        
        // Process uncached texts in batches
        if !textsToProcess.isEmpty {
            let processedEmbeddings = try await processBatchWithoutCache(textsToProcess.map { $0.text })
            
            // Cache and collect results
            for (embedding, (_, text)) in zip(processedEmbeddings, textsToProcess) {
                await cache?.set(text: text, modelIdentifier: modelIdentifier, embedding: embedding)
                results.append(embedding)
            }
        }
        
        return results
    }
    
    private func processBatchWithoutCache(_ texts: [String]) async throws -> [EmbeddingVector] {
        var results: [EmbeddingVector] = []
        results.reserveCapacity(texts.count)
        
        // Process in batches according to configuration
        for i in stride(from: 0, to: texts.count, by: configuration.resources.batchSize) {
            let batchEnd = min(i + configuration.resources.batchSize, texts.count)
            let batchTexts = Array(texts[i..<batchEnd])
            
            // Tokenize batch
            let tokenizedBatch = try await tokenizer.tokenize(batch: batchTexts)
            
            // Generate embeddings for batch
            let outputs = try await backend.generateEmbeddings(for: tokenizedBatch)
            
            // Process each output
            if let accelerator = metalAccelerator, configuration.performance.useMetalAcceleration {
                // Use Metal for batch processing
                var pooledBatch: [[Float]] = []
                
                for (output, tokenized) in zip(outputs, tokenizedBatch) {
                    let pooled = try await accelerator.poolEmbeddings(
                        output.tokenEmbeddings,
                        strategy: configuration.model.poolingStrategy,
                        attentionMask: tokenized.attentionMask,
                        attentionWeights: output.attentionWeights?.first // Use first batch if available
                    )
                    pooledBatch.append(pooled)
                }
                
                // Batch normalize if needed
                let finalBatch: [[Float]]
                if configuration.model.normalizeEmbeddings {
                    finalBatch = try await accelerator.normalizeVectors(pooledBatch)
                } else {
                    finalBatch = pooledBatch
                }
                
                for final in finalBatch {
                    results.append(EmbeddingVector(final))
                }
            } else {
                // CPU processing
                for output in outputs {
                    let pooled = try pool(
                        tokenEmbeddings: output.tokenEmbeddings, 
                        strategy: configuration.model.poolingStrategy,
                        attentionWeights: output.attentionWeights?.first
                    )
                    let final = configuration.model.normalizeEmbeddings ? normalize(pooled) : pooled
                    results.append(EmbeddingVector(final))
                }
            }
        }
        
        return results
    }
    
    // MARK: - Private Helpers
    
    /// Pool token embeddings into a single vector representation
    ///
    /// Implementation leverages Accelerate framework for SIMD operations where possible.
    /// This is a performance-critical function called for every embedding.
    ///
    /// Optimization opportunities:
    /// - Consider pre-allocating result buffer for repeated calls
    /// - Investigate vDSP batch operations for multiple sequences
    /// - Profile Metal kernel vs Accelerate for specific pool strategies
    private func pool(tokenEmbeddings: [[Float]], strategy: PoolingStrategy, attentionWeights: [Float]? = nil) throws -> [Float] {
        guard !tokenEmbeddings.isEmpty else {
            throw ContextualEmbeddingError.inferenceFailed(
                context: ErrorContext(
                    operation: .inference,
                    modelIdentifier: modelIdentifier,
                    metadata: ErrorMetadata()
                        .with(key: "reason", value: "No token embeddings to pool"),
                    sourceLocation: SourceLocation()
                )
            )
        }
        
        let embeddingSize = tokenEmbeddings[0].count
        
        switch strategy {
        case .mean:
            // Average all token embeddings
            var result = [Float](repeating: 0, count: embeddingSize)
            var count: Float = 0
            
            for embedding in tokenEmbeddings {
                guard embedding.count == embeddingSize else {
                    throw ContextualEmbeddingError.dimensionMismatch(
                        context: ErrorContext(
                            operation: .inference,
                            modelIdentifier: modelIdentifier,
                            metadata: ErrorMetadata()
                                .with(key: "poolingStrategy", value: "mean"),
                            sourceLocation: SourceLocation()
                        ),
                        expected: embeddingSize,
                        actual: embedding.count
                    )
                }
                
                vDSP_vadd(result, 1, embedding, 1, &result, 1, vDSP_Length(embeddingSize))
                count += 1
            }
            
            vDSP_vsdiv(result, 1, &count, &result, 1, vDSP_Length(embeddingSize))
            return result
            
        case .cls:
            // Use only the first (CLS) token embedding
            return tokenEmbeddings[0]
            
        case .max:
            // Max pooling across all tokens
            var result = tokenEmbeddings[0]
            
            for i in 1..<tokenEmbeddings.count {
                let embedding = tokenEmbeddings[i]
                guard embedding.count == embeddingSize else {
                    throw ContextualEmbeddingError.dimensionMismatch(
                        context: ErrorContext(
                            operation: .inference,
                            modelIdentifier: modelIdentifier,
                            metadata: ErrorMetadata()
                                .with(key: "poolingStrategy", value: "mean"),
                            sourceLocation: SourceLocation()
                        ),
                        expected: embeddingSize,
                        actual: embedding.count
                    )
                }
                
                vDSP_vmax(result, 1, embedding, 1, &result, 1, vDSP_Length(embeddingSize))
            }
            
            return result
            
        case .attentionWeighted:
            // Use provided attention weights or fall back to uniform weights (mean pooling)
            let weights = attentionWeights ?? [Float](repeating: 1.0 / Float(tokenEmbeddings.count), count: tokenEmbeddings.count)
            
            guard weights.count == tokenEmbeddings.count else {
                throw ContextualEmbeddingError.inferenceFailed(
                    context: ErrorContext(
                        operation: .inference,
                        modelIdentifier: modelIdentifier,
                        metadata: ErrorMetadata()
                            .with(key: "reason", value: "Attention weights count mismatch"),
                        sourceLocation: SourceLocation()
                    )
                )
            }
            
            var result = [Float](repeating: 0, count: embeddingSize)
            var weightSum: Float = 0
            
            // Apply attention weights to each token embedding
            for (embedding, weight) in zip(tokenEmbeddings, weights) {
                guard embedding.count == embeddingSize else {
                    throw ContextualEmbeddingError.dimensionMismatch(
                        context: ErrorContext(
                            operation: .inference,
                            modelIdentifier: modelIdentifier,
                            metadata: ErrorMetadata()
                                .with(key: "poolingStrategy", value: "attentionWeighted"),
                            sourceLocation: SourceLocation()
                        ),
                        expected: embeddingSize,
                        actual: embedding.count
                    )
                }
                
                // Scale embedding by weight and add to result
                var scaledEmbedding = embedding
                vDSP_vsmul(embedding, 1, [weight], &scaledEmbedding, 1, vDSP_Length(embeddingSize))
                vDSP_vadd(result, 1, scaledEmbedding, 1, &result, 1, vDSP_Length(embeddingSize))
                weightSum += weight
            }
            
            // Normalize by weight sum
            if weightSum > 0 {
                vDSP_vsdiv(result, 1, [weightSum], &result, 1, vDSP_Length(embeddingSize))
            }
            
            return result
        }
    }
    
    /// L2 normalize a vector to unit length
    ///
    /// Normalization ensures embeddings lie on the unit hypersphere,
    /// making cosine similarity equivalent to dot product.
    ///
    /// Performance note: Accelerate's vDSP functions leverage SIMD
    /// instructions for ~4-8x speedup over naive implementation
    private func normalize(_ vector: [Float]) -> [Float] {
        var result = vector
        var norm: Float = 0
        
        // Calculate L2 norm
        vDSP_svesq(vector, 1, &norm, vDSP_Length(vector.count))
        norm = sqrt(norm)
        
        // Avoid division by zero
        guard norm > 0 else { return result }
        
        // Normalize
        vDSP_vsdiv(vector, 1, &norm, &result, 1, vDSP_Length(vector.count))
        
        return result
    }
    
    /// Get model metadata
    public func getMetadata() async -> ModelMetadata? {
        await backend.metadata
    }
}

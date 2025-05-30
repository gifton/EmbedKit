import Foundation
import CoreML
import Accelerate
import OSLog

/// Core ML-based text embedder implementation
public actor CoreMLTextEmbedder: TextEmbedder {
    private let logger = Logger(subsystem: "EmbedKit", category: "CoreMLTextEmbedder")
    
    public let configuration: EmbeddingConfiguration
    public let modelIdentifier: String
    
    private let backend: CoreMLBackend
    private let tokenizer: any Tokenizer
    private var _dimensions: Int?
    private let metalAccelerator: MetalAccelerator?
    private let cache: EmbeddingCache?
    private let memoryAwareCache: MemoryAwareCache?
    
    public var dimensions: Int {
        _dimensions ?? 768 // Default BERT dimensions
    }
    
    public var isReady: Bool {
        get async {
            await backend.isLoaded
        }
    }
    
    public init(
        modelIdentifier: String,
        configuration: EmbeddingConfiguration = EmbeddingConfiguration(),
        tokenizer: (any Tokenizer)? = nil,
        enableCaching: Bool = true
    ) {
        self.modelIdentifier = modelIdentifier
        self.configuration = configuration
        self.backend = CoreMLBackend(identifier: modelIdentifier)
        self.tokenizer = tokenizer ?? SimpleTokenizer(
            maxSequenceLength: configuration.maxSequenceLength
        )
        
        // Setup Metal acceleration if requested and available
        if configuration.useGPUAcceleration {
            self.metalAccelerator = MetalAccelerator.shared
        } else {
            self.metalAccelerator = nil
        }
        
        // Setup caching if enabled
        if enableCaching {
            self.cache = EmbeddingCache()
            self.memoryAwareCache = MemoryAwareCache(embeddingCache: cache!)
        } else {
            self.cache = nil
            self.memoryAwareCache = nil
        }
    }
    
    public func loadModel() async throws {
        // For now, assume the model is bundled with the app
        guard let modelURL = Bundle.main.url(forResource: modelIdentifier, withExtension: "mlmodelc") else {
            // Try in Documents directory
            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            let modelPath = documentsPath.appendingPathComponent("\(modelIdentifier).mlmodelc")
            
            if FileManager.default.fileExists(atPath: modelPath.path) {
                try await backend.loadModel(from: modelPath)
            } else {
                throw EmbeddingError.resourceUnavailable("Model not found: \(modelIdentifier)")
            }
            return
        }
        
        try await backend.loadModel(from: modelURL)
        
        // Update dimensions from model
        if let outputDims = await backend.outputDimensions() {
            self._dimensions = outputDims
        }
    }
    
    public func unloadModel() async throws {
        try await backend.unloadModel()
        self._dimensions = nil
    }
    
    public func embed(_ text: String) async throws -> EmbeddingVector {
        // Check cache first
        if let cached = await cache?.get(text: text, modelIdentifier: modelIdentifier) {
            logger.trace("Using cached embedding")
            return cached
        }
        
        // Ensure model is loaded
        guard await backend.isLoaded else {
            throw EmbeddingError.modelNotLoaded
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
                strategy: configuration.poolingStrategy,
                attentionMask: tokenized.attentionMask
            )
        } else {
            pooled = try pool(tokenEmbeddings: output.tokenEmbeddings, strategy: configuration.poolingStrategy)
        }
        
        // Normalize if requested
        let final: [Float]
        if configuration.normalizeEmbeddings {
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
        // Ensure model is loaded
        guard await backend.isLoaded else {
            throw EmbeddingError.modelNotLoaded
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
        for i in stride(from: 0, to: texts.count, by: configuration.batchSize) {
            let batchEnd = min(i + configuration.batchSize, texts.count)
            let batchTexts = Array(texts[i..<batchEnd])
            
            // Tokenize batch
            let tokenizedBatch = try await tokenizer.tokenize(batch: batchTexts)
            
            // Generate embeddings for batch
            let outputs = try await backend.generateEmbeddings(for: tokenizedBatch)
            
            // Process each output
            if let accelerator = metalAccelerator, configuration.useGPUAcceleration {
                // Use Metal for batch processing
                var pooledBatch: [[Float]] = []
                
                for (output, tokenized) in zip(outputs, tokenizedBatch) {
                    let pooled = try await accelerator.poolEmbeddings(
                        output.tokenEmbeddings,
                        strategy: configuration.poolingStrategy,
                        attentionMask: tokenized.attentionMask
                    )
                    pooledBatch.append(pooled)
                }
                
                // Batch normalize if needed
                let finalBatch: [[Float]]
                if configuration.normalizeEmbeddings {
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
                    let pooled = try pool(tokenEmbeddings: output.tokenEmbeddings, strategy: configuration.poolingStrategy)
                    let final = configuration.normalizeEmbeddings ? normalize(pooled) : pooled
                    results.append(EmbeddingVector(final))
                }
            }
        }
        
        return results
    }
    
    // MARK: - Private Helpers
    
    private func pool(tokenEmbeddings: [[Float]], strategy: PoolingStrategy) throws -> [Float] {
        guard !tokenEmbeddings.isEmpty else {
            throw EmbeddingError.inferenceFailed("No token embeddings to pool")
        }
        
        let embeddingSize = tokenEmbeddings[0].count
        
        switch strategy {
        case .mean:
            // Average all token embeddings
            var result = [Float](repeating: 0, count: embeddingSize)
            var count: Float = 0
            
            for embedding in tokenEmbeddings {
                guard embedding.count == embeddingSize else {
                    throw EmbeddingError.dimensionMismatch(expected: embeddingSize, actual: embedding.count)
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
                    throw EmbeddingError.dimensionMismatch(expected: embeddingSize, actual: embedding.count)
                }
                
                vDSP_vmax(result, 1, embedding, 1, &result, 1, vDSP_Length(embeddingSize))
            }
            
            return result
            
        case .attentionWeighted:
            // For now, fall back to mean pooling
            // TODO: Implement attention-weighted pooling when attention weights are available
            return try pool(tokenEmbeddings: tokenEmbeddings, strategy: .mean)
        }
    }
    
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
}
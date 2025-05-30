import Foundation
import OSLog

/// Protocol for managing embedding models
public protocol EmbeddingModelManager: Actor {
    func loadModel(
        from url: URL,
        identifier: String,
        configuration: ModelBackendConfiguration?
    ) async throws -> ModelMetadata
    
    func unloadModel(identifier: String) async throws
    
    func getModel(identifier: String?) async -> (any TextEmbedder)?
}

/// Manager for multiple embedding models
public actor DefaultEmbeddingModelManager: EmbeddingModelManager {
    private let logger = Logger(subsystem: "EmbedKit", category: "ModelManager")
    
    private var models: [String: any TextEmbedder] = [:]
    private var defaultModelIdentifier: String?
    
    public init() {}
    
    public func loadModel(
        from url: URL,
        identifier: String,
        configuration: ModelBackendConfiguration? = nil
    ) async throws -> ModelMetadata {
        logger.info("Loading model: \(identifier)")
        
        // Create a new embedder for this model
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: identifier,
            configuration: EmbeddingConfiguration()
        )
        
        // Load the model
        try await embedder.loadModel()
        
        // Store the embedder
        models[identifier] = embedder
        
        // Set as default if it's the first model
        if defaultModelIdentifier == nil {
            defaultModelIdentifier = identifier
        }
        
        // Return metadata
        return ModelMetadata(
            name: identifier,
            version: "1.0",
            embeddingDimensions: await embedder.dimensions,
            maxSequenceLength: embedder.configuration.maxSequenceLength,
            vocabularySize: 30522, // Default
            modelType: "coreml",
            additionalInfo: [:]
        )
    }
    
    public func unloadModel(identifier: String) async throws {
        logger.info("Unloading model: \(identifier)")
        
        guard let embedder = models[identifier] else {
            throw EmbeddingError.resourceUnavailable("Model not found: \(identifier)")
        }
        
        try await embedder.unloadModel()
        models.removeValue(forKey: identifier)
        
        // Update default if needed
        if defaultModelIdentifier == identifier {
            defaultModelIdentifier = models.keys.first
        }
    }
    
    public func getModel(identifier: String?) async -> (any TextEmbedder)? {
        if let identifier = identifier {
            return models[identifier]
        } else if let defaultId = defaultModelIdentifier {
            return models[defaultId]
        }
        return nil
    }
    
    /// Set the default model identifier
    public func setDefaultModel(identifier: String) async throws {
        guard models[identifier] != nil else {
            throw EmbeddingError.resourceUnavailable("Model not found: \(identifier)")
        }
        defaultModelIdentifier = identifier
    }
    
    /// Get all loaded model identifiers
    public func loadedModels() async -> [String] {
        Array(models.keys)
    }
    
    /// Get information about a specific model
    public func modelInfo(for identifier: String?) async -> EmbeddingModelInfo? {
        guard let embedder = await getModel(identifier: identifier) else {
            return nil
        }
        
        let modelId = identifier ?? defaultModelIdentifier ?? "unknown"
        
        return EmbeddingModelInfo(
            identifier: modelId,
            dimensions: await embedder.dimensions,
            metadata: nil, // TODO: Store metadata
            isReady: await embedder.isReady,
            cacheStatistics: nil // TODO: Implement caching
        )
    }
}

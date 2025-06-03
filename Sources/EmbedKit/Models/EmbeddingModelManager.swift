import Foundation
import OSLog

/// Protocol for managing embedding models
public protocol EmbeddingModelManager: Actor {
    func loadModel(
        from url: URL,
        identifier: ModelIdentifier,
        configuration: ModelBackendConfiguration?
    ) async throws -> ModelMetadata
    
    func unloadModel(identifier: ModelIdentifier) async throws
    
    func getModel(identifier: ModelIdentifier?) async -> (any TextEmbedder)?
}

/// Manager for multiple embedding models
public actor DefaultEmbeddingModelManager: EmbeddingModelManager {
    private let logger = Logger(subsystem: "EmbedKit", category: "ModelManager")
    
    private var models: [String: any TextEmbedder] = [:]
    private var defaultModelIdentifier: ModelIdentifier?
    
    public init() {}
    
    public func loadModel(
        from url: URL,
        identifier: ModelIdentifier,
        configuration: ModelBackendConfiguration? = nil
    ) async throws -> ModelMetadata {
        logger.info("Loading model: \(identifier.rawValue)")
        
        // Create a new embedder for this model
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: identifier,
            configuration: Configuration()
        )
        
        // Load the model
        try await embedder.loadModel()
        
        // Store the embedder
        models[identifier.rawValue] = embedder
        
        // Set as default if it's the first model
        if defaultModelIdentifier == nil {
            defaultModelIdentifier = identifier
        }
        
        // Return metadata
        return ModelMetadata(
            name: identifier.rawValue,
            version: "1.0",
            embeddingDimensions: await embedder.dimensions,
            maxSequenceLength: embedder.configuration.model.maxSequenceLength,
            vocabularySize: 30522, // Default
            modelType: "coreml",
            additionalInfo: [:]
        )
    }
    
    public func unloadModel(identifier: ModelIdentifier) async throws {
        logger.info("Unloading model: \(identifier.rawValue)")
        
        guard let embedder = models[identifier.rawValue] else {
            throw ContextualEmbeddingError.resourceUnavailable(
                context: ErrorContext(
                    operation: .resourceManagement,
                    modelIdentifier: identifier,
                    metadata: ErrorMetadata()
                        .with(key: "action", value: "unload"),
                    sourceLocation: SourceLocation()
                ),
                resource: .model
            )
        }
        
        try await embedder.unloadModel()
        models.removeValue(forKey: identifier.rawValue)
        
        // Update default if needed
        if defaultModelIdentifier == identifier {
            defaultModelIdentifier = models.keys.first.map { try? ModelIdentifier($0) } ?? nil
        }
    }
    
    public func getModel(identifier: ModelIdentifier?) async -> (any TextEmbedder)? {
        if let identifier = identifier {
            return models[identifier.rawValue]
        } else if let defaultId = defaultModelIdentifier {
            return models[defaultId.rawValue]
        }
        return nil
    }
    
    /// Set the default model identifier
    public func setDefaultModel(identifier: ModelIdentifier) async throws {
        guard models[identifier.rawValue] != nil else {
            throw ContextualEmbeddingError.resourceUnavailable(
                context: ErrorContext(
                    operation: .resourceManagement,
                    modelIdentifier: identifier,
                    metadata: ErrorMetadata()
                        .with(key: "action", value: "unload"),
                    sourceLocation: SourceLocation()
                ),
                resource: .model
            )
        }
        defaultModelIdentifier = identifier
    }
    
    /// Get all loaded model identifiers
    public func loadedModels() async -> [ModelIdentifier] {
        models.keys.compactMap { try? ModelIdentifier($0) }
    }
    
    /// Get information about a specific model
    public func modelInfo(for identifier: ModelIdentifier?) async -> ModelInfo? {
        guard let embedder = await getModel(identifier: identifier) else {
            return nil
        }
        
        let modelId = identifier ?? defaultModelIdentifier ?? ModelIdentifier(family: "unknown")
        
        return ModelInfo(
            identifier: modelId,
            dimensions: await embedder.dimensions,
            isReady: await embedder.isReady
        )
    }
}

/// Information about a loaded model
public struct ModelInfo: Sendable {
    public let identifier: ModelIdentifier
    public let dimensions: Int
    public let isReady: Bool
    
    public init(identifier: ModelIdentifier, dimensions: Int, isReady: Bool) {
        self.identifier = identifier
        self.dimensions = dimensions
        self.isReady = isReady
    }
}

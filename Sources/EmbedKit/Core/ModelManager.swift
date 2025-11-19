import Foundation
import Logging

/// Actor responsible for managing embedding models.
/// Handles loading, unloading, and access to models.
public actor ModelManager {
    private var loadedModels: [String: any EmbeddingModel] = [:]
    private let logger = Logger(label: "com.embedkit.modelmanager")
    
    public init() {}
    
    /// Loads a model with the specified configuration
    /// - Parameters:
    ///   - id: Unique identifier for the model
    ///   - loader: Closure that loads the model
    /// - Returns: The loaded model
    public func loadModel(
        id: String,
        loader: () async throws -> any EmbeddingModel
    ) async throws -> any EmbeddingModel {
        if let existing = loadedModels[id] {
            return existing
        }
        
        logger.info("Loading model: \(id)")
        do {
            let model = try await loader()
            loadedModels[id] = model
            logger.info("Successfully loaded model: \(id)")
            return model
        } catch {
            logger.error("Failed to load model \(id): \(error)")
            throw EmbedKitError.modelLoadFailed(id, error)
        }
    }
    
    /// Retrieves a loaded model by ID
    /// - Parameter id: The model ID
    /// - Returns: The model if loaded, nil otherwise
    public func getModel(id: String) -> (any EmbeddingModel)? {
        return loadedModels[id]
    }
    
    /// Unloads a model to free resources
    /// - Parameter id: The model ID
    public func unloadModel(id: String) {
        if loadedModels.removeValue(forKey: id) != nil {
            logger.info("Unloaded model: \(id)")
        }
    }
    
    /// Unloads all models
    public func unloadAll() {
        loadedModels.removeAll()
        logger.info("Unloaded all models")
    }
}

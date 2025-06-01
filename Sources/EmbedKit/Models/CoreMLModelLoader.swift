import Foundation
import CoreML
import OSLog

/// Utility for loading and managing Core ML embedding models
public actor CoreMLModelLoader {
    private let logger = EmbedKitLogger.modelManagement()
    
    /// Available pre-converted models
    public enum PretrainedModel: String, CaseIterable {
        case miniLM = "all-MiniLM-L6-v2"
        case bgeSmall = "bge-small-en-v1.5"
        case gteSmall = "gte-small"
        
        var bundleName: String {
            rawValue.replacingOccurrences(of: "/", with: "_")
        }
        
        var expectedDimensions: Int {
            switch self {
            case .miniLM, .bgeSmall, .gteSmall:
                return 384
            }
        }
        
        var maxSequenceLength: Int {
            switch self {
            case .miniLM:
                return 256
            case .bgeSmall, .gteSmall:
                return 512
            }
        }
    }
    
    private var loadedModels: [String: MLModel] = [:]
    
    public init() {}
    
    /// Load a pretrained model from the app bundle
    public func loadPretrainedModel(_ model: PretrainedModel) async throws -> (MLModel, ModelMetadata) {
        logger.start("Loading pretrained model", details: model.rawValue)
        
        // Check if already loaded
        if let cached = loadedModels[model.rawValue] {
            logger.cache("Model already loaded", hitRate: nil, size: nil)
            let metadata = try await extractMetadata(from: cached, modelId: model.rawValue)
            return (cached, metadata)
        }
        
        // Find model in bundle
        guard let modelURL = Bundle.main.url(
            forResource: model.bundleName,
            withExtension: "mlmodelc"
        ) ?? Bundle.main.url(
            forResource: model.bundleName,
            withExtension: "mlpackage"
        ) else {
            throw EmbeddingError.resourceUnavailable("Model not found in bundle: \(model.rawValue)")
        }
        
        return try await loadModel(from: modelURL, identifier: model.rawValue)
    }
    
    /// Load a Core ML model from a URL
    public func loadModel(from url: URL, identifier: String) async throws -> (MLModel, ModelMetadata) {
        logger.start("Loading Core ML model", details: identifier)
        
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all  // Use Neural Engine when available
        
        // For iOS 16+ we can set additional options
        if #available(iOS 16.0, macOS 13.0, *) {
            configuration.modelDisplayName = identifier
        }
        
        do {
            let startTime = CFAbsoluteTimeGetCurrent()
            let model = try await MLModel.load(contentsOf: url, configuration: configuration)
            let loadTime = CFAbsoluteTimeGetCurrent() - startTime
            
            logger.performance("Model load time", duration: loadTime)
            
            // Cache the model
            loadedModels[identifier] = model
            
            // Extract metadata
            let metadata = try await extractMetadata(from: model, modelId: identifier)
            
            logger.complete("Model loaded", result: "\(metadata.embeddingDimensions) dimensions")
            
            return (model, metadata)
        } catch {
            logger.error("Failed to load Core ML model", error: error)
            throw EmbeddingError.resourceUnavailable("Failed to load model: \(error.localizedDescription)")
        }
    }
    
    /// Extract metadata from a Core ML model
    private func extractMetadata(from model: MLModel, modelId: String) async throws -> ModelMetadata {
        let description = model.modelDescription
        
        // Get embedding dimensions from output shape
        guard let outputFeature = description.outputDescriptionsByName.values.first,
              let multiArray = outputFeature.multiArrayConstraint else {
            throw EmbeddingError.invalidInput("Cannot determine output dimensions")
        }
        
        let dimensions = multiArray.shape.last?.intValue ?? 0
        
        // Extract custom metadata if available
        let userMetadata = description.metadata[MLModelMetadataKey.creatorDefinedKey] as? [String: String] ?? [:]
        
        let maxLength = Int(userMetadata["max_length"] ?? "256") ?? 256
        let poolingStrategy = userMetadata["pooling_strategy"] ?? "mean"
        let normalize = (userMetadata["normalize"] ?? "true").lowercased() == "true"
        
        // Calculate model size
        let modelSize = try? FileManager.default.attributesOfItem(atPath: model.modelDescription.url?.path ?? "")[.size] as? Int64
        
        return ModelMetadata(
            identifier: modelId,
            source: userMetadata["source_model"] ?? modelId,
            embeddingDimensions: dimensions,
            maxSequenceLength: maxLength,
            modelFormat: .coreML,
            fileSize: modelSize ?? 0,
            version: description.metadata[.versionString] as? String ?? "1.0",
            author: description.metadata[.author] as? String,
            description: description.metadata[.description] as? String,
            createdDate: Date(),
            metadata: [
                "pooling_strategy": poolingStrategy,
                "normalize": String(normalize),
                "compute_units": "all"
            ]
        )
    }
    
    /// Unload a model from memory
    public func unloadModel(_ identifier: String) async {
        logger.info("Unloading model", context: identifier)
        loadedModels.removeValue(forKey: identifier)
    }
    
    /// Get all loaded models
    public func loadedModelIdentifiers() async -> [String] {
        Array(loadedModels.keys)
    }
    
    /// Validate model compatibility
    public func validateModel(_ model: MLModel) async throws {
        let description = model.modelDescription
        
        // Check inputs
        guard let inputIds = description.inputDescriptionsByName["input_ids"],
              let attentionMask = description.inputDescriptionsByName["attention_mask"] else {
            throw EmbeddingError.invalidInput("Model missing required inputs: input_ids, attention_mask")
        }
        
        // Validate input types
        guard inputIds.type == .multiArray,
              attentionMask.type == .multiArray else {
            throw EmbeddingError.invalidInput("Model inputs must be MultiArray type")
        }
        
        // Check output
        guard let output = description.outputDescriptionsByName["embeddings"] ?? description.outputDescriptionsByName.values.first,
              output.type == .multiArray else {
            throw EmbeddingError.invalidInput("Model must output embeddings as MultiArray")
        }
        
        logger.success("Model validation passed")
    }
}

// MARK: - Model Discovery

public extension CoreMLModelLoader {
    /// Discover Core ML models in a directory
    func discoverModels(in directory: URL) async throws -> [ModelMetadata] {
        logger.start("Discovering models", details: directory.path)
        
        let fileManager = FileManager.default
        var discovered: [ModelMetadata] = []
        
        let enumerator = fileManager.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )
        
        while let url = enumerator?.nextObject() as? URL {
            if url.pathExtension == "mlmodelc" || url.pathExtension == "mlpackage" {
                do {
                    let (_, metadata) = try await loadModel(
                        from: url,
                        identifier: url.deletingPathExtension().lastPathComponent
                    )
                    discovered.append(metadata)
                    logger.info("Discovered model", context: metadata.identifier)
                } catch {
                    logger.warning("Failed to load model at \(url.path): \(error)")
                }
            }
        }
        
        logger.complete("Model discovery", result: "\(discovered.count) models found")
        return discovered
    }
}

// MARK: - Tokenizer Configuration Loader

public struct TokenizerConfigLoader {
    /// Load tokenizer configuration from a model directory
    public static func loadConfig(from directory: URL) throws -> TokenizerConfig {
        let configURL = directory.appendingPathComponent("tokenizer_config.json")
        let specialTokensURL = directory.appendingPathComponent("special_tokens.json")
        
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(RawTokenizerConfig.self, from: configData)
        
        var specialTokens: TokenizerSpecialTokens?
        if let specialData = try? Data(contentsOf: specialTokensURL) {
            specialTokens = try? JSONDecoder().decode(TokenizerSpecialTokens.self, from: specialData)
        }
        
        return TokenizerConfig(
            type: config.type,
            vocabSize: config.vocab_size,
            maxLength: config.max_length,
            vocab: config.vocab,
            specialTokens: specialTokens
        )
    }
    
    private struct RawTokenizerConfig: Decodable {
        let type: String
        let vocab_size: Int
        let max_length: Int
        let vocab: [String: Int]
    }
}

public struct TokenizerConfig {
    public let type: String
    public let vocabSize: Int
    public let maxLength: Int
    public let vocab: [String: Int]
    public let specialTokens: TokenizerSpecialTokens?
}

private struct TokenizerSpecialTokens: Codable {
    public let padTokenId: Int?
    public let unkTokenId: Int?
    public let clsTokenId: Int?
    public let sepTokenId: Int?
    public let maskTokenId: Int?
    
    enum CodingKeys: String, CodingKey {
        case padTokenId = "pad_token_id"
        case unkTokenId = "unk_token_id"
        case clsTokenId = "cls_token_id"
        case sepTokenId = "sep_token_id"
        case maskTokenId = "mask_token_id"
    }
}
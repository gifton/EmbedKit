import Foundation
@preconcurrency import CoreML
import OSLog

/// Enhanced metadata extraction for CoreML models
public struct CoreMLMetadataExtractor {
    private static let logger = Logger(subsystem: "EmbedKit", category: "MetadataExtractor")
    
    /// Extract comprehensive metadata from a CoreML model
    public static func extractMetadata(
        from model: MLModel,
        modelIdentifier: String,
        modelURL: URL? = nil
    ) async throws -> ModelMetadata {
        let description = model.modelDescription
        
        // Extract dimensions and model structure
        let structureInfo = extractModelStructure(from: description)
        
        // Extract metadata from model's user-defined metadata
        let userMetadata = description.metadata[MLModelMetadataKey.creatorDefinedKey] as? [String: Any] ?? [:]
        
        // Extract from standard metadata keys
        let standardMetadata = extractStandardMetadata(from: description)
        
        // Determine model type and configuration
        let modelTypeInfo = detectModelType(from: description, userMetadata: userMetadata)
        
        // Get file size if URL provided
        var fileSize: Int64 = 0
        if let url = modelURL {
            fileSize = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? 0
        }
        
        // Build comprehensive metadata
        let metadata = ModelMetadata(
            name: userMetadata["name"] as? String ?? standardMetadata.name ?? modelIdentifier,
            version: userMetadata["version"] as? String ?? standardMetadata.version ?? "1.0",
            embeddingDimensions: structureInfo.embeddingDimensions,
            maxSequenceLength: structureInfo.maxSequenceLength,
            vocabularySize: structureInfo.vocabularySize,
            modelType: modelTypeInfo.type,
            additionalInfo: buildAdditionalInfo(
                structureInfo: structureInfo,
                modelTypeInfo: modelTypeInfo,
                userMetadata: userMetadata,
                standardMetadata: standardMetadata,
                fileSize: fileSize
            )
        )
        
        logger.info("""
            Extracted metadata for \(modelIdentifier):
            - Embedding dimensions: \(metadata.embeddingDimensions)
            - Max sequence length: \(metadata.maxSequenceLength)
            - Vocabulary size: \(metadata.vocabularySize)
            - Model type: \(metadata.modelType)
            """)
        
        return metadata
    }
    
    // MARK: - Model Structure Analysis
    
    private struct ModelStructureInfo {
        let embeddingDimensions: Int
        let maxSequenceLength: Int
        let vocabularySize: Int
        let inputNames: [String]
        let outputNames: [String]
        let hasAttentionWeights: Bool
        let hasTokenTypeIds: Bool
        let outputShape: [Int]
    }
    
    private static func extractModelStructure(from description: MLModelDescription) -> ModelStructureInfo {
        var embeddingDimensions = 0
        var maxSequenceLength = 0
        var vocabularySize = 0
        var hasAttentionWeights = false
        var hasTokenTypeIds = false
        var outputShape: [Int] = []
        
        let inputNames = Array(description.inputDescriptionsByName.keys)
        let outputNames = Array(description.outputDescriptionsByName.keys)
        
        // Analyze outputs for embedding dimensions
        for (name, output) in description.outputDescriptionsByName {
            guard let shape = output.multiArrayConstraint?.shape else { continue }
            
            let nameLower = name.lowercased()
            
            // Check for embedding outputs
            if nameLower.contains("embedding") ||
               nameLower.contains("output") ||
               nameLower.contains("pooler") ||
               nameLower.contains("last_hidden_state") ||
               nameLower.contains("sentence_embedding") {
                
                outputShape = shape.map { $0.intValue }
                
                // Handle different output shapes
                if shape.count == 2 {
                    // [batch, embedding_dim]
                    embeddingDimensions = shape[1].intValue
                } else if shape.count == 3 {
                    // [batch, sequence, embedding_dim]
                    embeddingDimensions = shape[2].intValue
                    if maxSequenceLength == 0 {
                        maxSequenceLength = shape[1].intValue
                    }
                }
                
                logger.debug("Found embedding output '\(name)' with shape \(outputShape)")
            }
            
            // Check for attention weights
            if nameLower.contains("attention") {
                hasAttentionWeights = true
            }
        }
        
        // Analyze inputs for sequence length and vocabulary
        for (name, input) in description.inputDescriptionsByName {
            guard let shape = input.multiArrayConstraint?.shape else { continue }
            
            let nameLower = name.lowercased()
            
            // Check for token inputs
            if nameLower.contains("input_ids") || nameLower.contains("token") {
                if shape.count >= 2 {
                    maxSequenceLength = shape[1].intValue
                } else if shape.count == 1 && maxSequenceLength == 0 {
                    maxSequenceLength = shape[0].intValue
                }
            }
            
            // Check for token type IDs
            if nameLower.contains("token_type") {
                hasTokenTypeIds = true
            }
            
            // Try to infer vocabulary size from input constraints
            if let constraint = input.multiArrayConstraint {
                // Some models might have vocabulary size in shape constraints
                // Note: MLMultiArrayShapeConstraint API is limited, so we skip this for now
            }
        }
        
        return ModelStructureInfo(
            embeddingDimensions: embeddingDimensions,
            maxSequenceLength: maxSequenceLength,
            vocabularySize: vocabularySize,
            inputNames: inputNames,
            outputNames: outputNames,
            hasAttentionWeights: hasAttentionWeights,
            hasTokenTypeIds: hasTokenTypeIds,
            outputShape: outputShape
        )
    }
    
    // MARK: - Model Type Detection
    
    private struct ModelTypeInfo {
        let type: String
        let architecture: String?
        let poolingStrategy: String?
        let normalize: Bool
    }
    
    private static func detectModelType(
        from description: MLModelDescription,
        userMetadata: [String: Any]
    ) -> ModelTypeInfo {
        // Check user metadata first
        if let modelType = userMetadata["model_type"] as? String {
            return ModelTypeInfo(
                type: modelType,
                architecture: userMetadata["architecture"] as? String,
                poolingStrategy: userMetadata["pooling_strategy"] as? String,
                normalize: (userMetadata["normalize"] as? Bool) ?? true
            )
        }
        
        // Try to detect from input/output names
        let inputNames = Set(description.inputDescriptionsByName.keys.map { $0.lowercased() })
        let outputNames = Set(description.outputDescriptionsByName.keys.map { $0.lowercased() })
        
        var detectedType = "unknown"
        var architecture: String?
        
        // BERT-style models
        if inputNames.contains("input_ids") && inputNames.contains("attention_mask") {
            if inputNames.contains("token_type_ids") {
                detectedType = "bert"
                architecture = "transformer-bert"
            } else {
                detectedType = "sentence-transformer"
                architecture = "transformer-mpnet"
            }
        }
        
        // Check for specific architectures in output names
        if outputNames.contains("sentence_embedding") {
            detectedType = "sentence-transformer"
        } else if outputNames.contains("pooler_output") {
            architecture = "transformer-bert"
        }
        
        // Default pooling strategy based on type
        let poolingStrategy = userMetadata["pooling_strategy"] as? String ?? "mean"
        
        return ModelTypeInfo(
            type: detectedType,
            architecture: architecture,
            poolingStrategy: poolingStrategy,
            normalize: true
        )
    }
    
    // MARK: - Standard Metadata Extraction
    
    private struct StandardMetadata {
        let name: String?
        let version: String?
        let author: String?
        let license: String?
        let description: String?
    }
    
    private static func extractStandardMetadata(from description: MLModelDescription) -> StandardMetadata {
        let metadata = description.metadata
        
        return StandardMetadata(
            name: metadata[.description] as? String,
            version: metadata[.versionString] as? String,
            author: metadata[.author] as? String,
            license: metadata[.license] as? String,
            description: metadata[MLModelMetadataKey.description] as? String
        )
    }
    
    // MARK: - Additional Info Builder
    
    private static func buildAdditionalInfo(
        structureInfo: ModelStructureInfo,
        modelTypeInfo: ModelTypeInfo,
        userMetadata: [String: Any],
        standardMetadata: StandardMetadata,
        fileSize: Int64
    ) -> [String: String] {
        var info: [String: String] = [:]
        
        // Structure information
        info["input_names"] = structureInfo.inputNames.joined(separator: ",")
        info["output_names"] = structureInfo.outputNames.joined(separator: ",")
        info["output_shape"] = structureInfo.outputShape.map(String.init).joined(separator: "x")
        info["has_attention_weights"] = String(structureInfo.hasAttentionWeights)
        info["has_token_type_ids"] = String(structureInfo.hasTokenTypeIds)
        
        // Model type information
        if let arch = modelTypeInfo.architecture {
            info["architecture"] = arch
        }
        if let pooling = modelTypeInfo.poolingStrategy {
            info["pooling_strategy"] = pooling
        }
        info["normalize"] = String(modelTypeInfo.normalize)
        
        // Standard metadata
        if let author = standardMetadata.author {
            info["author"] = author
        }
        if let license = standardMetadata.license {
            info["license"] = license
        }
        if let desc = standardMetadata.description {
            info["description"] = desc
        }
        
        // File information
        if fileSize > 0 {
            info["file_size"] = String(fileSize)
            info["file_size_mb"] = String(format: "%.2f", Double(fileSize) / 1024 / 1024)
        }
        
        // User metadata passthrough
        for (key, value) in userMetadata {
            if !info.keys.contains(key) {
                info[key] = String(describing: value)
            }
        }
        
        // Compute configuration
        info["compute_units"] = "all"
        info["supports_gpu"] = "true"
        info["supports_neural_engine"] = "true"
        
        return info
    }
}

// MARK: - Model Metadata Validation

public extension CoreMLMetadataExtractor {
    /// Validate extracted metadata for completeness
    static func validateMetadata(_ metadata: ModelMetadata) -> [String] {
        var issues: [String] = []
        
        if metadata.embeddingDimensions <= 0 {
            issues.append("Invalid embedding dimensions: \(metadata.embeddingDimensions)")
        }
        
        if metadata.maxSequenceLength <= 0 {
            issues.append("Invalid max sequence length: \(metadata.maxSequenceLength)")
        }
        
        if metadata.vocabularySize <= 0 {
            issues.append("Invalid vocabulary size: \(metadata.vocabularySize)")
        }
        
        if metadata.modelType == "unknown" {
            issues.append("Could not determine model type")
        }
        
        return issues
    }
    
    /// Create default metadata for known model types
    static func defaultMetadata(for identifier: ModelIdentifier) -> ModelMetadata? {
        // Known model defaults - check both family and full raw value
        let identifierLower = identifier.rawValue.lowercased()
        let familyLower = identifier.family.lowercased()
        
        if identifierLower.contains("minilm") || familyLower.contains("minilm") {
            return ModelMetadata(
                name: "all-MiniLM-L6-v2",
                version: "2",
                embeddingDimensions: 384,
                maxSequenceLength: 256,
                vocabularySize: 30522,
                modelType: "sentence-transformer",
                additionalInfo: [
                    "architecture": "transformer-mpnet",
                    "pooling_strategy": "mean",
                    "normalize": "true"
                ]
            )
        } else if identifierLower.contains("bge-small") || familyLower == "bge" {
            return ModelMetadata(
                name: "bge-small-en-v1.5",
                version: "1.5",
                embeddingDimensions: 384,
                maxSequenceLength: 512,
                vocabularySize: 30522,
                modelType: "sentence-transformer",
                additionalInfo: [
                    "architecture": "transformer-bert",
                    "pooling_strategy": "cls",
                    "normalize": "true"
                ]
            )
        } else if identifierLower.contains("gte-small") || familyLower == "gte" {
            return ModelMetadata(
                name: "gte-small",
                version: "1",
                embeddingDimensions: 384,
                maxSequenceLength: 512,
                vocabularySize: 30522,
                modelType: "sentence-transformer",
                additionalInfo: [
                    "architecture": "transformer-bert",
                    "pooling_strategy": "mean",
                    "normalize": "true"
                ]
            )
        }
        
        return nil
    }
}
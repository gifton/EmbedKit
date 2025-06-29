import Foundation
import OSLog

/// Registry for Hugging Face hosted embedding models
public actor HuggingFaceModelRegistry: ModelRegistryProtocol {
    private let logger = Logger(subsystem: "EmbedKit", category: "HuggingFaceRegistry")
    
    /// Base URL for Hugging Face model hub
    private let baseURL = "https://huggingface.co"
    
    /// Known embedding models with their configurations
    private let knownModels: [String: ModelConfiguration] = [
        "sentence-transformers/all-MiniLM-L6-v2": ModelConfiguration(
            identifier: .miniLM_L6_v2,
            dimensions: 384,
            maxLength: 512,
            vocabSize: 30522
        ),
        "BAAI/bge-small-en-v1.5": ModelConfiguration(
            identifier: ModelIdentifier(family: "bge", variant: "small-en", version: "v1.5"),
            dimensions: 384,
            maxLength: 512,
            vocabSize: 30522
        ),
        "thenlper/gte-small": ModelConfiguration(
            identifier: ModelIdentifier(family: "gte", variant: "small", version: "v1"),
            dimensions: 384,
            maxLength: 512,
            vocabSize: 30522
        ),
        "sentence-transformers/all-mpnet-base-v2": ModelConfiguration(
            identifier: ModelIdentifier(family: "mpnet", variant: "base", version: "v2"),
            dimensions: 768,
            maxLength: 384,
            vocabSize: 30527
        )
    ]
    
    private struct ModelConfiguration {
        let identifier: ModelIdentifier
        let dimensions: Int
        let maxLength: Int
        let vocabSize: Int
    }
    
    public init() {}
    
    /// Get model information from the registry
    public func getModel(_ identifier: ModelIdentifier) async -> ModelRegistryEntry? {
        // Find the HuggingFace model ID for this identifier
        guard let (hfModelId, config) = findHuggingFaceModel(for: identifier) else {
            logger.warning("Model not found in registry: \(identifier.rawValue)")
            return nil
        }
        
        // Construct download URLs for Core ML model files
        let coreMLURL = constructCoreMLDownloadURL(for: hfModelId)
        
        // Get model info from HuggingFace API
        let modelInfo = await fetchModelInfo(hfModelId: hfModelId)
        
        return ModelRegistryEntry(
            identifier: identifier,
            downloadURL: coreMLURL,
            checksum: modelInfo?.checksum,
            fileSize: modelInfo?.fileSize,
            metadata: [
                "huggingface_id": hfModelId,
                "dimensions": String(config.dimensions),
                "max_length": String(config.maxLength),
                "vocab_size": String(config.vocabSize),
                "source": "huggingface"
            ]
        )
    }
    
    /// List all available models
    public func listModels() async -> [ModelRegistryEntry] {
        knownModels.compactMap { (hfModelId, config) in
            ModelRegistryEntry(
                identifier: config.identifier,
                downloadURL: constructCoreMLDownloadURL(for: hfModelId),
                metadata: [
                    "huggingface_id": hfModelId,
                    "dimensions": String(config.dimensions),
                    "max_length": String(config.maxLength),
                    "vocab_size": String(config.vocabSize),
                    "source": "huggingface"
                ]
            )
        }
    }
    
    /// Search for models by query
    public func searchModels(query: String, limit: Int = 10) async throws -> [ModelSearchResult] {
        let searchURL = URL(string: "\(baseURL)/api/models")!
            .appending(queryItems: [
                URLQueryItem(name: "search", value: query),
                URLQueryItem(name: "pipeline_tag", value: "sentence-similarity"),
                URLQueryItem(name: "library", value: "sentence-transformers"),
                URLQueryItem(name: "limit", value: String(limit))
            ])
        
        let (data, response) = try await URLSession.shared.data(from: searchURL)
        
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw ContextualEmbeddingError.networkError(
                context: ErrorContext(
                    operation: .initialization,
                    metadata: ErrorMetadata()
                        .with(key: "query", value: query),
                    sourceLocation: SourceLocation()
                ),
                statusCode: (response as? HTTPURLResponse)?.statusCode ?? -1
            )
        }
        
        let results = try JSONDecoder().decode([HuggingFaceModel].self, from: data)
        
        return results.map { model in
            ModelSearchResult(
                modelId: model.modelId,
                downloads: model.downloads ?? 0,
                likes: model.likes ?? 0,
                tags: model.tags ?? [],
                lastModified: model.lastModified
            )
        }
    }
    
    // MARK: - Private Methods
    
    private func findHuggingFaceModel(for identifier: ModelIdentifier) -> (String, ModelConfiguration)? {
        // Try exact match first
        for (hfId, config) in knownModels {
            if config.identifier == identifier {
                return (hfId, config)
            }
        }
        
        // Try fuzzy match
        let searchKey = identifier.rawValue.lowercased()
        for (hfId, config) in knownModels {
            if hfId.lowercased().contains(searchKey) || 
               config.identifier.rawValue.lowercased().contains(searchKey) {
                return (hfId, config)
            }
        }
        
        return nil
    }
    
    private func constructCoreMLDownloadURL(for hfModelId: String) -> URL {
        // Construct URL for Core ML model file
        // Format: https://huggingface.co/{model_id}/resolve/main/coreml/{model_name}.mlpackage
        let modelName = hfModelId.split(separator: "/").last ?? "model"
        
        return URL(string: "\(baseURL)/\(hfModelId)/resolve/main/coreml/\(modelName).mlpackage")!
    }
    
    private func fetchModelInfo(hfModelId: String) async -> (checksum: String?, fileSize: Int64?)? {
        // Fetch model file info from HuggingFace API
        let apiURL = URL(string: "\(baseURL)/api/models/\(hfModelId)")!
        
        do {
            let (data, response) = try await URLSession.shared.data(from: apiURL)
            
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                return nil
            }
            
            if let modelInfo = try? JSONDecoder().decode(HuggingFaceModelInfo.self, from: data) {
                // Look for Core ML file in siblings
                let coreMLFile = modelInfo.siblings?.first { sibling in
                    sibling.rfilename.contains("coreml") && 
                    (sibling.rfilename.hasSuffix(".mlpackage") || sibling.rfilename.hasSuffix(".mlmodel"))
                }
                
                return (coreMLFile?.lfs?.sha256, coreMLFile?.size)
            }
        } catch {
            logger.warning("Failed to fetch model info for \(hfModelId): \(error)")
        }
        
        return nil
    }
}

// MARK: - Data Models

public struct ModelSearchResult: Sendable {
    public let modelId: String
    public let downloads: Int
    public let likes: Int
    public let tags: [String]
    public let lastModified: String?
}

private struct HuggingFaceModel: Codable {
    let modelId: String
    let downloads: Int?
    let likes: Int?
    let tags: [String]?
    let lastModified: String?
}

private struct HuggingFaceModelInfo: Codable {
    let siblings: [ModelFile]?
    
    struct ModelFile: Codable {
        let rfilename: String
        let size: Int64?
        let lfs: LFSInfo?
        
        struct LFSInfo: Codable {
            let sha256: String?
        }
    }
}

// MARK: - URL Extension

private extension URL {
    func appending(queryItems: [URLQueryItem]) -> URL {
        var components = URLComponents(url: self, resolvingAgainstBaseURL: false)!
        components.queryItems = queryItems
        return components.url!
    }
}
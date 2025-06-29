import Foundation
import OSLog

/// Manages model loading, downloading, and lifecycle
public actor ModelManager {
    private let logger = Logger(subsystem: "EmbedKit", category: "ModelManager")
    
    /// Model manager loading options
    public struct ManagerLoadingOptions: Sendable {
        /// Whether to download models if not found locally
        public let allowDownload: Bool
        
        /// Custom model registry to use
        public let registry: (any ModelRegistryProtocol)?
        
        /// Download configuration
        public let downloadConfiguration: ModelDownloader.DownloadConfiguration
        
        /// Whether to verify model signatures
        public let verifySignature: Bool
        
        public init(
            allowDownload: Bool = true,
            registry: (any ModelRegistryProtocol)? = nil,
            downloadConfiguration: ModelDownloader.DownloadConfiguration = .init(),
            verifySignature: Bool = true
        ) {
            self.allowDownload = allowDownload
            self.registry = registry
            self.downloadConfiguration = downloadConfiguration
            self.verifySignature = verifySignature
        }
    }
    
    private let downloader: ModelDownloader
    private let loader: CoreMLModelLoader
    private let registry: any ModelRegistryProtocol
    private let options: ManagerLoadingOptions
    
    public init(
        options: ManagerLoadingOptions = ManagerLoadingOptions()
    ) {
        self.options = options
        self.downloader = ModelDownloader(
            configuration: options.downloadConfiguration,
            delegate: nil
        )
        self.loader = CoreMLModelLoader()
        self.registry = options.registry ?? HuggingFaceModelRegistry()
    }
    
    /// Initialize with a download delegate
    public init(
        options: ManagerLoadingOptions = ManagerLoadingOptions(),
        downloadDelegate: any ModelDownloader.DownloadDelegate & Sendable
    ) {
        self.options = options
        self.downloader = ModelDownloader(
            configuration: options.downloadConfiguration,
            delegate: downloadDelegate
        )
        self.loader = CoreMLModelLoader()
        self.registry = options.registry ?? HuggingFaceModelRegistry()
    }
    
    /// Load or download a model
    public func loadModel(_ identifier: ModelIdentifier) async throws -> URL {
        logger.info("Loading model: \(identifier.rawValue)")
        
        // Check bundled models first
        if let bundledURL = findBundledModel(identifier) {
            logger.info("Found bundled model at: \(bundledURL.path)")
            return bundledURL
        }
        
        // Check downloaded models
        let localURL = await downloader.localModelURL(for: identifier)
        if FileManager.default.fileExists(atPath: localURL.path) {
            logger.info("Found downloaded model at: \(localURL.path)")
            return localURL
        }
        
        // Download if allowed
        guard options.allowDownload else {
            throw ContextualEmbeddingError.resourceUnavailable(
                context: ErrorContext(
                    operation: .modelLoading,
                    modelIdentifier: identifier,
                    metadata: ErrorMetadata()
                        .with(key: "reason", value: "Model not found locally and downloading is disabled"),
                    sourceLocation: SourceLocation()
                ),
                resource: .model
            )
        }
        
        logger.info("Model not found locally. Attempting to download...")
        
        // Get model info from registry
        guard let registryEntry = await registry.getModel(identifier) else {
            throw ContextualEmbeddingError.resourceUnavailable(
                context: ErrorContext(
                    operation: .modelLoading,
                    modelIdentifier: identifier,
                    metadata: ErrorMetadata()
                        .with(key: "reason", value: "Model not found in registry"),
                    sourceLocation: SourceLocation()
                ),
                resource: .model
            )
        }
        
        // Download the model
        guard let downloadURL = registryEntry.downloadURL else {
            throw ContextualEmbeddingError.resourceUnavailable(
                context: ErrorContext(
                    operation: .modelLoading,
                    modelIdentifier: identifier,
                    metadata: ErrorMetadata()
                        .with(key: "reason", value: "Model has no download URL"),
                    sourceLocation: SourceLocation()
                ),
                resource: .model
            )
        }
        
        let source = ModelDownloader.ModelSource(
            url: downloadURL,
            expectedChecksum: registryEntry.checksum,
            expectedSize: registryEntry.fileSize,
            modelIdentifier: identifier
        )
        
        return try await downloader.downloadModel(from: source)
    }
    
    /// Create an embedder with automatic model management
    public func createEmbedder(
        identifier: ModelIdentifier,
        configuration: Configuration? = nil
    ) async throws -> CoreMLTextEmbedder {
        // Load or download model
        let modelURL = try await loadModel(identifier)
        
        // Create configuration
        let config = configuration ?? Configuration.default(for: identifier)
        
        // Create new configuration with custom model URL
        let modifiedConfig = Configuration(
            model: ModelConfiguration(
                identifier: config.model.identifier,
                maxSequenceLength: config.model.maxSequenceLength,
                normalizeEmbeddings: config.model.normalizeEmbeddings,
                poolingStrategy: config.model.poolingStrategy,
                loadingOptions: LoadingOptions(
                    preloadWeights: config.model.loadingOptions.preloadWeights,
                    enableOptimizations: config.model.loadingOptions.enableOptimizations,
                    verifyIntegrity: config.model.loadingOptions.verifyIntegrity,
                    customModelURL: modelURL
                ),
                computeUnits: config.model.computeUnits
            ),
            resources: config.resources,
            performance: config.performance,
            monitoring: config.monitoring,
            cache: config.cache,
            errorHandling: config.errorHandling
        )
        
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: identifier,
            configuration: modifiedConfig
        )
        
        // Load the model
        try await embedder.loadModel()
        
        // Extract metadata to configure tokenizer
        let metadata = await embedder.getMetadata()
        
        // Try to load appropriate tokenizer with metadata-informed configuration
        if let tokenizerURL = try await loadTokenizer(for: identifier) {
            let tokenizerConfig = TokenizerConfiguration(
                maxSequenceLength: metadata?.maxSequenceLength ?? config.model.maxSequenceLength,
                vocabularySize: metadata?.vocabularySize ?? 30522
            )
            
            logger.info("""
                Configuring tokenizer with metadata:
                - Max sequence length: \(tokenizerConfig.maxSequenceLength)
                - Vocabulary size: \(tokenizerConfig.vocabularySize)
                """)
            
            if let tokenizer = try? await TokenizerFactory.loadTokenizer(
                from: tokenizerURL,
                configuration: tokenizerConfig
            ) {
                await embedder.updateTokenizer(tokenizer)
                logger.info("Loaded custom tokenizer for model")
            }
        } else {
            // Try to create appropriate tokenizer based on model type
            if let metadata = metadata {
                logger.info("No tokenizer files found. Creating tokenizer based on model type: \(metadata.modelType)")
                
                // Use TokenizerFactory to create appropriate tokenizer
                if let tokenizer = await createTokenizerForModelType(
                    metadata.modelType,
                    maxLength: metadata.maxSequenceLength,
                    vocabSize: metadata.vocabularySize
                ) {
                    await embedder.updateTokenizer(tokenizer)
                    logger.info("Created \(metadata.modelType) tokenizer for model")
                }
            }
        }
        
        return embedder
    }
    
    /// List all available models (bundled + downloaded)
    public func listAvailableModels() async throws -> [DownloadedModelInfo] {
        var models: [DownloadedModelInfo] = []
        
        // Add bundled models
        for pretrainedModel in CoreMLModelLoader.PretrainedModel.allCases {
            if findBundledModel(ModelIdentifier(family: pretrainedModel.rawValue)) != nil {
                models.append(DownloadedModelInfo(
                    identifier: ModelIdentifier(family: pretrainedModel.rawValue),
                    location: .bundled,
                    size: nil,
                    lastModified: nil
                ))
            }
        }
        
        // Add downloaded models
        let downloadedModels = try await downloader.listDownloadedModels()
        for identifier in downloadedModels {
            let url = await downloader.localModelURL(for: identifier)
            let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
            
            models.append(DownloadedModelInfo(
                identifier: identifier,
                location: .downloaded,
                size: attributes?[.size] as? Int64,
                lastModified: attributes?[.modificationDate] as? Date
            ))
        }
        
        return models
    }
    
    /// Delete a downloaded model
    public func deleteModel(_ identifier: ModelIdentifier) async throws {
        try await downloader.deleteModel(identifier)
        logger.info("Deleted model: \(identifier.rawValue)")
    }
    
    /// Get download progress for a model
    public func downloadProgress(for identifier: ModelIdentifier) async -> Progress? {
        guard let entry = await registry.getModel(identifier),
              let downloadURL = entry.downloadURL else {
            return nil
        }
        
        return await downloader.progress(for: downloadURL)
    }
    
    // MARK: - Private Methods
    
    private func findBundledModel(_ identifier: ModelIdentifier) -> URL? {
        // Check for .mlmodelc
        if let url = Bundle.main.url(
            forResource: identifier.rawValue,
            withExtension: "mlmodelc"
        ) {
            return url
        }
        
        // Check for .mlpackage
        if let url = Bundle.main.url(
            forResource: identifier.rawValue,
            withExtension: "mlpackage"
        ) {
            return url
        }
        
        // Check with underscores instead of slashes
        let bundleName = identifier.rawValue.replacingOccurrences(of: "/", with: "_")
        if let url = Bundle.main.url(
            forResource: bundleName,
            withExtension: "mlmodelc"
        ) {
            return url
        }
        
        if let url = Bundle.main.url(
            forResource: bundleName,
            withExtension: "mlpackage"
        ) {
            return url
        }
        
        return nil
    }
    
    private func loadTokenizer(for identifier: ModelIdentifier) async throws -> URL? {
        // Check if tokenizer files exist alongside the model
        let modelURL = await downloader.localModelURL(for: identifier)
        let modelDir = modelURL.deletingLastPathComponent()
        
        // Look for tokenizer files
        let vocabURL = modelDir.appendingPathComponent("vocab.txt")
        let tokenizerConfigURL = modelDir.appendingPathComponent("tokenizer_config.json")
        
        if FileManager.default.fileExists(atPath: vocabURL.path) ||
           FileManager.default.fileExists(atPath: tokenizerConfigURL.path) {
            return modelDir
        }
        
        // Try to download tokenizer files from registry
        // This would require extending the registry to support tokenizer files
        // For now, return nil
        return nil
    }
    
    private func createTokenizerForModelType(
        _ modelType: String,
        maxLength: Int,
        vocabSize: Int
    ) async -> (any Tokenizer)? {
        let config = TokenizerConfiguration(
            maxSequenceLength: maxLength,
            vocabularySize: vocabSize
        )
        
        switch modelType.lowercased() {
        case "bert", "sentence-transformer":
            // Try to load BERT tokenizer from bundled vocabulary
            if let vocabURL = Bundle.main.url(forResource: "bert-base-uncased-vocab", withExtension: "txt") {
                return try? await TokenizerFactory.createTokenizer(
                    type: .bert,
                    configuration: config,
                    vocabularyPath: vocabURL.path
                )
            }
            
        case "gpt", "gpt2":
            // For GPT models, we'd need BPE tokenizer
            logger.info("GPT tokenizer not yet implemented, using simple tokenizer")
            
        default:
            logger.info("Unknown model type '\(modelType)', using simple tokenizer")
        }
        
        // Fallback to simple tokenizer
        return SimpleTokenizer(
            maxSequenceLength: config.maxSequenceLength,
            vocabularySize: config.vocabularySize
        )
    }
}

// MARK: - Supporting Types

public struct DownloadedModelInfo: Sendable {
    public let identifier: ModelIdentifier
    public let location: ModelLocation
    public let size: Int64?
    public let lastModified: Date?
    
    public enum ModelLocation: Sendable {
        case bundled
        case downloaded
        case remote
    }
}

// MARK: - Download Progress Delegate

public final class ConsoleDownloadDelegate: ModelDownloader.DownloadDelegate, @unchecked Sendable {
    private let logger = Logger(subsystem: "EmbedKit", category: "Download")
    
    public init() {}
    
    public func downloadDidStart(url: URL) {
        logger.info("Starting download: \(url.lastPathComponent)")
    }
    
    public func downloadDidProgress(bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpected: Int64) {
        let progress = Double(totalBytesWritten) / Double(totalBytesExpected) * 100
        logger.info("Download progress: \(String(format: "%.1f", progress))% (\(totalBytesWritten)/\(totalBytesExpected) bytes)")
    }
    
    public func downloadDidComplete(url: URL, localURL: URL) {
        logger.info("Download complete: \(localURL.lastPathComponent)")
    }
    
    public func downloadDidFail(url: URL, error: Error) {
        logger.error("Download failed: \(error)")
    }
}
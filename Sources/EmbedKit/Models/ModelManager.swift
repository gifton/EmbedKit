//
//  ModelManager.swift
//  EmbedKit
//
//  Main orchestrator for model lifecycle management
//

import Foundation

/// Errors that can occur during model management
public enum ModelManagerError: LocalizedError {
    case modelNotFound(String)
    case downloadFailed(String, Error)
    case invalidModelFormat(String)
    case loadingFailed(String, Error)
    case networkUnavailable
    case conversionRequired(String)
    case unsupportedPlatform

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let identifier):
            return "Model not found: \(identifier)"
        case .downloadFailed(let identifier, let error):
            return "Failed to download model \(identifier): \(error.localizedDescription)"
        case .invalidModelFormat(let details):
            return "Invalid model format: \(details)"
        case .loadingFailed(let identifier, let error):
            return "Failed to load model \(identifier): \(error.localizedDescription)"
        case .networkUnavailable:
            return "Network connection unavailable"
        case .conversionRequired(let format):
            return "Model requires conversion from \(format) format"
        case .unsupportedPlatform:
            return "Current platform not supported for this model"
        }
    }
}

/// Download progress information
public struct DownloadProgress: Sendable {
    public let bytesDownloaded: Int64
    public let totalBytes: Int64
    public let percentComplete: Double
    public let estimatedTimeRemaining: TimeInterval?

    public var formattedProgress: String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        let downloaded = formatter.string(fromByteCount: bytesDownloaded)
        let total = formatter.string(fromByteCount: totalBytes)
        return "\(downloaded) / \(total) (\(Int(percentComplete * 100))%)"
    }
}

/// Model manager for loading, downloading, and managing embedding models
public actor ModelManager {
    private let cache: ModelCache
    private let registry: ModelRegistry
    private let downloader: ModelDownloader
    private var loadedModels: [String: (backend: any ModelBackend, info: EmbeddingModelInfo)] = [:]
    private let configuration: Configuration

    public struct Configuration: Sendable {
        public let autoDownload: Bool
        public let preferredBackend: BackendType
        public let maxConcurrentDownloads: Int
        public let cacheDirectory: URL?
        public let maxCacheSizeMB: Int

        public enum BackendType: String, Sendable {
            case coreML
            case mps
            case auto
        }

        public init(
            autoDownload: Bool = true,
            preferredBackend: BackendType = .auto,
            maxConcurrentDownloads: Int = 2,
            cacheDirectory: URL? = nil,
            maxCacheSizeMB: Int = 5000
        ) {
            self.autoDownload = autoDownload
            self.preferredBackend = preferredBackend
            self.maxConcurrentDownloads = maxConcurrentDownloads
            self.cacheDirectory = cacheDirectory
            self.maxCacheSizeMB = maxCacheSizeMB
        }
    }

    /// Initialize model manager
    public init(configuration: Configuration = Configuration()) async throws {
        self.configuration = configuration
        self.registry = ModelRegistry()
        self.cache = try await ModelCache(
            cacheDirectory: configuration.cacheDirectory,
            maxCacheSizeMB: configuration.maxCacheSizeMB
        )
        self.downloader = ModelDownloader(
            maxConcurrentDownloads: configuration.maxConcurrentDownloads
        )
    }

    // MARK: - Model Loading

    /// Load a model by identifier
    public func loadModel(_ identifier: String) async throws -> any ModelBackend {
        // Check if already loaded
        if let loaded = loadedModels[identifier] {
            return loaded.backend
        }

        // Get model info from registry
        guard let modelInfo = await registry.getModel(identifier) else {
            throw ModelManagerError.modelNotFound(identifier)
        }

        // Get or download model
        let modelPath = try await getModelPath(for: identifier, info: modelInfo)

        // Create appropriate backend
        let backend = try await createBackend(for: modelInfo, at: modelPath)

        // Load the model
        try await backend.loadModel(from: modelPath)

        // Cache loaded model
        loadedModels[identifier] = (backend, modelInfo)

        return backend
    }

    /// Load a predefined model
    public func loadModel(_ model: PretrainedModel) async throws -> any ModelBackend {
        return try await loadModel(model.rawValue)
    }

    /// Get or create an embedding pipeline for a model
    public func getPipeline(
        for model: PretrainedModel,
        configuration: EmbeddingPipelineConfiguration = EmbeddingPipelineConfiguration()
    ) async throws -> EmbeddingPipeline {
        let backend = try await loadModel(model)
        let modelInfo = await registry.getModel(model)

        // Select appropriate tokenizer based on model type
        let tokenizer: any Tokenizer
        switch modelInfo.modelType {
        case .bert, .distilbert:
            tokenizer = try await BERTTokenizer(
                maxSequenceLength: modelInfo.maxSequenceLength
            )
        case .sentenceTransformer, .mpnet, .miniLM:
            tokenizer = try await BERTTokenizer(
                maxSequenceLength: modelInfo.maxSequenceLength
            )
        case .custom:
            tokenizer = SimpleTokenizer(
                maxSequenceLength: modelInfo.maxSequenceLength,
                vocabularySize: 30522  // Default BERT vocab size
            )
        }

        return EmbeddingPipeline(
            tokenizer: tokenizer,
            backend: backend,
            configuration: configuration
        )
    }

    /// Check if a model is loaded
    public func isLoaded(_ identifier: String) -> Bool {
        return loadedModels[identifier] != nil
    }

    /// Unload a model to free memory
    public func unloadModel(_ identifier: String) {
        loadedModels.removeValue(forKey: identifier)
    }

    // MARK: - Model Management

    /// Download a model
    public func downloadModel(
        _ identifier: String,
        progressHandler: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws -> URL {
        guard let modelInfo = await registry.getModel(identifier) else {
            throw ModelManagerError.modelNotFound(identifier)
        }

        guard let downloadURL = modelInfo.downloadURL else {
            throw ModelManagerError.downloadFailed(
                identifier,
                NSError(domain: "EmbedKit", code: 1, userInfo: [
                    NSLocalizedDescriptionKey: "No download URL available"
                ])
            )
        }

        // Download model
        let localPath = try await downloader.download(
            from: downloadURL,
            identifier: identifier,
            progressHandler: progressHandler
        )

        // Store in cache
        let cachedPath = try await cache.storeModel(
            identifier: identifier,
            from: localPath,
            modelInfo: modelInfo
        )

        // Clean up temporary file
        try? FileManager.default.removeItem(at: localPath)

        return cachedPath
    }

    /// List available models
    public func listAvailableModels() async -> [EmbeddingModelInfo] {
        return await registry.listModels()
    }

    /// List cached models
    public func listCachedModels() async -> [CacheEntry] {
        return await cache.listCachedModels()
    }

    /// Get cache statistics
    public func getCacheStatistics() async -> ModelCache.CacheStatistics {
        return await cache.getStatistics()
    }

    /// Clear model cache
    public func clearCache() async throws {
        // Unload all models first
        loadedModels.removeAll()

        // Clear cache
        try await cache.clear()
    }

    /// Verify model integrity
    public func verifyModel(_ identifier: String) async throws -> Bool {
        return try await cache.verifyModel(identifier)
    }

    // MARK: - Private Methods

    /// Get model path, downloading if necessary
    private func getModelPath(for identifier: String, info: EmbeddingModelInfo) async throws -> URL {
        // Check cache first
        if await cache.isCached(identifier) {
            return try await cache.getModelPath(identifier)
        }

        // Check for local path
        if let localPath = info.localPath {
            let url = URL(fileURLWithPath: localPath)
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }

        // Download if auto-download enabled
        if configuration.autoDownload {
            return try await downloadModel(identifier)
        } else {
            throw ModelManagerError.modelNotFound(identifier)
        }
    }

    /// Create appropriate backend for model
    private func createBackend(for info: EmbeddingModelInfo, at url: URL) async throws -> any ModelBackend {
        switch configuration.preferredBackend {
        case .coreML:
            return CoreMLBackend()
        case .mps:
            // MPS backend not yet implemented
            return CoreMLBackend()
        case .auto:
            // Auto-select based on model format and platform
            #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            return CoreMLBackend()
            #else
            throw ModelManagerError.unsupportedPlatform
            #endif
        }
    }
}

/// Model downloader for fetching models from remote sources
public actor ModelDownloader {
    private let session: URLSession
    private let maxConcurrentDownloads: Int
    private var activeDownloads: Set<String> = []

    init(maxConcurrentDownloads: Int = 2) {
        self.maxConcurrentDownloads = maxConcurrentDownloads

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 3600  // 1 hour for large models
        self.session = URLSession(configuration: config)
    }

    /// Download a model from URL
    func download(
        from url: URL,
        identifier: String,
        progressHandler: ((DownloadProgress) -> Void)? = nil
    ) async throws -> URL {
        // Wait if too many concurrent downloads
        while activeDownloads.count >= maxConcurrentDownloads {
            try await Task.sleep(nanoseconds: 100_000_000)  // 0.1 seconds
        }

        activeDownloads.insert(identifier)
        defer { activeDownloads.remove(identifier) }

        // Create temporary file
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("\(identifier).download")

        // Download with progress
        let (localURL, response) = try await session.download(from: url)

        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw ModelManagerError.downloadFailed(
                identifier,
                NSError(domain: "EmbedKit", code: 2, userInfo: [
                    NSLocalizedDescriptionKey: "Invalid response from server"
                ])
            )
        }

        // Move to final location
        try FileManager.default.moveItem(at: localURL, to: tempFile)

        return tempFile
    }
}
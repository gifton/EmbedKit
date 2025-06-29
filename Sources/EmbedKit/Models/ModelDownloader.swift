import Foundation
import OSLog
import CryptoKit

/// Handles downloading embedding models from remote sources
public actor ModelDownloader {
    private let logger = Logger(subsystem: "EmbedKit", category: "ModelDownloader")
    
    /// Download progress delegate
    public protocol DownloadDelegate: AnyObject, Sendable {
        func downloadDidStart(url: URL)
        func downloadDidProgress(bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpected: Int64)
        func downloadDidComplete(url: URL, localURL: URL)
        func downloadDidFail(url: URL, error: Error)
    }
    
    /// Model download configuration
    public struct DownloadConfiguration: Sendable {
        /// Maximum number of retry attempts
        public let maxRetries: Int
        
        /// Timeout interval for download
        public let timeoutInterval: TimeInterval
        
        /// Whether to verify checksums
        public let verifyChecksum: Bool
        
        /// Custom URLSession configuration
        public let sessionConfiguration: URLSessionConfiguration
        
        public init(
            maxRetries: Int = 3,
            timeoutInterval: TimeInterval = 300,
            verifyChecksum: Bool = true,
            sessionConfiguration: URLSessionConfiguration = .default
        ) {
            self.maxRetries = maxRetries
            self.timeoutInterval = timeoutInterval
            self.verifyChecksum = verifyChecksum
            self.sessionConfiguration = sessionConfiguration
        }
    }
    
    /// Model source configuration
    public struct ModelSource: Sendable {
        public let url: URL
        public let expectedChecksum: String?
        public let expectedSize: Int64?
        public let modelIdentifier: ModelIdentifier
        
        public init(
            url: URL,
            expectedChecksum: String? = nil,
            expectedSize: Int64? = nil,
            modelIdentifier: ModelIdentifier
        ) {
            self.url = url
            self.expectedChecksum = expectedChecksum
            self.expectedSize = expectedSize
            self.modelIdentifier = modelIdentifier
        }
    }
    
    private let configuration: DownloadConfiguration
    private var activeTasks: [URL: URLSessionDownloadTask] = [:]
    private var downloadProgress: [URL: Progress] = [:]
    private weak var delegate: (any DownloadDelegate)?
    
    private let documentsDirectory: URL = {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    }()
    
    private let modelsDirectory: URL
    
    public init(
        configuration: DownloadConfiguration = DownloadConfiguration(),
        delegate: (any DownloadDelegate)? = nil
    ) {
        self.configuration = configuration
        self.delegate = delegate
        
        // Create models directory
        self.modelsDirectory = documentsDirectory.appendingPathComponent("EmbedKitModels")
        try? FileManager.default.createDirectory(
            at: modelsDirectory,
            withIntermediateDirectories: true
        )
    }
    
    /// Download a model from a remote source
    public func downloadModel(from source: ModelSource) async throws -> URL {
        logger.info("Starting model download: \(source.modelIdentifier.rawValue)")
        
        // Check if already downloaded
        let localURL = localModelURL(for: source.modelIdentifier)
        if FileManager.default.fileExists(atPath: localURL.path) {
            logger.info("Model already exists locally: \(localURL.path)")
            
            // Verify checksum if required
            if configuration.verifyChecksum, let expectedChecksum = source.expectedChecksum {
                let actualChecksum = try await calculateChecksum(for: localURL)
                guard actualChecksum == expectedChecksum else {
                    logger.error("Checksum mismatch for existing model. Deleting and re-downloading.")
                    try FileManager.default.removeItem(at: localURL)
                    return try await performDownload(from: source)
                }
            }
            
            return localURL
        }
        
        // Perform download
        return try await performDownload(from: source)
    }
    
    /// Cancel a download in progress
    public func cancelDownload(for url: URL) async {
        if let task = activeTasks[url] {
            task.cancel()
            activeTasks.removeValue(forKey: url)
            downloadProgress.removeValue(forKey: url)
            logger.info("Cancelled download: \(url)")
        }
    }
    
    /// Get download progress for a URL
    public func progress(for url: URL) async -> Progress? {
        downloadProgress[url]
    }
    
    /// Get local URL for a model
    public func localModelURL(for identifier: ModelIdentifier) -> URL {
        modelsDirectory
            .appendingPathComponent(identifier.family)
            .appendingPathComponent(identifier.variant ?? "default")
            .appendingPathComponent("\(identifier.version ?? "1").mlmodelc")
    }
    
    /// List all downloaded models
    public func listDownloadedModels() async throws -> [ModelIdentifier] {
        var models: [ModelIdentifier] = []
        
        let enumerator = FileManager.default.enumerator(
            at: modelsDirectory,
            includingPropertiesForKeys: [.isDirectoryKey]
        )
        
        while let url = enumerator?.nextObject() as? URL {
            if url.pathExtension == "mlmodelc" || url.pathExtension == "mlpackage" {
                // Extract model identifier from path
                let components = url.pathComponents
                if components.count >= 3 {
                    let family = components[components.count - 3]
                    let variant = components[components.count - 2]
                    let version = url.deletingPathExtension().lastPathComponent
                    
                    let identifier = ModelIdentifier(
                        family: family,
                        variant: variant == "default" ? nil : variant,
                        version: version
                    )
                    models.append(identifier)
                }
            }
        }
        
        return models
    }
    
    /// Delete a downloaded model
    public func deleteModel(_ identifier: ModelIdentifier) async throws {
        let localURL = localModelURL(for: identifier)
        if FileManager.default.fileExists(atPath: localURL.path) {
            try FileManager.default.removeItem(at: localURL)
            logger.info("Deleted model: \(identifier.rawValue)")
        }
    }
    
    // MARK: - Private Methods
    
    private func performDownload(from source: ModelSource) async throws -> URL {
        let localURL = localModelURL(for: source.modelIdentifier)
        
        // Create parent directory
        try FileManager.default.createDirectory(
            at: localURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        
        // Create URLSession with custom configuration
        let session = URLSession(configuration: configuration.sessionConfiguration)
        
        var lastError: Error?
        for attempt in 1...configuration.maxRetries {
            do {
                logger.info("Download attempt \(attempt) for \(source.modelIdentifier.rawValue)")
                
                // Create download task
                let (tempURL, response) = try await session.download(from: source.url)
                
                // Verify response
                guard let httpResponse = response as? HTTPURLResponse,
                      httpResponse.statusCode == 200 else {
                    throw ContextualEmbeddingError.networkError(
                        context: ErrorContext(
                            operation: .modelLoading,
                            modelIdentifier: source.modelIdentifier,
                            metadata: ErrorMetadata()
                                .with(key: "statusCode", value: String((response as? HTTPURLResponse)?.statusCode ?? -1)),
                            sourceLocation: SourceLocation()
                        ),
                        statusCode: (response as? HTTPURLResponse)?.statusCode ?? -1
                    )
                }
                
                // Verify size if provided
                if let expectedSize = source.expectedSize {
                    let attributes = try FileManager.default.attributesOfItem(atPath: tempURL.path)
                    let actualSize = attributes[.size] as? Int64 ?? 0
                    
                    guard actualSize == expectedSize else {
                        throw ContextualEmbeddingError.validationFailed(
                            context: ErrorContext(
                                operation: .modelLoading,
                                modelIdentifier: source.modelIdentifier,
                                metadata: ErrorMetadata()
                                    .with(key: "expectedSize", value: String(expectedSize))
                                    .with(key: "actualSize", value: String(actualSize)),
                                sourceLocation: SourceLocation()
                            ),
                            reason: "Downloaded file size mismatch"
                        )
                    }
                }
                
                // Verify checksum if provided
                if configuration.verifyChecksum, let expectedChecksum = source.expectedChecksum {
                    let actualChecksum = try await calculateChecksum(for: tempURL)
                    guard actualChecksum == expectedChecksum else {
                        throw ContextualEmbeddingError.validationFailed(
                            context: ErrorContext(
                                operation: .modelLoading,
                                modelIdentifier: source.modelIdentifier,
                                metadata: ErrorMetadata()
                                    .with(key: "expectedChecksum", value: expectedChecksum)
                                    .with(key: "actualChecksum", value: actualChecksum),
                                sourceLocation: SourceLocation()
                            ),
                            reason: "Downloaded file checksum mismatch"
                        )
                    }
                }
                
                // Move to final location
                try FileManager.default.moveItem(at: tempURL, to: localURL)
                
                logger.info("Successfully downloaded model to: \(localURL.path)")
                delegate?.downloadDidComplete(url: source.url, localURL: localURL)
                
                return localURL
                
            } catch {
                lastError = error
                logger.error("Download attempt \(attempt) failed: \(error)")
                
                if attempt < configuration.maxRetries {
                    // Exponential backoff
                    let delay = Double(attempt) * 2.0
                    try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                }
            }
        }
        
        // All attempts failed
        let finalError = lastError ?? ContextualEmbeddingError.resourceUnavailable(
            context: ErrorContext(
                operation: .modelLoading,
                modelIdentifier: source.modelIdentifier,
                sourceLocation: SourceLocation()
            ),
            resource: .model
        )
        
        delegate?.downloadDidFail(url: source.url, error: finalError)
        throw finalError
    }
    
    private func calculateChecksum(for url: URL) async throws -> String {
        let data = try Data(contentsOf: url)
        let digest = SHA256.hash(data: data)
        
        return digest.compactMap { String(format: "%02x", $0) }.joined()
    }
}


// MARK: - Model Registry Integration

public extension ModelDownloader {
    /// Download a model from a registry entry
    func downloadModel(from registry: any ModelDownloadRegistryProtocol, identifier: ModelIdentifier) async throws -> URL {
        guard let entry = await registry.getModel(identifier) else {
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
        
        guard let downloadURL = entry.downloadURL else {
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
        
        let source = ModelSource(
            url: downloadURL,
            expectedChecksum: entry.checksum,
            expectedSize: entry.fileSize,
            modelIdentifier: identifier
        )
        
        return try await downloadModel(from: source)
    }
}

// MARK: - Model Registry Protocol

public protocol ModelDownloadRegistryProtocol: Sendable {
    func getModel(_ identifier: ModelIdentifier) async -> ModelRegistryEntry?
    func listModels() async -> [ModelRegistryEntry]
}

public struct ModelRegistryEntry: Sendable {
    public let identifier: ModelIdentifier
    public let downloadURL: URL?
    public let checksum: String?
    public let fileSize: Int64?
    public let metadata: [String: String]
    
    public init(
        identifier: ModelIdentifier,
        downloadURL: URL? = nil,
        checksum: String? = nil,
        fileSize: Int64? = nil,
        metadata: [String: String] = [:]
    ) {
        self.identifier = identifier
        self.downloadURL = downloadURL
        self.checksum = checksum
        self.fileSize = fileSize
        self.metadata = metadata
    }
}
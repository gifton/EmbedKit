// EmbedKit - Model Discovery
// Auto-discover local CoreML embedding models

import Foundation
import CoreML

// MARK: - Discovered Model

/// Information about a discovered local model.
public struct DiscoveredModel: Sendable, Identifiable {
    /// Unique identifier based on file path
    public var id: String { path.path }

    /// Path to the model file/package
    public let path: URL

    /// Model file type
    public let type: ModelFileType

    /// File size in bytes
    public let sizeBytes: Int64

    /// Last modification date
    public let modifiedDate: Date

    /// Model name derived from filename
    public let name: String

    /// Inferred model family (e.g., "sentence-transformers", "bert", "minilm")
    public let family: ModelFamily?

    /// Expected embedding dimensions (if detectable)
    public let inferredDimensions: Int?

    /// Whether the model appears to be an embedding model
    public let isLikelyEmbeddingModel: Bool

    public init(
        path: URL,
        type: ModelFileType,
        sizeBytes: Int64,
        modifiedDate: Date,
        name: String,
        family: ModelFamily?,
        inferredDimensions: Int?,
        isLikelyEmbeddingModel: Bool
    ) {
        self.path = path
        self.type = type
        self.sizeBytes = sizeBytes
        self.modifiedDate = modifiedDate
        self.name = name
        self.family = family
        self.inferredDimensions = inferredDimensions
        self.isLikelyEmbeddingModel = isLikelyEmbeddingModel
    }

    /// Human-readable size string
    public var formattedSize: String {
        ByteCountFormatter.string(fromByteCount: sizeBytes, countStyle: .file)
    }
}

// MARK: - Model File Type

/// Types of model files that can be discovered.
public enum ModelFileType: String, CaseIterable, Sendable {
    /// CoreML model package (.mlpackage)
    case mlpackage

    /// Compiled CoreML model (.mlmodel or .mlmodelc)
    case mlmodel

    /// ONNX model (.onnx)
    case onnx

    /// File extensions for this type
    public var extensions: [String] {
        switch self {
        case .mlpackage:
            return ["mlpackage"]
        case .mlmodel:
            return ["mlmodel", "mlmodelc"]
        case .onnx:
            return ["onnx"]
        }
    }

    /// Whether this model type requires the EmbedKitONNX module
    public var requiresONNXModule: Bool {
        self == .onnx
    }
}

// MARK: - Model Family

/// Known model families for embedding models.
public enum ModelFamily: String, CaseIterable, Sendable {
    case sentenceTransformers = "sentence-transformers"
    case bert = "bert"
    case miniLM = "minilm"
    case distilBERT = "distilbert"
    case roberta = "roberta"
    case mpnet = "mpnet"
    case e5 = "e5"
    case gte = "gte"
    case bge = "bge"
    case unknown

    /// Infer family from model name
    public static func infer(from name: String) -> ModelFamily? {
        let lowercased = name.lowercased()

        if lowercased.contains("minilm") || lowercased.contains("mini-lm") {
            return .miniLM
        }
        if lowercased.contains("distilbert") {
            return .distilBERT
        }
        if lowercased.contains("mpnet") {
            return .mpnet
        }
        if lowercased.contains("roberta") {
            return .roberta
        }
        if lowercased.contains("bert") {
            return .bert
        }
        if lowercased.contains("sentence-transformer") || lowercased.contains("sbert") {
            return .sentenceTransformers
        }
        if lowercased.contains("-e5-") || lowercased.hasPrefix("e5-") {
            return .e5
        }
        if lowercased.contains("-gte-") || lowercased.hasPrefix("gte-") {
            return .gte
        }
        if lowercased.contains("-bge-") || lowercased.hasPrefix("bge-") {
            return .bge
        }

        return nil
    }

    /// Common embedding dimensions for this family
    public var commonDimensions: [Int] {
        switch self {
        case .miniLM:
            return [384, 256, 128]
        case .bert, .distilBERT:
            return [768, 512]
        case .sentenceTransformers:
            return [384, 768, 512]
        case .mpnet:
            return [768]
        case .roberta:
            return [768, 1024]
        case .e5, .gte, .bge:
            return [384, 768, 1024]
        case .unknown:
            return []
        }
    }
}

// MARK: - Discovery Options

/// Options for model discovery.
public struct ModelDiscoveryOptions: Sendable {
    /// Whether to search recursively in directories
    public var recursive: Bool

    /// Maximum depth for recursive search (0 = unlimited)
    public var maxDepth: Int

    /// File types to search for
    public var fileTypes: Set<ModelFileType>

    /// Minimum file size to consider (filters out empty/stub models)
    public var minSizeBytes: Int64

    /// Whether to only return likely embedding models
    public var embeddingModelsOnly: Bool

    /// Default options
    public static let `default` = ModelDiscoveryOptions(
        recursive: true,
        maxDepth: 5,
        fileTypes: Set(ModelFileType.allCases),
        minSizeBytes: 1024,  // At least 1KB
        embeddingModelsOnly: false
    )

    /// Options for quick scan (non-recursive)
    public static let quick = ModelDiscoveryOptions(
        recursive: false,
        maxDepth: 1,
        fileTypes: Set(ModelFileType.allCases),
        minSizeBytes: 1024,
        embeddingModelsOnly: false
    )

    public init(
        recursive: Bool = true,
        maxDepth: Int = 5,
        fileTypes: Set<ModelFileType> = Set(ModelFileType.allCases),
        minSizeBytes: Int64 = 1024,
        embeddingModelsOnly: Bool = false
    ) {
        self.recursive = recursive
        self.maxDepth = maxDepth
        self.fileTypes = fileTypes
        self.minSizeBytes = minSizeBytes
        self.embeddingModelsOnly = embeddingModelsOnly
    }
}

// MARK: - Model Discovery

/// Discovers local CoreML models that may be used for embeddings.
///
/// Example:
/// ```swift
/// let discovery = ModelDiscovery()
///
/// // Scan common locations
/// let models = try await discovery.scanCommonLocations()
///
/// // Or scan specific directory
/// let customModels = try await discovery.scan(directory: myModelDir)
///
/// for model in models {
///     print("\(model.name): \(model.formattedSize)")
/// }
/// ```
public actor ModelDiscovery {

    private let fileManager: FileManager

    public init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
    }

    // MARK: - Scanning

    /// Scan a directory for models.
    ///
    /// - Parameters:
    ///   - directory: Directory to scan
    ///   - options: Discovery options
    /// - Returns: Array of discovered models
    public func scan(
        directory: URL,
        options: ModelDiscoveryOptions = .default
    ) async throws -> [DiscoveredModel] {
        var results: [DiscoveredModel] = []

        try await scanDirectory(
            directory,
            options: options,
            currentDepth: 0,
            results: &results
        )

        // Filter by embedding models if requested
        if options.embeddingModelsOnly {
            return results.filter { $0.isLikelyEmbeddingModel }
        }

        return results.sorted { $0.name < $1.name }
    }

    /// Scan common model locations.
    ///
    /// Scans:
    /// - Application bundle Resources
    /// - Application Support directory
    /// - Caches directory
    /// - Downloads directory
    ///
    /// - Parameter options: Discovery options
    /// - Returns: Array of discovered models
    public func scanCommonLocations(
        options: ModelDiscoveryOptions = .default
    ) async throws -> [DiscoveredModel] {
        var allModels: [DiscoveredModel] = []

        for location in commonModelLocations() {
            if fileManager.fileExists(atPath: location.path) {
                let models = try await scan(directory: location, options: options)
                allModels.append(contentsOf: models)
            }
        }

        // Remove duplicates by path
        var seen = Set<String>()
        let unique = allModels.filter { model in
            if seen.contains(model.id) {
                return false
            }
            seen.insert(model.id)
            return true
        }

        return unique.sorted { $0.name < $1.name }
    }

    /// Get common locations where models might be stored.
    public nonisolated func commonModelLocations() -> [URL] {
        let fm = FileManager.default
        var locations: [URL] = []

        // Application bundle
        if let resourceURL = Bundle.main.resourceURL {
            locations.append(resourceURL)
        }

        // Application Support
        if let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            locations.append(appSupport)
            locations.append(appSupport.appendingPathComponent("Models"))
            locations.append(appSupport.appendingPathComponent("EmbedKit"))
        }

        // Caches
        if let caches = fm.urls(for: .cachesDirectory, in: .userDomainMask).first {
            locations.append(caches)
            locations.append(caches.appendingPathComponent("Models"))
            locations.append(caches.appendingPathComponent("EmbedKit"))
        }

        // Downloads
        if let downloads = fm.urls(for: .downloadsDirectory, in: .userDomainMask).first {
            locations.append(downloads)
        }

        // Documents (iOS)
        #if os(iOS)
        if let documents = fm.urls(for: .documentDirectory, in: .userDomainMask).first {
            locations.append(documents)
        }
        #endif

        return locations
    }

    // MARK: - Single Model Analysis

    /// Analyze a single model file.
    ///
    /// - Parameter url: Path to the model file
    /// - Returns: Discovered model info, or nil if not a valid model
    public func analyze(modelAt url: URL) async -> DiscoveredModel? {
        guard let type = modelFileType(for: url) else {
            return nil
        }

        return await analyzeModel(at: url, type: type)
    }

    // MARK: - Private Implementation

    private func scanDirectory(
        _ directory: URL,
        options: ModelDiscoveryOptions,
        currentDepth: Int,
        results: inout [DiscoveredModel]
    ) async throws {
        // Check depth limit
        if options.maxDepth > 0 && currentDepth >= options.maxDepth {
            return
        }

        let contents: [URL]
        do {
            contents = try fileManager.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey, .contentModificationDateKey],
                options: [.skipsHiddenFiles]
            )
        } catch {
            // Directory not accessible, skip silently
            return
        }

        for url in contents {
            // Check if it's a model file
            if let type = modelFileType(for: url), options.fileTypes.contains(type) {
                if let model = await analyzeModel(at: url, type: type) {
                    if model.sizeBytes >= options.minSizeBytes {
                        results.append(model)
                    }
                }
                continue
            }

            // Recurse into directories
            if options.recursive {
                var isDirectory: ObjCBool = false
                if fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory),
                   isDirectory.boolValue {
                    try await scanDirectory(
                        url,
                        options: options,
                        currentDepth: currentDepth + 1,
                        results: &results
                    )
                }
            }
        }
    }

    private func modelFileType(for url: URL) -> ModelFileType? {
        let ext = url.pathExtension.lowercased()
        for type in ModelFileType.allCases {
            if type.extensions.contains(ext) {
                return type
            }
        }
        return nil
    }

    private func analyzeModel(at url: URL, type: ModelFileType) async -> DiscoveredModel? {
        // Get file attributes
        guard let attributes = try? fileManager.attributesOfItem(atPath: url.path) else {
            return nil
        }

        let size = (attributes[.size] as? Int64) ?? directorySize(at: url)
        let modified = (attributes[.modificationDate] as? Date) ?? Date()

        let name = url.deletingPathExtension().lastPathComponent
        let family = ModelFamily.infer(from: name)

        // Try to infer dimensions
        let dimensions = inferDimensions(name: name, family: family)

        // Determine if likely embedding model
        let isEmbedding = isLikelyEmbeddingModel(name: name, family: family)

        return DiscoveredModel(
            path: url,
            type: type,
            sizeBytes: size,
            modifiedDate: modified,
            name: name,
            family: family,
            inferredDimensions: dimensions,
            isLikelyEmbeddingModel: isEmbedding
        )
    }

    private func directorySize(at url: URL) -> Int64 {
        guard let enumerator = fileManager.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return 0
        }

        var size: Int64 = 0
        for case let fileURL as URL in enumerator {
            if let fileSize = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                size += Int64(fileSize)
            }
        }
        return size
    }

    private func inferDimensions(name: String, family: ModelFamily?) -> Int? {
        let lowercased = name.lowercased()

        // Check for explicit dimension in name
        let dimensionPatterns = [
            "384", "256", "128", "768", "512", "1024"
        ]

        for dim in dimensionPatterns {
            if lowercased.contains("-\(dim)") || lowercased.contains("_\(dim)") ||
               lowercased.hasSuffix(dim) {
                return Int(dim)
            }
        }

        // Check for L6, L12, etc. (layer count indicators for MiniLM)
        if lowercased.contains("l6") || lowercased.contains("-l6-") {
            return 384
        }
        if lowercased.contains("l12") || lowercased.contains("-l12-") {
            return 384
        }

        // Fall back to family defaults
        return family?.commonDimensions.first
    }

    private func isLikelyEmbeddingModel(name: String, family: ModelFamily?) -> Bool {
        let lowercased = name.lowercased()

        // Explicit embedding indicators
        let embeddingKeywords = [
            "embed", "sentence", "sbert", "transformer",
            "minilm", "mpnet", "e5-", "gte-", "bge-",
            "all-minilm", "paraphrase", "multi-qa"
        ]

        for keyword in embeddingKeywords {
            if lowercased.contains(keyword) {
                return true
            }
        }

        // Known embedding families
        if let family = family {
            switch family {
            case .miniLM, .sentenceTransformers, .mpnet, .e5, .gte, .bge:
                return true
            case .bert, .distilBERT, .roberta:
                // These could be for other tasks, need more indicators
                return lowercased.contains("embed") ||
                       lowercased.contains("sentence") ||
                       lowercased.contains("encoder")
            case .unknown:
                return false
            }
        }

        return false
    }
}

// MARK: - Convenience Extension

public extension ModelDiscovery {
    /// Find models matching a name pattern.
    ///
    /// - Parameters:
    ///   - pattern: Name pattern to match (case-insensitive contains)
    ///   - in: Directory to search, or nil for common locations
    /// - Returns: Matching models
    func findModels(
        matching pattern: String,
        in directory: URL? = nil
    ) async throws -> [DiscoveredModel] {
        let models: [DiscoveredModel]

        if let dir = directory {
            models = try await scan(directory: dir)
        } else {
            models = try await scanCommonLocations()
        }

        let lowercasedPattern = pattern.lowercased()
        return models.filter { $0.name.lowercased().contains(lowercasedPattern) }
    }

    /// Get the best match for a model family.
    ///
    /// - Parameter family: Model family to find
    /// - Returns: Best matching model, or nil
    func findModel(family: ModelFamily) async throws -> DiscoveredModel? {
        let models = try await scanCommonLocations(options: .init(embeddingModelsOnly: true))
        return models.first { $0.family == family }
    }
}

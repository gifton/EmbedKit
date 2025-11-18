//
//  ModelCache.swift
//  EmbedKit
//
//  Persistent caching system for downloaded models
//

import Foundation
import CryptoKit

/// Errors that can occur during cache operations
public enum ModelCacheError: LocalizedError {
    case invalidCacheDirectory
    case modelNotFound(String)
    case corruptedModel(String)
    case insufficientSpace(required: Int64, available: Int64)
    case checksumMismatch(expected: String, actual: String)
    case cacheLocked

    public var errorDescription: String? {
        switch self {
        case .invalidCacheDirectory:
            return "Invalid or inaccessible cache directory"
        case .modelNotFound(let identifier):
            return "Model not found in cache: \(identifier)"
        case .corruptedModel(let identifier):
            return "Cached model is corrupted: \(identifier)"
        case .insufficientSpace(let required, let available):
            return "Insufficient space. Required: \(required) bytes, Available: \(available) bytes"
        case .checksumMismatch(let expected, let actual):
            return "Checksum mismatch. Expected: \(expected), Actual: \(actual)"
        case .cacheLocked:
            return "Cache is locked by another operation"
        }
    }
}

/// Cache entry metadata
public struct CacheEntry: Codable, Sendable {
    public let identifier: String
    public let modelPath: URL
    public let metadataPath: URL
    public let sizeBytes: Int64
    public let lastAccessed: Date
    public let downloadDate: Date
    public let checksum: String?
    public let version: String?
    public let modelInfo: EmbeddingModelInfo?

    public init(
        identifier: String,
        modelPath: URL,
        metadataPath: URL,
        sizeBytes: Int64,
        lastAccessed: Date,
        downloadDate: Date,
        checksum: String?,
        version: String?,
        modelInfo: EmbeddingModelInfo?
    ) {
        self.identifier = identifier
        self.modelPath = modelPath
        self.metadataPath = metadataPath
        self.sizeBytes = sizeBytes
        self.lastAccessed = lastAccessed
        self.downloadDate = downloadDate
        self.checksum = checksum
        self.version = version
        self.modelInfo = modelInfo
    }
}

/// Model cache for managing downloaded models
public actor ModelCache {
    private let cacheDirectory: URL
    private let maxCacheSizeBytes: Int64
    private var entries: [String: CacheEntry] = [:]
    private let fileManager = FileManager.default
    private var isLocked = false

    /// Cache statistics
    private var statistics = CacheStatistics()

    public struct CacheStatistics: Sendable {
        public var totalModels: Int = 0
        public var totalSizeBytes: Int64 = 0
        public var hits: Int = 0
        public var misses: Int = 0
        public var evictions: Int = 0

        public var hitRate: Double {
            let total = hits + misses
            return total > 0 ? Double(hits) / Double(total) : 0
        }

        public var totalSizeMB: Double {
            return Double(totalSizeBytes) / (1024 * 1024)
        }
    }

    /// Initialize cache with directory and size limit
    public init(
        cacheDirectory: URL? = nil,
        maxCacheSizeMB: Int = 5000  // 5GB default
    ) async throws {
        // Determine cache directory
        if let customDir = cacheDirectory {
            self.cacheDirectory = customDir
        } else {
            let documentsPath = FileManager.default.urls(
                for: .cachesDirectory,
                in: .userDomainMask
            ).first!
            self.cacheDirectory = documentsPath.appendingPathComponent("EmbedKit/Models")
        }

        self.maxCacheSizeBytes = Int64(maxCacheSizeMB) * 1024 * 1024

        // Create cache directory if needed
        try await createCacheDirectory()

        // Load existing cache entries
        try await loadCacheManifest()
    }

    // MARK: - Public API

    /// Check if a model is cached
    public func isCached(_ identifier: String) -> Bool {
        return entries[identifier] != nil
    }

    /// Get cached model path
    public func getModelPath(_ identifier: String) async throws -> URL {
        guard let entry = entries[identifier] else {
            statistics.misses += 1
            throw ModelCacheError.modelNotFound(identifier)
        }

        // Update last accessed time
        let refreshedEntry = CacheEntry(
            identifier: entry.identifier,
            modelPath: entry.modelPath,
            metadataPath: entry.metadataPath,
            sizeBytes: entry.sizeBytes,
            lastAccessed: Date(),
            downloadDate: entry.downloadDate,
            checksum: entry.checksum,
            version: entry.version,
            modelInfo: entry.modelInfo
        )
        entries[identifier] = refreshedEntry

        statistics.hits += 1
        return entry.modelPath
    }

    /// Store a model in cache
    public func storeModel(
        identifier: String,
        from sourceURL: URL,
        modelInfo: EmbeddingModelInfo? = nil,
        version: String? = nil
    ) async throws -> URL {
        guard !isLocked else {
            throw ModelCacheError.cacheLocked
        }

        isLocked = true
        defer { isLocked = false }

        // Check available space
        let fileSize = try getFileSize(at: sourceURL)
        let availableSpace = try await getAvailableSpace()

        if fileSize > availableSpace {
            // Try to evict old models
            try await evictModelsForSpace(fileSize)
        }

        // Create model directory
        let modelDir = cacheDirectory.appendingPathComponent(sanitize(identifier))
        try fileManager.createDirectory(at: modelDir, withIntermediateDirectories: true)

        // Copy model file
        let modelPath = modelDir.appendingPathComponent("model.mlmodel")
        if fileManager.fileExists(atPath: modelPath.path) {
            try fileManager.removeItem(at: modelPath)
        }
        try fileManager.copyItem(at: sourceURL, to: modelPath)

        // Calculate checksum
        let checksum = try await calculateChecksum(for: modelPath)

        // Save metadata
        let metadataPath = modelDir.appendingPathComponent("metadata.json")
        let entry = CacheEntry(
            identifier: identifier,
            modelPath: modelPath,
            metadataPath: metadataPath,
            sizeBytes: fileSize,
            lastAccessed: Date(),
            downloadDate: Date(),
            checksum: checksum,
            version: version,
            modelInfo: modelInfo
        )

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let metadataData = try encoder.encode(entry)
        try metadataData.write(to: metadataPath)

        // Update cache entries
        entries[identifier] = entry
        statistics.totalModels += 1
        statistics.totalSizeBytes += fileSize

        // Save manifest
        try await saveCacheManifest()

        return modelPath
    }

    /// Remove a model from cache
    public func removeModel(_ identifier: String) async throws {
        guard let entry = entries[identifier] else {
            throw ModelCacheError.modelNotFound(identifier)
        }

        // Remove files
        let modelDir = entry.modelPath.deletingLastPathComponent()
        try fileManager.removeItem(at: modelDir)

        // Update statistics
        statistics.totalModels -= 1
        statistics.totalSizeBytes -= entry.sizeBytes

        // Remove from entries
        entries.removeValue(forKey: identifier)

        // Save manifest
        try await saveCacheManifest()
    }

    /// Clear entire cache
    public func clear() async throws {
        guard !isLocked else {
            throw ModelCacheError.cacheLocked
        }

        isLocked = true
        defer { isLocked = false }

        // Remove all model directories
        for entry in entries.values {
            let modelDir = entry.modelPath.deletingLastPathComponent()
            try? fileManager.removeItem(at: modelDir)
        }

        // Reset state
        entries.removeAll()
        statistics = CacheStatistics()

        // Save empty manifest
        try await saveCacheManifest()
    }

    /// Get cache statistics
    public func getStatistics() -> CacheStatistics {
        return statistics
    }

    /// List all cached models
    public func listCachedModels() -> [CacheEntry] {
        return Array(entries.values).sorted { $0.lastAccessed > $1.lastAccessed }
    }

    /// Verify model integrity
    public func verifyModel(_ identifier: String) async throws -> Bool {
        guard let entry = entries[identifier] else {
            throw ModelCacheError.modelNotFound(identifier)
        }

        // Check file exists
        guard fileManager.fileExists(atPath: entry.modelPath.path) else {
            return false
        }

        // Verify checksum if available
        if let expectedChecksum = entry.checksum {
            let actualChecksum = try await calculateChecksum(for: entry.modelPath)
            return expectedChecksum == actualChecksum
        }

        return true
    }

    // MARK: - Private Methods

    private func createCacheDirectory() async throws {
        if !fileManager.fileExists(atPath: cacheDirectory.path) {
            try fileManager.createDirectory(
                at: cacheDirectory,
                withIntermediateDirectories: true,
                attributes: nil
            )
        }
    }

    private func loadCacheManifest() async throws {
        let manifestPath = cacheDirectory.appendingPathComponent("manifest.json")

        guard fileManager.fileExists(atPath: manifestPath.path) else {
            return  // No manifest yet
        }

        let data = try Data(contentsOf: manifestPath)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        let manifest = try decoder.decode([String: CacheEntry].self, from: data)
        self.entries = manifest

        // Update statistics
        statistics.totalModels = entries.count
        statistics.totalSizeBytes = entries.values.reduce(0) { $0 + $1.sizeBytes }
    }

    private func saveCacheManifest() async throws {
        let manifestPath = cacheDirectory.appendingPathComponent("manifest.json")

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = .prettyPrinted

        let data = try encoder.encode(entries)
        try data.write(to: manifestPath)
    }

    private func getFileSize(at url: URL) throws -> Int64 {
        let attributes = try fileManager.attributesOfItem(atPath: url.path)
        return attributes[.size] as? Int64 ?? 0
    }

    private func getAvailableSpace() async throws -> Int64 {
        let values = try cacheDirectory.resourceValues(forKeys: [.volumeAvailableCapacityKey])
        let availableBytes = values.volumeAvailableCapacity ?? 0
        let usedBytes = statistics.totalSizeBytes

        return min(
            Int64(availableBytes),
            maxCacheSizeBytes - usedBytes
        )
    }

    private func evictModelsForSpace(_ requiredBytes: Int64) async throws {
        // Sort by last accessed (LRU)
        let sortedEntries = entries.values.sorted { $0.lastAccessed < $1.lastAccessed }

        var freedBytes: Int64 = 0
        var toEvict: [String] = []

        for entry in sortedEntries {
            if freedBytes >= requiredBytes {
                break
            }

            toEvict.append(entry.identifier)
            freedBytes += entry.sizeBytes
        }

        // Evict models
        for identifier in toEvict {
            try await removeModel(identifier)
            statistics.evictions += 1
        }

        if freedBytes < requiredBytes {
            let available = try await getAvailableSpace()
            throw ModelCacheError.insufficientSpace(
                required: requiredBytes,
                available: available
            )
        }
    }

    private func calculateChecksum(for url: URL) async throws -> String {
        let data = try Data(contentsOf: url)
        let hash = SHA256.hash(data: data)
        return hash.compactMap { String(format: "%02x", $0) }.joined()
    }

    private func sanitize(_ string: String) -> String {
        // Replace invalid filename characters
        let invalidCharacters = CharacterSet(charactersIn: "/\\?%*|\"<>:")
        let components = string.components(separatedBy: invalidCharacters)
        return components.joined(separator: "_")
    }
}

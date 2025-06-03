import Foundation
import OSLog
import CryptoKit

/// Enhanced model version registry with persistent storage
public actor PersistentModelVersionRegistry {
    private let logger = Logger(subsystem: "EmbedKit", category: "PersistentModelVersionRegistry")
    
    private let storage: PersistentModelRegistry
    private var memoryCache: [String: [ModelVersion]] = [:]
    private var activeVersionCache: [String: ModelVersion] = [:]
    private let maxCacheSize = 100
    
    public init(storageDirectory: URL? = nil) async throws {
        self.storage = try await PersistentModelRegistry(storageDirectory: storageDirectory)
        await loadCacheFromStorage()
    }
    
    /// Register a new model version with persistent storage
    public func register(
        version: ModelVersion,
        modelURL: URL,
        signature: String? = nil,
        forceReload: Bool = false
    ) async throws {
        // Verify the model file exists
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw ModelVersionError.modelFileNotFound(modelURL.path)
        }
        
        // Save to persistent storage
        try await storage.saveVersion(version, fileURL: modelURL, signature: signature)
        
        // Update memory cache
        memoryCache[version.identifier, default: []].append(version)
        memoryCache[version.identifier]?.sort { $0.isNewer(than: $1) }
        
        // Limit cache size
        if memoryCache[version.identifier]?.count ?? 0 > maxCacheSize {
            memoryCache[version.identifier]?.removeLast()
        }
        
        // If this is the first version or newer than current active, make it active
        if activeVersionCache[version.identifier] == nil ||
           version.isNewer(than: activeVersionCache[version.identifier]!) {
            try await setActiveVersion(version)
        }
        
        // Integrate with telemetry
        await telemetry.recordEvent(TelemetryEvent(
            name: "model_version_registered",
            description: "Registered model version \(version.semanticVersion) for \(version.identifier)",
            severity: .info,
            metadata: [
                "model_id": version.identifier,
                "version": version.semanticVersion,
                "has_signature": String(signature != nil)
            ]
        ))
        
        logger.info("Registered model version \(version.semanticVersion) for \(version.identifier)")
    }
    
    /// Get the active version for a model
    public func getActiveVersion(for modelId: String) async -> ModelVersion? {
        // Check memory cache first
        if let cached = activeVersionCache[modelId] {
            return cached
        }
        
        // Load from storage
        do {
            if let record = try await storage.getActiveVersion(for: modelId) {
                activeVersionCache[modelId] = record.version
                return record.version
            }
        } catch {
            logger.error("Failed to load active version for \(modelId): \(error)")
        }
        
        return nil
    }
    
    /// Set the active version for a model
    public func setActiveVersion(_ version: ModelVersion) async throws {
        // Verify version exists in storage
        let versionRecords = try await storage.loadVersions(for: version.identifier)
        guard versionRecords.contains(where: { 
            $0.version.identifier == version.identifier &&
            $0.version.version == version.version &&
            $0.version.buildNumber == version.buildNumber
        }) else {
            throw ModelVersionError.versionNotFound(version.semanticVersion)
        }
        
        // Update storage
        try await storage.setActiveVersion(version)
        
        // Update memory cache
        activeVersionCache[version.identifier] = version
        
        logger.info("Set active version to \(version.semanticVersion) for \(version.identifier)")
    }
    
    /// Get all versions for a model
    public func getVersions(for modelId: String) async -> [ModelVersion] {
        // Check memory cache first
        if let cached = memoryCache[modelId], !cached.isEmpty {
            return cached
        }
        
        // Load from storage
        do {
            let records = try await storage.loadVersions(for: modelId)
            let versions = records.map { $0.version }
            
            // Update cache
            memoryCache[modelId] = versions
            
            return versions
        } catch {
            logger.error("Failed to load versions for \(modelId): \(error)")
            return []
        }
    }
    
    /// Get the model file URL for a version
    public func getModelURL(for version: ModelVersion) async -> URL? {
        do {
            let records = try await storage.loadVersions(for: version.identifier)
            if let record = records.first(where: { 
                $0.version.identifier == version.identifier &&
                $0.version.version == version.version &&
                $0.version.buildNumber == version.buildNumber
            }) {
                return URL(fileURLWithPath: record.filePath)
            }
        } catch {
            logger.error("Failed to get model URL for \(version.semanticVersion): \(error)")
        }
        
        return nil
    }
    
    /// Get detailed version record with metadata
    public func getVersionRecord(for version: ModelVersion) async -> ModelVersionRecord? {
        do {
            let records = try await storage.loadVersions(for: version.identifier)
            return records.first(where: { 
                $0.version.identifier == version.identifier &&
                $0.version.version == version.version &&
                $0.version.buildNumber == version.buildNumber
            })
        } catch {
            logger.error("Failed to get version record for \(version.semanticVersion): \(error)")
            return nil
        }
    }
    
    /// Remove a model version
    public func removeVersion(_ version: ModelVersion) async throws {
        // Remove from storage
        try await storage.removeVersion(version)
        
        // Update memory cache
        memoryCache[version.identifier]?.removeAll { 
            $0.identifier == version.identifier &&
            $0.version == version.version &&
            $0.buildNumber == version.buildNumber
        }
        
        // If this was the active version, choose the latest remaining version
        if let activeVersion = activeVersionCache[version.identifier],
           activeVersion.identifier == version.identifier &&
           activeVersion.version == version.version &&
           activeVersion.buildNumber == version.buildNumber {
            let remainingVersions = await getVersions(for: version.identifier)
            activeVersionCache[version.identifier] = remainingVersions.first
            
            // Update storage if there's a new active version
            if let newActive = remainingVersions.first {
                try await storage.setActiveVersion(newActive)
            }
        }
        
        logger.info("Removed model version \(version.semanticVersion) for \(version.identifier)")
    }
    
    /// Verify model integrity using stored checksums
    public func verifyModelIntegrity(for version: ModelVersion) async throws -> ModelIntegrityResult {
        guard let record = await getVersionRecord(for: version) else {
            throw ModelVersionError.versionNotFound(version.semanticVersion)
        }
        
        let fileURL = URL(fileURLWithPath: record.filePath)
        
        // Check if file exists
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            return ModelIntegrityResult(
                version: version,
                isValid: false,
                issues: [.fileNotFound],
                checkedAt: Date()
            )
        }
        
        var issues: [ModelIntegrityIssue] = []
        
        // Verify file size
        do {
            let attributes = try FileManager.default.attributesOfItem(atPath: fileURL.path)
            let currentSize = attributes[.size] as? Int64 ?? 0
            
            if currentSize != record.fileSize {
                issues.append(.sizeMismatch(expected: record.fileSize, actual: currentSize))
            }
        } catch {
            issues.append(.fileAccessError(error.localizedDescription))
        }
        
        // Verify checksum
        do {
            let data = try Data(contentsOf: fileURL)
            let currentChecksum = data.sha256
            
            if currentChecksum != record.checksum {
                issues.append(.checksumMismatch)
            }
        } catch {
            issues.append(.checksumCalculationFailed(error.localizedDescription))
        }
        
        // Verify signature if available
        if let storedSignature = record.signatureHash {
            do {
                let isValidSignature = try await verifyModelSignature(fileURL: fileURL, expectedSignature: storedSignature)
                if !isValidSignature {
                    issues.append(.signatureVerificationFailed)
                }
            } catch {
                issues.append(.signatureVerificationFailed)
            }
        }
        
        return ModelIntegrityResult(
            version: version,
            isValid: issues.isEmpty,
            issues: issues,
            checkedAt: Date()
        )
    }
    
    /// Get registry statistics
    public func getStatistics() async -> EnhancedRegistryStatistics {
        do {
            let storageStats = try await storage.getStatistics()
            let cacheStats = ModelCacheStatistics(
                cachedModels: memoryCache.count,
                cachedVersions: memoryCache.values.map { $0.count }.reduce(0, +),
                activeVersionsCached: activeVersionCache.count
            )
            
            return EnhancedRegistryStatistics(
                storage: storageStats,
                cache: cacheStats
            )
        } catch {
            logger.error("Failed to get statistics: \(error)")
            return EnhancedRegistryStatistics(
                storage: StorageStatistics(totalModels: 0, totalVersions: 0, totalFileSize: 0, databaseSize: 0),
                cache: ModelCacheStatistics(cachedModels: 0, cachedVersions: 0, activeVersionsCached: 0)
            )
        }
    }
    
    /// Cleanup orphaned records and optimize storage
    public func cleanup() async throws {
        try await storage.cleanup()
        
        // Clear and reload cache
        memoryCache.removeAll()
        activeVersionCache.removeAll()
        await loadCacheFromStorage()
        
        logger.info("Completed registry cleanup")
    }
    
    /// Export registry data for backup
    public func exportData() async throws -> RegistryExportData {
        let storageStats = try await storage.getStatistics()
        
        var modelData: [String: [ModelVersionRecord]] = [:]
        
        // Load all versions for all models
        // This is a simple approach - in production, you might want to paginate
        for modelId in Set(memoryCache.keys) {
            let records = try await storage.loadVersions(for: modelId)
            modelData[modelId] = records
        }
        
        return RegistryExportData(
            exportedAt: Date(),
            statistics: storageStats,
            modelData: modelData
        )
    }
    
    /// Health check for the registry
    public func performHealthCheck() async -> RegistryHealthStatus {
        var issues: [RegistryHealthIssue] = []
        
        // Check storage connectivity
        do {
            _ = try await storage.getStatistics()
        } catch {
            issues.append(.storageUnavailable(error.localizedDescription))
        }
        
        // Check for orphaned files
        let cachedModels = Array(memoryCache.keys)
        for modelId in cachedModels {
            let versions = await getVersions(for: modelId)
            for version in versions {
                if let url = await getModelURL(for: version) {
                    if !FileManager.default.fileExists(atPath: url.path) {
                        issues.append(.orphanedRecord(version.identifier, version.semanticVersion))
                    }
                }
            }
        }
        
        // Check cache consistency
        let cacheSize = memoryCache.values.map { $0.count }.reduce(0, +)
        if cacheSize > maxCacheSize * memoryCache.count {
            issues.append(.cacheOverflow)
        }
        
        return RegistryHealthStatus(
            isHealthy: issues.isEmpty,
            issues: issues,
            checkedAt: Date()
        )
    }
    
    // MARK: - Private Methods
    
    private func loadCacheFromStorage() async {
        // This is a simplified cache loading - in production you might want to be more selective
        do {
            let stats = try await storage.getStatistics()
            if stats.totalModels > 0 {
                logger.info("Loaded model registry with \(stats.totalModels) models, \(stats.totalVersions) versions")
            }
        } catch {
            logger.error("Failed to load cache from storage: \(error)")
        }
    }
    
    private func verifyModelSignature(fileURL: URL, expectedSignature: String) async throws -> Bool {
        // Compute SHA256 hash of the model file
        let fileHash = try await computeFileHash(fileURL: fileURL)
        
        // Compare with expected signature
        // Note: In a production system, you would verify a cryptographic signature
        // using public key cryptography (e.g., RSA or ECDSA signatures)
        // For now, we're using SHA256 hash comparison as a basic integrity check
        return fileHash == expectedSignature
    }
    
    private func computeFileHash(fileURL: URL) async throws -> String {
        return try await withCheckedThrowingContinuation { continuation in
            Task {
                do {
                    let fileData = try Data(contentsOf: fileURL)
                    let digest = SHA256.hash(data: fileData)
                    let hash = digest.compactMap { String(format: "%02x", $0) }.joined()
                    continuation.resume(returning: hash)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}

// MARK: - Supporting Types

public struct ModelIntegrityResult: Sendable {
    public let version: ModelVersion
    public let isValid: Bool
    public let issues: [ModelIntegrityIssue]
    public let checkedAt: Date
    
    public init(version: ModelVersion, isValid: Bool, issues: [ModelIntegrityIssue], checkedAt: Date) {
        self.version = version
        self.isValid = isValid
        self.issues = issues
        self.checkedAt = checkedAt
    }
}

public enum ModelIntegrityIssue: Sendable {
    case fileNotFound
    case sizeMismatch(expected: Int64, actual: Int64)
    case checksumMismatch
    case signatureVerificationFailed
    case fileAccessError(String)
    case checksumCalculationFailed(String)
}

public struct ModelCacheStatistics: Sendable {
    public let cachedModels: Int
    public let cachedVersions: Int
    public let activeVersionsCached: Int
    
    public init(cachedModels: Int, cachedVersions: Int, activeVersionsCached: Int) {
        self.cachedModels = cachedModels
        self.cachedVersions = cachedVersions
        self.activeVersionsCached = activeVersionsCached
    }
}

public struct EnhancedRegistryStatistics: Sendable {
    public let storage: StorageStatistics
    public let cache: ModelCacheStatistics
    
    public init(storage: StorageStatistics, cache: ModelCacheStatistics) {
        self.storage = storage
        self.cache = cache
    }
}

public struct RegistryExportData: Sendable {
    public let exportedAt: Date
    public let statistics: StorageStatistics
    public let modelData: [String: [ModelVersionRecord]]
    
    public init(exportedAt: Date, statistics: StorageStatistics, modelData: [String: [ModelVersionRecord]]) {
        self.exportedAt = exportedAt
        self.statistics = statistics
        self.modelData = modelData
    }
}

public struct RegistryHealthStatus: Sendable {
    public let isHealthy: Bool
    public let issues: [RegistryHealthIssue]
    public let checkedAt: Date
    
    public init(isHealthy: Bool, issues: [RegistryHealthIssue], checkedAt: Date) {
        self.isHealthy = isHealthy
        self.issues = issues
        self.checkedAt = checkedAt
    }
}

public enum RegistryHealthIssue: Sendable {
    case storageUnavailable(String)
    case orphanedRecord(String, String) // modelId, version
    case cacheOverflow
    case integrityCheckFailed(String)
}
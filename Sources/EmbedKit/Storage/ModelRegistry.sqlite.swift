import Foundation
import SQLite3
import OSLog

/// SQLite-based persistent storage for model registry data
public final class PersistentModelRegistry: @unchecked Sendable {
    private let logger = Logger(subsystem: "EmbedKit", category: "PersistentModelRegistry")
    
    private var db: OpaquePointer?
    private let dbPath: String
    private let queue = DispatchQueue(label: "model-registry-db", qos: .userInitiated)
    
    // SQL statements
    private let createVersionsTable = """
        CREATE TABLE IF NOT EXISTS model_versions (
            id TEXT PRIMARY KEY,
            identifier TEXT NOT NULL,
            version TEXT NOT NULL,
            build_number INTEGER NOT NULL,
            created_at REAL NOT NULL,
            metadata TEXT NOT NULL,
            file_path TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 0,
            signature_hash TEXT,
            file_size INTEGER NOT NULL DEFAULT 0,
            checksum TEXT
        );
    """
    
    private let createIndexes = """
        CREATE INDEX IF NOT EXISTS idx_model_identifier ON model_versions(identifier);
        CREATE INDEX IF NOT EXISTS idx_model_active ON model_versions(identifier, is_active);
        CREATE INDEX IF NOT EXISTS idx_created_at ON model_versions(created_at);
    """
    
    public init(storageDirectory: URL? = nil) async throws {
        // Determine storage location
        let baseURL: URL
        if let storageDirectory = storageDirectory {
            baseURL = storageDirectory
        } else {
            baseURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
                .appendingPathComponent("EmbedKit")
        }
        
        // Ensure directory exists
        try FileManager.default.createDirectory(at: baseURL, withIntermediateDirectories: true)
        
        // Set database path
        self.dbPath = baseURL.appendingPathComponent("model_registry.sqlite").path
        
        // Initialize database
        try await initializeDatabase()
    }
    
    deinit {
        if let db = db {
            sqlite3_close(db)
        }
    }
    
    private func initializeDatabase() async throws {
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    // Open database
                    if sqlite3_open(self.dbPath, &self.db) != SQLITE_OK {
                        let error = String(cString: sqlite3_errmsg(self.db))
                        sqlite3_close(self.db)
                        throw ModelRegistryError.databaseError("Failed to open database: \(error)")
                    }
                    
                    // Enable foreign keys and WAL mode
                    try self.executeSQL("PRAGMA foreign_keys = ON;")
                    try self.executeSQL("PRAGMA journal_mode = WAL;")
                    try self.executeSQL("PRAGMA synchronous = NORMAL;")
                    
                    // Create tables
                    try self.executeSQL(self.createVersionsTable)
                    try self.executeSQL(self.createIndexes)
                    
                    self.logger.info("Database initialized at: \(self.dbPath)")
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    private func executeSQL(_ sql: String) throws {
        guard let db = db else {
            throw ModelRegistryError.databaseError("Database not initialized")
        }
        
        if sqlite3_exec(db, sql, nil, nil, nil) != SQLITE_OK {
            let error = String(cString: sqlite3_errmsg(db))
            throw ModelRegistryError.databaseError("SQL execution failed: \(error)")
        }
    }
    
    /// Save a model version to persistent storage
    public func saveVersion(_ version: ModelVersion, fileURL: URL, signature: String? = nil) async throws {
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    guard let db = self.db else {
                        throw ModelRegistryError.databaseError("Database not initialized")
                    }
                    
                    // Get file attributes
                    let attributes = try FileManager.default.attributesOfItem(atPath: fileURL.path)
                    let fileSize = attributes[.size] as? Int64 ?? 0
                    
                    // Calculate checksum
                    let checksum = try self.calculateChecksum(for: fileURL)
                    
                    // Serialize metadata
                    let metadataData = try JSONSerialization.data(withJSONObject: version.metadata)
                    let metadataString = String(data: metadataData, encoding: .utf8) ?? "{}"
                    
                    // Create unique ID
                    let uniqueID = "\(version.identifier)_\(version.version)_\(version.buildNumber)"
                    
                    // Insert statement
                    let sql = """
                        INSERT OR REPLACE INTO model_versions 
                        (id, identifier, version, build_number, created_at, metadata, file_path, signature_hash, file_size, checksum)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """
                    
                    var statement: OpaquePointer?
                    if sqlite3_prepare_v2(db, sql, -1, &statement, nil) != SQLITE_OK {
                        let error = String(cString: sqlite3_errmsg(db))
                        throw ModelRegistryError.databaseError("Failed to prepare statement: \(error)")
                    }
                    
                    defer { sqlite3_finalize(statement) }
                    
                    // Bind parameters
                    sqlite3_bind_text(statement, 1, uniqueID, -1, nil)
                    sqlite3_bind_text(statement, 2, version.identifier, -1, nil)
                    sqlite3_bind_text(statement, 3, version.version, -1, nil)
                    sqlite3_bind_int64(statement, 4, Int64(version.buildNumber))
                    sqlite3_bind_double(statement, 5, version.createdAt.timeIntervalSince1970)
                    sqlite3_bind_text(statement, 6, metadataString, -1, nil)
                    sqlite3_bind_text(statement, 7, fileURL.path, -1, nil)
                    
                    if let signature = signature {
                        sqlite3_bind_text(statement, 8, signature, -1, nil)
                    } else {
                        sqlite3_bind_null(statement, 8)
                    }
                    
                    sqlite3_bind_int64(statement, 9, fileSize)
                    sqlite3_bind_text(statement, 10, checksum, -1, nil)
                    
                    // Execute
                    if sqlite3_step(statement) != SQLITE_DONE {
                        let error = String(cString: sqlite3_errmsg(db))
                        throw ModelRegistryError.databaseError("Failed to insert version: \(error)")
                    }
                    
                    self.logger.info("Saved model version \(version.identifier) v\(version.version) to database")
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    /// Load all versions for a model identifier
    public func loadVersions(for identifier: String) async throws -> [ModelVersionRecord] {
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    guard let db = self.db else {
                        throw ModelRegistryError.databaseError("Database not initialized")
                    }
                    
                    let sql = """
                        SELECT id, identifier, version, build_number, created_at, metadata, file_path, 
                               is_active, signature_hash, file_size, checksum
                        FROM model_versions 
                        WHERE identifier = ? 
                        ORDER BY created_at DESC;
                    """
                    
                    var statement: OpaquePointer?
                    if sqlite3_prepare_v2(db, sql, -1, &statement, nil) != SQLITE_OK {
                        let error = String(cString: sqlite3_errmsg(db))
                        throw ModelRegistryError.databaseError("Failed to prepare statement: \(error)")
                    }
                    
                    defer { sqlite3_finalize(statement) }
                    
                    sqlite3_bind_text(statement, 1, identifier, -1, nil)
                    
                    var records: [ModelVersionRecord] = []
                    
                    while sqlite3_step(statement) == SQLITE_ROW {
                        let record = try self.parseVersionRecord(from: statement)
                        records.append(record)
                    }
                    
                    continuation.resume(returning: records)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    /// Get the active version for a model
    public func getActiveVersion(for identifier: String) async throws -> ModelVersionRecord? {
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    guard let db = self.db else {
                        throw ModelRegistryError.databaseError("Database not initialized")
                    }
                    
                    let sql = """
                        SELECT id, identifier, version, build_number, created_at, metadata, file_path, 
                               is_active, signature_hash, file_size, checksum
                        FROM model_versions 
                        WHERE identifier = ? AND is_active = 1
                        LIMIT 1;
                    """
                    
                    var statement: OpaquePointer?
                    if sqlite3_prepare_v2(db, sql, -1, &statement, nil) != SQLITE_OK {
                        let error = String(cString: sqlite3_errmsg(db))
                        throw ModelRegistryError.databaseError("Failed to prepare statement: \(error)")
                    }
                    
                    defer { sqlite3_finalize(statement) }
                    
                    sqlite3_bind_text(statement, 1, identifier, -1, nil)
                    
                    if sqlite3_step(statement) == SQLITE_ROW {
                        let record = try self.parseVersionRecord(from: statement)
                        continuation.resume(returning: record)
                    } else {
                        continuation.resume(returning: nil)
                    }
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    /// Set the active version for a model
    public func setActiveVersion(_ version: ModelVersion) async throws {
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    guard let db = self.db else {
                        throw ModelRegistryError.databaseError("Database not initialized")
                    }
                    
                    // Start transaction
                    try self.executeSQL("BEGIN TRANSACTION;")
                    
                    // Clear existing active version
                    let clearSql = "UPDATE model_versions SET is_active = 0 WHERE identifier = ?;"
                    var clearStatement: OpaquePointer?
                    if sqlite3_prepare_v2(db, clearSql, -1, &clearStatement, nil) != SQLITE_OK {
                        try self.executeSQL("ROLLBACK;")
                        let error = String(cString: sqlite3_errmsg(db))
                        throw ModelRegistryError.databaseError("Failed to prepare clear statement: \(error)")
                    }
                    
                    sqlite3_bind_text(clearStatement, 1, version.identifier, -1, nil)
                    
                    if sqlite3_step(clearStatement) != SQLITE_DONE {
                        sqlite3_finalize(clearStatement)
                        try self.executeSQL("ROLLBACK;")
                        let error = String(cString: sqlite3_errmsg(db))
                        throw ModelRegistryError.databaseError("Failed to clear active versions: \(error)")
                    }
                    
                    sqlite3_finalize(clearStatement)
                    
                    // Set new active version
                    let setSql = """
                        UPDATE model_versions SET is_active = 1 
                        WHERE identifier = ? AND version = ? AND build_number = ?;
                    """
                    var setStatement: OpaquePointer?
                    if sqlite3_prepare_v2(db, setSql, -1, &setStatement, nil) != SQLITE_OK {
                        try self.executeSQL("ROLLBACK;")
                        let error = String(cString: sqlite3_errmsg(db))
                        throw ModelRegistryError.databaseError("Failed to prepare set statement: \(error)")
                    }
                    
                    sqlite3_bind_text(setStatement, 1, version.identifier, -1, nil)
                    sqlite3_bind_text(setStatement, 2, version.version, -1, nil)
                    sqlite3_bind_int64(setStatement, 3, Int64(version.buildNumber))
                    
                    if sqlite3_step(setStatement) != SQLITE_DONE {
                        sqlite3_finalize(setStatement)
                        try self.executeSQL("ROLLBACK;")
                        let error = String(cString: sqlite3_errmsg(db))
                        throw ModelRegistryError.databaseError("Failed to set active version: \(error)")
                    }
                    
                    sqlite3_finalize(setStatement)
                    
                    // Commit transaction
                    try self.executeSQL("COMMIT;")
                    
                    self.logger.info("Set active version to \(version.identifier) v\(version.version)")
                    continuation.resume()
                } catch {
                    // Rollback on error
                    try? self.executeSQL("ROLLBACK;")
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    /// Remove a version from storage
    public func removeVersion(_ version: ModelVersion) async throws {
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    guard let db = self.db else {
                        throw ModelRegistryError.databaseError("Database not initialized")
                    }
                    
                    let sql = """
                        DELETE FROM model_versions 
                        WHERE identifier = ? AND version = ? AND build_number = ?;
                    """
                    
                    var statement: OpaquePointer?
                    if sqlite3_prepare_v2(db, sql, -1, &statement, nil) != SQLITE_OK {
                        let error = String(cString: sqlite3_errmsg(db))
                        throw ModelRegistryError.databaseError("Failed to prepare statement: \(error)")
                    }
                    
                    defer { sqlite3_finalize(statement) }
                    
                    sqlite3_bind_text(statement, 1, version.identifier, -1, nil)
                    sqlite3_bind_text(statement, 2, version.version, -1, nil)
                    sqlite3_bind_int64(statement, 3, Int64(version.buildNumber))
                    
                    if sqlite3_step(statement) != SQLITE_DONE {
                        let error = String(cString: sqlite3_errmsg(db))
                        throw ModelRegistryError.databaseError("Failed to remove version: \(error)")
                    }
                    
                    self.logger.info("Removed model version \(version.identifier) v\(version.version)")
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    /// Get storage statistics
    public func getStatistics() async throws -> StorageStatistics {
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    guard let db = self.db else {
                        throw ModelRegistryError.databaseError("Database not initialized")
                    }
                    
                    // Count versions
                    let countSql = "SELECT COUNT(*) FROM model_versions;"
                    var countStatement: OpaquePointer?
                    if sqlite3_prepare_v2(db, countSql, -1, &countStatement, nil) != SQLITE_OK {
                        let error = String(cString: sqlite3_errmsg(db))
                        throw ModelRegistryError.databaseError("Failed to prepare count statement: \(error)")
                    }
                    
                    var totalVersions = 0
                    if sqlite3_step(countStatement) == SQLITE_ROW {
                        totalVersions = Int(sqlite3_column_int(countStatement, 0))
                    }
                    sqlite3_finalize(countStatement)
                    
                    // Count unique models
                    let modelsSql = "SELECT COUNT(DISTINCT identifier) FROM model_versions;"
                    var modelsStatement: OpaquePointer?
                    if sqlite3_prepare_v2(db, modelsSql, -1, &modelsStatement, nil) != SQLITE_OK {
                        let error = String(cString: sqlite3_errmsg(db))
                        throw ModelRegistryError.databaseError("Failed to prepare models statement: \(error)")
                    }
                    
                    var totalModels = 0
                    if sqlite3_step(modelsStatement) == SQLITE_ROW {
                        totalModels = Int(sqlite3_column_int(modelsStatement, 0))
                    }
                    sqlite3_finalize(modelsStatement)
                    
                    // Sum file sizes
                    let sizeSql = "SELECT SUM(file_size) FROM model_versions;"
                    var sizeStatement: OpaquePointer?
                    if sqlite3_prepare_v2(db, sizeSql, -1, &sizeStatement, nil) != SQLITE_OK {
                        let error = String(cString: sqlite3_errmsg(db))
                        throw ModelRegistryError.databaseError("Failed to prepare size statement: \(error)")
                    }
                    
                    var totalSize: Int64 = 0
                    if sqlite3_step(sizeStatement) == SQLITE_ROW {
                        totalSize = sqlite3_column_int64(sizeStatement, 0)
                    }
                    sqlite3_finalize(sizeStatement)
                    
                    // Get database file size
                    let dbAttributes = try FileManager.default.attributesOfItem(atPath: self.dbPath)
                    let dbSize = dbAttributes[.size] as? Int64 ?? 0
                    
                    let statistics = StorageStatistics(
                        totalModels: totalModels,
                        totalVersions: totalVersions,
                        totalFileSize: totalSize,
                        databaseSize: dbSize
                    )
                    
                    continuation.resume(returning: statistics)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    /// Cleanup orphaned records and optimize database
    public func cleanup() async throws {
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    guard let db = self.db else {
                        throw ModelRegistryError.databaseError("Database not initialized")
                    }
                    
                    // Remove records for files that no longer exist
                    let sql = "SELECT id, file_path FROM model_versions;"
                    var statement: OpaquePointer?
                    if sqlite3_prepare_v2(db, sql, -1, &statement, nil) != SQLITE_OK {
                        let error = String(cString: sqlite3_errmsg(db))
                        throw ModelRegistryError.databaseError("Failed to prepare statement: \(error)")
                    }
                    
                    var orphanedIds: [String] = []
                    
                    while sqlite3_step(statement) == SQLITE_ROW {
                        let id = String(cString: sqlite3_column_text(statement, 0))
                        let filePath = String(cString: sqlite3_column_text(statement, 1))
                        
                        if !FileManager.default.fileExists(atPath: filePath) {
                            orphanedIds.append(id)
                        }
                    }
                    sqlite3_finalize(statement)
                    
                    // Remove orphaned records
                    for orphanedId in orphanedIds {
                        let deleteSql = "DELETE FROM model_versions WHERE id = ?;"
                        var deleteStatement: OpaquePointer?
                        if sqlite3_prepare_v2(db, deleteSql, -1, &deleteStatement, nil) != SQLITE_OK {
                            continue
                        }
                        sqlite3_bind_text(deleteStatement, 1, orphanedId, -1, nil)
                        sqlite3_step(deleteStatement)
                        sqlite3_finalize(deleteStatement)
                    }
                    
                    // Vacuum database to reclaim space
                    try self.executeSQL("VACUUM;")
                    
                    self.logger.info("Cleaned up \(orphanedIds.count) orphaned records")
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    // MARK: - Helper Methods
    
    private func parseVersionRecord(from statement: OpaquePointer?) throws -> ModelVersionRecord {
        guard let statement = statement else {
            throw ModelRegistryError.databaseError("Invalid statement")
        }
        
        let id = String(cString: sqlite3_column_text(statement, 0))
        let identifier = String(cString: sqlite3_column_text(statement, 1))
        let version = String(cString: sqlite3_column_text(statement, 2))
        let buildNumber = Int(sqlite3_column_int(statement, 3))
        let createdAt = Date(timeIntervalSince1970: sqlite3_column_double(statement, 4))
        let metadataString = String(cString: sqlite3_column_text(statement, 5))
        let filePath = String(cString: sqlite3_column_text(statement, 6))
        let isActive = sqlite3_column_int(statement, 7) == 1
        
        let signatureHash: String?
        if sqlite3_column_type(statement, 8) != SQLITE_NULL {
            signatureHash = String(cString: sqlite3_column_text(statement, 8))
        } else {
            signatureHash = nil
        }
        
        let fileSize = sqlite3_column_int64(statement, 9)
        let checksum = String(cString: sqlite3_column_text(statement, 10))
        
        // Parse metadata
        let metadataData = metadataString.data(using: .utf8) ?? Data()
        let metadata = (try? JSONSerialization.jsonObject(with: metadataData) as? [String: String]) ?? [:]
        
        let modelVersion = ModelVersion(
            identifier: identifier,
            version: version,
            buildNumber: buildNumber,
            createdAt: createdAt,
            metadata: metadata
        )
        
        return ModelVersionRecord(
            id: id,
            version: modelVersion,
            filePath: filePath,
            isActive: isActive,
            signatureHash: signatureHash,
            fileSize: fileSize,
            checksum: checksum
        )
    }
    
    private func calculateChecksum(for url: URL) throws -> String {
        let data = try Data(contentsOf: url)
        return data.sha256
    }
}

// MARK: - Supporting Types

public struct ModelVersionRecord: Sendable {
    public let id: String
    public let version: ModelVersion
    public let filePath: String
    public let isActive: Bool
    public let signatureHash: String?
    public let fileSize: Int64
    public let checksum: String
    
    public init(id: String, version: ModelVersion, filePath: String, isActive: Bool, signatureHash: String?, fileSize: Int64, checksum: String) {
        self.id = id
        self.version = version
        self.filePath = filePath
        self.isActive = isActive
        self.signatureHash = signatureHash
        self.fileSize = fileSize
        self.checksum = checksum
    }
}

public struct StorageStatistics: Sendable {
    public let totalModels: Int
    public let totalVersions: Int
    public let totalFileSize: Int64
    public let databaseSize: Int64
    
    public init(totalModels: Int, totalVersions: Int, totalFileSize: Int64, databaseSize: Int64) {
        self.totalModels = totalModels
        self.totalVersions = totalVersions
        self.totalFileSize = totalFileSize
        self.databaseSize = databaseSize
    }
}

public enum ModelRegistryError: LocalizedError {
    case databaseError(String)
    case fileNotFound(String)
    case checksumMismatch
    case invalidData(String)
    
    public var errorDescription: String? {
        switch self {
        case .databaseError(let message):
            return "Database error: \(message)"
        case .fileNotFound(let path):
            return "File not found: \(path)"
        case .checksumMismatch:
            return "File checksum verification failed"
        case .invalidData(let message):
            return "Invalid data: \(message)"
        }
    }
}

// MARK: - Extensions

extension Data {
    var sha256: String {
        let hash = self.withUnsafeBytes { bytes in
            var hash = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
            CC_SHA256(bytes.bindMemory(to: UInt8.self).baseAddress, CC_LONG(self.count), &hash)
            return hash
        }
        return hash.map { String(format: "%02x", $0) }.joined()
    }
}

// Need to import CommonCrypto for SHA256
import CommonCrypto
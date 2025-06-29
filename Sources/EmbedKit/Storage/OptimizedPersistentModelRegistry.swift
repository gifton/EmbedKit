import Foundation
import SQLite3
import OSLog
import CommonCrypto

/// Actor-based persistent storage for model registry data with connection pooling
public actor OptimizedPersistentModelRegistry {
    private let logger = Logger(subsystem: "EmbedKit", category: "OptimizedPersistentModelRegistry")
    
    private let connectionPool: SQLiteConnectionPool
    private let dbPath: String
    
    // SQL statements as constants for better performance
    private enum SQL {
        static let createVersionsTable = """
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
        
        static let createIndexes = """
            CREATE INDEX IF NOT EXISTS idx_model_identifier ON model_versions(identifier);
            CREATE INDEX IF NOT EXISTS idx_model_active ON model_versions(identifier, is_active);
            CREATE INDEX IF NOT EXISTS idx_created_at ON model_versions(created_at);
        """
        
        static let insertVersion = """
            INSERT OR REPLACE INTO model_versions 
            (id, identifier, version, build_number, created_at, metadata, file_path, signature_hash, file_size, checksum)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        
        static let selectVersionsForIdentifier = """
            SELECT id, identifier, version, build_number, created_at, metadata, file_path, 
                   is_active, signature_hash, file_size, checksum
            FROM model_versions 
            WHERE identifier = ? 
            ORDER BY created_at DESC;
        """
        
        static let selectActiveVersion = """
            SELECT id, identifier, version, build_number, created_at, metadata, file_path, 
                   is_active, signature_hash, file_size, checksum
            FROM model_versions 
            WHERE identifier = ? AND is_active = 1
            LIMIT 1;
        """
        
        static let clearActiveVersions = "UPDATE model_versions SET is_active = 0 WHERE identifier = ?;"
        
        static let setActiveVersion = """
            UPDATE model_versions SET is_active = 1 
            WHERE identifier = ? AND version = ? AND build_number = ?;
        """
        
        static let deleteVersion = """
            DELETE FROM model_versions 
            WHERE identifier = ? AND version = ? AND build_number = ?;
        """
    }
    
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
        
        // Create connection pool
        self.connectionPool = try await SQLiteConnectionPool(
            dbPath: dbPath,
            minConnections: 2,
            maxConnections: 10
        )
        
        // Initialize database schema
        try await initializeDatabase()
        
        logger.info("Optimized database initialized with connection pool")
    }
    
    private func initializeDatabase() async throws {
        try await connectionPool.execute { db in
            // Enable optimizations
            try self.executeSQL("PRAGMA foreign_keys = ON;", on: db)
            try self.executeSQL("PRAGMA journal_mode = WAL;", on: db)
            try self.executeSQL("PRAGMA synchronous = NORMAL;", on: db)
            try self.executeSQL("PRAGMA temp_store = MEMORY;", on: db)
            try self.executeSQL("PRAGMA mmap_size = 268435456;", on: db) // 256MB
            
            // Create schema
            try self.executeSQL(SQL.createVersionsTable, on: db)
            try self.executeSQL(SQL.createIndexes, on: db)
            
            return ()
        }
    }
    
    /// Save a model version to persistent storage with optimized performance
    public func saveVersion(_ version: ModelVersion, fileURL: URL, signature: String? = nil) async throws {
        // Pre-calculate values outside of database operation
        let attributes = try FileManager.default.attributesOfItem(atPath: fileURL.path)
        let fileSize = attributes[.size] as? Int64 ?? 0
        let checksum = try await calculateChecksum(for: fileURL)
        let metadataString = try serializeMetadata(version.metadata)
        let uniqueID = "\(version.identifier)_\(version.version)_\(version.buildNumber)"
        
        try await connectionPool.executePrepared(
            sql: SQL.insertVersion,
            parameters: [
                uniqueID,
                version.identifier,
                version.version,
                version.buildNumber,
                version.createdAt.timeIntervalSince1970,
                metadataString,
                fileURL.path,
                signature,
                fileSize,
                checksum
            ]
        ) { statement in
            // The prepared statement is already executed by executePrepared
            return ()
        }
        
        logger.info("Saved model version \(version.identifier) v\(version.version) to database")
    }
    
    /// Load all versions for a model identifier with prepared statements
    public func loadVersions(for identifier: String) async throws -> [ModelVersionRecord] {
        try await connectionPool.executePrepared(
            sql: SQL.selectVersionsForIdentifier,
            parameters: [identifier]
        ) { statement in
            var records: [ModelVersionRecord] = []
            
            while sqlite3_step(statement) == SQLITE_ROW {
                let record = try self.parseVersionRecord(from: statement)
                records.append(record)
            }
            
            return records
        }
    }
    
    /// Get the active version for a model using cached statements
    public func getActiveVersion(for identifier: String) async throws -> ModelVersionRecord? {
        try await connectionPool.executePrepared(
            sql: SQL.selectActiveVersion,
            parameters: [identifier]
        ) { statement in
            if sqlite3_step(statement) == SQLITE_ROW {
                return try self.parseVersionRecord(from: statement)
            } else {
                return nil
            }
        }
    }
    
    /// Set the active version for a model within a transaction
    public func setActiveVersion(_ version: ModelVersion) async throws {
        try await connectionPool.transaction { db in
            // Clear existing active versions
            var clearStatement: OpaquePointer?
            defer { sqlite3_finalize(clearStatement) }
            
            guard sqlite3_prepare_v2(db, SQL.clearActiveVersions, -1, &clearStatement, nil) == SQLITE_OK else {
                throw ModelRegistryError.databaseError("Failed to prepare clear statement")
            }
            
            let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
            sqlite3_bind_text(clearStatement, 1, version.identifier, -1, SQLITE_TRANSIENT)
            
            guard sqlite3_step(clearStatement) == SQLITE_DONE else {
                throw ModelRegistryError.databaseError("Failed to clear active versions")
            }
            
            // Set new active version
            var setStatement: OpaquePointer?
            defer { sqlite3_finalize(setStatement) }
            
            guard sqlite3_prepare_v2(db, SQL.setActiveVersion, -1, &setStatement, nil) == SQLITE_OK else {
                throw ModelRegistryError.databaseError("Failed to prepare set statement")
            }
            
            sqlite3_bind_text(setStatement, 1, version.identifier, -1, SQLITE_TRANSIENT)
            sqlite3_bind_text(setStatement, 2, version.version, -1, SQLITE_TRANSIENT)
            sqlite3_bind_int64(setStatement, 3, Int64(version.buildNumber))
            
            guard sqlite3_step(setStatement) == SQLITE_DONE else {
                throw ModelRegistryError.databaseError("Failed to set active version")
            }
            
            return ()
        }
        
        logger.info("Set active version to \(version.identifier) v\(version.version)")
    }
    
    /// Remove a version from storage
    public func removeVersion(_ version: ModelVersion) async throws {
        try await connectionPool.executePrepared(
            sql: SQL.deleteVersion,
            parameters: [version.identifier, version.version, version.buildNumber]
        ) { _ in
            return ()
        }
        
        logger.info("Removed model version \(version.identifier) v\(version.version)")
    }
    
    /// Get storage statistics with connection pooling benefits
    public func getStatistics() async throws -> StorageStatistics {
        try await connectionPool.execute { db in
            var totalVersions = 0
            var totalModels = 0
            var totalSize: Int64 = 0
            
            // Use a single query with aggregation for better performance
            let statsSql = """
                SELECT 
                    COUNT(*) as total_versions,
                    COUNT(DISTINCT identifier) as total_models,
                    SUM(file_size) as total_size
                FROM model_versions;
            """
            
            var statement: OpaquePointer?
            defer { sqlite3_finalize(statement) }
            
            guard sqlite3_prepare_v2(db, statsSql, -1, &statement, nil) == SQLITE_OK else {
                throw ModelRegistryError.databaseError("Failed to prepare statistics query")
            }
            
            if sqlite3_step(statement) == SQLITE_ROW {
                totalVersions = Int(sqlite3_column_int(statement, 0))
                totalModels = Int(sqlite3_column_int(statement, 1))
                totalSize = sqlite3_column_int64(statement, 2)
            }
            
            // Get database file size
            let dbAttributes = try FileManager.default.attributesOfItem(atPath: self.dbPath)
            let dbSize = dbAttributes[.size] as? Int64 ?? 0
            
            return StorageStatistics(
                totalModels: totalModels,
                totalVersions: totalVersions,
                totalFileSize: totalSize,
                databaseSize: dbSize
            )
        }
    }
    
    /// Cleanup orphaned records and optimize database
    public func cleanup() async throws {
        try await connectionPool.execute { db in
            // Find orphaned records
            let selectSql = "SELECT id, file_path FROM model_versions;"
            var statement: OpaquePointer?
            defer { sqlite3_finalize(statement) }
            
            guard sqlite3_prepare_v2(db, selectSql, -1, &statement, nil) == SQLITE_OK else {
                throw ModelRegistryError.databaseError("Failed to prepare cleanup query")
            }
            
            var orphanedIds: [String] = []
            
            while sqlite3_step(statement) == SQLITE_ROW {
                let id = String(cString: sqlite3_column_text(statement, 0))
                let filePath = String(cString: sqlite3_column_text(statement, 1))
                
                if !FileManager.default.fileExists(atPath: filePath) {
                    orphanedIds.append(id)
                }
            }
            
            // Delete orphaned records in batch
            if !orphanedIds.isEmpty {
                let deleteSql = "DELETE FROM model_versions WHERE id IN (" +
                    orphanedIds.map { _ in "?" }.joined(separator: ",") + ");"
                
                var deleteStatement: OpaquePointer?
                defer { sqlite3_finalize(deleteStatement) }
                
                guard sqlite3_prepare_v2(db, deleteSql, -1, &deleteStatement, nil) == SQLITE_OK else {
                    throw ModelRegistryError.databaseError("Failed to prepare delete statement")
                }
                
                let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
                for (index, id) in orphanedIds.enumerated() {
                    sqlite3_bind_text(deleteStatement, Int32(index + 1), id, -1, SQLITE_TRANSIENT)
                }
                
                sqlite3_step(deleteStatement)
            }
            
            // Vacuum database
            try self.executeSQL("VACUUM;", on: db)
            
            self.logger.info("Cleaned up \(orphanedIds.count) orphaned records")
            
            return ()
        }
        
        // Perform connection pool maintenance
        await connectionPool.performMaintenance()
    }
    
    /// Get connection pool statistics
    public func getPoolStatistics() async -> SQLiteConnectionPool.PoolStatistics {
        await connectionPool.getStatistics()
    }
    
    // MARK: - Helper Methods
    
    private func parseVersionRecord(from statement: OpaquePointer?) throws -> ModelVersionRecord {
        guard let statement = statement else {
            throw ModelRegistryError.databaseError("Invalid statement")
        }
        
        let id = sqlite3_column_text(statement, 0).map { String(cString: $0) } ?? ""
        let identifier = sqlite3_column_text(statement, 1).map { String(cString: $0) } ?? ""
        let version = sqlite3_column_text(statement, 2).map { String(cString: $0) } ?? ""
        let buildNumber = Int(sqlite3_column_int(statement, 3))
        let createdAt = Date(timeIntervalSince1970: sqlite3_column_double(statement, 4))
        let metadataString = sqlite3_column_text(statement, 5).map { String(cString: $0) } ?? "{}"
        let filePath = sqlite3_column_text(statement, 6).map { String(cString: $0) } ?? ""
        let isActive = sqlite3_column_int(statement, 7) == 1
        
        let signatureHash: String?
        if sqlite3_column_type(statement, 8) != SQLITE_NULL {
            signatureHash = String(cString: sqlite3_column_text(statement, 8))
        } else {
            signatureHash = nil
        }
        
        let fileSize = sqlite3_column_int64(statement, 9)
        let checksum = sqlite3_column_text(statement, 10).map { String(cString: $0) } ?? ""
        
        let metadata = try deserializeMetadata(metadataString)
        
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
    
    private func serializeMetadata(_ metadata: [String: String]) throws -> String {
        let data = try JSONSerialization.data(withJSONObject: metadata)
        return String(data: data, encoding: .utf8) ?? "{}"
    }
    
    private func deserializeMetadata(_ string: String) throws -> [String: String] {
        let data = string.data(using: .utf8) ?? Data()
        return (try? JSONSerialization.jsonObject(with: data) as? [String: String]) ?? [:]
    }
    
    private func calculateChecksum(for url: URL) async throws -> String {
        return try await withCheckedThrowingContinuation { continuation in
            Task {
                do {
                    let data = try Data(contentsOf: url)
                    let checksum = data.sha256
                    continuation.resume(returning: checksum)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    private func executeSQL(_ sql: String, on db: OpaquePointer) throws {
        if sqlite3_exec(db, sql, nil, nil, nil) != SQLITE_OK {
            let error = String(cString: sqlite3_errmsg(db))
            throw ModelRegistryError.databaseError("SQL execution failed: \(error)")
        }
    }
}
// EmbedKit - SQLite Connection Wrapper
// Low-level SQLite3 interface for persistent cache

import Foundation

#if canImport(SQLite3)
import SQLite3
#endif

// MARK: - SQLite Connection

/// A lightweight SQLite connection wrapper for the persistent cache.
/// This class is NOT thread-safe - use from a single actor context.
final class SQLiteConnection: @unchecked Sendable {
    private var db: OpaquePointer?
    private let path: String
    private var preparedStatements: [String: OpaquePointer] = [:]

    /// Whether the connection is currently open.
    var isOpen: Bool { db != nil }

    /// Create a connection to a SQLite database.
    /// - Parameter path: Path to database file, or ":memory:" for in-memory database.
    init(path: String) throws {
        self.path = path
        try open()
    }

    deinit {
        close()
    }

    // MARK: - Connection Management

    /// Open the database connection.
    func open() throws {
        guard db == nil else { return }

        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        let result = sqlite3_open_v2(path, &db, flags, nil)

        guard result == SQLITE_OK else {
            let message = String(cString: sqlite3_errmsg(db))
            db = nil
            throw CacheError.databaseError("Failed to open database: \(message)")
        }

        // Set busy timeout to handle concurrent access
        sqlite3_busy_timeout(db, 5000)  // 5 seconds
    }

    /// Close the database connection.
    func close() {
        // Finalize all prepared statements
        for (_, stmt) in preparedStatements {
            sqlite3_finalize(stmt)
        }
        preparedStatements.removeAll()

        // Close database
        if let db = db {
            sqlite3_close_v2(db)
        }
        db = nil
    }

    // MARK: - Configuration

    /// Enable or disable Write-Ahead Logging mode.
    func setWALMode(_ enabled: Bool) throws {
        let mode = enabled ? "WAL" : "DELETE"
        try execute("PRAGMA journal_mode = \(mode)")
    }

    /// Set the cache size in pages (negative = KB).
    func setCacheSize(_ sizeKB: Int) throws {
        try execute("PRAGMA cache_size = -\(sizeKB)")
    }

    /// Enable or disable foreign keys.
    func setForeignKeys(_ enabled: Bool) throws {
        try execute("PRAGMA foreign_keys = \(enabled ? "ON" : "OFF")")
    }

    // MARK: - Execution

    /// Execute a SQL statement without returning results.
    @discardableResult
    func execute(_ sql: String) throws -> Int {
        guard let db = db else {
            throw CacheError.databaseError("Database not open")
        }

        var errmsg: UnsafeMutablePointer<CChar>?
        let result = sqlite3_exec(db, sql, nil, nil, &errmsg)

        if result != SQLITE_OK {
            let message = errmsg.map { String(cString: $0) } ?? "Unknown error"
            sqlite3_free(errmsg)
            throw CacheError.databaseError("Execute failed: \(message)")
        }

        return Int(sqlite3_changes(db))
    }

    /// Execute a parameterized statement.
    @discardableResult
    func execute(_ sql: String, parameters: [SQLiteValue]) throws -> Int {
        let stmt = try prepareStatement(sql)
        defer { sqlite3_reset(stmt) }

        try bindParameters(stmt, parameters: parameters)

        let result = sqlite3_step(stmt)
        if result != SQLITE_DONE && result != SQLITE_ROW {
            let message = String(cString: sqlite3_errmsg(db))
            throw CacheError.databaseError("Step failed: \(message)")
        }

        return Int(sqlite3_changes(db))
    }

    /// Query and return all matching rows.
    func query(_ sql: String, parameters: [SQLiteValue] = []) throws -> [[String: SQLiteValue]] {
        let stmt = try prepareStatement(sql)
        defer { sqlite3_reset(stmt) }

        try bindParameters(stmt, parameters: parameters)

        var rows: [[String: SQLiteValue]] = []
        let columnCount = sqlite3_column_count(stmt)

        while sqlite3_step(stmt) == SQLITE_ROW {
            var row: [String: SQLiteValue] = [:]

            for i in 0..<columnCount {
                let name = String(cString: sqlite3_column_name(stmt, i))
                row[name] = extractValue(stmt, column: i)
            }

            rows.append(row)
        }

        return rows
    }

    /// Query and return the first matching row.
    func queryOne(_ sql: String, parameters: [SQLiteValue] = []) throws -> [String: SQLiteValue]? {
        let stmt = try prepareStatement(sql)
        defer { sqlite3_reset(stmt) }

        try bindParameters(stmt, parameters: parameters)

        guard sqlite3_step(stmt) == SQLITE_ROW else {
            return nil
        }

        var row: [String: SQLiteValue] = [:]
        let columnCount = sqlite3_column_count(stmt)

        for i in 0..<columnCount {
            let name = String(cString: sqlite3_column_name(stmt, i))
            row[name] = extractValue(stmt, column: i)
        }

        return row
    }

    /// Get the last inserted row ID.
    var lastInsertRowID: Int64 {
        guard let db = db else { return 0 }
        return sqlite3_last_insert_rowid(db)
    }

    // MARK: - Transactions

    /// Begin a transaction.
    func beginTransaction() throws {
        try execute("BEGIN TRANSACTION")
    }

    /// Commit the current transaction.
    func commit() throws {
        try execute("COMMIT")
    }

    /// Rollback the current transaction.
    func rollback() throws {
        try execute("ROLLBACK")
    }

    /// Execute a block within a transaction.
    func transaction<T>(_ block: () throws -> T) throws -> T {
        try beginTransaction()
        do {
            let result = try block()
            try commit()
            return result
        } catch {
            try? rollback()
            throw error
        }
    }

    // MARK: - Private Helpers

    private func prepareStatement(_ sql: String) throws -> OpaquePointer {
        // Check cache first
        if let cached = preparedStatements[sql] {
            sqlite3_reset(cached)
            return cached
        }

        guard let db = db else {
            throw CacheError.databaseError("Database not open")
        }

        var stmt: OpaquePointer?
        let result = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)

        guard result == SQLITE_OK, let preparedStmt = stmt else {
            let message = String(cString: sqlite3_errmsg(db))
            throw CacheError.databaseError("Prepare failed: \(message)")
        }

        // Cache for reuse
        preparedStatements[sql] = preparedStmt
        return preparedStmt
    }

    private func bindParameters(_ stmt: OpaquePointer, parameters: [SQLiteValue]) throws {
        for (index, value) in parameters.enumerated() {
            let sqlIndex = Int32(index + 1)  // SQLite uses 1-based indexing

            let result: Int32
            switch value {
            case .null:
                result = sqlite3_bind_null(stmt, sqlIndex)
            case .integer(let i):
                result = sqlite3_bind_int64(stmt, sqlIndex, i)
            case .real(let d):
                result = sqlite3_bind_double(stmt, sqlIndex, d)
            case .text(let s):
                result = sqlite3_bind_text(stmt, sqlIndex, s, -1, SQLITE_TRANSIENT)
            case .blob(let data):
                result = data.withUnsafeBytes { ptr in
                    sqlite3_bind_blob(stmt, sqlIndex, ptr.baseAddress, Int32(data.count), SQLITE_TRANSIENT)
                }
            }

            guard result == SQLITE_OK else {
                let message = String(cString: sqlite3_errmsg(db))
                throw CacheError.databaseError("Bind failed at index \(index): \(message)")
            }
        }
    }

    private func extractValue(_ stmt: OpaquePointer, column: Int32) -> SQLiteValue {
        let type = sqlite3_column_type(stmt, column)

        switch type {
        case SQLITE_NULL:
            return .null
        case SQLITE_INTEGER:
            return .integer(sqlite3_column_int64(stmt, column))
        case SQLITE_FLOAT:
            return .real(sqlite3_column_double(stmt, column))
        case SQLITE_TEXT:
            if let text = sqlite3_column_text(stmt, column) {
                return .text(String(cString: text))
            }
            return .null
        case SQLITE_BLOB:
            let size = sqlite3_column_bytes(stmt, column)
            if let ptr = sqlite3_column_blob(stmt, column), size > 0 {
                return .blob(Data(bytes: ptr, count: Int(size)))
            }
            return .blob(Data())
        default:
            return .null
        }
    }
}

// MARK: - SQLite Value

/// A value that can be stored in SQLite.
enum SQLiteValue: Sendable {
    case null
    case integer(Int64)
    case real(Double)
    case text(String)
    case blob(Data)

    /// Convert to optional Int64.
    var intValue: Int64? {
        if case .integer(let i) = self { return i }
        return nil
    }

    /// Convert to optional Double.
    var doubleValue: Double? {
        if case .real(let d) = self { return d }
        if case .integer(let i) = self { return Double(i) }
        return nil
    }

    /// Convert to optional String.
    var stringValue: String? {
        if case .text(let s) = self { return s }
        return nil
    }

    /// Convert to optional Data.
    var dataValue: Data? {
        if case .blob(let d) = self { return d }
        return nil
    }
}

// MARK: - SQLITE_TRANSIENT

/// SQLite transient destructor - tells SQLite to copy the data.
private let SQLITE_TRANSIENT = unsafeBitCast(OpaquePointer(bitPattern: -1), to: sqlite3_destructor_type.self)

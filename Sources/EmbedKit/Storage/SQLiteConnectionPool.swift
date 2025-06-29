import Foundation
import SQLite3
import OSLog

/// A pooled SQLite connection with prepared statement cache
private final class PooledConnection: @unchecked Sendable {
    let id: UUID = UUID()
    var db: OpaquePointer?
    var preparedStatements: [String: OpaquePointer] = [:]
    var lastUsed: Date = Date()
    var useCount: Int = 0
    var isInUse: Bool = false
    
    init(db: OpaquePointer) {
        self.db = db
    }
    
    deinit {
        // Clean up prepared statements
        for (_, statement) in preparedStatements {
            sqlite3_finalize(statement)
        }
        
        // Close database connection
        if let db = db {
            sqlite3_close(db)
        }
    }
    
    func markUsed() {
        lastUsed = Date()
        useCount += 1
        isInUse = true
    }
    
    func markReleased() {
        isInUse = false
    }
}

/// High-performance SQLite connection pool with prepared statement caching
///
/// This actor-based implementation provides:
/// - Connection pooling with configurable min/max connections
/// - Prepared statement caching per connection
/// - Automatic connection health checking
/// - Fair connection distribution
/// - Transaction support
public actor SQLiteConnectionPool {
    private let logger = Logger(subsystem: "EmbedKit", category: "SQLiteConnectionPool")
    
    // Pool configuration
    private let dbPath: String
    private let minConnections: Int
    private let maxConnections: Int
    
    // Connection pool
    private var connections: [PooledConnection] = []
    private var waitingRequests: [CheckedContinuation<PooledConnection, Error>] = []
    
    // Statistics
    private var totalRequests: Int = 0
    private var totalWaits: Int = 0
    private var totalErrors: Int = 0
    
    public struct PoolStatistics: Sendable {
        public let totalConnections: Int
        public let activeConnections: Int
        public let idleConnections: Int
        public let totalRequests: Int
        public let totalWaits: Int
        public let averageUseCount: Double
        public let connectionUtilization: Double
        
        public init(connections: [PooledConnection], totalRequests: Int, totalWaits: Int) {
            self.totalConnections = connections.count
            self.activeConnections = connections.filter { $0.isInUse }.count
            self.idleConnections = totalConnections - activeConnections
            self.totalRequests = totalRequests
            self.totalWaits = totalWaits
            
            let totalUseCount = connections.reduce(0) { $0 + $1.useCount }
            self.averageUseCount = connections.isEmpty ? 0 : Double(totalUseCount) / Double(connections.count)
            self.connectionUtilization = totalConnections > 0 ? Double(activeConnections) / Double(totalConnections) : 0
        }
    }
    
    public init(dbPath: String, minConnections: Int = 2, maxConnections: Int = 10) async throws {
        self.dbPath = dbPath
        self.minConnections = max(1, minConnections)
        self.maxConnections = max(minConnections, maxConnections)
        
        // Create initial connections
        for _ in 0..<minConnections {
            let connection = try createConnection()
            connections.append(connection)
        }
        
        logger.info("SQLite connection pool initialized with \(minConnections) connections")
    }
    
    deinit {
        // Clean up all connections
        for connection in connections {
            connection.markReleased()
        }
        connections.removeAll()
    }
    
    /// Execute a query using a pooled connection
    public func execute<T>(_ operation: @escaping (OpaquePointer) async throws -> T) async throws -> T {
        let connection = try await acquireConnection()
        
        do {
            let result = try await operation(connection.db!)
            await releaseConnection(connection)
            return result
        } catch {
            await releaseConnection(connection)
            totalErrors += 1
            throw error
        }
    }
    
    /// Execute a query with a prepared statement
    public func executePrepared<T>(
        sql: String,
        parameters: [Any?] = [],
        operation: @escaping (OpaquePointer) async throws -> T
    ) async throws -> T {
        let connection = try await acquireConnection()
        
        do {
            // Get or create prepared statement
            let statement = try getPreparedStatement(for: sql, connection: connection)
            
            // Bind parameters
            try bindParameters(parameters, to: statement)
            
            // Execute operation
            let result = try await operation(statement)
            
            // Reset statement for reuse
            sqlite3_reset(statement)
            sqlite3_clear_bindings(statement)
            
            await releaseConnection(connection)
            return result
        } catch {
            await releaseConnection(connection)
            totalErrors += 1
            throw error
        }
    }
    
    /// Execute a transaction
    public func transaction<T>(_ operations: @escaping (OpaquePointer) async throws -> T) async throws -> T {
        let connection = try await acquireConnection()
        
        do {
            // Begin transaction
            try executeSQL("BEGIN TRANSACTION", on: connection.db!)
            
            do {
                let result = try await operations(connection.db!)
                
                // Commit transaction
                try executeSQL("COMMIT", on: connection.db!)
                
                await releaseConnection(connection)
                return result
            } catch {
                // Rollback on error
                try? executeSQL("ROLLBACK", on: connection.db!)
                throw error
            }
        } catch {
            await releaseConnection(connection)
            totalErrors += 1
            throw error
        }
    }
    
    /// Get pool statistics
    public func getStatistics() async -> PoolStatistics {
        PoolStatistics(
            connections: connections,
            totalRequests: totalRequests,
            totalWaits: totalWaits
        )
    }
    
    /// Perform maintenance on the pool
    public func performMaintenance() async {
        logger.debug("Performing connection pool maintenance")
        
        // Remove excess idle connections
        let idleConnections = connections.filter { !$0.isInUse }
        let excessCount = connections.count - minConnections
        
        if excessCount > 0 && idleConnections.count > 0 {
            let toRemove = min(excessCount, idleConnections.count)
            
            // Remove least recently used connections
            let sortedIdle = idleConnections.sorted { $0.lastUsed < $1.lastUsed }
            
            for i in 0..<toRemove {
                if let index = connections.firstIndex(where: { $0.id == sortedIdle[i].id }) {
                    connections.remove(at: index)
                    logger.debug("Removed idle connection from pool")
                }
            }
        }
        
        // Check health of remaining connections
        for connection in connections where !connection.isInUse {
            if !isConnectionHealthy(connection) {
                if let index = connections.firstIndex(where: { $0.id == connection.id }) {
                    connections.remove(at: index)
                    logger.warning("Removed unhealthy connection from pool")
                    
                    // Replace with new connection if below minimum
                    if connections.count < minConnections {
                        if let newConnection = try? createConnection() {
                            connections.append(newConnection)
                        }
                    }
                }
            }
        }
    }
    
    // MARK: - Private Methods
    
    private func acquireConnection() async throws -> PooledConnection {
        totalRequests += 1
        
        // Try to find an idle connection
        if let connection = connections.first(where: { !$0.isInUse }) {
            connection.markUsed()
            logger.trace("Acquired idle connection")
            return connection
        }
        
        // Try to create a new connection if below max
        if connections.count < maxConnections {
            let connection = try createConnection()
            connection.markUsed()
            connections.append(connection)
            logger.debug("Created new connection, pool size: \(connections.count)")
            return connection
        }
        
        // Need to wait for a connection
        totalWaits += 1
        logger.debug("All connections in use, waiting...")
        
        return try await withCheckedThrowingContinuation { continuation in
            waitingRequests.append(continuation)
        }
    }
    
    private func releaseConnection(_ connection: PooledConnection) async {
        connection.markReleased()
        
        // Check if anyone is waiting
        if !waitingRequests.isEmpty {
            let continuation = waitingRequests.removeFirst()
            connection.markUsed()
            continuation.resume(returning: connection)
            logger.trace("Handed connection to waiting request")
        } else {
            logger.trace("Released connection back to pool")
        }
    }
    
    private func createConnection() throws -> PooledConnection {
        var db: OpaquePointer?
        
        // Open database with optimized settings
        var flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        
        if sqlite3_open_v2(dbPath, &db, flags, nil) != SQLITE_OK {
            let error = db.map { String(cString: sqlite3_errmsg($0)) } ?? "Unknown error"
            sqlite3_close(db)
            throw SQLitePoolError.connectionFailed(error)
        }
        
        // Configure connection for performance
        try executeSQL("PRAGMA journal_mode = WAL", on: db!)
        try executeSQL("PRAGMA synchronous = NORMAL", on: db!)
        try executeSQL("PRAGMA temp_store = MEMORY", on: db!)
        try executeSQL("PRAGMA mmap_size = 268435456", on: db!) // 256MB memory map
        
        return PooledConnection(db: db!)
    }
    
    private func getPreparedStatement(for sql: String, connection: PooledConnection) throws -> OpaquePointer {
        // Check cache first
        if let cached = connection.preparedStatements[sql] {
            return cached
        }
        
        // Prepare new statement
        var statement: OpaquePointer?
        
        guard sqlite3_prepare_v2(connection.db, sql, -1, &statement, nil) == SQLITE_OK else {
            let error = String(cString: sqlite3_errmsg(connection.db))
            throw SQLitePoolError.prepareFailed(error)
        }
        
        // Cache it
        connection.preparedStatements[sql] = statement
        
        return statement!
    }
    
    private func bindParameters(_ parameters: [Any?], to statement: OpaquePointer) throws {
        for (index, parameter) in parameters.enumerated() {
            let position = Int32(index + 1)
            
            if let parameter = parameter {
                switch parameter {
                case let value as Int:
                    sqlite3_bind_int64(statement, position, Int64(value))
                case let value as Int64:
                    sqlite3_bind_int64(statement, position, value)
                case let value as Double:
                    sqlite3_bind_double(statement, position, value)
                case let value as String:
                    let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
                    sqlite3_bind_text(statement, position, value, -1, SQLITE_TRANSIENT)
                case let value as Data:
                    value.withUnsafeBytes { bytes in
                        sqlite3_bind_blob(statement, position, bytes.baseAddress, Int32(value.count), nil)
                    }
                case let value as Bool:
                    sqlite3_bind_int(statement, position, value ? 1 : 0)
                default:
                    throw SQLitePoolError.unsupportedParameterType("\(type(of: parameter))")
                }
            } else {
                sqlite3_bind_null(statement, position)
            }
        }
    }
    
    private func executeSQL(_ sql: String, on db: OpaquePointer) throws {
        if sqlite3_exec(db, sql, nil, nil, nil) != SQLITE_OK {
            let error = String(cString: sqlite3_errmsg(db))
            throw SQLitePoolError.executionFailed(error)
        }
    }
    
    private func isConnectionHealthy(_ connection: PooledConnection) -> Bool {
        guard let db = connection.db else { return false }
        
        // Simple health check - try to execute a basic query
        var statement: OpaquePointer?
        defer { sqlite3_finalize(statement) }
        
        let sql = "SELECT 1"
        
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else {
            return false
        }
        
        guard sqlite3_step(statement) == SQLITE_ROW else {
            return false
        }
        
        return true
    }
}

/// Errors that can occur in connection pool operations
public enum SQLitePoolError: LocalizedError {
    case connectionFailed(String)
    case prepareFailed(String)
    case executionFailed(String)
    case unsupportedParameterType(String)
    case poolExhausted
    
    public var errorDescription: String? {
        switch self {
        case .connectionFailed(let message):
            return "Failed to create database connection: \(message)"
        case .prepareFailed(let message):
            return "Failed to prepare statement: \(message)"
        case .executionFailed(let message):
            return "Failed to execute SQL: \(message)"
        case .unsupportedParameterType(let type):
            return "Unsupported parameter type: \(type)"
        case .poolExhausted:
            return "Connection pool exhausted"
        }
    }
}
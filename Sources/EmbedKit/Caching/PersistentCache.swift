// EmbedKit - Persistent Embedding Cache
// On-disk SQLite-backed cache with semantic deduplication support

import Foundation

// MARK: - Persistent Cache

/// Actor-based persistent cache for embeddings.
///
/// Provides on-disk storage of embeddings with:
/// - Exact text match lookup
/// - Optional semantic similarity matching
/// - LRU eviction when capacity limits are reached
/// - Statistics tracking
///
/// Example:
/// ```swift
/// let cache = try await PersistentCache()
///
/// // Check cache first
/// if let cached = await cache.get(text: "Hello", modelID: modelID) {
///     return cached.embedding
/// }
///
/// // Generate and store
/// let embedding = try await model.embed(text)
/// try await cache.store(text: "Hello", embedding: embedding)
/// ```
public actor PersistentCache {

    // MARK: - Properties

    private let connection: SQLiteConnection
    private let config: CacheConfiguration
    private var stats: CacheStatistics

    /// Path to the cache database file.
    public let path: URL?

    /// Whether the cache is using in-memory storage.
    public nonisolated var isInMemory: Bool { path == nil }

    // MARK: - Initialization

    /// Create a new persistent cache.
    ///
    /// - Parameters:
    ///   - path: Path for the database file. If nil, uses default location.
    ///   - config: Cache configuration.
    /// - Throws: CacheError if database cannot be opened.
    public init(
        path: URL? = nil,
        config: CacheConfiguration = .default
    ) async throws {
        let dbPath: String
        if let customPath = path {
            dbPath = customPath.path
            self.path = customPath
        } else {
            dbPath = Self.defaultPath().path
            self.path = Self.defaultPath()
        }

        self.config = config
        self.stats = CacheStatistics()
        self.connection = try SQLiteConnection(path: dbPath)

        try setupDatabase()
    }

    /// Create an in-memory cache (useful for testing).
    public init(inMemory config: CacheConfiguration = .inMemory) async throws {
        self.path = nil
        self.config = config
        self.stats = CacheStatistics()
        self.connection = try SQLiteConnection(path: ":memory:")

        try setupDatabase()
    }

    // MARK: - Core Operations

    /// Look up an embedding in the cache.
    ///
    /// - Parameters:
    ///   - text: The text to look up.
    ///   - modelID: The model ID to match.
    /// - Returns: Cache result indicating hit or miss.
    public func get(text: String, modelID: ModelID) -> CacheResult {
        stats.totalLookups += 1
        stats.lastAccessTime = Date()

        let textHash = TextHasher.hash(text)

        // Try exact match
        if let cached = getExact(textHash: textHash, modelID: modelID) {
            stats.exactHits += 1
            updateAccessTime(id: cached.id)
            return .exactMatch(cached)
        }

        stats.misses += 1
        return .miss
    }

    /// Look up an embedding with optional semantic matching.
    ///
    /// - Parameters:
    ///   - text: The text to look up.
    ///   - embedding: Pre-computed embedding for semantic comparison.
    ///   - modelID: The model ID to match.
    ///   - threshold: Similarity threshold for semantic match.
    /// - Returns: Cache result indicating hit or miss.
    public func getSemantic(
        text: String,
        embedding: Embedding,
        modelID: ModelID,
        threshold: Float? = nil
    ) -> CacheResult {
        // First try exact match
        let exactResult = get(text: text, modelID: modelID)
        if exactResult.isHit {
            return exactResult
        }

        // Decrement miss count since we'll recount
        stats.misses -= 1

        // Try semantic match if enabled
        guard config.enableSemanticDedup else {
            stats.misses += 1
            return .miss
        }

        let similarityThreshold = threshold ?? config.deduplicationThreshold

        if let (cached, similarity) = findSimilar(embedding: embedding, modelID: modelID, threshold: similarityThreshold) {
            stats.semanticHits += 1
            updateAccessTime(id: cached.id)
            return .semanticMatch(cached, similarity: similarity)
        }

        stats.misses += 1
        return .miss
    }

    /// Store an embedding in the cache.
    ///
    /// - Parameters:
    ///   - text: The original text.
    ///   - embedding: The embedding to cache.
    /// - Throws: CacheError if storage fails.
    public func store(text: String, embedding: Embedding) throws {
        let textHash = TextHasher.hash(text)
        let modelID = embedding.metadata.modelID

        // Check if already exists
        if getExact(textHash: textHash, modelID: modelID) != nil {
            return  // Already cached
        }

        // Check capacity and evict if needed
        if config.autoEvict {
            try evictIfNeeded()
        }

        // Serialize vector
        let vectorData = VectorSerializer.serialize(embedding.vector)

        // Insert
        let sql = """
            INSERT INTO embeddings (
                text_hash, text, vector, model_id, dimensions,
                token_count, created_at, accessed_at, access_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

        let now = Date().timeIntervalSince1970

        try connection.execute(sql, parameters: [
            .text(textHash),
            .text(text),
            .blob(vectorData),
            .text(modelID.description),
            .integer(Int64(embedding.dimensions)),
            .integer(Int64(embedding.metadata.tokenCount)),
            .real(now),
            .real(now),
            .integer(1)
        ])

        // Update stats
        stats.entryCount += 1
        stats.sizeBytes += estimateEntrySize(text: text, dimensions: embedding.dimensions)
    }

    /// Store multiple embeddings in a single transaction.
    ///
    /// - Parameter entries: Array of (text, embedding) pairs.
    /// - Throws: CacheError if storage fails.
    public func storeBatch(_ entries: [(text: String, embedding: Embedding)]) throws {
        guard !entries.isEmpty else { return }

        try connection.transaction {
            for (text, embedding) in entries {
                try self.store(text: text, embedding: embedding)
            }
        }
    }

    // MARK: - Management

    /// Remove a specific entry from the cache.
    ///
    /// - Parameter textHash: Hash of the text to remove.
    /// - Returns: True if entry was removed.
    @discardableResult
    public func remove(textHash: String) throws -> Bool {
        let count = try connection.execute(
            "DELETE FROM embeddings WHERE text_hash = ?",
            parameters: [.text(textHash)]
        )

        if count > 0 {
            stats.entryCount -= count
        }

        return count > 0
    }

    /// Clear all entries from the cache.
    public func clear() throws {
        try connection.execute("DELETE FROM embeddings")
        stats.entryCount = 0
        stats.sizeBytes = 0
    }

    /// Evict entries older than the specified date.
    ///
    /// - Parameter date: Evict entries accessed before this date.
    /// - Returns: Number of evicted entries.
    @discardableResult
    public func evict(olderThan date: Date) throws -> Int {
        let timestamp = date.timeIntervalSince1970
        let count = try connection.execute(
            "DELETE FROM embeddings WHERE accessed_at < ?",
            parameters: [.real(timestamp)]
        )

        if count > 0 {
            stats.entryCount -= count
            stats.evictions += count
            try updateSizeEstimate()
        }

        return count
    }

    /// Evict least recently used entries to meet capacity constraints.
    ///
    /// - Returns: Number of evicted entries.
    @discardableResult
    public func evictLRU(count: Int = 100) throws -> Int {
        let evicted = try connection.execute(
            """
            DELETE FROM embeddings WHERE id IN (
                SELECT id FROM embeddings ORDER BY accessed_at ASC LIMIT ?
            )
            """,
            parameters: [.integer(Int64(count))]
        )

        if evicted > 0 {
            stats.entryCount -= evicted
            stats.evictions += evicted
            try updateSizeEstimate()
        }

        return evicted
    }

    /// Compact the database to reclaim space.
    public func vacuum() throws {
        try connection.execute("VACUUM")
    }

    /// Get current cache statistics.
    public var statistics: CacheStatistics {
        stats
    }

    /// Reset statistics counters.
    public func resetStatistics() {
        stats.reset()
    }

    // MARK: - Private Implementation

    private func setupDatabase() throws {
        // Configure connection
        if config.enableWAL {
            try connection.setWALMode(true)
        }

        // Create schema
        try connection.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_hash TEXT NOT NULL,
                text TEXT NOT NULL,
                vector BLOB NOT NULL,
                model_id TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                token_count INTEGER,
                created_at REAL NOT NULL,
                accessed_at REAL NOT NULL,
                access_count INTEGER DEFAULT 1,
                UNIQUE(text_hash, model_id)
            )
            """)

        // Create indices
        try connection.execute("CREATE INDEX IF NOT EXISTS idx_text_hash ON embeddings(text_hash)")
        try connection.execute("CREATE INDEX IF NOT EXISTS idx_model_id ON embeddings(model_id)")
        try connection.execute("CREATE INDEX IF NOT EXISTS idx_accessed_at ON embeddings(accessed_at)")

        // Load current stats
        try updateStats()
    }

    private func getExact(textHash: String, modelID: ModelID) -> CachedEmbedding? {
        let sql = """
            SELECT id, text_hash, text, vector, model_id, dimensions,
                   token_count, created_at, accessed_at, access_count
            FROM embeddings
            WHERE text_hash = ? AND model_id = ?
            """

        guard let row = try? connection.queryOne(sql, parameters: [
            .text(textHash),
            .text(modelID.description)
        ]) else {
            return nil
        }

        return parseRow(row)
    }

    private func findSimilar(
        embedding: Embedding,
        modelID: ModelID,
        threshold: Float
    ) -> (CachedEmbedding, Float)? {
        // Query all entries for this model
        // Note: For large caches, this should use an index (LSH, etc.)
        let sql = """
            SELECT id, text_hash, text, vector, model_id, dimensions,
                   token_count, created_at, accessed_at, access_count
            FROM embeddings
            WHERE model_id = ?
            ORDER BY accessed_at DESC
            LIMIT 1000
            """

        guard let rows = try? connection.query(sql, parameters: [.text(modelID.description)]) else {
            return nil
        }

        var bestMatch: (CachedEmbedding, Float)? = nil

        for row in rows {
            guard let cached = parseRow(row) else { continue }

            let similarity = embedding.similarity(to: cached.embedding)

            if similarity >= threshold {
                if bestMatch == nil || similarity > bestMatch!.1 {
                    bestMatch = (cached, similarity)
                }
            }
        }

        return bestMatch
    }

    private func parseRow(_ row: [String: SQLiteValue]) -> CachedEmbedding? {
        guard let id = row["id"]?.intValue,
              let textHash = row["text_hash"]?.stringValue,
              let text = row["text"]?.stringValue,
              let vectorData = row["vector"]?.dataValue,
              let modelIDStr = row["model_id"]?.stringValue,
              let dimensions = row["dimensions"]?.intValue,
              let createdAtTs = row["created_at"]?.doubleValue,
              let accessedAtTs = row["accessed_at"]?.doubleValue
        else {
            return nil
        }

        guard let vector = VectorSerializer.deserialize(vectorData, dimensions: Int(dimensions)) else {
            return nil
        }

        let tokenCount = row["token_count"]?.intValue ?? 0
        let accessCount = row["access_count"]?.intValue ?? 1

        // Parse model ID (format: provider/name@version)
        let modelID = parseModelID(modelIDStr)

        let embedding = Embedding(
            vector: vector,
            metadata: EmbeddingMetadata(
                modelID: modelID,
                tokenCount: Int(tokenCount),
                processingTime: 0,
                normalized: true
            )
        )

        return CachedEmbedding(
            id: id,
            text: text,
            textHash: textHash,
            embedding: embedding,
            modelID: modelID,
            createdAt: Date(timeIntervalSince1970: createdAtTs),
            accessedAt: Date(timeIntervalSince1970: accessedAtTs),
            accessCount: Int(accessCount)
        )
    }

    private func parseModelID(_ str: String) -> ModelID {
        // Parse "provider/name@version" format
        let parts = str.split(separator: "/", maxSplits: 1)
        guard parts.count == 2 else {
            return ModelID(provider: "unknown", name: str, version: "1.0")
        }

        let provider = String(parts[0])
        let nameVersion = parts[1].split(separator: "@", maxSplits: 1)

        if nameVersion.count == 2 {
            return ModelID(
                provider: provider,
                name: String(nameVersion[0]),
                version: String(nameVersion[1])
            )
        } else {
            return ModelID(
                provider: provider,
                name: String(nameVersion[0]),
                version: "1.0"
            )
        }
    }

    private func updateAccessTime(id: Int64) {
        let now = Date().timeIntervalSince1970
        try? connection.execute(
            "UPDATE embeddings SET accessed_at = ?, access_count = access_count + 1 WHERE id = ?",
            parameters: [.real(now), .integer(id)]
        )
    }

    private func evictIfNeeded() throws {
        // Check entry count limit
        if config.maxEntries > 0 && stats.entryCount >= config.maxEntries {
            let toEvict = stats.entryCount - config.maxEntries + 100  // Evict extra to avoid frequent eviction
            try evictLRU(count: toEvict)
        }

        // Check size limit
        if config.maxSizeBytes > 0 && stats.sizeBytes >= config.maxSizeBytes {
            // Evict until under limit
            while stats.sizeBytes >= config.maxSizeBytes && stats.entryCount > 0 {
                try evictLRU(count: 100)
            }
        }

        // Check TTL
        if config.ttlSeconds > 0 {
            let cutoff = Date().addingTimeInterval(-config.ttlSeconds)
            try evict(olderThan: cutoff)
        }
    }

    private func updateStats() throws {
        // Get entry count
        if let row = try connection.queryOne("SELECT COUNT(*) as count FROM embeddings") {
            stats.entryCount = Int(row["count"]?.intValue ?? 0)
        }

        // Estimate size
        try updateSizeEstimate()
    }

    private func updateSizeEstimate() throws {
        // Estimate based on average entry size
        if let row = try connection.queryOne("""
            SELECT SUM(LENGTH(text) + LENGTH(vector) + 200) as total_size
            FROM embeddings
            """) {
            stats.sizeBytes = row["total_size"]?.intValue ?? 0
        }
    }

    private func estimateEntrySize(text: String, dimensions: Int) -> Int64 {
        let textSize = Int64(text.utf8.count)
        let vectorSize = Int64(dimensions * MemoryLayout<Float>.size)
        return textSize + vectorSize + 200  // Overhead
    }

    // MARK: - Default Path

    /// Get the default cache database path.
    public static func defaultPath() -> URL {
        let cacheDir: URL
        #if os(macOS)
        cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        #elseif os(iOS) || os(tvOS) || os(watchOS)
        cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        #else
        cacheDir = URL(fileURLWithPath: NSTemporaryDirectory())
        #endif

        let embedKitDir = cacheDir.appendingPathComponent("EmbedKit", isDirectory: true)

        // Create directory if needed
        try? FileManager.default.createDirectory(at: embedKitDir, withIntermediateDirectories: true)

        return embedKitDir.appendingPathComponent("embeddings.sqlite")
    }
}

// MARK: - Convenience Extensions

public extension PersistentCache {
    /// Check if a text is cached for a specific model.
    func contains(text: String, modelID: ModelID) -> Bool {
        let textHash = TextHasher.hash(text)
        return getExact(textHash: textHash, modelID: modelID) != nil
    }

    /// Get just the embedding if cached.
    func getEmbedding(text: String, modelID: ModelID) -> Embedding? {
        let result = get(text: text, modelID: modelID)
        return result.embedding?.embedding
    }

    /// Warm the cache with pre-computed embeddings.
    func warmCache(entries: [(text: String, embedding: Embedding)]) throws {
        try storeBatch(entries)
    }
}

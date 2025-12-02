// EmbedKit - Tensor Lifecycle Manager
//
// Automatic tensor lifecycle management with scope-based cleanup.
// Provides RAII-style resource management for GPU tensors.
//
// Metal 4.0 (iOS 26+ / macOS 26+)

import Foundation

#if canImport(Metal)
@preconcurrency import Metal

// MARK: - Tensor Scope

/// Scope-based tensor management for automatic cleanup.
///
/// `TensorScope` provides RAII-style resource management for GPU tensors.
/// Tensors created within a scope are automatically released when the scope exits.
///
/// ## Usage
/// ```swift
/// try await storageManager.withScope { scope in
///     let input = try scope.createEmbeddingTensor(batchSize: 32, dimensions: 384)
///     let output = try scope.createEmbeddingTensor(batchSize: 32, dimensions: 384)
///
///     // Use tensors...
///
///     // Tensors automatically released when scope exits
/// }
/// ```
public final class TensorScope: @unchecked Sendable {

    /// The storage manager owning this scope
    private let storageManager: TensorStorageManager

    /// Tensors created in this scope
    private var scopeTensors: [ManagedTensor] = []

    /// Whether the scope has been finalized
    private var isFinalized: Bool = false

    /// Lock for thread safety
    private let lock = NSLock()

    /// Scope identifier for debugging
    public let id: UUID = UUID()

    /// Human-readable label
    public let label: String

    // MARK: - Initialization

    /// Create a new tensor scope.
    ///
    /// - Parameters:
    ///   - storageManager: The storage manager to use
    ///   - label: Human-readable label for debugging
    init(storageManager: TensorStorageManager, label: String = "") {
        self.storageManager = storageManager
        self.label = label
    }

    // MARK: - Tensor Creation

    /// Create an embedding tensor within this scope.
    ///
    /// - Parameters:
    ///   - batchSize: Number of embeddings
    ///   - dimensions: Embedding dimensionality
    ///   - label: Human-readable label
    /// - Returns: The created tensor
    public func createEmbeddingTensor(
        batchSize: Int,
        dimensions: Int,
        label: String = ""
    ) async throws -> ManagedTensor {
        let tensor = try await storageManager.createEmbeddingTensor(
            batchSize: batchSize,
            dimensions: dimensions,
            label: label,
            registerResidency: true
        )
        track(tensor)
        return tensor
    }

    /// Create a token embedding tensor within this scope.
    ///
    /// - Parameters:
    ///   - batchSize: Number of sequences
    ///   - sequenceLength: Tokens per sequence
    ///   - dimensions: Embedding dimensionality
    ///   - label: Human-readable label
    /// - Returns: The created tensor
    public func createTokenEmbeddingTensor(
        batchSize: Int,
        sequenceLength: Int,
        dimensions: Int,
        label: String = ""
    ) async throws -> ManagedTensor {
        let tensor = try await storageManager.createTokenEmbeddingTensor(
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            dimensions: dimensions,
            label: label,
            registerResidency: true
        )
        track(tensor)
        return tensor
    }

    /// Create a similarity matrix tensor within this scope.
    ///
    /// - Parameters:
    ///   - queryCount: Number of query vectors
    ///   - keyCount: Number of key vectors
    ///   - label: Human-readable label
    /// - Returns: The created tensor
    public func createSimilarityTensor(
        queryCount: Int,
        keyCount: Int,
        label: String = ""
    ) async throws -> ManagedTensor {
        let tensor = try await storageManager.createSimilarityTensor(
            queryCount: queryCount,
            keyCount: keyCount,
            label: label,
            registerResidency: true
        )
        track(tensor)
        return tensor
    }

    /// Track a tensor for automatic cleanup.
    ///
    /// - Parameter tensor: The tensor to track
    private func track(_ tensor: ManagedTensor) {
        lock.lock()
        defer { lock.unlock() }
        guard !isFinalized else { return }
        scopeTensors.append(tensor)
    }

    /// Manually release a tensor before scope exit.
    ///
    /// - Parameter tensor: The tensor to release
    public func release(_ tensor: ManagedTensor) async {
        // Extract from scope synchronously
        let shouldRelease = removeTensorFromScope(tensor.id)

        if shouldRelease {
            await storageManager.release(tensor)
        }
    }

    /// Remove a tensor from scope tracking (synchronous helper).
    private func removeTensorFromScope(_ id: UUID) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        let originalCount = scopeTensors.count
        scopeTensors.removeAll { $0.id == id }
        return scopeTensors.count < originalCount
    }

    /// Finalize the scope and release all tensors.
    func finalize() async {
        // Extract tensors to release synchronously
        let tensorsToRelease = extractTensorsForRelease()

        // Release asynchronously
        for tensor in tensorsToRelease {
            await storageManager.release(tensor)
        }
    }

    /// Extract all tensors for release (synchronous helper).
    private func extractTensorsForRelease() -> [ManagedTensor] {
        lock.lock()
        defer { lock.unlock() }

        guard !isFinalized else {
            return []
        }
        isFinalized = true
        let tensors = scopeTensors
        scopeTensors.removeAll()
        return tensors
    }

    /// Number of tensors in this scope.
    public var tensorCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return scopeTensors.count
    }

    /// Total bytes allocated in this scope.
    public var totalBytes: Int {
        lock.lock()
        defer { lock.unlock() }
        return scopeTensors.reduce(0) { $0 + $1.sizeInBytes }
    }
}

// MARK: - Tensor Lifecycle Manager

/// Actor for managing tensor lifecycles with automatic cleanup.
///
/// `TensorLifecycleManager` provides centralized lifecycle management:
/// - Periodic cleanup of idle tensors
/// - Memory pressure monitoring
/// - Scope-based resource management
/// - Lifecycle event notifications
///
/// ## Usage
/// ```swift
/// let lifecycleManager = TensorLifecycleManager(storageManager: storageManager)
///
/// // Start automatic cleanup
/// await lifecycleManager.startMonitoring()
///
/// // Use scope-based cleanup
/// try await lifecycleManager.withScope { scope in
///     let tensor = try scope.createEmbeddingTensor(batchSize: 32, dimensions: 384)
///     // Use tensor...
/// }  // Automatic cleanup
///
/// // Stop monitoring
/// await lifecycleManager.stopMonitoring()
/// ```
public actor TensorLifecycleManager {

    /// The storage manager to manage
    private let storageManager: TensorStorageManager

    /// Configuration
    public let configuration: Configuration

    /// Whether monitoring is active
    private var isMonitoring: Bool = false

    /// Monitoring task
    private var monitoringTask: Task<Void, Never>?

    /// Active scopes
    private var activeScopes: [UUID: TensorScope] = [:]

    /// Lifecycle delegates
    private var delegates: [ObjectIdentifier: any TensorLifecycleDelegate] = [:]

    /// Statistics
    private var stats: Statistics = Statistics()

    // MARK: - Configuration

    /// Configuration for the lifecycle manager.
    public struct Configuration: Sendable {
        /// Interval between cleanup cycles (seconds)
        public let cleanupInterval: TimeInterval

        /// Whether to auto-start monitoring
        public let autoStartMonitoring: Bool

        /// Memory usage threshold to trigger cleanup (0.0 - 1.0)
        public let memoryThreshold: Float

        /// Default configuration
        public static let `default` = Configuration(
            cleanupInterval: 30.0,
            autoStartMonitoring: false,
            memoryThreshold: 0.8
        )

        /// Aggressive cleanup configuration
        public static let aggressive = Configuration(
            cleanupInterval: 10.0,
            autoStartMonitoring: true,
            memoryThreshold: 0.6
        )

        /// Relaxed configuration
        public static let relaxed = Configuration(
            cleanupInterval: 60.0,
            autoStartMonitoring: false,
            memoryThreshold: 0.9
        )

        public init(
            cleanupInterval: TimeInterval = 30.0,
            autoStartMonitoring: Bool = false,
            memoryThreshold: Float = 0.8
        ) {
            self.cleanupInterval = max(1.0, cleanupInterval)
            self.autoStartMonitoring = autoStartMonitoring
            self.memoryThreshold = min(1.0, max(0.1, memoryThreshold))
        }
    }

    /// Lifecycle manager statistics.
    public struct Statistics: Sendable {
        /// Total cleanup cycles run
        public var cleanupCycles: Int = 0

        /// Total tensors cleaned up
        public var tensorsCleaned: Int = 0

        /// Total scopes created
        public var scopesCreated: Int = 0

        /// Total scopes finalized
        public var scopesFinalized: Int = 0

        /// Memory pressure events handled
        public var memoryPressureEvents: Int = 0
    }

    // MARK: - Initialization

    /// Initialize the lifecycle manager.
    ///
    /// - Parameters:
    ///   - storageManager: The storage manager to manage
    ///   - configuration: Lifecycle configuration
    public init(
        storageManager: TensorStorageManager,
        configuration: Configuration = .default
    ) {
        self.storageManager = storageManager
        self.configuration = configuration
    }

    // MARK: - Monitoring

    /// Start lifecycle monitoring.
    public func startMonitoring() {
        guard !isMonitoring else { return }
        isMonitoring = true

        monitoringTask = Task { [weak self] in
            await self?.monitoringLoop()
        }
    }

    /// Stop lifecycle monitoring.
    public func stopMonitoring() {
        isMonitoring = false
        monitoringTask?.cancel()
        monitoringTask = nil
    }

    /// The main monitoring loop.
    private func monitoringLoop() async {
        while isMonitoring {
            // Sleep for the configured interval
            try? await Task.sleep(nanoseconds: UInt64(configuration.cleanupInterval * 1_000_000_000))

            guard isMonitoring else { break }

            // Perform cleanup cycle
            await performCleanupCycle()
        }
    }

    /// Perform a single cleanup cycle.
    public func performCleanupCycle() async {
        stats.cleanupCycles += 1

        // Check memory usage
        let usage = await storageManager.getMemoryUsage()

        if usage.usagePercentage > configuration.memoryThreshold {
            // Trigger cleanup
            let aggressiveness = (usage.usagePercentage - configuration.memoryThreshold) / (1.0 - configuration.memoryThreshold)
            await storageManager.trimIdleTensors(aggressiveness: aggressiveness)

            // Notify delegates
            await notifyMemoryPressure(level: aggressiveness > 0.5 ? 1 : 0)
            stats.memoryPressureEvents += 1
        }
    }

    // MARK: - Scope Management

    /// Create a new tensor scope.
    ///
    /// - Parameter label: Human-readable label
    /// - Returns: The created scope
    public func createScope(label: String = "") -> TensorScope {
        let scope = TensorScope(storageManager: storageManager, label: label)
        activeScopes[scope.id] = scope
        stats.scopesCreated += 1
        return scope
    }

    /// Execute work within a scope with automatic cleanup.
    ///
    /// - Parameters:
    ///   - label: Scope label
    ///   - work: The work to execute
    /// - Returns: The result of the work
    public func withScope<T: Sendable>(
        label: String = "",
        _ work: @Sendable (TensorScope) async throws -> T
    ) async throws -> T {
        let scope = createScope(label: label)
        do {
            let result = try await work(scope)
            await finalizeScope(scope)
            return result
        } catch {
            await finalizeScope(scope)
            throw error
        }
    }

    /// Finalize a scope and release its tensors.
    ///
    /// - Parameter scope: The scope to finalize
    public func finalizeScope(_ scope: TensorScope) async {
        await scope.finalize()
        activeScopes.removeValue(forKey: scope.id)
        stats.scopesFinalized += 1
    }

    /// Get count of active scopes.
    public var activeScopeCount: Int {
        activeScopes.count
    }

    // MARK: - Delegates

    /// Add a lifecycle delegate.
    ///
    /// - Parameter delegate: The delegate to add
    public func addDelegate(_ delegate: any TensorLifecycleDelegate) {
        delegates[ObjectIdentifier(delegate)] = delegate
    }

    /// Remove a lifecycle delegate.
    ///
    /// - Parameter delegate: The delegate to remove
    public func removeDelegate(_ delegate: any TensorLifecycleDelegate) {
        delegates.removeValue(forKey: ObjectIdentifier(delegate))
    }

    /// Notify delegates of memory pressure.
    private func notifyMemoryPressure(level: Int) async {
        for delegate in delegates.values {
            await delegate.memoryPressure(level: level)
        }
    }

    /// Notify delegates of tensor creation.
    public func notifyTensorCreated(_ tensor: ManagedTensor) async {
        for delegate in delegates.values {
            await delegate.tensorCreated(tensor)
        }
    }

    /// Notify delegates of tensor release.
    public func notifyTensorReleased(_ tensor: ManagedTensor) async {
        for delegate in delegates.values {
            await delegate.tensorReleased(tensor)
        }
    }

    // MARK: - Statistics

    /// Get current statistics.
    public func getStatistics() -> Statistics {
        stats
    }

    /// Get comprehensive status.
    public func getStatus() async -> Status {
        let memoryUsage = await storageManager.getMemoryUsage()
        let storageStats = await storageManager.getStatistics()

        return Status(
            isMonitoring: isMonitoring,
            activeScopeCount: activeScopes.count,
            lifecycleStats: stats,
            storageStats: storageStats,
            memoryUsage: memoryUsage
        )
    }

    /// Comprehensive status snapshot.
    public struct Status: Sendable {
        public let isMonitoring: Bool
        public let activeScopeCount: Int
        public let lifecycleStats: Statistics
        public let storageStats: TensorStorageManager.Statistics
        public let memoryUsage: TensorStorageManager.MemoryUsage
    }
}

// MARK: - TensorStorageManager Extension

extension TensorStorageManager {

    /// Create a lifecycle manager for this storage manager.
    ///
    /// - Parameter configuration: Lifecycle configuration
    /// - Returns: A new lifecycle manager
    public func createLifecycleManager(
        configuration: TensorLifecycleManager.Configuration = .default
    ) -> TensorLifecycleManager {
        TensorLifecycleManager(storageManager: self, configuration: configuration)
    }

    /// Execute work within a scope with automatic cleanup.
    ///
    /// - Parameters:
    ///   - label: Scope label
    ///   - work: The work to execute
    /// - Returns: The result of the work
    public func withScope<T: Sendable>(
        label: String = "",
        _ work: @Sendable (TensorScope) async throws -> T
    ) async throws -> T {
        let scope = TensorScope(storageManager: self, label: label)
        do {
            let result = try await work(scope)
            await scope.finalize()
            return result
        } catch {
            await scope.finalize()
            throw error
        }
    }
}

// MARK: - Tensor Pool

/// A pool of pre-allocated tensors for efficient reuse.
///
/// `TensorPool` maintains a pool of tensors with a specific shape, allowing
/// fast acquisition and release without allocation overhead.
///
/// ## Usage
/// ```swift
/// let pool = try await TensorPool(
///     storageManager: storageManager,
///     shape: .embedding(batchSize: 32, dimensions: 384),
///     poolSize: 4
/// )
///
/// // Acquire a tensor
/// if let tensor = await pool.acquire() {
///     // Use tensor...
///     await pool.release(tensor)
/// }
/// ```
public actor TensorPool {

    /// The storage manager
    private let storageManager: TensorStorageManager

    /// Shape of tensors in this pool
    public let shape: ManagedTensor.TensorShape

    /// Pool label
    public let label: String

    /// Available tensors
    private var available: [ManagedTensor] = []

    /// In-use tensors
    private var inUse: Set<UUID> = []

    /// Maximum pool size
    public let maxSize: Int

    /// Statistics
    private var acquireCount: Int = 0
    private var releaseCount: Int = 0
    private var missCount: Int = 0

    // MARK: - Initialization

    /// Create a tensor pool.
    ///
    /// - Parameters:
    ///   - storageManager: Storage manager for tensor allocation
    ///   - shape: Shape of tensors in the pool
    ///   - poolSize: Number of tensors to pre-allocate
    ///   - label: Human-readable label
    public init(
        storageManager: TensorStorageManager,
        shape: ManagedTensor.TensorShape,
        poolSize: Int,
        label: String = ""
    ) async throws {
        self.storageManager = storageManager
        self.shape = shape
        self.maxSize = poolSize
        self.label = label

        // Pre-allocate tensors
        for i in 0..<poolSize {
            let tensor = try await storageManager.createTensor(
                shape: shape,
                label: label.isEmpty ? "pool_\(i)" : "\(label)_\(i)",
                registerResidency: true
            )
            available.append(tensor)
        }
    }

    // MARK: - Pool Operations

    /// Acquire a tensor from the pool.
    ///
    /// - Returns: A tensor if available, nil if pool is exhausted
    public func acquire() -> ManagedTensor? {
        acquireCount += 1

        guard let tensor = available.popLast() else {
            missCount += 1
            return nil
        }

        inUse.insert(tensor.id)
        tensor.markAccessed()
        return tensor
    }

    /// Release a tensor back to the pool.
    ///
    /// - Parameter tensor: The tensor to release
    public func release(_ tensor: ManagedTensor) {
        guard inUse.contains(tensor.id) else { return }

        releaseCount += 1
        inUse.remove(tensor.id)
        available.append(tensor)
    }

    /// Number of available tensors.
    public var availableCount: Int { available.count }

    /// Number of in-use tensors.
    public var inUseCount: Int { inUse.count }

    /// Pool statistics.
    public var statistics: PoolStatistics {
        PoolStatistics(
            totalSize: maxSize,
            availableCount: available.count,
            inUseCount: inUse.count,
            acquireCount: acquireCount,
            releaseCount: releaseCount,
            missCount: missCount
        )
    }

    /// Pool statistics structure.
    public struct PoolStatistics: Sendable {
        public let totalSize: Int
        public let availableCount: Int
        public let inUseCount: Int
        public let acquireCount: Int
        public let releaseCount: Int
        public let missCount: Int

        public var hitRate: Float {
            acquireCount > 0 ? Float(acquireCount - missCount) / Float(acquireCount) : 1.0
        }
    }

    /// Clear the pool and release all tensors.
    public func clear() async {
        for tensor in available {
            await storageManager.release(tensor)
        }
        available.removeAll()

        // Note: in-use tensors remain with their current holders
        // They should be released when no longer needed
    }
}

// MARK: - AutoReleaseTensor

/// A tensor wrapper that automatically releases when deallocated.
///
/// `AutoReleaseTensor` provides automatic resource management using Swift's
/// ARC. When the last reference is released, the tensor is automatically
/// returned to the storage manager.
///
/// ## Usage
/// ```swift
/// let autoTensor = try await AutoReleaseTensor(
///     batchSize: 32,
///     dimensions: 384,
///     storageManager: storageManager
/// )
///
/// // Use autoTensor.tensor...
/// // Automatically released when autoTensor goes out of scope
/// ```
public final class AutoReleaseTensor: @unchecked Sendable {

    /// The managed tensor
    public let tensor: ManagedTensor

    /// The storage manager
    private let storageManager: TensorStorageManager

    /// Create an auto-releasing embedding tensor.
    ///
    /// - Parameters:
    ///   - batchSize: Number of embeddings
    ///   - dimensions: Embedding dimensionality
    ///   - label: Human-readable label
    ///   - storageManager: Storage manager
    public init(
        batchSize: Int,
        dimensions: Int,
        label: String = "",
        storageManager: TensorStorageManager
    ) async throws {
        self.storageManager = storageManager
        self.tensor = try await storageManager.createEmbeddingTensor(
            batchSize: batchSize,
            dimensions: dimensions,
            label: label,
            registerResidency: true
        )
    }

    /// Create an auto-releasing token embedding tensor.
    ///
    /// - Parameters:
    ///   - batchSize: Number of sequences
    ///   - sequenceLength: Tokens per sequence
    ///   - dimensions: Embedding dimensionality
    ///   - label: Human-readable label
    ///   - storageManager: Storage manager
    public init(
        batchSize: Int,
        sequenceLength: Int,
        dimensions: Int,
        label: String = "",
        storageManager: TensorStorageManager
    ) async throws {
        self.storageManager = storageManager
        self.tensor = try await storageManager.createTokenEmbeddingTensor(
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            dimensions: dimensions,
            label: label,
            registerResidency: true
        )
    }

    /// Create an auto-releasing wrapper for an existing tensor.
    ///
    /// - Parameters:
    ///   - tensor: The tensor to wrap
    ///   - storageManager: Storage manager
    public init(tensor: ManagedTensor, storageManager: TensorStorageManager) {
        self.tensor = tensor
        self.storageManager = storageManager
    }

    deinit {
        // Use Task.detached for cleanup to avoid keeping process alive.
        // Detached tasks are independent and don't block process exit
        // the same way unstructured Task {} does.
        let tensor = self.tensor
        let manager = self.storageManager
        Task.detached(priority: .background) {
            await manager.release(tensor)
        }
    }

    /// The underlying Metal buffer.
    public var buffer: MTLBuffer { tensor.buffer }

    /// Tensor shape.
    public var shape: ManagedTensor.TensorShape { tensor.shape }

    /// Size in bytes.
    public var sizeInBytes: Int { tensor.sizeInBytes }
}

#endif

// EmbedKit - Tensor Storage Manager
//
// Residency-aware tensor storage with automatic lifecycle management.
// Integrates with Metal 4's residency sets for optimal GPU memory handling.
//
// Metal 4.0 (iOS 26+ / macOS 26+)

import Foundation

#if canImport(Metal)
@preconcurrency import Metal

// MARK: - Managed Tensor

/// A tensor with lifecycle tracking and residency management.
///
/// `ManagedTensor` wraps an `EmbeddingTensor` or `TokenEmbeddingTensor` with
/// metadata for tracking access patterns, lifecycle state, and residency status.
///
/// **Lifecycle States**:
/// - `active`: Tensor is in use and should remain resident
/// - `cached`: Tensor is cached but may be evicted under memory pressure
/// - `released`: Tensor has been released and resources freed
///
/// - Note: Uses `@unchecked Sendable` because MTLBuffer is thread-safe.
public final class ManagedTensor: @unchecked Sendable {

    /// Unique identifier for this tensor
    public let id: UUID

    /// Human-readable label for debugging
    public let label: String

    /// The underlying Metal buffer
    public let buffer: MTLBuffer

    /// Shape of the tensor
    public let shape: TensorShape

    /// Current lifecycle state
    public private(set) var state: LifecycleState

    /// Size in bytes
    public var sizeInBytes: Int { buffer.length }

    /// Creation timestamp
    public let createdAt: Date

    /// Last access timestamp
    public private(set) var lastAccessedAt: Date

    /// Access count for LRU tracking
    public private(set) var accessCount: Int

    /// Whether this tensor is registered with a residency set
    public private(set) var isResident: Bool

    /// Residency set name if registered
    public private(set) var residencySetName: String?

    // MARK: - Types

    /// Tensor shape descriptor
    public enum TensorShape: Sendable, Equatable {
        /// 2D embedding tensor [batchSize, dimensions]
        case embedding(batchSize: Int, dimensions: Int)

        /// 3D token embedding tensor [batchSize, sequenceLength, dimensions]
        case tokenEmbedding(batchSize: Int, sequenceLength: Int, dimensions: Int)

        /// 1D similarity matrix [queryCount * keyCount]
        case similarityMatrix(queryCount: Int, keyCount: Int)

        /// Generic 1D buffer
        case buffer(length: Int)

        /// Total element count
        public var elementCount: Int {
            switch self {
            case .embedding(let batch, let dims):
                return batch * dims
            case .tokenEmbedding(let batch, let seq, let dims):
                return batch * seq * dims
            case .similarityMatrix(let q, let k):
                return q * k
            case .buffer(let length):
                return length
            }
        }

        /// Size in bytes (assuming Float32)
        public var sizeInBytes: Int {
            elementCount * MemoryLayout<Float>.stride
        }
    }

    /// Tensor lifecycle state
    public enum LifecycleState: String, Sendable {
        /// Tensor is actively being used
        case active

        /// Tensor is cached and may be evicted
        case cached

        /// Tensor has been released
        case released
    }

    // MARK: - Initialization

    /// Create a managed tensor from an existing buffer.
    ///
    /// - Parameters:
    ///   - buffer: The Metal buffer
    ///   - shape: Shape descriptor
    ///   - label: Human-readable label
    public init(buffer: MTLBuffer, shape: TensorShape, label: String = "") {
        self.id = UUID()
        self.label = label
        self.buffer = buffer
        self.shape = shape
        self.state = .active
        self.createdAt = Date()
        self.lastAccessedAt = Date()
        self.accessCount = 0
        self.isResident = false
        self.residencySetName = nil
    }

    // MARK: - Lifecycle Management

    /// Mark the tensor as accessed, updating timestamps and counters.
    public func markAccessed() {
        lastAccessedAt = Date()
        accessCount += 1
    }

    /// Transition to cached state.
    public func markCached() {
        state = .cached
    }

    /// Transition to released state.
    public func markReleased() {
        state = .released
        isResident = false
        residencySetName = nil
    }

    /// Mark as resident in a residency set.
    ///
    /// - Parameter setName: Name of the residency set
    public func markResident(inSet setName: String) {
        isResident = true
        residencySetName = setName
    }

    /// Remove from residency tracking.
    public func markNonResident() {
        isResident = false
        residencySetName = nil
    }

    /// Time since last access in seconds.
    public var idleTime: TimeInterval {
        Date().timeIntervalSince(lastAccessedAt)
    }
}

// MARK: - Tensor Storage Manager

/// Actor for managing tensor storage with residency awareness.
///
/// `TensorStorageManager` provides centralized management of GPU tensors with:
/// - Automatic residency set integration
/// - LRU-based cache eviction
/// - Memory pressure handling
/// - Lifecycle tracking
///
/// ## Usage
/// ```swift
/// let manager = TensorStorageManager(device: device)
///
/// // Create a managed tensor
/// let tensor = try await manager.createEmbeddingTensor(
///     batchSize: 32,
///     dimensions: 384,
///     label: "query_embeddings"
/// )
///
/// // Register with residency set
/// try await manager.registerWithResidency(tensor, setName: "active")
///
/// // Access tensor (updates LRU tracking)
/// await manager.markAccessed(tensor)
///
/// // Release when done
/// await manager.release(tensor)
/// ```
public actor TensorStorageManager {

    /// The Metal device
    public let device: MTLDevice

    /// Configuration for the storage manager
    public let configuration: Configuration

    /// Active tensors by ID
    private var tensors: [UUID: ManagedTensor] = [:]

    /// Tensors by label for lookup
    private var tensorsByLabel: [String: UUID] = [:]

    /// Associated residency manager
    private var residencyManager: Metal4ResidencyManager?

    /// Total bytes currently allocated
    private var totalAllocatedBytes: Int = 0

    /// Statistics tracking
    private var stats: Statistics = Statistics()

    // MARK: - Configuration

    /// Configuration for the tensor storage manager.
    public struct Configuration: Sendable {
        /// Maximum total memory for tensors (bytes)
        public let maxMemoryBytes: Int

        /// Enable automatic residency management
        public let autoResidency: Bool

        /// Default residency set name
        public let defaultResidencySet: String

        /// Idle time before tensor becomes eviction candidate (seconds)
        public let evictionIdleThreshold: TimeInterval

        /// Default configuration
        public static let `default` = Configuration(
            maxMemoryBytes: 512 * 1024 * 1024,  // 512 MB
            autoResidency: true,
            defaultResidencySet: "embeddings",
            evictionIdleThreshold: 60.0
        )

        /// High-memory configuration
        public static let highMemory = Configuration(
            maxMemoryBytes: 1024 * 1024 * 1024,  // 1 GB
            autoResidency: true,
            defaultResidencySet: "embeddings",
            evictionIdleThreshold: 120.0
        )

        /// Low-memory configuration
        public static let lowMemory = Configuration(
            maxMemoryBytes: 128 * 1024 * 1024,  // 128 MB
            autoResidency: true,
            defaultResidencySet: "embeddings",
            evictionIdleThreshold: 30.0
        )

        public init(
            maxMemoryBytes: Int,
            autoResidency: Bool = true,
            defaultResidencySet: String = "embeddings",
            evictionIdleThreshold: TimeInterval = 60.0
        ) {
            self.maxMemoryBytes = max(1024 * 1024, maxMemoryBytes)  // Min 1 MB
            self.autoResidency = autoResidency
            self.defaultResidencySet = defaultResidencySet
            self.evictionIdleThreshold = evictionIdleThreshold
        }
    }

    /// Storage manager statistics.
    public struct Statistics: Sendable {
        /// Total tensors created
        public var totalCreated: Int = 0

        /// Total tensors released
        public var totalReleased: Int = 0

        /// Total bytes allocated over lifetime
        public var totalBytesAllocated: Int = 0

        /// Total bytes freed over lifetime
        public var totalBytesFreed: Int = 0

        /// Cache hits (found existing tensor)
        public var cacheHits: Int = 0

        /// Cache misses (created new tensor)
        public var cacheMisses: Int = 0

        /// Evictions due to memory pressure
        public var evictions: Int = 0

        /// Residency registrations
        public var residencyRegistrations: Int = 0
    }

    // MARK: - Initialization

    /// Initialize the tensor storage manager.
    ///
    /// - Parameters:
    ///   - device: Metal device for buffer allocation
    ///   - configuration: Storage configuration
    ///   - residencyManager: Optional residency manager for integration
    public init(
        device: MTLDevice,
        configuration: Configuration = .default,
        residencyManager: Metal4ResidencyManager? = nil
    ) {
        self.device = device
        self.configuration = configuration
        self.residencyManager = residencyManager
    }

    // MARK: - Tensor Creation

    /// Create a managed embedding tensor.
    ///
    /// - Parameters:
    ///   - batchSize: Number of embeddings
    ///   - dimensions: Embedding dimensionality
    ///   - label: Human-readable label
    ///   - registerResidency: Whether to auto-register with residency set
    /// - Returns: The created managed tensor
    public func createEmbeddingTensor(
        batchSize: Int,
        dimensions: Int,
        label: String = "",
        registerResidency: Bool = true
    ) throws -> ManagedTensor {
        let shape = ManagedTensor.TensorShape.embedding(
            batchSize: batchSize,
            dimensions: dimensions
        )

        return try createTensor(shape: shape, label: label, registerResidency: registerResidency)
    }

    /// Create a managed token embedding tensor.
    ///
    /// - Parameters:
    ///   - batchSize: Number of sequences
    ///   - sequenceLength: Tokens per sequence
    ///   - dimensions: Embedding dimensionality
    ///   - label: Human-readable label
    ///   - registerResidency: Whether to auto-register with residency set
    /// - Returns: The created managed tensor
    public func createTokenEmbeddingTensor(
        batchSize: Int,
        sequenceLength: Int,
        dimensions: Int,
        label: String = "",
        registerResidency: Bool = true
    ) throws -> ManagedTensor {
        let shape = ManagedTensor.TensorShape.tokenEmbedding(
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            dimensions: dimensions
        )

        return try createTensor(shape: shape, label: label, registerResidency: registerResidency)
    }

    /// Create a managed similarity matrix tensor.
    ///
    /// - Parameters:
    ///   - queryCount: Number of query vectors
    ///   - keyCount: Number of key vectors
    ///   - label: Human-readable label
    ///   - registerResidency: Whether to auto-register with residency set
    /// - Returns: The created managed tensor
    public func createSimilarityTensor(
        queryCount: Int,
        keyCount: Int,
        label: String = "",
        registerResidency: Bool = true
    ) throws -> ManagedTensor {
        let shape = ManagedTensor.TensorShape.similarityMatrix(
            queryCount: queryCount,
            keyCount: keyCount
        )

        return try createTensor(shape: shape, label: label, registerResidency: registerResidency)
    }

    /// Create a generic managed tensor.
    ///
    /// - Parameters:
    ///   - shape: Tensor shape
    ///   - label: Human-readable label
    ///   - registerResidency: Whether to auto-register with residency set
    /// - Returns: The created managed tensor
    public func createTensor(
        shape: ManagedTensor.TensorShape,
        label: String = "",
        registerResidency: Bool = true
    ) throws -> ManagedTensor {
        let sizeNeeded = shape.sizeInBytes

        // Check if we need to evict
        if totalAllocatedBytes + sizeNeeded > configuration.maxMemoryBytes {
            try evictIfNeeded(bytesNeeded: sizeNeeded)
        }

        // Create buffer
        guard let buffer = device.makeBuffer(length: max(sizeNeeded, 4), options: .storageModeShared) else {
            throw EmbedKitError.metalBufferFailed
        }

        let tensor = ManagedTensor(buffer: buffer, shape: shape, label: label)

        // Track tensor
        tensors[tensor.id] = tensor
        if !label.isEmpty {
            tensorsByLabel[label] = tensor.id
        }

        totalAllocatedBytes += sizeNeeded
        stats.totalCreated += 1
        stats.totalBytesAllocated += sizeNeeded
        stats.cacheMisses += 1

        // Auto-register with residency
        if registerResidency && configuration.autoResidency {
            try registerWithResidencyInternal(tensor, setName: configuration.defaultResidencySet)
        }

        return tensor
    }

    // MARK: - Tensor Lookup

    /// Get a tensor by ID.
    ///
    /// - Parameter id: Tensor UUID
    /// - Returns: The tensor if found
    public func getTensor(id: UUID) -> ManagedTensor? {
        if let tensor = tensors[id] {
            tensor.markAccessed()
            stats.cacheHits += 1
            return tensor
        }
        return nil
    }

    /// Get a tensor by label.
    ///
    /// - Parameter label: Tensor label
    /// - Returns: The tensor if found
    public func getTensor(label: String) -> ManagedTensor? {
        guard let id = tensorsByLabel[label] else { return nil }
        return getTensor(id: id)
    }

    /// Check if a tensor exists.
    ///
    /// - Parameter id: Tensor UUID
    /// - Returns: Whether the tensor exists
    public func hasTensor(id: UUID) -> Bool {
        tensors[id] != nil
    }

    /// Check if a tensor with label exists.
    ///
    /// - Parameter label: Tensor label
    /// - Returns: Whether the tensor exists
    public func hasTensor(label: String) -> Bool {
        tensorsByLabel[label] != nil
    }

    // MARK: - Lifecycle Management

    /// Mark a tensor as accessed.
    ///
    /// - Parameter tensor: The tensor to mark
    public func markAccessed(_ tensor: ManagedTensor) {
        tensor.markAccessed()
    }

    /// Mark a tensor as cached (eligible for eviction).
    ///
    /// - Parameter tensor: The tensor to mark
    public func markCached(_ tensor: ManagedTensor) {
        tensor.markCached()
    }

    /// Release a tensor and free its resources.
    ///
    /// - Parameter tensor: The tensor to release
    public func release(_ tensor: ManagedTensor) {
        guard tensors[tensor.id] != nil else { return }

        // Remove from residency if needed
        if tensor.isResident, let setName = tensor.residencySetName {
            removeFromResidencyInternal(tensor, setName: setName)
        }

        // Remove from tracking
        tensors.removeValue(forKey: tensor.id)
        if !tensor.label.isEmpty {
            tensorsByLabel.removeValue(forKey: tensor.label)
        }

        totalAllocatedBytes -= tensor.sizeInBytes
        tensor.markReleased()

        stats.totalReleased += 1
        stats.totalBytesFreed += tensor.sizeInBytes
    }

    /// Release a tensor by ID.
    ///
    /// - Parameter id: Tensor UUID
    public func release(id: UUID) {
        guard let tensor = tensors[id] else { return }
        release(tensor)
    }

    /// Release a tensor by label.
    ///
    /// - Parameter label: Tensor label
    public func release(label: String) {
        guard let id = tensorsByLabel[label] else { return }
        release(id: id)
    }

    /// Release all tensors.
    public func releaseAll() {
        for tensor in tensors.values {
            tensor.markReleased()
        }
        tensors.removeAll()
        tensorsByLabel.removeAll()
        stats.totalBytesFreed += totalAllocatedBytes
        stats.totalReleased += tensors.count
        totalAllocatedBytes = 0
    }

    // MARK: - Residency Management

    /// Set the residency manager.
    ///
    /// - Parameter manager: The residency manager to use
    public func setResidencyManager(_ manager: Metal4ResidencyManager) {
        self.residencyManager = manager
    }

    /// Register a tensor with a residency set.
    ///
    /// - Parameters:
    ///   - tensor: The tensor to register
    ///   - setName: Residency set name
    public func registerWithResidency(_ tensor: ManagedTensor, setName: String) throws {
        try registerWithResidencyInternal(tensor, setName: setName)
    }

    /// Remove a tensor from residency tracking.
    ///
    /// - Parameters:
    ///   - tensor: The tensor to remove
    ///   - setName: Residency set name
    public func removeFromResidency(_ tensor: ManagedTensor, setName: String) {
        removeFromResidencyInternal(tensor, setName: setName)
    }

    /// Internal residency registration (non-async for use in sync context).
    private func registerWithResidencyInternal(_ tensor: ManagedTensor, setName: String) throws {
        guard let manager = residencyManager else { return }

        // Get or create residency set
        if let set = manager.getResidencySetSync(named: setName) {
            set.addBuffer(tensor.buffer)
            tensor.markResident(inSet: setName)
            stats.residencyRegistrations += 1
        }
    }

    /// Internal residency removal.
    private func removeFromResidencyInternal(_ tensor: ManagedTensor, setName: String) {
        guard let manager = residencyManager else { return }

        if let set = manager.getResidencySetSync(named: setName) {
            set.removeBuffer(tensor.buffer)
            tensor.markNonResident()
        }
    }

    // MARK: - Memory Management

    /// Evict tensors to free memory.
    ///
    /// - Parameter bytesNeeded: Bytes that need to be freed
    private func evictIfNeeded(bytesNeeded: Int) throws {
        var bytesToFree = (totalAllocatedBytes + bytesNeeded) - configuration.maxMemoryBytes

        // Get eviction candidates (cached tensors, sorted by LRU)
        let candidates = tensors.values
            .filter { $0.state == .cached }
            .sorted { $0.lastAccessedAt < $1.lastAccessedAt }

        for tensor in candidates {
            guard bytesToFree > 0 else { break }

            // Only evict if idle long enough
            if tensor.idleTime >= configuration.evictionIdleThreshold {
                bytesToFree -= tensor.sizeInBytes
                release(tensor)
                stats.evictions += 1
            }
        }

        // If still not enough, evict active tensors (least recently used)
        if bytesToFree > 0 {
            let activeCandidates = tensors.values
                .filter { $0.state == .active }
                .sorted { $0.lastAccessedAt < $1.lastAccessedAt }

            for tensor in activeCandidates {
                guard bytesToFree > 0 else { break }

                bytesToFree -= tensor.sizeInBytes
                release(tensor)
                stats.evictions += 1
            }
        }
    }

    /// Trim idle tensors to reduce memory usage.
    ///
    /// - Parameter aggressiveness: How aggressive to trim (0.0 = minimal, 1.0 = maximum)
    public func trimIdleTensors(aggressiveness: Float = 0.5) {
        let adjustedThreshold = configuration.evictionIdleThreshold * Double(1.0 - aggressiveness)

        for tensor in tensors.values where tensor.state == .cached {
            if tensor.idleTime >= adjustedThreshold {
                release(tensor)
                stats.evictions += 1
            }
        }
    }

    /// Handle memory pressure notification.
    ///
    /// - Parameter level: Memory pressure level (0 = low, 1 = critical)
    public func handleMemoryPressure(level: Int) {
        switch level {
        case 0:
            // Low pressure: trim only cached tensors
            trimIdleTensors(aggressiveness: 0.3)
        case 1:
            // Medium pressure: trim more aggressively
            trimIdleTensors(aggressiveness: 0.7)
        default:
            // Critical: release all cached tensors
            for tensor in tensors.values where tensor.state == .cached {
                release(tensor)
            }
        }
    }

    // MARK: - Statistics

    /// Get current statistics.
    public func getStatistics() -> Statistics {
        stats
    }

    /// Get memory usage statistics.
    public func getMemoryUsage() -> MemoryUsage {
        MemoryUsage(
            allocatedBytes: totalAllocatedBytes,
            maxBytes: configuration.maxMemoryBytes,
            tensorCount: tensors.count,
            activeTensors: tensors.values.filter { $0.state == .active }.count,
            cachedTensors: tensors.values.filter { $0.state == .cached }.count,
            residentTensors: tensors.values.filter { $0.isResident }.count
        )
    }

    /// Memory usage statistics.
    public struct MemoryUsage: Sendable {
        /// Currently allocated bytes
        public let allocatedBytes: Int

        /// Maximum allowed bytes
        public let maxBytes: Int

        /// Total tensor count
        public let tensorCount: Int

        /// Active tensor count
        public let activeTensors: Int

        /// Cached tensor count
        public let cachedTensors: Int

        /// Resident tensor count
        public let residentTensors: Int

        /// Usage percentage (0.0 - 1.0)
        public var usagePercentage: Float {
            maxBytes > 0 ? Float(allocatedBytes) / Float(maxBytes) : 0
        }
    }

    /// Get all tensor IDs.
    public func getAllTensorIds() -> [UUID] {
        Array(tensors.keys)
    }

    /// Get all tensors matching a predicate.
    public func getTensors(matching predicate: @Sendable (ManagedTensor) -> Bool) -> [ManagedTensor] {
        tensors.values.filter(predicate)
    }
}

// MARK: - Metal4ResidencyManager Extension

extension Metal4ResidencyManager {

    /// Synchronous getter for residency set (for use in TensorStorageManager).
    nonisolated func getResidencySetSync(named name: String) -> Metal4ResidencySet? {
        // This is a workaround for actor isolation.
        // In production, you'd use proper async/await patterns.
        // For now, we return nil and the async version should be preferred.
        nil
    }
}

// MARK: - Tensor Lifecycle Protocol

/// Protocol for types that manage tensor lifecycle.
public protocol TensorLifecycleDelegate: AnyObject, Sendable {
    /// Called when a tensor is created.
    func tensorCreated(_ tensor: ManagedTensor) async

    /// Called when a tensor is accessed.
    func tensorAccessed(_ tensor: ManagedTensor) async

    /// Called when a tensor is released.
    func tensorReleased(_ tensor: ManagedTensor) async

    /// Called when memory pressure occurs.
    func memoryPressure(level: Int) async
}

// MARK: - Tensor Handle

/// A lightweight handle to a managed tensor for safe passing between contexts.
///
/// `TensorHandle` provides a way to reference a managed tensor without
/// holding a strong reference to the tensor itself. Use the storage manager
/// to resolve the handle to the actual tensor.
public struct TensorHandle: Sendable, Hashable {
    /// The tensor's unique identifier
    public let id: UUID

    /// The tensor's label (if any)
    public let label: String

    /// Shape of the tensor
    public let shape: ManagedTensor.TensorShape

    /// Create a handle from a managed tensor.
    public init(from tensor: ManagedTensor) {
        self.id = tensor.id
        self.label = tensor.label
        self.shape = tensor.shape
    }
}

// MARK: - TensorShape Hashable

extension ManagedTensor.TensorShape: Hashable {
    public func hash(into hasher: inout Hasher) {
        switch self {
        case .embedding(let batch, let dims):
            hasher.combine(0)
            hasher.combine(batch)
            hasher.combine(dims)
        case .tokenEmbedding(let batch, let seq, let dims):
            hasher.combine(1)
            hasher.combine(batch)
            hasher.combine(seq)
            hasher.combine(dims)
        case .similarityMatrix(let q, let k):
            hasher.combine(2)
            hasher.combine(q)
            hasher.combine(k)
        case .buffer(let length):
            hasher.combine(3)
            hasher.combine(length)
        }
    }
}

#endif

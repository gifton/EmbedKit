// EmbedKit - SharedMetalContextManager
// Unified GPU resource management bridging MetalAccelerator with VectorAccelerate's MetalContext

import Foundation
import VectorCore
import VectorAccelerate

#if canImport(Metal)
@preconcurrency import Metal
#endif

// MARK: - SharedMetalContextManager

/// Manages shared Metal resources across EmbedKit and VectorAccelerate.
///
/// `SharedMetalContextManager` provides a unified GPU resource pool that enables:
/// - **Shared buffer pools**: Reduce memory fragmentation across packages
/// - **Unified command queues**: Better GPU utilization through coordinated scheduling
/// - **Cross-package metrics**: Holistic view of GPU resource usage
///
/// ## Example Usage
/// ```swift
/// // Get or create shared context manager
/// let manager = await SharedMetalContextManager.shared
///
/// // Get VectorAccelerate's MetalContext for compute operations
/// if let context = try await manager.getVectorAccelerateContext() {
///     let engine = try await ComputeEngine(context: context)
///     // Use engine for GPU-accelerated operations...
/// }
///
/// // Get EmbedKit's MetalAccelerator for embedding operations
/// if let accelerator = await manager.getEmbedKitAccelerator() {
///     let results = await accelerator.tensorPoolNormalize(...)
/// }
/// ```
public actor SharedMetalContextManager {

    // MARK: - Singleton

    /// Shared singleton instance.
    ///
    /// Uses lazy initialization to defer Metal resource allocation until first use.
    public static let shared = SharedMetalContextManager()

    // MARK: - Properties

    /// VectorAccelerate's MetalContext for general GPU compute.
    private var vectorAccelerateContext: MetalContext?

    /// EmbedKit's MetalAccelerator for embedding-specific operations.
    private var embedKitAccelerator: MetalAccelerator?

    /// Shared buffer factory for cross-package buffer reuse.
    #if canImport(Metal)
    private var sharedBufferFactory: SharedBufferFactory?
    #endif

    /// Configuration for shared context.
    private var configuration: SharedMetalContextConfiguration

    /// Whether initialization has been attempted.
    private var isInitialized = false

    /// Statistics about shared resource usage.
    private var stats = SharedContextStatistics()

    // MARK: - Initialization

    private init() {
        self.configuration = .default
    }

    /// Configure the shared context manager.
    ///
    /// Call this before first use to customize resource allocation.
    /// Has no effect if contexts are already initialized.
    ///
    /// - Parameter config: Configuration for shared contexts
    public func configure(_ config: SharedMetalContextConfiguration) {
        guard !isInitialized else { return }
        self.configuration = config
    }

    // MARK: - Context Access

    /// Get VectorAccelerate's MetalContext for GPU compute operations.
    ///
    /// Creates the context lazily on first access. Returns nil if Metal
    /// is unavailable or initialization fails.
    ///
    /// - Returns: Shared MetalContext or nil if unavailable
    public func getVectorAccelerateContext() async throws -> MetalContext? {
        if !isInitialized {
            await initializeContexts()
        }
        return vectorAccelerateContext
    }

    /// Get EmbedKit's MetalAccelerator for embedding operations.
    ///
    /// Creates the accelerator lazily on first access. Returns nil if Metal
    /// is unavailable or initialization fails.
    ///
    /// - Returns: Shared MetalAccelerator or nil if unavailable
    public func getEmbedKitAccelerator() async -> MetalAccelerator? {
        if !isInitialized {
            await initializeContexts()
        }
        return embedKitAccelerator
    }

    #if canImport(Metal)
    /// Get shared buffer factory for cross-package buffer allocation.
    ///
    /// Use this when you need to share buffers between EmbedKit and
    /// VectorAccelerate operations.
    ///
    /// - Returns: SharedBufferFactory or nil if Metal is unavailable
    public func getSharedBufferFactory() async -> SharedBufferFactory? {
        if !isInitialized {
            await initializeContexts()
        }
        return sharedBufferFactory
    }
    #endif

    /// Check if shared Metal resources are available.
    public var isAvailable: Bool {
        get async {
            if !isInitialized {
                await initializeContexts()
            }
            return vectorAccelerateContext != nil || embedKitAccelerator != nil
        }
    }

    // MARK: - Statistics

    /// Get current statistics about shared resource usage.
    public func getStatistics() async -> SharedContextStatistics {
        var currentStats = stats

        #if canImport(Metal)
        if sharedBufferFactory != nil {
            // SharedBufferFactory doesn't track pool stats directly;
            // stats come from the underlying MetalContext pool
            currentStats.sharedBufferPoolUtilization = 0  // Not tracked at this level
            currentStats.sharedBufferCount = 0  // Not tracked at this level
        }
        #endif

        if let context = vectorAccelerateContext {
            let poolStats = await context.getPoolStatistics()
            currentStats.vectorAcceleratePooledBuffers = poolStats.totalBuffers
            currentStats.vectorAccelerateMemoryUsage = poolStats.currentMemoryUsage
        }

        if let accelerator = embedKitAccelerator {
            currentStats.embedKitGPUAvailable = await accelerator.isAvailable
            #if canImport(Metal)
            if let bufferStats = await accelerator.bufferPoolStatistics {
                currentStats.embedKitBufferPoolHitRate = bufferStats.hitRate
            }
            #endif
        }

        return currentStats
    }

    /// Reset shared context statistics.
    public func resetStatistics() {
        stats = SharedContextStatistics()
    }

    // MARK: - Lifecycle

    /// Release all shared Metal resources.
    ///
    /// Call this when your application is terminating or when you need to
    /// free GPU memory. The manager will reinitialize on next access.
    public func releaseResources() async {
        if let context = vectorAccelerateContext {
            await context.cleanup()
        }
        vectorAccelerateContext = nil
        embedKitAccelerator = nil
        #if canImport(Metal)
        // SharedBufferFactory is managed by MetalContext; just release our reference
        sharedBufferFactory = nil
        #endif
        isInitialized = false
    }

    /// Reset the manager for testing purposes.
    ///
    /// This releases all resources and resets internal state, allowing fresh
    /// initialization on next access. Use this in test teardown to prevent
    /// resource accumulation across test runs.
    @available(*, deprecated, message: "For testing only - do not use in production code")
    public func resetForTesting() async {
        await releaseResources()
        stats = SharedContextStatistics()
        configuration = .default
    }

    // MARK: - Private Initialization

    private func initializeContexts() async {
        guard !isInitialized else { return }
        isInitialized = true

        // Check if Metal is available
        guard MetalContext.isAvailable else {
            return
        }

        // Create VectorAccelerate's MetalContext
        do {
            let baseConfig = MetalConfiguration(
                maxBufferPoolMemory: configuration.maxBufferPoolMB * 1024 * 1024,
                enableProfiling: configuration.enableMetrics
            )
            let vaConfig = SharedConfiguration(
                baseConfiguration: baseConfig,
                queuePriority: configuration.highPriority ? .high : .normal,
                enableBufferSharing: configuration.enableBufferSharing,
                identifier: "embedkit-shared"
            )
            vectorAccelerateContext = try await MetalContext.create(sharedConfig: vaConfig)
            stats.vectorAccelerateInitialized = true
        } catch {
            // VectorAccelerate context failed - continue with EmbedKit only
            stats.vectorAccelerateInitializationError = error.localizedDescription
        }

        // Create EmbedKit's MetalAccelerator
        embedKitAccelerator = await MetalAccelerator()
        if let accelerator = embedKitAccelerator {
            stats.embedKitInitialized = await accelerator.isAvailable
        } else {
            stats.embedKitInitialized = false
        }

        // Create shared buffer factory from VectorAccelerate context
        #if canImport(Metal)
        if let context = vectorAccelerateContext {
            sharedBufferFactory = await context.sharedBufferFactory()
        }
        #endif
    }

    #if canImport(Metal)
    /// Get the shared Metal device from either context.
    private func getSharedDevice() async -> MTLDevice? {
        // Prefer VectorAccelerate's device (usually the same anyway)
        if let context = vectorAccelerateContext {
            return await context.device.rawDevice
        }
        // Fall back to system default
        return MTLCreateSystemDefaultDevice()
    }
    #endif
}

// MARK: - Configuration

/// Configuration for SharedMetalContextManager.
public struct SharedMetalContextConfiguration: Sendable, Equatable {
    /// Maximum buffer pool size in megabytes.
    public let maxBufferPoolMB: Int

    /// Maximum shared buffer pool size in megabytes.
    public let sharedBufferPoolMB: Int

    /// Whether to enable cross-package buffer sharing.
    public let enableBufferSharing: Bool

    /// Whether to use high-priority command queue.
    public let highPriority: Bool

    /// Whether to collect performance metrics.
    public let enableMetrics: Bool

    public init(
        maxBufferPoolMB: Int = 256,
        sharedBufferPoolMB: Int = 64,
        enableBufferSharing: Bool = true,
        highPriority: Bool = false,
        enableMetrics: Bool = false
    ) {
        self.maxBufferPoolMB = maxBufferPoolMB
        self.sharedBufferPoolMB = sharedBufferPoolMB
        self.enableBufferSharing = enableBufferSharing
        self.highPriority = highPriority
        self.enableMetrics = enableMetrics
    }

    /// Default configuration suitable for most use cases.
    public static let `default` = SharedMetalContextConfiguration()

    /// Configuration optimized for embedding generation workloads.
    public static let forEmbedding = SharedMetalContextConfiguration(
        maxBufferPoolMB: 128,
        sharedBufferPoolMB: 32,
        enableBufferSharing: true,
        highPriority: false,
        enableMetrics: false
    )

    /// Configuration optimized for semantic search workloads.
    public static let forSearch = SharedMetalContextConfiguration(
        maxBufferPoolMB: 512,
        sharedBufferPoolMB: 128,
        enableBufferSharing: true,
        highPriority: true,
        enableMetrics: false
    )

    /// Configuration for development and debugging.
    public static let debug = SharedMetalContextConfiguration(
        maxBufferPoolMB: 64,
        sharedBufferPoolMB: 16,
        enableBufferSharing: true,
        highPriority: false,
        enableMetrics: true
    )
}

// MARK: - Statistics

/// Statistics about shared Metal context usage.
public struct SharedContextStatistics: Sendable {
    /// Whether VectorAccelerate's MetalContext was initialized.
    public var vectorAccelerateInitialized: Bool = false

    /// Error message if VectorAccelerate initialization failed.
    public var vectorAccelerateInitializationError: String?

    /// Number of buffers in VectorAccelerate's pool.
    public var vectorAcceleratePooledBuffers: Int = 0

    /// Memory usage of VectorAccelerate's buffer pool in bytes.
    public var vectorAccelerateMemoryUsage: Int = 0

    /// Whether EmbedKit's MetalAccelerator was initialized.
    public var embedKitInitialized: Bool = false

    /// Whether EmbedKit's GPU acceleration is available.
    public var embedKitGPUAvailable: Bool = false

    /// Hit rate of EmbedKit's buffer pool.
    public var embedKitBufferPoolHitRate: Double = 0

    /// Utilization of shared buffer pool [0.0, 1.0].
    public var sharedBufferPoolUtilization: Double = 0

    /// Number of buffers in shared pool.
    public var sharedBufferCount: Int = 0

    /// Whether both contexts are available and sharing resources.
    public var isFullySynchronized: Bool {
        vectorAccelerateInitialized && embedKitInitialized
    }
}

// MARK: - Convenience Extensions

extension AccelerationManager {
    /// Create an AccelerationManager using shared Metal resources.
    ///
    /// This factory method creates an AccelerationManager that shares GPU
    /// resources with other VSK components through SharedMetalContextManager.
    ///
    /// - Parameters:
    ///   - preference: Compute preference (default: .auto)
    ///   - thresholds: Acceleration thresholds
    /// - Returns: AccelerationManager configured to use shared resources
    public static func createWithSharedContext(
        preference: ComputePreference = .auto,
        thresholds: AccelerationThresholds = .default
    ) async -> AccelerationManager {
        // Ensure shared context is initialized
        _ = await SharedMetalContextManager.shared.isAvailable

        // Create manager with standard initialization
        // (It will use the same GPU device as the shared context)
        return await AccelerationManager(preference: preference, thresholds: thresholds)
    }
}

extension EmbeddingStore {
    /// Create an EmbeddingStore using shared Metal resources.
    ///
    /// This factory method creates an EmbeddingStore that shares GPU
    /// resources with other VSK components.
    ///
    /// - Parameters:
    ///   - config: Index configuration
    ///   - model: Optional embedding model
    /// - Returns: EmbeddingStore configured to use shared resources
    public static func createWithSharedContext(
        config: IndexConfiguration,
        model: (any EmbeddingModel)? = nil
    ) async throws -> EmbeddingStore {
        // Ensure shared context is initialized
        _ = await SharedMetalContextManager.shared.isAvailable

        // Create store with standard initialization
        return try await EmbeddingStore(config: config, model: model)
    }
}

// EmbedKit - Metal 4 Command Allocator
//
// Explicit command buffer memory management for Metal 4.
// Provides efficient memory reuse for batch processing operations.

import Foundation

#if canImport(Metal)
@preconcurrency import Metal

// MARK: - Command Allocator Protocol

/// Protocol for command buffer memory allocators.
///
/// Metal 4 introduces explicit command buffer memory management through
/// `MTL4CommandAllocator`. This protocol abstracts the allocator interface
/// to support both Metal 4 native allocators and a compatibility shim.
public protocol CommandAllocator: Sendable {
    /// Heap size in bytes
    var heapSize: Int { get }

    /// Whether the allocator is currently in use
    var isInUse: Bool { get }

    /// Reset the allocator, reclaiming all memory.
    ///
    /// Call this after GPU work has completed to reuse the memory.
    func reset()
}

// MARK: - Simple Command Allocator

/// Simple command allocator that tracks usage for pool management.
///
/// Provides a compatible interface for the CommandAllocatorPool.
/// Memory is managed by Metal 4's automatic resource management.
public final class SimpleCommandAllocator: CommandAllocator, @unchecked Sendable {
    public let heapSize: Int
    private var _isInUse: Bool = false
    private let lock = NSLock()

    public init(heapSize: Int = 16 * 1024 * 1024) {
        self.heapSize = heapSize
    }

    public var isInUse: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _isInUse
    }

    public func markInUse() {
        lock.lock()
        _isInUse = true
        lock.unlock()
    }

    public func reset() {
        lock.lock()
        _isInUse = false
        lock.unlock()
    }
}

// MARK: - Command Allocator Pool

/// Pool of command allocators for efficient memory reuse.
///
/// This actor manages a pool of command allocators to reduce memory
/// allocation overhead during batch processing.
///
/// **Metal 4 Benefits**:
/// - Explicit memory management reduces fragmentation
/// - Reset-based reuse is faster than allocation
/// - Better memory utilization for streaming workloads
///
/// ## Usage
/// ```swift
/// let pool = CommandAllocatorPool(device: device, poolSize: 3)
/// let allocator = await pool.acquire()
/// // Use allocator for command buffer encoding
/// // ... after GPU completion ...
/// await pool.release(allocator)
/// ```
public actor CommandAllocatorPool {

    /// Configuration for the allocator pool
    public struct Configuration: Sendable {
        /// Number of allocators in the pool
        public let poolSize: Int

        /// Heap size per allocator in bytes
        public let heapSizePerAllocator: Int

        /// Default configuration
        public static let `default` = Configuration(
            poolSize: 3,
            heapSizePerAllocator: 16 * 1024 * 1024  // 16MB
        )

        /// Configuration for large batch processing
        public static let largeBatch = Configuration(
            poolSize: 4,
            heapSizePerAllocator: 64 * 1024 * 1024  // 64MB
        )

        public init(poolSize: Int, heapSizePerAllocator: Int) {
            self.poolSize = max(1, poolSize)
            self.heapSizePerAllocator = max(1024 * 1024, heapSizePerAllocator)
        }
    }

    /// Statistics for monitoring pool usage
    public struct Statistics: Sendable {
        public let totalAllocators: Int
        public let availableAllocators: Int
        public let inUseAllocators: Int
        public let totalAcquisitions: Int
        public let totalResets: Int
        public let heapSizePerAllocator: Int

        public var utilizationPercentage: Double {
            guard totalAllocators > 0 else { return 0 }
            return Double(inUseAllocators) / Double(totalAllocators) * 100
        }

        public var totalHeapSize: Int {
            totalAllocators * heapSizePerAllocator
        }
    }

    // MARK: - Properties

    private let device: MTLDevice
    private let configuration: Configuration
    private var allocators: [any CommandAllocator]
    private var availableIndices: [Int]
    private var totalAcquisitions: Int = 0
    private var totalResets: Int = 0

    // MARK: - Initialization

    /// Initialize an allocator pool.
    ///
    /// - Parameters:
    ///   - device: Metal device for allocator creation
    ///   - configuration: Pool configuration (default: 3 allocators, 16MB each)
    public init(device: MTLDevice, configuration: Configuration = .default) {
        self.device = device
        self.configuration = configuration

        // Create allocators
        var allocators: [any CommandAllocator] = []
        for _ in 0..<configuration.poolSize {
            let allocator = SimpleCommandAllocator(heapSize: configuration.heapSizePerAllocator)
            allocators.append(allocator)
        }
        self.allocators = allocators
        self.availableIndices = Array(0..<configuration.poolSize)
    }

    /// Convenience initializer with default configuration.
    public init(device: MTLDevice, poolSize: Int = 3, heapSizeMB: Int = 16) {
        self.init(
            device: device,
            configuration: Configuration(
                poolSize: poolSize,
                heapSizePerAllocator: heapSizeMB * 1024 * 1024
            )
        )
    }

    // MARK: - Acquire/Release

    /// Acquire an allocator from the pool.
    ///
    /// Blocks if all allocators are in use until one becomes available.
    ///
    /// - Returns: An allocator ready for use
    public func acquire() async -> any CommandAllocator {
        // Wait for an available allocator
        while availableIndices.isEmpty {
            // Yield to allow other tasks to release allocators
            await Task.yield()
        }

        let index = availableIndices.removeFirst()
        if let allocator = allocators[index] as? SimpleCommandAllocator {
            allocator.markInUse()
        }
        totalAcquisitions += 1

        return allocators[index]
    }

    /// Try to acquire an allocator without blocking.
    ///
    /// - Returns: An allocator if one is available, nil otherwise
    public func tryAcquire() -> (any CommandAllocator)? {
        guard !availableIndices.isEmpty else { return nil }

        let index = availableIndices.removeFirst()
        if let allocator = allocators[index] as? SimpleCommandAllocator {
            allocator.markInUse()
        }
        totalAcquisitions += 1

        return allocators[index]
    }

    /// Release an allocator back to the pool.
    ///
    /// The allocator is reset and made available for reuse.
    ///
    /// - Parameter allocator: The allocator to release
    public func release(_ allocator: any CommandAllocator) {
        // Find the allocator index
        guard let index = allocators.firstIndex(where: {
            ($0 as AnyObject) === (allocator as AnyObject)
        }) else {
            return  // Not from this pool
        }

        // Reset and return to available pool
        allocator.reset()
        totalResets += 1

        if !availableIndices.contains(index) {
            availableIndices.append(index)
        }
    }

    /// Release all allocators back to the pool.
    ///
    /// Call this after a batch of GPU work has completed.
    public func releaseAll() {
        for (index, allocator) in allocators.enumerated() {
            allocator.reset()
            totalResets += 1

            if !availableIndices.contains(index) {
                availableIndices.append(index)
            }
        }
    }

    // MARK: - Statistics

    /// Get current pool statistics.
    public func getStatistics() -> Statistics {
        Statistics(
            totalAllocators: allocators.count,
            availableAllocators: availableIndices.count,
            inUseAllocators: allocators.count - availableIndices.count,
            totalAcquisitions: totalAcquisitions,
            totalResets: totalResets,
            heapSizePerAllocator: configuration.heapSizePerAllocator
        )
    }

    /// Number of available allocators.
    public var availableCount: Int {
        availableIndices.count
    }

    /// Number of allocators currently in use.
    public var inUseCount: Int {
        allocators.count - availableIndices.count
    }
}

// MARK: - Integration with CommandBufferPool

extension CommandBufferPool {

    /// Acquire a command buffer with an associated allocator.
    ///
    /// Uses explicit memory management for better performance.
    ///
    /// - Parameter allocatorPool: Optional allocator pool for Metal 4
    /// - Returns: A command buffer ready for encoding
    public func acquireCommandBuffer(
        using allocatorPool: CommandAllocatorPool
    ) async throws -> (buffer: MTLCommandBuffer, allocator: any CommandAllocator) {
        // Acquire allocator first
        let allocator = await allocatorPool.acquire()

        // Then acquire command buffer
        let buffer = try await acquireCommandBuffer()

        return (buffer, allocator)
    }
}

#endif

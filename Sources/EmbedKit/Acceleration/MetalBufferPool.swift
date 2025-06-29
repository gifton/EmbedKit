import Foundation
@preconcurrency import Metal
import OSLog

/// Size bucket for buffer pooling
private struct BufferSizeBucket: Hashable {
    let sizeClass: Int
    
    init(requestedSize: Int) {
        // Round up to nearest power of 2 for better allocation patterns
        let powerOf2 = requestedSize.nextPowerOf2()
        self.sizeClass = powerOf2
    }
}

private extension Int {
    /// Returns the next power of 2 greater than or equal to self
    func nextPowerOf2() -> Int {
        guard self > 0 else { return 1 }
        
        var n = self - 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        n |= n >> 32
        
        return n + 1
    }
}

/// Buffer wrapper that tracks allocation metadata
private final class PooledBuffer: @unchecked Sendable {
    let buffer: MTLBuffer
    let actualSize: Int
    let bucket: BufferSizeBucket
    var lastUsed: Date
    var useCount: Int = 0
    
    init(buffer: MTLBuffer, actualSize: Int, bucket: BufferSizeBucket) {
        self.buffer = buffer
        self.actualSize = actualSize
        self.bucket = bucket
        self.lastUsed = Date()
    }
    
    func markUsed() {
        lastUsed = Date()
        useCount += 1
    }
}

/// High-performance Metal buffer pool with automatic growth and memory pressure handling
///
/// This pool uses MTLHeap for efficient memory suballocation and implements
/// size-based bucketing to minimize fragmentation.
///
/// Features:
/// - O(1) buffer acquisition and release
/// - Automatic heap growth based on usage patterns
/// - Memory pressure responsive
/// - Usage statistics and monitoring
/// - Thread-safe actor isolation
public actor MetalBufferPool {
    nonisolated private let logger = EmbedKitLogger.metal()
    
    // Core resources
    nonisolated private let device: MTLDevice
    private var heaps: [MTLHeap] = []
    
    // Buffer pools organized by size buckets
    private var availableBuffers: [BufferSizeBucket: [PooledBuffer]] = [:]
    private var inUseBuffers: Set<ObjectIdentifier> = []
    
    // Configuration
    private let initialHeapSize: Int
    private let maxHeapSize: Int
    private let heapGrowthFactor: Double = 1.5
    
    // Statistics
    private var totalAllocatedBytes: Int = 0
    private var totalBuffersCreated: Int = 0
    private var bufferHits: Int = 0
    private var bufferMisses: Int = 0
    
    // Memory pressure handling
    private var isUnderMemoryPressure: Bool = false
    private let maxIdleBuffersPerBucket: Int = 10
    
    public struct PoolStatistics: Sendable {
        public let totalHeaps: Int
        public let totalAllocatedBytes: Int
        public let totalBuffersCreated: Int
        public let bufferHitRate: Double
        public let bucketsInUse: Int
        public let averageBuffersPerBucket: Double
        
        public init(totalHeaps: Int, totalAllocatedBytes: Int, totalBuffersCreated: Int,
                    bufferHits: Int, bufferMisses: Int, bucketsInUse: Int, totalAvailableBuffers: Int) {
            self.totalHeaps = totalHeaps
            self.totalAllocatedBytes = totalAllocatedBytes
            self.totalBuffersCreated = totalBuffersCreated
            let totalRequests = bufferHits + bufferMisses
            self.bufferHitRate = totalRequests > 0 ? Double(bufferHits) / Double(totalRequests) : 0
            self.bucketsInUse = bucketsInUse
            self.averageBuffersPerBucket = bucketsInUse > 0 ? Double(totalAvailableBuffers) / Double(bucketsInUse) : 0
        }
    }
    
    public init(device: MTLDevice, initialHeapSizeMB: Int = 64, maxHeapSizeMB: Int = 512) async throws {
        self.device = device
        self.initialHeapSize = initialHeapSizeMB * 1024 * 1024
        self.maxHeapSize = maxHeapSizeMB * 1024 * 1024
        
        // Create initial heap
        try createHeap(size: initialHeapSize)
        
        logger.info("MetalBufferPool initialized with \(initialHeapSizeMB)MB initial heap")
    }
    
    /// Acquire a buffer from the pool or create a new one
    public func acquireBuffer(size: Int) async throws -> MTLBuffer {
        let bucket = BufferSizeBucket(requestedSize: size)
        
        // Try to get from pool first
        if var bucketBuffers = availableBuffers[bucket], !bucketBuffers.isEmpty {
            let pooledBuffer = bucketBuffers.removeLast()
            availableBuffers[bucket] = bucketBuffers
            
            pooledBuffer.markUsed()
            inUseBuffers.insert(ObjectIdentifier(pooledBuffer.buffer))
            
            bufferHits += 1
            logger.debug("Buffer hit for size \(size) (bucket: \(bucket.sizeClass))")
            
            return pooledBuffer.buffer
        }
        
        // Need to allocate new buffer
        bufferMisses += 1
        logger.trace("Buffer miss for size \(size), allocating new")
        
        // Try to allocate from existing heaps
        for heap in heaps {
            if let buffer = allocateFromHeap(heap, size: bucket.sizeClass) {
                let pooledBuffer = PooledBuffer(buffer: buffer, actualSize: bucket.sizeClass, bucket: bucket)
                inUseBuffers.insert(ObjectIdentifier(buffer))
                totalBuffersCreated += 1
                return buffer
            }
        }
        
        // Need to grow heap
        try await growHeapIfPossible()
        
        // Try allocation again after growth
        for heap in heaps {
            if let buffer = allocateFromHeap(heap, size: bucket.sizeClass) {
                let pooledBuffer = PooledBuffer(buffer: buffer, actualSize: bucket.sizeClass, bucket: bucket)
                inUseBuffers.insert(ObjectIdentifier(buffer))
                totalBuffersCreated += 1
                return buffer
            }
        }
        
        throw MetalBufferPoolError.allocationFailed(size: size)
    }
    
    /// Release a buffer back to the pool
    public func releaseBuffer(_ buffer: MTLBuffer) async {
        let bufferId = ObjectIdentifier(buffer)
        guard inUseBuffers.remove(bufferId) != nil else {
            logger.warning("Attempted to release untracked buffer")
            return
        }
        
        // Determine bucket size from buffer length
        let bucket = BufferSizeBucket(requestedSize: buffer.length)
        let pooledBuffer = PooledBuffer(buffer: buffer, actualSize: buffer.length, bucket: bucket)
        
        // Add back to pool if not under memory pressure
        if !isUnderMemoryPressure {
            var bucketBuffers = availableBuffers[bucket] ?? []
            
            // Limit number of idle buffers per bucket
            if bucketBuffers.count < maxIdleBuffersPerBucket {
                bucketBuffers.append(pooledBuffer)
                availableBuffers[bucket] = bucketBuffers
                logger.trace("Buffer returned to pool (bucket: \(bucket.sizeClass))")
            } else {
                // Buffer pool for this size is full, let it be deallocated
                logger.trace("Buffer pool full for bucket \(bucket.sizeClass), discarding buffer")
            }
        }
    }
    
    /// Handle memory pressure by clearing idle buffers
    public func handleMemoryPressure() async {
        logger.memory("Memory pressure detected • Clearing idle buffers", bytes: 0)
        
        isUnderMemoryPressure = true
        
        // Clear all available buffers
        let totalBuffers = availableBuffers.values.reduce(0) { $0 + $1.count }
        availableBuffers.removeAll()
        
        logger.memory("Cleared \(totalBuffers) idle buffers from pool", bytes: 0)
        
        // Reset memory pressure flag after a delay
        Task {
            try? await Task.sleep(nanoseconds: 5_000_000_000) // 5 seconds
            await self.clearMemoryPressureFlag()
        }
    }
    
    /// Get current pool statistics
    public func getStatistics() async -> PoolStatistics {
        let totalAvailable = availableBuffers.values.reduce(0) { $0 + $1.count }
        
        return PoolStatistics(
            totalHeaps: heaps.count,
            totalAllocatedBytes: totalAllocatedBytes,
            totalBuffersCreated: totalBuffersCreated,
            bufferHits: bufferHits,
            bufferMisses: bufferMisses,
            bucketsInUse: availableBuffers.count,
            totalAvailableBuffers: totalAvailable
        )
    }
    
    /// Perform maintenance on the pool (remove old unused buffers)
    public func performMaintenance() async {
        let cutoffDate = Date().addingTimeInterval(-300) // 5 minutes
        var removedCount = 0
        
        for (bucket, buffers) in availableBuffers {
            let activeBuffers = buffers.filter { buffer in
                if buffer.lastUsed > cutoffDate {
                    return true
                } else {
                    removedCount += 1
                    return false
                }
            }
            
            if activeBuffers.isEmpty {
                availableBuffers.removeValue(forKey: bucket)
            } else {
                availableBuffers[bucket] = activeBuffers
            }
        }
        
        if removedCount > 0 {
            logger.debug("Pool maintenance removed \(removedCount) stale buffers")
        }
    }
    
    // MARK: - Private Methods
    
    private func createHeap(size: Int) throws {
        let descriptor = MTLHeapDescriptor()
        descriptor.size = size
        descriptor.storageMode = .shared
        
        // Enable hazard tracking for automatic synchronization
        if device.supportsFamily(.apple6) {
            descriptor.hazardTrackingMode = .tracked
        }
        
        guard let heap = device.makeHeap(descriptor: descriptor) else {
            throw MetalBufferPoolError.heapCreationFailed(size: size)
        }
        
        heaps.append(heap)
        totalAllocatedBytes += size
        
        logger.info("Created new heap of size \(size / 1024 / 1024)MB")
    }
    
    private func allocateFromHeap(_ heap: MTLHeap, size: Int) -> MTLBuffer? {
        // Check if heap has enough space
        guard heap.maxAvailableSize(alignment: 256) >= size else {
            return nil
        }
        
        return heap.makeBuffer(length: size, options: [])
    }
    
    private func growHeapIfPossible() async throws {
        let currentTotalSize = heaps.reduce(0) { $0 + $1.size }
        
        guard currentTotalSize < maxHeapSize else {
            throw MetalBufferPoolError.maxHeapSizeReached
        }
        
        let newHeapSize = min(
            Int(Double(heaps.last?.size ?? initialHeapSize) * heapGrowthFactor),
            maxHeapSize - currentTotalSize
        )
        
        try createHeap(size: newHeapSize)
    }
    
    private func clearMemoryPressureFlag() async {
        isUnderMemoryPressure = false
        logger.debug("Memory pressure flag cleared")
    }
}

/// Errors that can occur in buffer pool operations
public enum MetalBufferPoolError: LocalizedError {
    case heapCreationFailed(size: Int)
    case allocationFailed(size: Int)
    case maxHeapSizeReached
    
    public var errorDescription: String? {
        switch self {
        case .heapCreationFailed(let size):
            return "Failed to create Metal heap of size \(size) bytes"
        case .allocationFailed(let size):
            return "Failed to allocate buffer of size \(size) bytes"
        case .maxHeapSizeReached:
            return "Maximum heap size reached, cannot grow further"
        }
    }
}

/// Extension to integrate buffer pool with MetalResourceManager
public extension MetalResourceManager {
    /// Shared buffer pool instance
    nonisolated(unsafe) static var sharedBufferPool: MetalBufferPool?
    
    /// Initialize the shared buffer pool
    func initializeBufferPool() async throws {
        if MetalResourceManager.sharedBufferPool == nil {
            MetalResourceManager.sharedBufferPool = try await MetalBufferPool(
                device: device,
                initialHeapSizeMB: 64,
                maxHeapSizeMB: 512
            )
        }
    }
    
    /// Create a buffer using the pool
    func createPooledBuffer(length: Int) async throws -> MTLBuffer {
        guard let pool = MetalResourceManager.sharedBufferPool else {
            // Fallback to direct allocation
            guard let buffer = device.makeBuffer(length: length, options: optimalStorageMode) else {
                throw MetalError.bufferCreationFailed
            }
            return buffer
        }
        
        return try await pool.acquireBuffer(size: length)
    }
    
    /// Release a buffer back to the pool
    func releasePooledBuffer(_ buffer: MTLBuffer) async {
        guard let pool = MetalResourceManager.sharedBufferPool else {
            return
        }
        
        await pool.releaseBuffer(buffer)
    }
}
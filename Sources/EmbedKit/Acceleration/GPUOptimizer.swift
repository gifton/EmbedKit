// EmbedKit - GPU Optimizer (Metal 4 Phase 4)
//
// Comprehensive GPU optimization infrastructure for M-series chips:
// - Device capability detection
// - Adaptive threadgroup sizing
// - Intelligent kernel selection
// - Progressive computation for large batches
// - Buffer residency management

import Foundation

#if canImport(Metal)
@preconcurrency import Metal
#endif

// MARK: - GPU Device Capabilities

/// Detected capabilities of the current GPU device.
///
/// Provides information about the GPU architecture to enable
/// optimal threadgroup sizing and kernel selection.
public struct GPUDeviceCapabilities: Sendable {
    /// GPU family identifier for Apple Silicon
    public enum GPUFamily: String, Sendable {
        case m1 = "Apple M1"
        case m1Pro = "Apple M1 Pro"
        case m1Max = "Apple M1 Max"
        case m1Ultra = "Apple M1 Ultra"
        case m2 = "Apple M2"
        case m2Pro = "Apple M2 Pro"
        case m2Max = "Apple M2 Max"
        case m2Ultra = "Apple M2 Ultra"
        case m3 = "Apple M3"
        case m3Pro = "Apple M3 Pro"
        case m3Max = "Apple M3 Max"
        case m4 = "Apple M4"
        case m4Pro = "Apple M4 Pro"
        case m4Max = "Apple M4 Max"
        case aSeriesRecent = "A-series (A14+)"
        case aSeriesOlder = "A-series (older)"
        case intelIntegrated = "Intel Integrated"
        case intelDiscrete = "Intel Discrete"
        case amdDiscrete = "AMD Discrete"
        case unknown = "Unknown"

        /// Optimal SIMD width for this GPU family
        public var optimalSimdWidth: Int {
            switch self {
            case .m1, .m1Pro, .m1Max, .m1Ultra,
                 .m2, .m2Pro, .m2Max, .m2Ultra,
                 .m3, .m3Pro, .m3Max,
                 .m4, .m4Pro, .m4Max,
                 .aSeriesRecent:
                return 32  // Apple Silicon SIMD width
            case .aSeriesOlder:
                return 32
            case .intelIntegrated, .intelDiscrete:
                return 16  // Intel SIMD width varies
            case .amdDiscrete:
                return 64  // AMD wavefront size
            case .unknown:
                return 32  // Safe default
            }
        }

        /// Recommended max threads per threadgroup
        public var recommendedMaxThreads: Int {
            switch self {
            case .m3, .m3Pro, .m3Max, .m4, .m4Pro, .m4Max:
                return 1024  // M3/M4 have improved occupancy
            case .m1, .m1Pro, .m1Max, .m1Ultra,
                 .m2, .m2Pro, .m2Max, .m2Ultra:
                return 1024  // M1/M2 standard
            case .aSeriesRecent:
                return 512   // Mobile chips have less resources
            case .aSeriesOlder:
                return 256
            case .intelIntegrated:
                return 256
            case .intelDiscrete, .amdDiscrete:
                return 1024
            case .unknown:
                return 256   // Conservative default
            }
        }

        /// GPU generation (higher = newer)
        public var generation: Int {
            switch self {
            case .m4, .m4Pro, .m4Max: return 4
            case .m3, .m3Pro, .m3Max: return 3
            case .m2, .m2Pro, .m2Max, .m2Ultra: return 2
            case .m1, .m1Pro, .m1Max, .m1Ultra: return 1
            case .aSeriesRecent: return 1
            case .aSeriesOlder: return 0
            case .intelIntegrated, .intelDiscrete, .amdDiscrete: return 0
            case .unknown: return 0
            }
        }
    }

    /// Detected GPU family
    public let family: GPUFamily

    /// Whether the device has unified memory (Apple Silicon)
    public let hasUnifiedMemory: Bool

    /// Maximum threads per threadgroup
    public let maxThreadsPerThreadgroup: Int

    /// Maximum threadgroup memory in bytes
    public let maxThreadgroupMemory: Int

    /// Maximum buffer size in bytes
    public let maxBufferLength: Int

    /// Whether Float16 is supported
    public let supportsFloat16: Bool

    /// Whether simdgroup matrix operations are supported
    public let supportsSimdgroupMatrix: Bool

    /// Device name
    public let deviceName: String

    #if canImport(Metal)
    /// Initialize from a Metal device
    public init(device: MTLDevice) {
        self.deviceName = device.name
        self.hasUnifiedMemory = device.hasUnifiedMemory
        self.maxThreadsPerThreadgroup = device.maxThreadsPerThreadgroup.width
        self.maxBufferLength = device.maxBufferLength
        self.supportsFloat16 = device.supports32BitFloatFiltering

        // Threadgroup memory varies by platform
        #if os(iOS) || os(tvOS) || os(visionOS)
        self.maxThreadgroupMemory = 32768
        #else
        self.maxThreadgroupMemory = 65536
        #endif

        // Detect GPU family from device name and capabilities
        let name = device.name.lowercased()
        if name.contains("m4 max") {
            self.family = .m4Max
        } else if name.contains("m4 pro") {
            self.family = .m4Pro
        } else if name.contains("m4") {
            self.family = .m4
        } else if name.contains("m3 max") {
            self.family = .m3Max
        } else if name.contains("m3 pro") {
            self.family = .m3Pro
        } else if name.contains("m3") {
            self.family = .m3
        } else if name.contains("m2 ultra") {
            self.family = .m2Ultra
        } else if name.contains("m2 max") {
            self.family = .m2Max
        } else if name.contains("m2 pro") {
            self.family = .m2Pro
        } else if name.contains("m2") {
            self.family = .m2
        } else if name.contains("m1 ultra") {
            self.family = .m1Ultra
        } else if name.contains("m1 max") {
            self.family = .m1Max
        } else if name.contains("m1 pro") {
            self.family = .m1Pro
        } else if name.contains("m1") {
            self.family = .m1
        } else if name.contains("a14") || name.contains("a15") || name.contains("a16") || name.contains("a17") {
            self.family = .aSeriesRecent
        } else if name.contains("apple") {
            self.family = .aSeriesOlder
        } else if name.contains("intel") {
            self.family = device.isLowPower ? .intelIntegrated : .intelDiscrete
        } else if name.contains("amd") || name.contains("radeon") {
            self.family = .amdDiscrete
        } else {
            self.family = .unknown
        }

        // Simdgroup matrix support (Apple7+)
        #if os(macOS)
        if #available(macOS 13.0, *) {
            self.supportsSimdgroupMatrix = device.supportsFamily(.apple7) ||
                                          device.supportsFamily(.apple8) ||
                                          device.supportsFamily(.apple9)
        } else {
            self.supportsSimdgroupMatrix = false
        }
        #else
        if #available(iOS 16.0, tvOS 16.0, *) {
            self.supportsSimdgroupMatrix = device.supportsFamily(.apple7) ||
                                          device.supportsFamily(.apple8) ||
                                          device.supportsFamily(.apple9)
        } else {
            self.supportsSimdgroupMatrix = false
        }
        #endif
    }
    #endif

    /// Recommended threadgroup width for given dimensions
    public func recommendedThreadgroupWidth(forDimensions dims: Int) -> Int {
        let simdWidth = family.optimalSimdWidth
        let maxWidth = min(family.recommendedMaxThreads, maxThreadsPerThreadgroup)

        // For small dimensions, use power of 2 that fits
        if dims <= simdWidth {
            return max(1, dims)
        }

        // For medium dimensions, align to SIMD width
        if dims <= 256 {
            return min(dims, maxWidth)
        }

        // For large dimensions, cap at recommended max
        return min(256, maxWidth)
    }
}

// MARK: - Threadgroup Optimizer

/// Optimizes threadgroup sizes for different operation types and workloads.
public struct ThreadgroupOptimizer: Sendable {
    /// Operation type for threadgroup optimization
    public enum OperationType: Sendable {
        case pooling          // Sequence reduction
        case normalization    // Vector normalization
        case similarity       // Pairwise similarity
        case fusedPoolNorm    // Combined pool+norm
        case batchProcess     // Batch operations
    }

    private let capabilities: GPUDeviceCapabilities

    public init(capabilities: GPUDeviceCapabilities) {
        self.capabilities = capabilities
    }

    /// Calculate optimal threadgroup size for an operation
    ///
    /// - Parameters:
    ///   - operation: Type of operation
    ///   - batchSize: Number of items in batch
    ///   - sequenceLength: Sequence length (for pooling)
    ///   - dimensions: Vector dimensions
    /// - Returns: Tuple of (threadgroupSize, gridSize) for dispatch
    public func optimalThreadgroup(
        for operation: OperationType,
        batchSize: Int,
        sequenceLength: Int = 1,
        dimensions: Int
    ) -> (threadgroup: (width: Int, height: Int, depth: Int),
          grid: (width: Int, height: Int, depth: Int)) {

        switch operation {
        case .pooling:
            return poolingThreadgroup(batchSize: batchSize, sequenceLength: sequenceLength, dimensions: dimensions)

        case .normalization:
            return normalizationThreadgroup(batchSize: batchSize, dimensions: dimensions)

        case .similarity:
            return similarityThreadgroup(queryBatch: batchSize, keyBatch: batchSize, dimensions: dimensions)

        case .fusedPoolNorm:
            return fusedPoolNormThreadgroup(batchSize: batchSize, sequenceLength: sequenceLength, dimensions: dimensions)

        case .batchProcess:
            return batchProcessThreadgroup(batchSize: batchSize, dimensions: dimensions)
        }
    }

    // MARK: - Private Threadgroup Calculations

    private func poolingThreadgroup(
        batchSize: Int,
        sequenceLength: Int,
        dimensions: Int
    ) -> (threadgroup: (Int, Int, Int), grid: (Int, Int, Int)) {
        let simdWidth = capabilities.family.optimalSimdWidth
        let maxThreads = capabilities.family.recommendedMaxThreads

        // For tensor pooling: grid = (dimensions, batchSize)
        // Each thread handles one dimension for one batch item
        let threadWidth = min(dimensions, maxThreads)

        // Align to SIMD width for efficiency
        let alignedWidth = ((threadWidth + simdWidth - 1) / simdWidth) * simdWidth

        // Grid covers all dimensions and batch items
        let gridWidth = (dimensions + alignedWidth - 1) / alignedWidth
        let gridHeight = batchSize

        return (
            threadgroup: (min(alignedWidth, maxThreads), 1, 1),
            grid: (gridWidth, gridHeight, 1)
        )
    }

    private func normalizationThreadgroup(
        batchSize: Int,
        dimensions: Int
    ) -> (threadgroup: (Int, Int, Int), grid: (Int, Int, Int)) {
        let simdWidth = capabilities.family.optimalSimdWidth
        let maxThreads = capabilities.family.recommendedMaxThreads

        // Fused normalization: one threadgroup per vector
        // Threads cooperatively reduce to compute norm
        let threadWidth = min(dimensions, maxThreads)
        let alignedWidth = ((threadWidth + simdWidth - 1) / simdWidth) * simdWidth

        return (
            threadgroup: (min(alignedWidth, maxThreads), 1, 1),
            grid: (1, batchSize, 1)
        )
    }

    private func similarityThreadgroup(
        queryBatch: Int,
        keyBatch: Int,
        dimensions: Int
    ) -> (threadgroup: (Int, Int, Int), grid: (Int, Int, Int)) {
        // Similarity matrix: each thread computes one similarity value
        // Use 2D threadgroup for coalesced memory access

        let tileSize: Int
        switch capabilities.family {
        case .m3, .m3Pro, .m3Max, .m4, .m4Pro, .m4Max:
            tileSize = 16  // Larger tiles for newer chips
        case .m1, .m1Pro, .m1Max, .m1Ultra, .m2, .m2Pro, .m2Max, .m2Ultra:
            tileSize = 16
        case .aSeriesRecent:
            tileSize = 8   // Smaller tiles for mobile
        default:
            tileSize = 8
        }

        let gridWidth = (keyBatch + tileSize - 1) / tileSize
        let gridHeight = (queryBatch + tileSize - 1) / tileSize

        return (
            threadgroup: (tileSize, tileSize, 1),
            grid: (gridWidth, gridHeight, 1)
        )
    }

    private func fusedPoolNormThreadgroup(
        batchSize: Int,
        sequenceLength: Int,
        dimensions: Int
    ) -> (threadgroup: (Int, Int, Int), grid: (Int, Int, Int)) {
        let simdWidth = capabilities.family.optimalSimdWidth
        let maxThreads = capabilities.family.recommendedMaxThreads

        // Fused operations: one threadgroup per sequence
        // Threads cooperatively pool then normalize
        let threadWidth = min(dimensions, maxThreads)
        let alignedWidth = min(((threadWidth + simdWidth - 1) / simdWidth) * simdWidth, 256)

        return (
            threadgroup: (alignedWidth, 1, 1),
            grid: (1, batchSize, 1)
        )
    }

    private func batchProcessThreadgroup(
        batchSize: Int,
        dimensions: Int
    ) -> (threadgroup: (Int, Int, Int), grid: (Int, Int, Int)) {
        let simdWidth = capabilities.family.optimalSimdWidth

        // Batch processing: 2D grid (dimensions, batchSize)
        let threadWidth = min(dimensions, 256)
        let alignedWidth = ((threadWidth + simdWidth - 1) / simdWidth) * simdWidth

        let gridWidth = (dimensions + alignedWidth - 1) / alignedWidth
        let gridHeight = batchSize

        return (
            threadgroup: (min(alignedWidth, 256), 1, 1),
            grid: (gridWidth, gridHeight, 1)
        )
    }
}

// MARK: - Adaptive Kernel Selector

/// Intelligently selects between fused and separate kernels based on workload.
public actor AdaptiveKernelSelector {
    /// Kernel selection decision
    public enum KernelChoice: Sendable {
        case fused           // Use fused kernel
        case separate        // Use separate kernels
        case cpu             // Fall back to CPU
        case progressive     // Use progressive/tiled computation
    }

    /// Operation types for selection
    public enum EmbeddingOperation: Sendable {
        case poolOnly
        case normalizeOnly
        case poolAndNormalize
        case similarityMatrix
        case fullPipeline
    }

    private let capabilities: GPUDeviceCapabilities
    private var performanceHistory: [EmbeddingOperation: [PerformanceRecord]] = [:]
    private let historyLimit = 50
    private let adaptiveLearningEnabled: Bool

    private struct PerformanceRecord {
        let choice: KernelChoice
        let workloadSize: Int
        let executionTime: TimeInterval
        let throughput: Double  // items per second
    }

    public init(capabilities: GPUDeviceCapabilities, adaptiveLearning: Bool = true) {
        self.capabilities = capabilities
        self.adaptiveLearningEnabled = adaptiveLearning
    }

    /// Select the best kernel for the given operation and workload
    public func selectKernel(
        for operation: EmbeddingOperation,
        batchSize: Int,
        sequenceLength: Int = 128,
        dimensions: Int = 384
    ) -> KernelChoice {
        let workloadSize = batchSize * sequenceLength * dimensions

        // Very small workloads: CPU is faster (no GPU dispatch overhead)
        if workloadSize < 1024 {
            return .cpu
        }

        // Check if workload is too large for single dispatch
        let maxGPUWorkload = capabilities.maxBufferLength / MemoryLayout<Float>.size
        if workloadSize > maxGPUWorkload / 2 {
            return .progressive
        }

        // Use adaptive learning if enabled and we have history
        if adaptiveLearningEnabled,
           let history = performanceHistory[operation],
           !history.isEmpty {
            return selectFromHistory(history, workloadSize: workloadSize)
        }

        // Default heuristics based on operation type
        return defaultSelection(for: operation, batchSize: batchSize, sequenceLength: sequenceLength, dimensions: dimensions)
    }

    private func defaultSelection(
        for operation: EmbeddingOperation,
        batchSize: Int,
        sequenceLength: Int,
        dimensions: Int
    ) -> KernelChoice {
        let workloadSize = batchSize * sequenceLength * dimensions

        switch operation {
        case .poolOnly, .normalizeOnly:
            // Single operations: fused overhead not worth it for small batches
            return batchSize >= 4 ? .fused : (workloadSize > 4096 ? .separate : .cpu)

        case .poolAndNormalize:
            // Combined operation: fused almost always better
            if batchSize < 2 && workloadSize < 8192 {
                return .cpu
            }
            return .fused

        case .similarityMatrix:
            // Similarity: depends on matrix size
            let matrixSize = batchSize * batchSize
            if matrixSize < 64 {
                return .cpu
            }
            if matrixSize > 10000 {
                return .progressive
            }
            return .fused

        case .fullPipeline:
            // Full pipeline: always use fused if workload is large enough
            if workloadSize < 4096 {
                return .cpu
            }
            return .fused
        }
    }

    private func selectFromHistory(_ history: [PerformanceRecord], workloadSize: Int) -> KernelChoice {
        // Find records with similar workload sizes (within 2x)
        let relevantRecords = history.filter { record in
            let ratio = Double(workloadSize) / Double(record.workloadSize)
            return ratio >= 0.5 && ratio <= 2.0
        }

        guard !relevantRecords.isEmpty else {
            return .fused  // Default to fused
        }

        // Group by choice and average throughput
        var choiceThroughputs: [KernelChoice: (total: Double, count: Int)] = [:]
        for record in relevantRecords {
            let current = choiceThroughputs[record.choice] ?? (0, 0)
            choiceThroughputs[record.choice] = (current.total + record.throughput, current.count + 1)
        }

        // Select choice with highest average throughput
        var bestChoice: KernelChoice = .fused
        var bestThroughput: Double = 0

        for (choice, stats) in choiceThroughputs {
            let avgThroughput = stats.total / Double(stats.count)
            if avgThroughput > bestThroughput {
                bestThroughput = avgThroughput
                bestChoice = choice
            }
        }

        return bestChoice
    }

    /// Record performance for adaptive learning
    public func recordPerformance(
        operation: EmbeddingOperation,
        choice: KernelChoice,
        workloadSize: Int,
        executionTime: TimeInterval
    ) {
        guard adaptiveLearningEnabled, executionTime > 0 else { return }

        let throughput = Double(workloadSize) / executionTime

        let record = PerformanceRecord(
            choice: choice,
            workloadSize: workloadSize,
            executionTime: executionTime,
            throughput: throughput
        )

        if performanceHistory[operation] == nil {
            performanceHistory[operation] = []
        }

        performanceHistory[operation]?.append(record)

        // Keep only recent history
        if let count = performanceHistory[operation]?.count, count > historyLimit {
            performanceHistory[operation]?.removeFirst(count - historyLimit)
        }
    }

    /// Get performance statistics for an operation
    public func getPerformanceStats(for operation: EmbeddingOperation) -> PerformanceStats? {
        guard let history = performanceHistory[operation], !history.isEmpty else {
            return nil
        }

        let fusedRecords = history.filter { $0.choice == .fused }
        let separateRecords = history.filter { $0.choice == .separate }
        let cpuRecords = history.filter { $0.choice == .cpu }

        return PerformanceStats(
            fusedThroughput: fusedRecords.isEmpty ? nil : fusedRecords.map(\.throughput).reduce(0, +) / Double(fusedRecords.count),
            separateThroughput: separateRecords.isEmpty ? nil : separateRecords.map(\.throughput).reduce(0, +) / Double(separateRecords.count),
            cpuThroughput: cpuRecords.isEmpty ? nil : cpuRecords.map(\.throughput).reduce(0, +) / Double(cpuRecords.count),
            totalOperations: history.count
        )
    }

    public struct PerformanceStats: Sendable {
        public let fusedThroughput: Double?
        public let separateThroughput: Double?
        public let cpuThroughput: Double?
        public let totalOperations: Int
    }
}

// MARK: - Progressive Similarity Computation

/// Handles large similarity matrix computations by tiling.
public struct ProgressiveSimilarityComputer: Sendable {
    /// Configuration for progressive computation
    public struct Configuration: Sendable {
        /// Maximum elements per tile (based on GPU memory)
        public let maxTileElements: Int

        /// Preferred tile size (for cache efficiency)
        public let preferredTileSize: Int

        /// Whether to overlap computation and data transfer
        public let enableOverlap: Bool

        public init(
            maxTileElements: Int = 1_000_000,
            preferredTileSize: Int = 1024,
            enableOverlap: Bool = true
        ) {
            self.maxTileElements = maxTileElements
            self.preferredTileSize = preferredTileSize
            self.enableOverlap = enableOverlap
        }

        public static let `default` = Configuration()

        public static func forDevice(_ capabilities: GPUDeviceCapabilities) -> Configuration {
            // Adjust based on device capabilities
            let maxElements = capabilities.maxBufferLength / MemoryLayout<Float>.size / 4
            let tileSize: Int

            switch capabilities.family {
            case .m3Max, .m4, .m4Pro, .m4Max:
                tileSize = 2048
            case .m2Max, .m2Ultra, .m3, .m3Pro:
                tileSize = 1536
            case .m1Max, .m1Ultra, .m2, .m2Pro:
                tileSize = 1024
            default:
                tileSize = 512
            }

            return Configuration(
                maxTileElements: min(maxElements, 4_000_000),
                preferredTileSize: tileSize,
                enableOverlap: capabilities.hasUnifiedMemory
            )
        }
    }

    private let config: Configuration

    public init(configuration: Configuration = .default) {
        self.config = configuration
    }

    /// Calculate tiles for a similarity matrix computation
    public func calculateTiles(
        queryBatchSize: Int,
        keyBatchSize: Int
    ) -> [SimilarityTile] {
        let totalElements = queryBatchSize * keyBatchSize

        // If small enough, single tile
        if totalElements <= config.maxTileElements {
            return [SimilarityTile(
                queryStart: 0, queryEnd: queryBatchSize,
                keyStart: 0, keyEnd: keyBatchSize
            )]
        }

        // Calculate optimal tile dimensions
        let tileSize = min(config.preferredTileSize, Int(sqrt(Double(config.maxTileElements))))

        var tiles: [SimilarityTile] = []

        var queryStart = 0
        while queryStart < queryBatchSize {
            let queryEnd = min(queryStart + tileSize, queryBatchSize)

            var keyStart = 0
            while keyStart < keyBatchSize {
                let keyEnd = min(keyStart + tileSize, keyBatchSize)

                tiles.append(SimilarityTile(
                    queryStart: queryStart, queryEnd: queryEnd,
                    keyStart: keyStart, keyEnd: keyEnd
                ))

                keyStart = keyEnd
            }

            queryStart = queryEnd
        }

        return tiles
    }

    public struct SimilarityTile: Sendable {
        public let queryStart: Int
        public let queryEnd: Int
        public let keyStart: Int
        public let keyEnd: Int

        public var queryCount: Int { queryEnd - queryStart }
        public var keyCount: Int { keyEnd - keyStart }
        public var elementCount: Int { queryCount * keyCount }
    }
}

// MARK: - Buffer Residency Manager

#if canImport(Metal)
/// Manages buffer residency hints for frequently used embeddings.
public actor BufferResidencyManager {
    private let device: MTLDevice
    private var residentBuffers: [ObjectIdentifier: ResidentBufferInfo] = [:]
    private let maxResidentBytes: Int

    private struct ResidentBufferInfo {
        let buffer: MTLBuffer
        let size: Int
        var lastAccess: Date
        var accessCount: Int
    }

    public init(device: MTLDevice, maxResidentMB: Int = 512) {
        self.device = device
        self.maxResidentBytes = maxResidentMB * 1024 * 1024
    }

    /// Mark a buffer as frequently accessed (hint to keep resident)
    public func markFrequent(_ buffer: MTLBuffer) {
        let id = ObjectIdentifier(buffer)

        if var info = residentBuffers[id] {
            info.lastAccess = Date()
            info.accessCount += 1
            residentBuffers[id] = info
        } else {
            // Check if we need to evict
            evictIfNeeded(newSize: buffer.length)

            residentBuffers[id] = ResidentBufferInfo(
                buffer: buffer,
                size: buffer.length,
                lastAccess: Date(),
                accessCount: 1
            )

            // Set residency hint if available
            #if os(macOS)
            if #available(macOS 13.0, *) {
                // Metal 3 residency sets would go here
                // For now, just track in our data structure
            }
            #endif
        }
    }

    /// Mark buffer as no longer frequently used
    public func markInfrequent(_ buffer: MTLBuffer) {
        let id = ObjectIdentifier(buffer)
        residentBuffers.removeValue(forKey: id)
    }

    /// Get current residency statistics
    public func getStatistics() -> ResidencyStatistics {
        let totalSize = residentBuffers.values.reduce(0) { $0 + $1.size }
        let avgAccess = residentBuffers.isEmpty ? 0 :
            Double(residentBuffers.values.reduce(0) { $0 + $1.accessCount }) / Double(residentBuffers.count)

        return ResidencyStatistics(
            residentBufferCount: residentBuffers.count,
            totalResidentBytes: totalSize,
            averageAccessCount: avgAccess,
            maxResidentBytes: maxResidentBytes
        )
    }

    private func evictIfNeeded(newSize: Int) {
        let currentTotal = residentBuffers.values.reduce(0) { $0 + $1.size }

        guard currentTotal + newSize > maxResidentBytes else { return }

        // Sort by access count (ascending) and last access (oldest first)
        let sorted = residentBuffers.sorted { a, b in
            if a.value.accessCount != b.value.accessCount {
                return a.value.accessCount < b.value.accessCount
            }
            return a.value.lastAccess < b.value.lastAccess
        }

        var freedBytes = 0
        let targetFree = (currentTotal + newSize) - maxResidentBytes

        for (id, info) in sorted {
            if freedBytes >= targetFree { break }
            residentBuffers.removeValue(forKey: id)
            freedBytes += info.size
        }
    }

    public struct ResidencyStatistics: Sendable {
        public let residentBufferCount: Int
        public let totalResidentBytes: Int
        public let averageAccessCount: Double
        public let maxResidentBytes: Int

        public var utilizationPercent: Double {
            guard maxResidentBytes > 0 else { return 0 }
            return Double(totalResidentBytes) / Double(maxResidentBytes) * 100
        }
    }
}
#endif

// MARK: - Combined Optimizer

/// Main entry point for GPU optimization, combining all Phase 4 components.
public actor GPUOptimizer {
    #if canImport(Metal)
    public let capabilities: GPUDeviceCapabilities
    public let threadgroupOptimizer: ThreadgroupOptimizer
    public let kernelSelector: AdaptiveKernelSelector
    public let similarityComputer: ProgressiveSimilarityComputer
    public let residencyManager: BufferResidencyManager

    public init(device: MTLDevice) {
        self.capabilities = GPUDeviceCapabilities(device: device)
        self.threadgroupOptimizer = ThreadgroupOptimizer(capabilities: capabilities)
        self.kernelSelector = AdaptiveKernelSelector(capabilities: capabilities)
        self.similarityComputer = ProgressiveSimilarityComputer(
            configuration: .forDevice(capabilities)
        )
        self.residencyManager = BufferResidencyManager(device: device)
    }

    /// Get optimal dispatch parameters for an operation
    public func getDispatchParameters(
        operation: ThreadgroupOptimizer.OperationType,
        batchSize: Int,
        sequenceLength: Int = 1,
        dimensions: Int
    ) -> DispatchParameters {
        let (threadgroup, grid) = threadgroupOptimizer.optimalThreadgroup(
            for: operation,
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            dimensions: dimensions
        )

        return DispatchParameters(
            threadgroupSize: MTLSize(width: threadgroup.0, height: threadgroup.1, depth: threadgroup.2),
            gridSize: MTLSize(width: grid.0, height: grid.1, depth: grid.2)
        )
    }

    /// Select kernel and get dispatch parameters in one call
    public func optimizeOperation(
        _ operation: AdaptiveKernelSelector.EmbeddingOperation,
        batchSize: Int,
        sequenceLength: Int = 128,
        dimensions: Int = 384
    ) async -> OptimizedOperation {
        let kernelChoice = await kernelSelector.selectKernel(
            for: operation,
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            dimensions: dimensions
        )

        let threadgroupOp: ThreadgroupOptimizer.OperationType = switch operation {
        case .poolOnly: .pooling
        case .normalizeOnly: .normalization
        case .poolAndNormalize: .fusedPoolNorm
        case .similarityMatrix: .similarity
        case .fullPipeline: .fusedPoolNorm
        }

        let (threadgroup, grid) = threadgroupOptimizer.optimalThreadgroup(
            for: threadgroupOp,
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            dimensions: dimensions
        )

        return OptimizedOperation(
            kernelChoice: kernelChoice,
            threadgroupSize: MTLSize(width: threadgroup.0, height: threadgroup.1, depth: threadgroup.2),
            gridSize: MTLSize(width: grid.0, height: grid.1, depth: grid.2),
            tiles: kernelChoice == .progressive ?
                similarityComputer.calculateTiles(queryBatchSize: batchSize, keyBatchSize: batchSize) : nil
        )
    }

    public struct DispatchParameters: Sendable {
        public let threadgroupSize: MTLSize
        public let gridSize: MTLSize
    }

    public struct OptimizedOperation: Sendable {
        public let kernelChoice: AdaptiveKernelSelector.KernelChoice
        public let threadgroupSize: MTLSize
        public let gridSize: MTLSize
        public let tiles: [ProgressiveSimilarityComputer.SimilarityTile]?
    }
    #endif
}

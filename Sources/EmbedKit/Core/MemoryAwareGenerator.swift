// EmbedKit - MemoryAwareGenerator
// Dynamic batch sizing and resource management under memory pressure

import Foundation
import VectorCore

#if canImport(Darwin)
import Darwin
#endif

// MARK: - Memory Pressure Level

/// System memory pressure levels.
public enum MemoryPressureLevel: Int, Sendable, Comparable, CaseIterable {
    /// Normal operation - full batch sizes allowed.
    case normal = 0
    /// Warning - reduce batch sizes and defer non-essential work.
    case warning = 1
    /// Critical - minimize memory usage, may need to release resources.
    case critical = 2

    public static func < (lhs: MemoryPressureLevel, rhs: MemoryPressureLevel) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    /// Suggested batch size multiplier for this pressure level.
    public var batchSizeMultiplier: Double {
        switch self {
        case .normal: return 1.0
        case .warning: return 0.5
        case .critical: return 0.25
        }
    }
}

// MARK: - Memory Monitor

/// Monitors system memory and provides pressure notifications.
public final class MemoryMonitor: @unchecked Sendable {
    /// Shared singleton instance.
    public static let shared = MemoryMonitor()

    private let lock = NSLock()
    private var _currentLevel: MemoryPressureLevel = .normal
    private var _handlers: [(MemoryPressureLevel) -> Void] = []

    #if canImport(Darwin)
    private var source: DispatchSourceMemoryPressure?
    #endif

    private init() {}

    /// Current memory pressure level.
    public var currentLevel: MemoryPressureLevel {
        lock.lock()
        defer { lock.unlock() }
        return _currentLevel
    }

    /// Start monitoring system memory pressure.
    public func startMonitoring() {
        #if canImport(Darwin)
        guard source == nil else { return }

        let newSource = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical],
            queue: .global(qos: .utility)
        )

        newSource.setEventHandler { [weak self] in
            guard let self = self else { return }
            let event = newSource.data
            let level: MemoryPressureLevel
            if event.contains(.critical) {
                level = .critical
            } else if event.contains(.warning) {
                level = .warning
            } else {
                level = .normal
            }
            self.updateLevel(level)
        }

        newSource.setCancelHandler { [weak self] in
            self?.lock.lock()
            self?._currentLevel = .normal
            self?.lock.unlock()
        }

        source = newSource
        newSource.resume()
        #endif
    }

    /// Stop monitoring system memory pressure.
    public func stopMonitoring() {
        #if canImport(Darwin)
        source?.cancel()
        source = nil
        #endif
    }

    /// Register a handler for memory pressure changes.
    ///
    /// - Parameter handler: Called when memory pressure level changes.
    /// - Returns: An ID that can be used to unregister the handler.
    @discardableResult
    public func onPressureChange(_ handler: @escaping (MemoryPressureLevel) -> Void) -> Int {
        lock.lock()
        defer { lock.unlock() }
        _handlers.append(handler)
        return _handlers.count - 1
    }

    /// Unregister a handler.
    public func removeHandler(at index: Int) {
        lock.lock()
        defer { lock.unlock() }
        if index < _handlers.count {
            _handlers[index] = { _ in }  // Replace with no-op
        }
    }

    /// Manually set memory pressure level (for testing).
    public func simulatePressure(_ level: MemoryPressureLevel) {
        updateLevel(level)
    }

    private func updateLevel(_ level: MemoryPressureLevel) {
        lock.lock()
        let oldLevel = _currentLevel
        _currentLevel = level
        let handlers = _handlers
        lock.unlock()

        if level != oldLevel {
            for handler in handlers {
                handler(level)
            }
        }
    }

    /// Get current memory statistics.
    public var memoryStats: MemoryStats {
        #if canImport(Darwin)
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }

        if result == KERN_SUCCESS {
            return MemoryStats(
                residentSize: Int64(info.resident_size),
                virtualSize: Int64(info.virtual_size),
                physicalMemory: Int64(ProcessInfo.processInfo.physicalMemory),
                pressure: currentLevel
            )
        }
        #endif

        return MemoryStats(
            residentSize: 0,
            virtualSize: 0,
            physicalMemory: Int64(ProcessInfo.processInfo.physicalMemory),
            pressure: currentLevel
        )
    }
}

// MARK: - Memory Stats

/// Current memory usage statistics.
public struct MemoryStats: Sendable {
    /// Resident memory size in bytes.
    public let residentSize: Int64

    /// Virtual memory size in bytes.
    public let virtualSize: Int64

    /// Total physical memory on the system.
    public let physicalMemory: Int64

    /// Current memory pressure level.
    public let pressure: MemoryPressureLevel

    /// Resident memory as a fraction of physical memory.
    public var memoryUtilization: Double {
        guard physicalMemory > 0 else { return 0 }
        return Double(residentSize) / Double(physicalMemory)
    }

    /// Resident memory in megabytes.
    public var residentSizeMB: Double {
        Double(residentSize) / (1024 * 1024)
    }
}

// MARK: - Memory Aware Configuration

/// Configuration for memory-aware embedding generation.
public struct MemoryAwareConfig: Sendable {
    /// Base batch size under normal memory pressure.
    public var baseBatchSize: Int

    /// Minimum batch size even under critical pressure.
    public var minBatchSize: Int

    /// Whether to automatically adjust batch size based on memory pressure.
    public var adaptiveBatching: Bool

    /// Memory utilization threshold to start reducing batch size (0.0-1.0).
    public var pressureThreshold: Double

    /// Whether to release model resources under critical pressure.
    public var releaseOnCritical: Bool

    /// Callback when batch size is adjusted.
    public var onBatchSizeAdjusted: (@Sendable (Int, MemoryPressureLevel) -> Void)?

    public init(
        baseBatchSize: Int = 32,
        minBatchSize: Int = 4,
        adaptiveBatching: Bool = true,
        pressureThreshold: Double = 0.7,
        releaseOnCritical: Bool = false,
        onBatchSizeAdjusted: (@Sendable (Int, MemoryPressureLevel) -> Void)? = nil
    ) {
        self.baseBatchSize = baseBatchSize
        self.minBatchSize = minBatchSize
        self.adaptiveBatching = adaptiveBatching
        self.pressureThreshold = pressureThreshold
        self.releaseOnCritical = releaseOnCritical
        self.onBatchSizeAdjusted = onBatchSizeAdjusted
    }

    /// Default configuration for most use cases.
    public static let `default` = MemoryAwareConfig()

    /// Configuration for memory-constrained environments.
    public static let conservative = MemoryAwareConfig(
        baseBatchSize: 16,
        minBatchSize: 2,
        adaptiveBatching: true,
        pressureThreshold: 0.5,
        releaseOnCritical: true
    )

    /// Configuration for high-memory systems.
    public static let aggressive = MemoryAwareConfig(
        baseBatchSize: 64,
        minBatchSize: 8,
        adaptiveBatching: true,
        pressureThreshold: 0.85,
        releaseOnCritical: false
    )
}

// MARK: - Memory Aware Generator

/// An embedding generator that automatically adjusts batch sizes based on memory pressure.
///
/// `MemoryAwareGenerator` wraps an `EmbeddingGenerator` and monitors system memory
/// to dynamically adjust batch sizes. Under memory pressure, it reduces batch sizes
/// to prevent out-of-memory conditions while maintaining throughput.
///
/// ## Example Usage
/// ```swift
/// let generator = try await modelManager.createGenerator()
/// let memoryAware = MemoryAwareGenerator(generator: generator)
///
/// // Start monitoring memory pressure
/// memoryAware.startMonitoring()
///
/// // Generate embeddings - batch size automatically adjusts
/// let embeddings = try await memoryAware.produce(largeTextArray)
///
/// // Check current effective batch size
/// print("Current batch size: \(memoryAware.effectiveBatchSize)")
/// ```
public actor MemoryAwareGenerator: VectorProducer {
    // MARK: - Properties

    private let generator: EmbeddingGenerator
    private var config: MemoryAwareConfig
    private var currentPressure: MemoryPressureLevel = .normal
    private var monitorHandlerID: Int?

    // MARK: - Statistics

    private var stats = GeneratorStats()

    private struct GeneratorStats {
        var totalBatches: Int = 0
        var batchesAtNormal: Int = 0
        var batchesAtWarning: Int = 0
        var batchesAtCritical: Int = 0
        var totalItemsProcessed: Int = 0
        var batchSizeAdjustments: Int = 0
    }

    // MARK: - VectorProducer Requirements

    public nonisolated var dimensions: Int {
        generator.dimensions
    }

    public nonisolated var producesNormalizedVectors: Bool {
        generator.producesNormalizedVectors
    }

    // MARK: - Initialization

    /// Creates a memory-aware generator wrapping an existing generator.
    ///
    /// - Parameters:
    ///   - generator: The underlying embedding generator.
    ///   - config: Memory-aware configuration.
    public init(
        generator: EmbeddingGenerator,
        config: MemoryAwareConfig = .default
    ) {
        self.generator = generator
        self.config = config
    }

    // MARK: - Monitoring

    /// Start monitoring system memory pressure.
    ///
    /// When enabled, batch sizes are automatically adjusted based on
    /// system memory pressure.
    public func startMonitoring() {
        guard monitorHandlerID == nil else { return }

        MemoryMonitor.shared.startMonitoring()

        monitorHandlerID = MemoryMonitor.shared.onPressureChange { [weak self] level in
            Task { [weak self] in
                await self?.handlePressureChange(level)
            }
        }
    }

    /// Stop monitoring system memory pressure.
    public func stopMonitoring() {
        if let id = monitorHandlerID {
            MemoryMonitor.shared.removeHandler(at: id)
            monitorHandlerID = nil
        }
    }

    private func handlePressureChange(_ level: MemoryPressureLevel) {
        let oldLevel = currentPressure
        currentPressure = level

        if level != oldLevel {
            stats.batchSizeAdjustments += 1
            let newSize = effectiveBatchSize
            config.onBatchSizeAdjusted?(newSize, level)
        }

        // Handle critical pressure
        if level == .critical && config.releaseOnCritical {
            Task {
                try? await generator.release()
            }
        }
    }

    // MARK: - Batch Size Calculation

    /// The current effective batch size based on memory pressure.
    public var effectiveBatchSize: Int {
        guard config.adaptiveBatching else {
            return config.baseBatchSize
        }

        let multiplier = currentPressure.batchSizeMultiplier
        let adjusted = Int(Double(config.baseBatchSize) * multiplier)
        return max(config.minBatchSize, adjusted)
    }

    /// Current memory pressure level.
    public var pressureLevel: MemoryPressureLevel {
        currentPressure
    }

    /// Manually set memory pressure (for testing or external monitoring).
    public func setPressure(_ level: MemoryPressureLevel) {
        handlePressureChange(level)
    }

    // MARK: - VectorProducer Implementation

    /// Produces embeddings with memory-aware batch sizing.
    public func produce(_ texts: [String]) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }

        try Task.checkCancellation()

        var results: [[Float]] = []
        results.reserveCapacity(texts.count)

        var position = 0
        while position < texts.count {
            try Task.checkCancellation()

            // Get current effective batch size (may change during processing)
            let batchSize = effectiveBatchSize
            let end = min(position + batchSize, texts.count)
            let batch = Array(texts[position..<end])

            // Process batch through underlying generator
            let batchResults = try await generator.produce(batch)
            results.append(contentsOf: batchResults)

            // Update stats
            stats.totalBatches += 1
            stats.totalItemsProcessed += batch.count
            switch currentPressure {
            case .normal: stats.batchesAtNormal += 1
            case .warning: stats.batchesAtWarning += 1
            case .critical: stats.batchesAtCritical += 1
            }

            position = end
        }

        return results
    }

    /// Produces an embedding for a single text.
    public func produce(_ text: String) async throws -> [Float] {
        try await generator.produce(text)
    }

    // MARK: - Extended API

    /// Generate embeddings with progress tracking and memory-aware batching.
    public func produceWithProgress(
        _ texts: [String],
        onProgress: (@Sendable (BatchProgress) -> Void)? = nil
    ) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }

        let startTime = CFAbsoluteTimeGetCurrent()
        var results: [[Float]] = []
        results.reserveCapacity(texts.count)

        var position = 0
        var batchIndex = 0

        // Estimate total batches (may change due to adaptive sizing)
        let estimatedBatches = (texts.count + effectiveBatchSize - 1) / effectiveBatchSize

        onProgress?(.started(total: texts.count, totalBatches: estimatedBatches))

        while position < texts.count {
            try Task.checkCancellation()

            let batchSize = effectiveBatchSize
            let end = min(position + batchSize, texts.count)
            let batch = Array(texts[position..<end])

            let batchResults = try await generator.produce(batch)
            results.append(contentsOf: batchResults)

            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let progress = BatchProgress.batchCompleted(
                itemsCompleted: results.count,
                totalItems: texts.count,
                batchIndex: batchIndex,
                totalBatches: estimatedBatches,
                batchSize: batch.count,
                tokensInBatch: 0,  // Not tracked at this level
                totalTokens: 0,
                elapsedTime: elapsed
            )
            onProgress?(progress)

            position = end
            batchIndex += 1
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        let throughput = elapsed > 0 ? Double(texts.count) / elapsed : nil
        onProgress?(.completed(
            total: texts.count,
            totalBatches: batchIndex,
            tokensProcessed: 0,
            itemsPerSecond: throughput
        ))

        return results
    }

    // MARK: - Statistics

    /// Statistics about memory-aware generation.
    public struct Statistics: Sendable {
        public let totalBatches: Int
        public let batchesAtNormal: Int
        public let batchesAtWarning: Int
        public let batchesAtCritical: Int
        public let totalItemsProcessed: Int
        public let batchSizeAdjustments: Int
        public let currentPressure: MemoryPressureLevel
        public let currentBatchSize: Int

        /// Fraction of batches processed under memory pressure.
        public var pressuredBatchFraction: Double {
            guard totalBatches > 0 else { return 0 }
            return Double(batchesAtWarning + batchesAtCritical) / Double(totalBatches)
        }
    }

    /// Get current generation statistics.
    public func getStatistics() -> Statistics {
        Statistics(
            totalBatches: stats.totalBatches,
            batchesAtNormal: stats.batchesAtNormal,
            batchesAtWarning: stats.batchesAtWarning,
            batchesAtCritical: stats.batchesAtCritical,
            totalItemsProcessed: stats.totalItemsProcessed,
            batchSizeAdjustments: stats.batchSizeAdjustments,
            currentPressure: currentPressure,
            currentBatchSize: effectiveBatchSize
        )
    }

    /// Reset generation statistics.
    public func resetStatistics() {
        stats = GeneratorStats()
    }

    // MARK: - Configuration

    /// Update the memory-aware configuration.
    public func updateConfig(_ newConfig: MemoryAwareConfig) {
        config = newConfig
    }

    /// Get the current configuration.
    public func getConfig() -> MemoryAwareConfig {
        config
    }

    // MARK: - Lifecycle

    /// Warm up the underlying generator.
    public func warmup() async throws {
        try await generator.warmup()
    }

    /// Release resources.
    public func release() async throws {
        stopMonitoring()
        try await generator.release()
    }
}

// MARK: - EmbeddingGenerator Extension

extension EmbeddingGenerator {
    /// Create a memory-aware wrapper for this generator.
    ///
    /// - Parameter config: Memory-aware configuration.
    /// - Returns: A `MemoryAwareGenerator` wrapping this generator.
    public func memoryAware(
        config: MemoryAwareConfig = .default
    ) -> MemoryAwareGenerator {
        MemoryAwareGenerator(generator: self, config: config)
    }
}

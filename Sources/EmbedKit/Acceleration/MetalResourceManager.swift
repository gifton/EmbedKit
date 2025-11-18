import Foundation
@preconcurrency import Metal
import OSLog

/// Actor responsible for managing Metal resources (device, queues, library, pipelines)
///
/// This component centralizes all Metal resource lifecycle management, providing
/// thread-safe access to Metal infrastructure components.
public actor MetalResourceManager {
    nonisolated private let logger = EmbedKitLogger.metal()

    // Core Metal resources
    nonisolated public let device: MTLDevice
    nonisolated public let commandQueue: MTLCommandQueue
    nonisolated public let library: MTLLibrary

    // Metal 3 optimization: async compute for parallel operations
    nonisolated public let asyncCommandQueue: MTLCommandQueue?

    // Pipeline cache - actor-isolated for thread safety
    private var computePipelines: [String: MTLComputePipelineState] = [:]

    // Numeric configuration affecting specialization of certain kernels
    private var numerics: AccelerationNumerics = AccelerationNumerics()

    /// Metal 3 optimization: Optimal storage mode for current platform
    nonisolated public var optimalStorageMode: MTLResourceOptions {
        #if arch(arm64) && !os(iOS) // Apple Silicon Mac
        return .storageModeManaged  // Zero-copy between CPU/GPU
        #else
        return .storageModeShared
        #endif
    }

    public init(device: MTLDevice) throws {
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MetalError.commandQueueCreationFailed
        }
        self.commandQueue = queue

        // Metal 3 optimization: Create async command queue for parallel operations
        if device.supportsFamily(.metal3) {
            self.asyncCommandQueue = device.makeCommandQueue()
        } else {
            self.asyncCommandQueue = nil
        }

        // Load Metal library using synchronous dual-mode loader
        // This tries precompiled metallib first, then falls back to string
        let (library, source) = try MetalLibraryLoader.loadLibrarySync(device: device)
        self.library = library

        // Log which loading method was used
        if source.isPrecompiled {
            logger.success("Metal library loaded from precompiled metallib (fast path)")
        } else {
            logger.warning("Metal library compiled from string (slow path)")
            logger.warning("Run ./Scripts/CompileMetalShaders.sh to enable fast path")
        }

        logger.info("MetalResourceManager initialized with Metal 3 support: \(device.supportsFamily(.metal3))")
    }

    /// Setup compute pipelines for Metal kernels
    public func setupPipelines() async throws {
        self.computePipelines = try setupComputePipelines()
    }

    /// Get a compute pipeline by name, setting up pipelines lazily if needed
    public func getPipeline(_ name: String) async throws -> MTLComputePipelineState? {
        if computePipelines.isEmpty {
            try await setupPipelines()
        }

        // Use specialization-aware key for normalization kernels
        let key = pipelineKey(for: name)
        if let pipeline = computePipelines[key] {
            return pipeline
        }

        // Lazily build pipeline if missing (e.g., numerics changed and specific variant not built yet)
        if let (builtKey, pipeline) = try? buildPipeline(for: name) {
            computePipelines[builtKey] = pipeline
            return pipeline
        }
        return nil
    }

    /// Create a Metal buffer with optimal storage configuration
    nonisolated public func createBuffer(bytes: UnsafeRawPointer, length: Int) -> MTLBuffer? {
        return device.makeBuffer(bytes: bytes, length: length, options: optimalStorageMode)
    }

    /// Create a Metal buffer with specified length
    nonisolated public func createBuffer(length: Int) -> MTLBuffer? {
        return device.makeBuffer(length: length, options: optimalStorageMode)
    }

    /// Handle memory pressure by clearing pipeline cache
    public func handleMemoryPressure() async {
        logger.memory("Memory pressure detected • Clearing GPU pipeline cache", bytes: 0)
        computePipelines.removeAll()

        // Recreate essential pipelines
        do {
            self.computePipelines = try setupComputePipelines()
            logger.success("GPU pipelines recreated after memory pressure")
        } catch {
            logger.error("Failed to recreate pipelines after memory pressure", error: error)
        }
    }

    /// Get current GPU memory usage in bytes
    nonisolated public func getCurrentMemoryUsage() -> Int64 {
        return Int64(device.currentAllocatedSize)
    }

    /// Check if Metal acceleration is available with required features
    nonisolated public var isAvailable: Bool {
        // Require Metal 3 for advanced ML features
        guard device.supportsFamily(.metal3) else { return false }

        // On macOS, also ensure it's not an external GPU
#if os(macOS)
        return !device.isRemovable
#else
        return true
#endif
    }

    // MARK: - Private Implementation

    private func setupComputePipelines() throws -> [String: MTLComputePipelineState] {
        logger.start("GPU pipeline compilation")

        var pipelines: [String: MTLComputePipelineState] = [:]

        // Create pipelines for all available kernels
        for kernelName in MetalShaderLibrary.KernelName.allCases {
            let name = kernelName.rawValue
            if let (key, pipeline) = try? buildPipeline(for: name) {
                pipelines[key] = pipeline
            }
        }

        logger.complete("GPU pipeline compilation", result: "\(pipelines.count) pipelines ready")

        return pipelines
    }

    // Build a pipeline for a given function name, specializing constants if needed
    private func buildPipeline(for name: String) throws -> (String, MTLComputePipelineState) {
        if isSpecializedKernelName(name) {
            let constants = MTLFunctionConstantValues()
            var stable = numerics.stableNormalizationEnabled
            var eps = numerics.epsilon
            constants.setConstantValue(&stable, type: .bool, index: 0)
            constants.setConstantValue(&eps, type: .float, index: 1)

            guard let function = try? library.makeFunction(name: name, constantValues: constants) else {
                throw MetalError.functionNotFound(name)
            }
            let pipeline = try device.makeComputePipelineState(function: function)
            let key = pipelineKey(for: name)
            logger.debug("Built specialized pipeline \(name) [stable=\(stable), eps=\(eps)]")
            return (key, pipeline)
        } else {
            guard let function = library.makeFunction(name: name) else {
                throw MetalError.functionNotFound(name)
            }
            let pipeline = try device.makeComputePipelineState(function: function)
            return (name, pipeline)
        }
    }

    private func isSpecializedKernelName(_ name: String) -> Bool {
        // Any kernel that references function constants (USE_STABLE_NORMALIZATION, EPSILON_NORMAL)
        return name == MetalShaderLibrary.KernelName.l2Normalize.rawValue ||
               name == MetalShaderLibrary.KernelName.l2NormalizeBatchOptimized.rawValue ||
               name == MetalShaderLibrary.KernelName.cosineSimilarity.rawValue ||
               name == MetalShaderLibrary.KernelName.cosineSimilarityBatch.rawValue ||
               name == MetalShaderLibrary.KernelName.attentionWeightedPool.rawValue
    }

    private func pipelineKey(for name: String) -> String {
        if isSpecializedKernelName(name) {
            let stable = numerics.stableNormalizationEnabled ? 1 : 0
            return "\(name)|stable=\(stable)|eps=\(numerics.epsilon)"
        }
        return name
    }

    /// Update numerics and rebuild only affected pipelines (normalization variants)
    public func updateNumerics(stable: Bool, epsilon: Float) async {
        numerics.stableNormalizationEnabled = stable
        numerics.epsilon = epsilon

        // Remove existing pipelines for specialized kernels (any variants)
        let specializedNames = [
            MetalShaderLibrary.KernelName.l2Normalize.rawValue,
            MetalShaderLibrary.KernelName.l2NormalizeBatchOptimized.rawValue,
            MetalShaderLibrary.KernelName.cosineSimilarity.rawValue,
            MetalShaderLibrary.KernelName.cosineSimilarityBatch.rawValue,
            MetalShaderLibrary.KernelName.attentionWeightedPool.rawValue
        ]
        computePipelines = computePipelines.filter { key, _ in
            !specializedNames.contains { key.hasPrefix($0) }
        }

        // Rebuild current-variant pipelines for specialized kernels
        do {
            for name in specializedNames {
                let (key, pipeline) = try buildPipeline(for: name)
                computePipelines[key] = pipeline
            }
            logger.success("Numerics updated • stable=\(stable) eps=\(epsilon) • specialized pipelines rebuilt")
        } catch {
            logger.error("Failed to rebuild pipelines after numerics update", error: error)
        }
    }
}

/// Factory for creating MetalResourceManager instances
public extension MetalResourceManager {
    /// Create a shared MetalResourceManager instance for the default GPU
    static func createShared() -> MetalResourceManager? {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return nil
        }
        guard let manager = try? MetalResourceManager(device: device) else {
            return nil
        }
        return manager
    }
}

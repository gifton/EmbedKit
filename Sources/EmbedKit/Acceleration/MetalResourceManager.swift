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
        
        // Load Metal shaders from source
        let source = MetalShaderLibrary.source
        
        // Metal 3 optimization: Enable fast math and other optimizations
        let compileOptions = MTLCompileOptions()
        if device.supportsFamily(.metal3) {
            compileOptions.fastMathEnabled = true
            compileOptions.languageVersion = .version3_0
        }
        
        self.library = try device.makeLibrary(source: source, options: compileOptions)
        
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
        return computePipelines[name]
    }
    
    /// Create a Metal buffer with optimal storage configuration
    public func createBuffer(bytes: UnsafeRawPointer, length: Int) -> MTLBuffer? {
        return device.makeBuffer(bytes: bytes, length: length, options: optimalStorageMode)
    }
    
    /// Create a Metal buffer with specified length
    public func createBuffer(length: Int) -> MTLBuffer? {
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
    
    private nonisolated func setupComputePipelines() throws -> [String: MTLComputePipelineState] {
        logger.start("GPU pipeline compilation")
        
        var pipelines: [String: MTLComputePipelineState] = [:]
        
        // Create pipelines for all available kernels
        for kernelName in MetalShaderLibrary.KernelName.allCases {
            if let function = library.makeFunction(name: kernelName.rawValue) {
                let pipeline = try device.makeComputePipelineState(function: function)
                pipelines[kernelName.rawValue] = pipeline
            }
        }
        
        logger.complete("GPU pipeline compilation", result: "\(pipelines.count) pipelines ready")
        
        return pipelines
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
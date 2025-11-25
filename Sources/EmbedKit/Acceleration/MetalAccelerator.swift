// EmbedKit - Metal Accelerator (Hybrid Loader)

import Foundation
#if canImport(Dispatch)
import Dispatch
#endif

#if canImport(Metal)
import Metal
#endif

/// Optional GPU accelerator for vector post-processing. Uses a hybrid loader:
/// 1) App-provided metallib (override URL) → 2) SPM Bundle.module metallib → 3) CPU fallback.
public actor MetalAccelerator {
    // MARK: - Override control via config actor (thread-safe)

    // MARK: - GPU resources (when available)
    #if canImport(Metal)
    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    private var library: MTLLibrary?

    // Pipelines (optional, only when library is loaded)
    private var psoL2Normalize: MTLComputePipelineState?
    private var psoL2NormalizeBatch: MTLComputePipelineState?
    private var psoMeanPool: MTLComputePipelineState?
    private var psoMaxPool: MTLComputePipelineState?
    private var psoAttentionWeightedPool: MTLComputePipelineState?
    private var psoCosine: MTLComputePipelineState?
    private var psoCosineBatch: MTLComputePipelineState?
    #else
    private let device: Any? = nil
    private let commandQueue: Any? = nil
    #endif

    // MARK: - Init
    public init() async {
        #if canImport(Metal)
        if let dev = MTLCreateSystemDefaultDevice() {
            device = dev
            commandQueue = dev.makeCommandQueue()
        } else {
            device = nil
            commandQueue = nil
        }
        await loadLibraryIfPossible()
        #else
        // No Metal on this platform
        #endif
    }

    // MARK: - Public helpers
    /// Indicates whether GPU acceleration is currently available (device + pipelines loaded).
    public var isAvailable: Bool {
        #if canImport(Metal)
        return library != nil && device != nil && commandQueue != nil
        #else
        return false
        #endif
    }

    // MARK: - Operations (GPU when available, otherwise CPU fallback)
    /// L2-normalize a batch of vectors (shape: N x D). Returns normalized vectors of same shape.
    public func l2Normalize(_ vectors: [[Float]]) async -> [[Float]] {
        // CPU fallback always available
        func cpu(_ v: [[Float]]) -> [[Float]] {
            v.map { row in
                let norm = max(1e-12, sqrt(row.reduce(0) { $0 + Double($1) * Double($1) }))
                return row.map { $0 / Float(norm) }
            }
        }
        #if canImport(Metal)
        // Use the standard normalization kernel (batch-optimized requires additional params)
        guard isAvailable,
              let dev = device,
              let queue = commandQueue,
              let pso = psoL2Normalize
        else { return cpu(vectors) }

        // Flatten input
        guard let width = vectors.first?.count, width > 0 else { return vectors }
        let batchSize = vectors.count
        let flat: [Float] = vectors.flatMap { $0 }
        let lengthBytes = flat.count * MemoryLayout<Float>.size
        guard let inBuf = dev.makeBuffer(bytes: flat, length: lengthBytes),
              let outBuf = dev.makeBuffer(length: lengthBytes) else {
            return cpu(vectors)
        }

        // Create dimensions buffer (required by kernel)
        var dims = Int32(width)
        guard let dimsBuf = dev.makeBuffer(bytes: &dims, length: MemoryLayout<Int32>.size) else {
            return cpu(vectors)
        }

        guard let cmd = queue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() else { return cpu(vectors) }
        enc.setComputePipelineState(pso)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBuffer(dimsBuf, offset: 0, index: 2)

        // The kernel uses cooperative reduction within a threadgroup.
        // Each threadgroup processes ONE vector, with threads cooperating for the reduction.
        // Grid: (1, batchSize, 1) - one threadgroup per vector
        // Threads per threadgroup: (dimensions, 1, 1) - one thread per dimension
        // Metal limit is typically 1024 threads per threadgroup
        let threadsPerVector = min(1024, width)
        let threadgroupSize = MTLSize(width: threadsPerVector, height: 1, depth: 1)
        let gridSize = MTLSize(width: 1, height: batchSize, depth: 1)
        enc.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        enc.endEncoding()

        let _: Void = await withCheckedContinuation { cont in
            cmd.addCompletedHandler { _ in cont.resume(returning: ()) }
            cmd.commit()
        }

        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: flat.count)
        let out = Array(UnsafeBufferPointer(start: outPtr, count: flat.count))
        return strideSplit(out, width: width)
        #else
        return cpu(vectors)
        #endif
    }

    // MARK: - Pooling Operations

    /// Mean pool token embeddings to a single vector.
    ///
    /// Reduces a sequence of token embeddings [sequenceLength, dimensions] to a single
    /// embedding [dimensions] by computing the mean across the sequence dimension.
    ///
    /// - Parameters:
    ///   - embeddings: Token embeddings as flat array [sequenceLength * dimensions] in row-major order
    ///   - sequenceLength: Number of tokens in the sequence
    ///   - dimensions: Embedding dimensions per token
    ///   - mask: Optional attention mask [sequenceLength] where 1=valid, 0=masked (padding)
    /// - Returns: Pooled embedding vector [dimensions]
    public func meanPool(
        embeddings: [Float],
        sequenceLength: Int,
        dimensions: Int,
        mask: [Int]? = nil
    ) async -> [Float] {
        // CPU fallback
        func cpu() -> [Float] {
            var result = [Float](repeating: 0, count: dimensions)
            var count = 0
            for t in 0..<sequenceLength {
                let isValid = mask == nil || (mask![t] == 1)
                if isValid {
                    for d in 0..<dimensions {
                        result[d] += embeddings[t * dimensions + d]
                    }
                    count += 1
                }
            }
            if count > 0 {
                let scale = 1.0 / Float(count)
                for d in 0..<dimensions {
                    result[d] *= scale
                }
            }
            return result
        }

        #if canImport(Metal)
        guard isAvailable,
              let dev = device,
              let queue = commandQueue,
              let pso = psoMeanPool,
              sequenceLength > 0,
              dimensions > 0
        else { return cpu() }

        // Threshold: GPU only beneficial for larger workloads
        if sequenceLength * dimensions < 1024 { return cpu() }

        // Create buffers
        let inputBytes = embeddings.count * MemoryLayout<Float>.size
        let outputBytes = dimensions * MemoryLayout<Float>.size
        guard let inputBuf = dev.makeBuffer(bytes: embeddings, length: inputBytes),
              let outputBuf = dev.makeBuffer(length: outputBytes)
        else { return cpu() }

        // Create mask buffer (or nil buffer if no mask)
        var maskBuf: MTLBuffer? = nil
        if let mask = mask {
            let maskInt32 = mask.map { Int32($0) }
            maskBuf = dev.makeBuffer(bytes: maskInt32, length: mask.count * MemoryLayout<Int32>.size)
        }

        // Create params buffer
        var params = PoolingParams(sequenceLength: sequenceLength, dimensions: dimensions)
        guard let paramsBuf = dev.makeBuffer(bytes: &params, length: MemoryLayout<PoolingParams>.size)
        else { return cpu() }

        // Encode and dispatch
        guard let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder()
        else { return cpu() }

        enc.setComputePipelineState(pso)
        enc.setBuffer(inputBuf, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        enc.setBuffer(maskBuf, offset: 0, index: 2)  // nil is valid for optional mask
        enc.setBuffer(paramsBuf, offset: 0, index: 3)

        // Dispatch one thread per dimension
        let threadgroups = MTLSize(width: (dimensions + 31) / 32, height: 1, depth: 1)
        let threadsPerGroup = MTLSize(width: min(32, dimensions), height: 1, depth: 1)
        enc.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        enc.endEncoding()

        // Wait for completion
        let _: Void = await withCheckedContinuation { cont in
            cmd.addCompletedHandler { _ in cont.resume(returning: ()) }
            cmd.commit()
        }

        // Read results
        let outPtr = outputBuf.contents().bindMemory(to: Float.self, capacity: dimensions)
        return Array(UnsafeBufferPointer(start: outPtr, count: dimensions))
        #else
        return cpu()
        #endif
    }

    /// Max pool token embeddings to a single vector.
    ///
    /// Reduces a sequence of token embeddings [sequenceLength, dimensions] to a single
    /// embedding [dimensions] by taking the element-wise maximum across the sequence.
    ///
    /// - Parameters:
    ///   - embeddings: Token embeddings as flat array [sequenceLength * dimensions] in row-major order
    ///   - sequenceLength: Number of tokens in the sequence
    ///   - dimensions: Embedding dimensions per token
    ///   - mask: Optional attention mask [sequenceLength] where 1=valid, 0=masked (padding)
    /// - Returns: Pooled embedding vector [dimensions]
    public func maxPool(
        embeddings: [Float],
        sequenceLength: Int,
        dimensions: Int,
        mask: [Int]? = nil
    ) async -> [Float] {
        // CPU fallback
        func cpu() -> [Float] {
            var result = [Float](repeating: -.greatestFiniteMagnitude, count: dimensions)
            var foundValid = false
            for t in 0..<sequenceLength {
                let isValid = mask == nil || (mask![t] == 1)
                if isValid {
                    foundValid = true
                    for d in 0..<dimensions {
                        let val = embeddings[t * dimensions + d]
                        if val > result[d] {
                            result[d] = val
                        }
                    }
                }
            }
            // If no valid tokens, return zeros
            if !foundValid {
                return [Float](repeating: 0, count: dimensions)
            }
            return result
        }

        #if canImport(Metal)
        guard isAvailable,
              let dev = device,
              let queue = commandQueue,
              let pso = psoMaxPool,
              sequenceLength > 0,
              dimensions > 0
        else { return cpu() }

        // Threshold: GPU only beneficial for larger workloads
        if sequenceLength * dimensions < 1024 { return cpu() }

        // Create buffers
        let inputBytes = embeddings.count * MemoryLayout<Float>.size
        let outputBytes = dimensions * MemoryLayout<Float>.size
        guard let inputBuf = dev.makeBuffer(bytes: embeddings, length: inputBytes),
              let outputBuf = dev.makeBuffer(length: outputBytes)
        else { return cpu() }

        // Create mask buffer (or nil buffer if no mask)
        var maskBuf: MTLBuffer? = nil
        if let mask = mask {
            let maskInt32 = mask.map { Int32($0) }
            maskBuf = dev.makeBuffer(bytes: maskInt32, length: mask.count * MemoryLayout<Int32>.size)
        }

        // Create params buffer
        var params = PoolingParams(sequenceLength: sequenceLength, dimensions: dimensions)
        guard let paramsBuf = dev.makeBuffer(bytes: &params, length: MemoryLayout<PoolingParams>.size)
        else { return cpu() }

        // Encode and dispatch
        guard let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder()
        else { return cpu() }

        enc.setComputePipelineState(pso)
        enc.setBuffer(inputBuf, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        enc.setBuffer(maskBuf, offset: 0, index: 2)  // nil is valid for optional mask
        enc.setBuffer(paramsBuf, offset: 0, index: 3)

        // Dispatch one thread per dimension
        let threadgroups = MTLSize(width: (dimensions + 31) / 32, height: 1, depth: 1)
        let threadsPerGroup = MTLSize(width: min(32, dimensions), height: 1, depth: 1)
        enc.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        enc.endEncoding()

        // Wait for completion
        let _: Void = await withCheckedContinuation { cont in
            cmd.addCompletedHandler { _ in cont.resume(returning: ()) }
            cmd.commit()
        }

        // Read results
        let outPtr = outputBuf.contents().bindMemory(to: Float.self, capacity: dimensions)
        return Array(UnsafeBufferPointer(start: outPtr, count: dimensions))
        #else
        return cpu()
        #endif
    }

    // MARK: - Similarity Operations

    /// Compute cosine similarity matrix for a batch of vectors (NxD → NxN).
    ///
    /// For N vectors of dimension D, computes an NxN matrix where element [i,j]
    /// is the cosine similarity between vector i and vector j.
    ///
    /// **Performance Notes**:
    /// - GPU acceleration used when N >= 32 and Metal is available
    /// - For very large N (> 1024), computation is tiled to manage memory
    /// - CPU fallback uses symmetry optimization (only computes upper triangle)
    ///
    /// - Parameters:
    ///   - vectors: Array of N vectors, each of dimension D
    ///   - tileSize: Optional tile size for GPU computation (0 = auto, typically 512)
    /// - Returns: NxN similarity matrix where [i][j] = cosine_similarity(vectors[i], vectors[j])
    public func cosineSimilarityMatrix(_ vectors: [[Float]], tileSize: Int = 0) async -> [[Float]] {
        let n = vectors.count
        guard n > 0 else { return [] }
        guard let dimensions = vectors.first?.count, dimensions > 0 else { return [] }

        // CPU baseline with symmetry optimization
        func cpu(_ v: [[Float]]) -> [[Float]] {
            let norms = v.map { row -> Float in
                let s = row.reduce(0) { $0 + Double($1) * Double($1) }
                return max(1e-12, Float(s).squareRoot())
            }
            var out = Array(repeating: Array(repeating: Float(0), count: n), count: n)
            for i in 0..<n {
                out[i][i] = 1.0  // Self-similarity is always 1
                for j in (i+1)..<n {
                    let dot = zip(v[i], v[j]).reduce(0) { $0 + $1.0 * $1.1 }
                    let cos = dot / (norms[i] * norms[j])
                    out[i][j] = cos
                    out[j][i] = cos
                }
            }
            return out
        }

        #if canImport(Metal)
        guard isAvailable,
              let dev = device,
              let queue = commandQueue,
              let pso = psoCosine  // Use pairwise matrix kernel
        else { return cpu(vectors) }

        // Threshold: GPU beneficial for larger matrices
        // Below this, CPU is faster due to GPU dispatch overhead
        if n < 32 { return cpu(vectors) }

        // Flatten input vectors to row-major format
        let flatVectors: [Float] = vectors.flatMap { $0 }
        let inputBytes = flatVectors.count * MemoryLayout<Float>.size

        // Create input buffer (used as both queries and keys for self-similarity)
        guard let inputBuf = dev.makeBuffer(bytes: flatVectors, length: inputBytes)
        else { return cpu(vectors) }

        // Determine effective tile size
        // For large N, tile to avoid allocating huge output buffers
        // Memory for NxN output: N*N*4 bytes
        // Tile if output would exceed ~64MB or N > 1024
        let maxUntiled = 1024
        let effectiveTileSize: Int
        if tileSize > 0 {
            effectiveTileSize = tileSize
        } else if n > maxUntiled {
            effectiveTileSize = 512  // Default tile size for large matrices
        } else {
            effectiveTileSize = n  // No tiling needed
        }

        // If no tiling needed, compute full matrix in one dispatch
        if effectiveTileSize >= n {
            return await computeFullSimilarityMatrix(
                dev: dev, queue: queue, pso: pso,
                inputBuf: inputBuf, n: n, dimensions: dimensions,
                cpuFallback: { cpu(vectors) }
            )
        }

        // Tiled computation for large matrices
        return await computeTiledSimilarityMatrix(
            dev: dev, queue: queue, pso: pso,
            flatVectors: flatVectors, n: n, dimensions: dimensions,
            tileSize: effectiveTileSize,
            cpuFallback: { cpu(vectors) }
        )
        #else
        return cpu(vectors)
        #endif
    }

    #if canImport(Metal)
    /// Compute full similarity matrix in a single GPU dispatch
    private func computeFullSimilarityMatrix(
        dev: MTLDevice,
        queue: MTLCommandQueue,
        pso: MTLComputePipelineState,
        inputBuf: MTLBuffer,
        n: Int,
        dimensions: Int,
        cpuFallback: () -> [[Float]]
    ) async -> [[Float]] {
        let outputBytes = n * n * MemoryLayout<Float>.size
        guard let outputBuf = dev.makeBuffer(length: outputBytes) else {
            return cpuFallback()
        }

        // Create params buffer
        var params = SimilarityParams(queryCount: n, keyCount: n, dimensions: dimensions)
        guard let paramsBuf = dev.makeBuffer(bytes: &params, length: MemoryLayout<SimilarityParams>.size)
        else { return cpuFallback() }

        // Encode and dispatch
        guard let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder()
        else { return cpuFallback() }

        enc.setComputePipelineState(pso)
        enc.setBuffer(inputBuf, offset: 0, index: 0)  // queries
        enc.setBuffer(inputBuf, offset: 0, index: 1)  // keys (same as queries for self-similarity)
        enc.setBuffer(outputBuf, offset: 0, index: 2) // output
        enc.setBuffer(paramsBuf, offset: 0, index: 3) // params

        // Dispatch grid: (keyCount, queryCount) = (n, n)
        // Each thread computes one similarity value
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(
            width: (n + threadgroupSize.width - 1) / threadgroupSize.width,
            height: (n + threadgroupSize.height - 1) / threadgroupSize.height,
            depth: 1
        )
        enc.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        enc.endEncoding()

        // Wait for completion
        let _: Void = await withCheckedContinuation { cont in
            cmd.addCompletedHandler { _ in cont.resume(returning: ()) }
            cmd.commit()
        }

        // Read results into 2D array
        let outPtr = outputBuf.contents().bindMemory(to: Float.self, capacity: n * n)
        var result: [[Float]] = []
        result.reserveCapacity(n)
        for i in 0..<n {
            let rowStart = i * n
            result.append(Array(UnsafeBufferPointer(start: outPtr + rowStart, count: n)))
        }
        return result
    }

    /// Compute similarity matrix using tiled approach for large N
    private func computeTiledSimilarityMatrix(
        dev: MTLDevice,
        queue: MTLCommandQueue,
        pso: MTLComputePipelineState,
        flatVectors: [Float],
        n: Int,
        dimensions: Int,
        tileSize: Int,
        cpuFallback: () -> [[Float]]
    ) async -> [[Float]] {
        // Initialize output matrix
        var result = Array(repeating: Array(repeating: Float(0), count: n), count: n)

        // Create full input buffer
        let inputBytes = flatVectors.count * MemoryLayout<Float>.size
        guard let inputBuf = dev.makeBuffer(bytes: flatVectors, length: inputBytes)
        else { return cpuFallback() }

        // Process tiles
        // We compute tiles of the output matrix, where each tile is at most tileSize x tileSize
        let numTiles = (n + tileSize - 1) / tileSize

        for tileRow in 0..<numTiles {
            let queryStart = tileRow * tileSize
            let queryEnd = min(queryStart + tileSize, n)
            let queryCount = queryEnd - queryStart

            for tileCol in 0..<numTiles {
                let keyStart = tileCol * tileSize
                let keyEnd = min(keyStart + tileSize, n)
                let keyCount = keyEnd - keyStart

                // For self-similarity, we can skip lower triangle tiles if tileRow > tileCol
                // and copy from the transpose. But for simplicity, compute all tiles.
                // (Symmetry optimization would require more complex bookkeeping)

                // Compute this tile
                guard let tileResult = await computeSimilarityTile(
                    dev: dev, queue: queue, pso: pso,
                    inputBuf: inputBuf,
                    queryStart: queryStart, queryCount: queryCount,
                    keyStart: keyStart, keyCount: keyCount,
                    dimensions: dimensions
                ) else {
                    return cpuFallback()
                }

                // Copy tile results to output matrix
                for i in 0..<queryCount {
                    for j in 0..<keyCount {
                        result[queryStart + i][keyStart + j] = tileResult[i * keyCount + j]
                    }
                }
            }
        }

        return result
    }

    /// Compute a single tile of the similarity matrix
    private func computeSimilarityTile(
        dev: MTLDevice,
        queue: MTLCommandQueue,
        pso: MTLComputePipelineState,
        inputBuf: MTLBuffer,
        queryStart: Int,
        queryCount: Int,
        keyStart: Int,
        keyCount: Int,
        dimensions: Int
    ) async -> [Float]? {
        let outputCount = queryCount * keyCount
        let outputBytes = outputCount * MemoryLayout<Float>.size
        guard let outputBuf = dev.makeBuffer(length: outputBytes) else { return nil }

        // Create params for this tile
        var params = SimilarityParams(queryCount: queryCount, keyCount: keyCount, dimensions: dimensions)
        guard let paramsBuf = dev.makeBuffer(bytes: &params, length: MemoryLayout<SimilarityParams>.size)
        else { return nil }

        // Calculate buffer offsets for query and key regions
        let queryOffset = queryStart * dimensions * MemoryLayout<Float>.size
        let keyOffset = keyStart * dimensions * MemoryLayout<Float>.size

        guard let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder()
        else { return nil }

        enc.setComputePipelineState(pso)
        enc.setBuffer(inputBuf, offset: queryOffset, index: 0)  // queries starting at queryStart
        enc.setBuffer(inputBuf, offset: keyOffset, index: 1)    // keys starting at keyStart
        enc.setBuffer(outputBuf, offset: 0, index: 2)
        enc.setBuffer(paramsBuf, offset: 0, index: 3)

        // Dispatch for this tile
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(
            width: (keyCount + threadgroupSize.width - 1) / threadgroupSize.width,
            height: (queryCount + threadgroupSize.height - 1) / threadgroupSize.height,
            depth: 1
        )
        enc.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        enc.endEncoding()

        let _: Void = await withCheckedContinuation { cont in
            cmd.addCompletedHandler { _ in cont.resume(returning: ()) }
            cmd.commit()
        }

        let outPtr = outputBuf.contents().bindMemory(to: Float.self, capacity: outputCount)
        return Array(UnsafeBufferPointer(start: outPtr, count: outputCount))
    }
    #endif

    // MARK: - Private: Library Loading
    #if canImport(Metal)
    private func loadLibraryIfPossible() async {
        guard let dev = device else { return }
        // 1) Try override URL
        if let url = await MetalAcceleratorConfig.shared.getOverride() {
            if let lib = try? dev.makeLibrary(URL: url) {
                library = lib
                await buildPipelines(from: lib)
                return
            }
        }
        // 2) Try Bundle.module metallib
        // The resource is named EmbedKitShaders.metallib in Sources/EmbedKit/Resources/
        #if SWIFT_PACKAGE
        if let url = Bundle.module.url(forResource: "EmbedKitShaders", withExtension: "metallib"),
           let lib = try? dev.makeLibrary(URL: url) {
            library = lib
            await buildPipelines(from: lib)
            return
        }
        #endif
        // Else: stay in CPU mode
    }

    private func buildPipelines(from lib: MTLLibrary) async {
        psoL2Normalize = try? makePSO(lib, name: "l2_normalize")
        psoL2NormalizeBatch = try? makePSO(lib, name: "l2_normalize_batch_optimized")
        psoMeanPool = try? makePSO(lib, name: "mean_pool")
        psoMaxPool = try? makePSO(lib, name: "max_pool")
        psoAttentionWeightedPool = try? makePSO(lib, name: "attention_weighted_pool")
        psoCosine = try? makePSO(lib, name: "cosine_similarity")
        psoCosineBatch = try? makePSO(lib, name: "cosine_similarity_batch")
    }

    private func makePSO(_ lib: MTLLibrary, name: String) throws -> MTLComputePipelineState {
        // Create function constant values for shader specialization
        let constantValues = MTLFunctionConstantValues()

        // Function constant 0: USE_STABLE_NORMALIZATION (bool) - enable stable two-pass algorithms
        var useStable: Bool = true
        constantValues.setConstantValue(&useStable, type: .bool, index: 0)

        // Function constant 1: EPSILON_NORMAL (float) - epsilon for division safety
        var epsilon: Float = 1e-8
        constantValues.setConstantValue(&epsilon, type: .float, index: 1)

        // Create specialized function with constants
        let fn = try lib.makeFunction(name: name, constantValues: constantValues)
        return try device!.makeComputePipelineState(function: fn)
    }
    #endif

    // MARK: - Utilities
    private nonisolated func strideSplit(_ flat: [Float], width: Int) -> [[Float]] {
        guard width > 0 else { return [] }
        var out: [[Float]] = []
        out.reserveCapacity(flat.count / width)
        var i = 0
        while i < flat.count {
            let j = i + width
            out.append(Array(flat[i..<min(j, flat.count)]))
            i = j
        }
        return out
    }
}

// MARK: - Metal Accelerator Configuration

/// Thread-safe configuration store for Metal accelerator overrides.
///
/// Use this actor to provide a custom metallib URL before initializing `MetalAccelerator`.
/// The override URL is checked first during library loading.
///
/// **Example**:
/// ```swift
/// // Set override before creating accelerator
/// await MetalAcceleratorConfig.shared.setOverride(url: myCustomMetallibURL)
/// let accelerator = await MetalAccelerator()
/// ```
public actor MetalAcceleratorConfig {
    /// Shared singleton instance
    public static let shared = MetalAcceleratorConfig()

    /// Custom metallib URL override (checked before Bundle.module)
    private var overrideURL: URL? = nil

    /// Set a custom metallib URL to use instead of the bundled one
    /// - Parameter url: URL to custom metallib, or nil to clear override
    public func setOverride(url: URL?) {
        self.overrideURL = url
    }

    /// Get the current override URL
    /// - Returns: Custom metallib URL if set, nil otherwise
    public func getOverride() -> URL? {
        overrideURL
    }
}

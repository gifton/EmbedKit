// EmbedKitV2 - CoreML Backend (Scaffold)
// Phase 1: API surface and lifecycle only. No CoreML imports yet.

import Foundation
#if canImport(CoreML)
import CoreML
#endif

/// Input to the CoreML backend prior to tensor conversion.
public struct CoreMLInputV2: Sendable {
    public let tokenIDs: [Int]
    public let attentionMask: [Int]

    public init(tokenIDs: [Int], attentionMask: [Int]) {
        self.tokenIDs = tokenIDs
        self.attentionMask = attentionMask
    }
}

/// Output from the CoreML backend prior to pooling and normalization.
public struct CoreMLOutputV2: Sendable {
    /// Raw token embeddings or pooled vector depending on later phases.
    public let values: [Float]

    public init(values: [Float]) {
        self.values = values
    }
}

/// Minimal CoreML backend actor that exposes a stable API for later wiring.
public actor CoreMLBackendV2: ModelBackend {
    public typealias Input = CoreMLInputV2
    public typealias Output = CoreMLOutputV2

    // MARK: - Configuration

    private let modelURL: URL?
    private let device: ComputeDevice

    // MARK: - State

    private(set) public var isLoaded: Bool = false
    public var memoryUsage: Int64 { 0 }

    #if canImport(CoreML)
    private var mlModel: MLModel?
    #endif

    // MARK: - Init

    public init(modelURL: URL?, device: ComputeDevice = .auto) {
        self.modelURL = modelURL
        self.device = device
    }

    // MARK: - Lifecycle

    public func load() async throws {
        if isLoaded { return }

        #if canImport(CoreML)
        let url = try resolveModelURL()
        let config = MLModelConfiguration()
        config.computeUnits = mapComputeUnits(device)

        do {
            self.mlModel = try await MLModel.load(contentsOf: url, configuration: config)
            self.isLoaded = true
        } catch {
            throw EmbedKitError.modelLoadFailed("CoreML load failed: \(error.localizedDescription)")
        }
        #else
        throw EmbedKitError.modelLoadFailed("CoreML not available on this platform")
        #endif
    }

    public func unload() async throws {
        #if canImport(CoreML)
        mlModel = nil
        #endif
        isLoaded = false
    }

    // MARK: - Inference

    public func process(_ input: Input) async throws -> Output {
        #if canImport(CoreML)
        guard let model = mlModel else {
            throw EmbedKitError.modelLoadFailed("CoreML model not loaded")
        }

        // Resolve input names dynamically
        let inputs = model.modelDescription.inputDescriptionsByName
        let tokenKey = try resolveTokenKey(from: inputs)
        let maskKey = resolveMaskKey(from: inputs)
        let typeKey = resolveTokenTypeKey(from: inputs)
        let posKey = resolvePositionKey(from: inputs)

        let tokenArray = try makeMLMultiArray(values: input.tokenIDs, key: tokenKey, inputs: inputs)
        var maskArray: MLMultiArray? = nil
        if let mk = maskKey {
            maskArray = try? makeMLMultiArray(values: input.attentionMask, key: mk, inputs: inputs)
        }
        var typeArray: MLMultiArray? = nil
        if let tk = typeKey {
            let zeros = Array(repeating: 0, count: input.tokenIDs.count)
            typeArray = try? makeMLMultiArray(values: zeros, key: tk, inputs: inputs)
        }
        var posArray: MLMultiArray? = nil
        if let pk = posKey {
            let pos = (0..<input.tokenIDs.count).map { $0 }
            posArray = try? makeMLMultiArray(values: pos, key: pk, inputs: inputs)
        }

        var dict: [String: MLFeatureValue] = [tokenKey: MLFeatureValue(multiArray: tokenArray)]
        if let maskKey, let arr = maskArray { dict[maskKey] = MLFeatureValue(multiArray: arr) }
        if let typeKey, let arr = typeArray { dict[typeKey] = MLFeatureValue(multiArray: arr) }
        if let posKey, let arr = posArray { dict[posKey] = MLFeatureValue(multiArray: arr) }

        let provider = try MLDictionaryFeatureProvider(dictionary: dict)
        let output = try predict(model, provider: provider)

        // Find the first float output multiarray
        for name in output.featureNames {
            let val = output.featureValue(for: name)
            if val?.type == .multiArray, let ma = val?.multiArrayValue {
                if ma.dataType == .float32 || ma.dataType == .double {
                    let floats = try flattenFloatArray(ma)
                    return CoreMLOutputV2(values: floats)
                }
            }
        }
        throw EmbedKitError.modelLoadFailed("No float multiarray output found")
        #else
        throw EmbedKitError.modelLoadFailed("CoreML not available on this platform")
        #endif
    }

    public func processBatch(_ inputs: [Input]) async throws -> [Output] {
        // Phase 2: simple sequential batch; optimize later
        var results: [Output] = []
        results.reserveCapacity(inputs.count)
        for inp in inputs { results.append(try await process(inp)) }
        return results
    }

    // MARK: - Helpers

    private func resolveModelURL() throws -> URL {
        if let url = modelURL { return url }
        // Attempt to find a bundled or workspace model for development convenience.
        let cwd = FileManager.default.currentDirectoryPath
        let candidate = URL(fileURLWithPath: cwd).appendingPathComponent("MiniLM-L12-v2.mlmodelc", isDirectory: true)
        if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
        throw EmbedKitError.modelLoadFailed("Model URL not provided and default path not found")
    }

    #if canImport(CoreML)
    private func resolveTokenKey(from inputs: [String: MLFeatureDescription]) throws -> String {
        // Prefer common names
        let preferred = ["input_ids", "token_ids", "tokens", "ids"]
        for key in preferred where inputs[key] != nil { return key }
        // Fallback: first multiarray integer input
        for (k, desc) in inputs {
            if let c = desc.multiArrayConstraint,
               c.dataType == .int32 { return k }
        }
        throw EmbedKitError.modelLoadFailed("No suitable token input found")
    }

    private func resolveMaskKey(from inputs: [String: MLFeatureDescription]) -> String? {
        let preferred = ["attention_mask", "mask"]
        for key in preferred where inputs[key] != nil { return key }
        for (k, desc) in inputs {
            if k.localizedCaseInsensitiveContains("mask"), desc.type == .multiArray { return k }
        }
        return nil
    }

    private func resolveTokenTypeKey(from inputs: [String: MLFeatureDescription]) -> String? {
        let preferred = ["token_type_ids", "type_ids"]
        for key in preferred where inputs[key] != nil { return key }
        return nil
    }

    private func resolvePositionKey(from inputs: [String: MLFeatureDescription]) -> String? {
        let preferred = ["position_ids", "positions"]
        for key in preferred where inputs[key] != nil { return key }
        return nil
    }

    private func makeMLMultiArray(values: [Int], key: String, inputs: [String: MLFeatureDescription]) throws -> MLMultiArray {
        let desc = inputs[key]!
        let constraint = desc.multiArrayConstraint
        let desiredShape: [NSNumber]
        if let shape = constraint?.shape, shape.count > 0 {
            // If shape has 2 dims, assume [1, seq]
            if shape.count == 2 { desiredShape = [1, NSNumber(value: values.count)] }
            else { desiredShape = [NSNumber(value: values.count)] }
        } else {
            desiredShape = [NSNumber(value: values.count)]
        }
        let dtype: MLMultiArrayDataType = constraint?.dataType ?? .int32
        let array = try MLMultiArray(shape: desiredShape, dataType: dtype)
        // Fill
        if dtype == .int32 {
            let ptr = UnsafeMutablePointer<Int32>(OpaquePointer(array.dataPointer))
            for (i, v) in values.enumerated() { ptr[i] = Int32(v) }
        } else {
            // Fallback, cast to float
            let ptr = UnsafeMutablePointer<Float>(OpaquePointer(array.dataPointer))
            for (i, v) in values.enumerated() { ptr[i] = Float(v) }
        }
        return array
    }

    private func flattenFloatArray(_ array: MLMultiArray) throws -> [Float] {
        let count = array.count
        switch array.dataType {
        case .float32:
            let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(array.dataPointer))
            return (0..<count).map { Float(ptr[$0]) }
        case .double:
            let ptr = UnsafeMutablePointer<Double>(OpaquePointer(array.dataPointer))
            return (0..<count).map { Float(ptr[$0]) }
        default:
            throw EmbedKitError.modelLoadFailed("Unsupported output data type \(array.dataType)")
        }
    }
    private func mapComputeUnits(_ device: ComputeDevice) -> MLComputeUnits {
        switch device {
        case .cpu: return .cpuOnly
        case .gpu: return .cpuAndGPU
        case .ane: return .all
        case .auto: return .all
        }
    }

    // Using nonisolated(unsafe) to call CoreML's non-concurrency-annotated API safely.
    nonisolated(unsafe) private func predict(_ model: MLModel, provider: MLFeatureProvider) throws -> MLFeatureProvider {
        try model.prediction(from: provider)
    }
    #endif
}

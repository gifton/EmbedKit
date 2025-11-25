// EmbedKit - CoreML Backend
// Phase 1: API surface and lifecycle only. No CoreML imports yet.

import Foundation
import Logging
#if canImport(CoreML)
import CoreML
#endif

/// Input to the CoreML backend prior to tensor conversion.
public struct CoreMLInput: Sendable {
    public let tokenIDs: [Int]
    public let attentionMask: [Int]

    public init(tokenIDs: [Int], attentionMask: [Int]) {
        self.tokenIDs = tokenIDs
        self.attentionMask = attentionMask
    }
}

/// Output from the CoreML backend prior to pooling and normalization.
public struct CoreMLOutput: Sendable {
    /// Flattened tensor values in row-major order (e.g., [tokens × dim]).
    public let values: [Float]
    /// Tensor shape reported by CoreML (e.g., [1, seq, dim] or [seq, dim]).
    public let shape: [Int]

    public init(values: [Float], shape: [Int]) {
        self.values = values
        self.shape = shape
    }
}

/// CoreML backend actor that manages model lifecycle and flexible input/output resolution.
/// - Resolves input feature keys dynamically and caches them.
/// - Returns raw output tensor and shape for pooling/validation at the model layer.
public actor CoreMLBackend: ModelBackend {
    public typealias Input = CoreMLInput
    public typealias Output = CoreMLOutput

    // MARK: - Configuration

    private let modelURL: URL?
    private let device: ComputeDevice
    // Auto device selection hint (set before load when device == .auto)
    private var autoHint: (paddedLength: Int, batchSize: Int)? = nil
    private let logger = Logger(label: "EmbedKit.CoreMLBackend")

    // MARK: - State

    private(set) public var isLoaded: Bool = false
    public var memoryUsage: Int64 { 0 }

    #if canImport(CoreML)
    private var mlModel: MLModel?
    private struct InputKeys { let token: String; let mask: String?; let type: String?; let pos: String? }
    private var cachedKeys: InputKeys?
    private struct Overrides { var token: String?; var mask: String?; var type: String?; var pos: String?; var output: String? }
    private var overrides = Overrides()
    #endif

    // MARK: - Init

    public init(modelURL: URL?, device: ComputeDevice = .auto) {
        self.modelURL = modelURL
        self.device = device
    }

    // MARK: - Auto device policy
    public enum ComputeClass: String { case cpu, hybrid, all }
    public static func chooseComputeClass(paddedLength: Int, batchSize: Int) -> ComputeClass {
        // Simple heuristics tuned for transformer embeddings
        let L = paddedLength
        let B = batchSize
        if B <= 2 && L <= 64 { return .cpu }
        if B <= 8 && L <= 256 { return .hybrid }
        return .all
    }
    public func setAutoWorkloadHint(paddedLength: Int, batchSize: Int) {
        // Accept hints only before load to affect compute units
        if !isLoaded { autoHint = (paddedLength, batchSize) }
    }

    // MARK: - Input/Output Overrides API
    public func setInputKeyOverrides(token: String? = nil, mask: String? = nil, type: String? = nil, pos: String? = nil) {
        #if canImport(CoreML)
        overrides.token = token
        overrides.mask = mask
        overrides.type = type
        overrides.pos = pos
        #endif
    }
    public func setOutputKeyOverride(_ key: String?) {
        #if canImport(CoreML)
        overrides.output = key
        #endif
    }

    // MARK: - Shape validation (platform-independent helper)
    // constraint: array of optional fixed dims; nil means flexible at that position
    static func validateShape(desired: [Int], against constraint: [Int?]) throws {
        guard !constraint.isEmpty else { return } // no constraints
        guard desired.count == constraint.count else {
            throw EmbedKitError.invalidConfiguration("Input shape rank mismatch: desired=\(desired.count), expected=\(constraint.count)")
        }
        for (i, (d, c)) in zip(desired, constraint).enumerated() {
            if let fixed = c, fixed != d {
                throw EmbedKitError.invalidConfiguration("Input shape mismatch at dim \(i): desired=\(d), expected=\(fixed)")
            }
        }
    }

    // MARK: - Lifecycle

    public func load() async throws {
        if isLoaded { return }

        #if canImport(CoreML)
        let url = try resolveModelURL()
        let config = MLModelConfiguration()
        var units = mapComputeUnits(device)
        if case .auto = device, let hint = autoHint {
            switch Self.chooseComputeClass(paddedLength: hint.paddedLength, batchSize: hint.batchSize) {
            case .cpu: units = .cpuOnly
            case .hybrid: units = .cpuAndGPU
            case .all: units = .all
            }
            logger.info("Auto device selection", metadata: [
                "paddedLength": .string("\(hint.paddedLength)"),
                "batchSize": .string("\(hint.batchSize)"),
                "computeClass": .string("\(units)")
            ])
        }
        config.computeUnits = units

        do {
            self.mlModel = try await MLModel.load(contentsOf: url, configuration: config)
            self.isLoaded = true
            self.cachedKeys = nil
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

        // Resolve input names and cache
        if cachedKeys == nil {
            let inputs = model.modelDescription.inputDescriptionsByName
            // Soft-warn if overrides are provided but not present
            if let o = overrides.token, inputs[o] == nil { logger.warning("Input override not found: token", metadata: ["key": .string(o)]) }
            if let o = overrides.mask, inputs[o] == nil { logger.warning("Input override not found: mask", metadata: ["key": .string(o)]) }
            if let o = overrides.type, inputs[o] == nil { logger.warning("Input override not found: type", metadata: ["key": .string(o)]) }
            if let o = overrides.pos, inputs[o] == nil { logger.warning("Input override not found: pos", metadata: ["key": .string(o)]) }

            // Apply overrides if present and valid; otherwise resolve heuristically
            let tokenKey = try resolveOrOverrideTokenKey(from: inputs)
            let maskKey = resolveOrOverrideMaskKey(from: inputs)
            let typeKey = resolveOrOverrideTypeKey(from: inputs)
            let posKey = resolveOrOverridePosKey(from: inputs)
            cachedKeys = InputKeys(token: tokenKey, mask: maskKey, type: typeKey, pos: posKey)
            logger.info("Resolved CoreML input keys", metadata: [
                "token": .string(tokenKey),
                "mask": .string(maskKey ?? "none"),
                "type": .string(typeKey ?? "none"),
                "pos": .string(posKey ?? "none")
            ])
        }
        guard let keys = cachedKeys else { throw EmbedKitError.modelLoadFailed("Input keys not resolved") }

        let inputs = model.modelDescription.inputDescriptionsByName
        let tokenArray = try makeMLMultiArray(values: input.tokenIDs, key: keys.token, inputs: inputs)
        var maskArray: MLMultiArray? = nil
        if let mk = keys.mask {
            maskArray = try? makeMLMultiArray(values: input.attentionMask, key: mk, inputs: inputs)
        }
        var typeArray: MLMultiArray? = nil
        if let tk = keys.type {
            let zeros = Array(repeating: 0, count: input.tokenIDs.count)
            typeArray = try? makeMLMultiArray(values: zeros, key: tk, inputs: inputs)
        }
        var posArray: MLMultiArray? = nil
        if let pk = keys.pos {
            let pos = (0..<input.tokenIDs.count).map { $0 }
            posArray = try? makeMLMultiArray(values: pos, key: pk, inputs: inputs)
        }

        var dict: [String: MLFeatureValue] = [keys.token: MLFeatureValue(multiArray: tokenArray)]
        if let mk = keys.mask, let arr = maskArray { dict[mk] = MLFeatureValue(multiArray: arr) }
        if let tk = keys.type, let arr = typeArray { dict[tk] = MLFeatureValue(multiArray: arr) }
        if let pk = keys.pos, let arr = posArray { dict[pk] = MLFeatureValue(multiArray: arr) }

        let provider = try MLDictionaryFeatureProvider(dictionary: dict)
        let output = try predict(model, provider: provider)

        // Preferred output key if provided
        if let preferred = overrides.output, let feat = output.featureValue(for: preferred), feat.type == .multiArray, let ma = feat.multiArrayValue, (ma.dataType == .float32 || ma.dataType == .double) {
            let floats = try flattenFloatArray(ma)
            let shape = ma.shape.map { $0.intValue }
            return CoreMLOutput(values: floats, shape: shape)
        }

        // Otherwise find the first float output multiarray
        if let preferred = overrides.output, output.featureValue(for: preferred) == nil {
            logger.debug("Output override not found; falling back to first float multiarray", metadata: ["key": .string(preferred)])
        }
        for name in output.featureNames {
            let val = output.featureValue(for: name)
            if val?.type == .multiArray, let ma = val?.multiArrayValue {
                if ma.dataType == .float32 || ma.dataType == .double {
                    let floats = try flattenFloatArray(ma)
                    let shape = ma.shape.map { $0.intValue }
                    return CoreMLOutput(values: floats, shape: shape)
                }
            }
        }
        throw EmbedKitError.modelLoadFailed("No float multiarray output found")
        #else
        throw EmbedKitError.modelLoadFailed("CoreML not available on this platform")
        #endif
    }

    public func processBatch(_ inputs: [Input]) async throws -> [Output] {
        #if canImport(CoreML)
        guard let model = mlModel else {
            throw EmbedKitError.modelLoadFailed("CoreML model not loaded")
        }

        // Resolve input keys once and cache
        let inputDescs = model.modelDescription.inputDescriptionsByName
        if cachedKeys == nil {
            let tokenKey = try resolveTokenKey(from: inputDescs)
            let maskKey = resolveMaskKey(from: inputDescs)
            let typeKey = resolveTokenTypeKey(from: inputDescs)
            let posKey = resolvePositionKey(from: inputDescs)
            cachedKeys = InputKeys(token: tokenKey, mask: maskKey, type: typeKey, pos: posKey)
            logger.info("Resolved CoreML input keys", metadata: [
                "token": .string(tokenKey),
                "mask": .string(maskKey ?? "none"),
                "type": .string(typeKey ?? "none"),
                "pos": .string(posKey ?? "none")
            ])
        }
        guard let keys = cachedKeys else { throw EmbedKitError.modelLoadFailed("Input keys not resolved") }

        // Build providers for the whole batch
        var providers: [MLFeatureProvider] = []
        providers.reserveCapacity(inputs.count)
        for inp in inputs {
            let tokenArray = try makeMLMultiArray(values: inp.tokenIDs, key: keys.token, inputs: inputDescs)
            var dict: [String: MLFeatureValue] = [keys.token: MLFeatureValue(multiArray: tokenArray)]

            if let mk = keys.mask {
                if let arr = try? makeMLMultiArray(values: inp.attentionMask, key: mk, inputs: inputDescs) {
                    dict[mk] = MLFeatureValue(multiArray: arr)
                }
            }
            if let tk = keys.type {
                let zeros = Array(repeating: 0, count: inp.tokenIDs.count)
                if let arr = try? makeMLMultiArray(values: zeros, key: tk, inputs: inputDescs) {
                    dict[tk] = MLFeatureValue(multiArray: arr)
                }
            }
            if let pk = keys.pos {
                let pos = (0..<inp.tokenIDs.count).map { $0 }
                if let arr = try? makeMLMultiArray(values: pos, key: pk, inputs: inputDescs) {
                    dict[pk] = MLFeatureValue(multiArray: arr)
                }
            }

            let provider = try MLDictionaryFeatureProvider(dictionary: dict)
            providers.append(provider)
        }

        let batchProvider = MLArrayBatchProvider(array: providers)
        let batchOutput = try model.predictions(from: batchProvider, options: MLPredictionOptions())

        // Extract outputs
        var outputs: [Output] = []
        outputs.reserveCapacity(batchOutput.count)
        for i in 0..<batchOutput.count {
            let feat = batchOutput.features(at: i)
            var found: Output? = nil
            // Try preferred key first
            if let preferred = overrides.output, let val = feat.featureValue(for: preferred), val.type == .multiArray, let ma = val.multiArrayValue, (ma.dataType == .float32 || ma.dataType == .double) {
                let floats = try flattenFloatArray(ma)
                let shape = ma.shape.map { $0.intValue }
                found = Output(values: floats, shape: shape)
            }
            if found == nil, let preferred = overrides.output, feat.featureValue(for: preferred) == nil {
                logger.debug("Batch output override not found for item; falling back", metadata: ["key": .string(preferred)])
            }
            for name in feat.featureNames where found == nil {
                guard let val = feat.featureValue(for: name) else { continue }
                if val.type == .multiArray, let ma = val.multiArrayValue {
                    if ma.dataType == .float32 || ma.dataType == .double {
                        let floats = try flattenFloatArray(ma)
                        let shape = ma.shape.map { $0.intValue }
                        found = Output(values: floats, shape: shape)
                        break
                    }
                }
            }
            if let out = found { outputs.append(out) }
            else { throw EmbedKitError.modelLoadFailed("No float multiarray output found for batch item \(i)") }
        }
        return outputs
        #else
        throw EmbedKitError.modelLoadFailed("CoreML not available on this platform")
        #endif
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
    
    // Override-aware resolvers
    private func resolveOrOverrideTokenKey(from inputs: [String: MLFeatureDescription]) throws -> String {
        if let o = overrides.token, inputs[o] != nil { return o }
        return try resolveTokenKey(from: inputs)
    }
    private func resolveOrOverrideMaskKey(from inputs: [String: MLFeatureDescription]) -> String? {
        if let o = overrides.mask, inputs[o] != nil { return o }
        return resolveMaskKey(from: inputs)
    }
    private func resolveOrOverrideTypeKey(from inputs: [String: MLFeatureDescription]) -> String? {
        if let o = overrides.type, inputs[o] != nil { return o }
        return resolveTokenTypeKey(from: inputs)
    }
    private func resolveOrOverridePosKey(from inputs: [String: MLFeatureDescription]) -> String? {
        if let o = overrides.pos, inputs[o] != nil { return o }
        return resolvePositionKey(from: inputs)
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
        // Validate desired shape against constraint if provided
        if let c = constraint?.shape, !c.isEmpty {
            // Interpret non-positive values as flexible
            let expected: [Int?] = c.map { $0.intValue > 0 ? $0.intValue : nil }
            let want: [Int] = desiredShape.map { $0.intValue }
            try Self.validateShape(desired: want, against: expected)
            logger.debug("Validated input array shape", metadata: [
                "key": .string(key),
                "desired": .string("\(want)"),
                "expected": .string("\(expected)")
            ])
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

    // Using nonisolated to call CoreML's API from actor context.
    nonisolated private func predict(_ model: MLModel, provider: MLFeatureProvider) throws -> MLFeatureProvider {
        try model.prediction(from: provider)
    }
    #endif
}

// Conform to the narrow processing interface for easy injection/testing
extension CoreMLBackend: CoreMLProcessingBackend {}

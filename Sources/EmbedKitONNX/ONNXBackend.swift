// EmbedKitONNX - ONNX Runtime Backend
// Inference wrapper for ONNX models

import Foundation
import EmbedKit
import OnnxRuntimeBindings

// MARK: - ONNX Input/Output Types

/// Input to the ONNX backend.
public struct ONNXInput: Sendable {
    public let tokenIDs: [Int]
    public let attentionMask: [Int]
    public let tokenTypeIDs: [Int]?

    public init(tokenIDs: [Int], attentionMask: [Int], tokenTypeIDs: [Int]? = nil) {
        self.tokenIDs = tokenIDs
        self.attentionMask = attentionMask
        self.tokenTypeIDs = tokenTypeIDs
    }

    /// Create from CoreMLInput for compatibility
    public init(from coreMLInput: CoreMLInput) {
        self.tokenIDs = coreMLInput.tokenIDs
        self.attentionMask = coreMLInput.attentionMask
        self.tokenTypeIDs = nil
    }
}

/// Output from the ONNX backend.
public struct ONNXOutput: Sendable {
    /// Flattened tensor values in row-major order
    public let values: [Float]
    /// Tensor shape (e.g., [1, seq_len, hidden_dim] or [seq_len, hidden_dim])
    public let shape: [Int]

    public init(values: [Float], shape: [Int]) {
        self.values = values
        self.shape = shape
    }

    /// Convert to CoreMLOutput for compatibility with existing pooling code
    public func toCoreMLOutput() -> CoreMLOutput {
        CoreMLOutput(values: values, shape: shape)
    }
}

// MARK: - ONNX Backend Configuration

/// Configuration options for the ONNX backend.
public struct ONNXBackendConfiguration: Sendable {
    /// Number of threads for intra-op parallelism (0 = auto)
    public var intraOpNumThreads: Int

    /// Number of threads for inter-op parallelism (0 = auto)
    public var interOpNumThreads: Int

    /// Whether to use CoreML execution provider on Apple platforms
    public var useCoreMLProvider: Bool

    /// Graph optimization level
    public var graphOptimizationLevel: GraphOptimizationLevel

    /// Default configuration
    public static let `default` = ONNXBackendConfiguration(
        intraOpNumThreads: 0,
        interOpNumThreads: 0,
        useCoreMLProvider: true,
        graphOptimizationLevel: .all
    )

    /// CPU-only configuration
    public static let cpuOnly = ONNXBackendConfiguration(
        intraOpNumThreads: 4,
        interOpNumThreads: 2,
        useCoreMLProvider: false,
        graphOptimizationLevel: .all
    )

    public init(
        intraOpNumThreads: Int = 0,
        interOpNumThreads: Int = 0,
        useCoreMLProvider: Bool = true,
        graphOptimizationLevel: GraphOptimizationLevel = .all
    ) {
        self.intraOpNumThreads = intraOpNumThreads
        self.interOpNumThreads = interOpNumThreads
        self.useCoreMLProvider = useCoreMLProvider
        self.graphOptimizationLevel = graphOptimizationLevel
    }

    public enum GraphOptimizationLevel: Int, Sendable {
        case disabled = 0
        case basic = 1
        case extended = 2
        case all = 99
    }
}

// MARK: - ONNX Backend Errors

/// Errors specific to ONNX backend operations.
public enum ONNXBackendError: Error, LocalizedError, Sendable {
    case modelNotLoaded
    case modelLoadFailed(String)
    case inferenceError(String)
    case invalidInput(String)
    case invalidOutput(String)
    case sessionCreationFailed(String)
    case unsupportedModel(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "ONNX model is not loaded"
        case .modelLoadFailed(let msg):
            return "Failed to load ONNX model: \(msg)"
        case .inferenceError(let msg):
            return "ONNX inference error: \(msg)"
        case .invalidInput(let msg):
            return "Invalid ONNX input: \(msg)"
        case .invalidOutput(let msg):
            return "Invalid ONNX output: \(msg)"
        case .sessionCreationFailed(let msg):
            return "Failed to create ONNX session: \(msg)"
        case .unsupportedModel(let msg):
            return "Unsupported ONNX model: \(msg)"
        }
    }
}

// MARK: - ONNX Backend

/// ONNX Runtime backend for embedding model inference.
///
/// Wraps the ONNX Runtime Objective-C bindings to provide model loading and inference
/// compatible with the EmbedKit backend protocol.
///
/// Example:
/// ```swift
/// let backend = ONNXBackend(modelPath: modelURL, config: .default)
/// try await backend.load()
///
/// let input = ONNXInput(tokenIDs: [101, 2054, 102], attentionMask: [1, 1, 1])
/// let output = try await backend.process(input)
/// ```
public actor ONNXBackend: ModelBackend {
    public typealias Input = ONNXInput
    public typealias Output = ONNXOutput

    // MARK: - Properties

    private let modelPath: URL
    private let config: ONNXBackendConfiguration

    private var env: ORTEnv?
    private var session: ORTSession?
    private var inputNames: [String] = []
    private var outputNames: [String] = []

    public private(set) var isLoaded: Bool = false
    public var memoryUsage: Int64 { 0 }

    // MARK: - Initialization

    /// Create an ONNX backend for a model file.
    ///
    /// - Parameters:
    ///   - modelPath: Path to the .onnx model file
    ///   - config: Backend configuration options
    public init(modelPath: URL, config: ONNXBackendConfiguration = .default) {
        self.modelPath = modelPath
        self.config = config
    }

    // MARK: - ModelBackend Protocol

    /// Load the ONNX model into memory.
    public func load() async throws {
        guard !isLoaded else { return }

        do {
            // Create environment
            env = try ORTEnv(loggingLevel: .warning)

            // Create session options
            let sessionOptions = try ORTSessionOptions()

            // Configure threading
            if config.intraOpNumThreads > 0 {
                try sessionOptions.setIntraOpNumThreads(Int32(config.intraOpNumThreads))
            }

            // Set graph optimization level
            try sessionOptions.setGraphOptimizationLevel(
                ORTGraphOptimizationLevel(rawValue: Int32(config.graphOptimizationLevel.rawValue)) ?? .all
            )

            // Add CoreML execution provider if requested
            #if os(macOS) || os(iOS)
            if config.useCoreMLProvider {
                try? sessionOptions.appendCoreMLExecutionProvider(with: ORTCoreMLExecutionProviderOptions())
            }
            #endif

            // Create session
            session = try ORTSession(
                env: env!,
                modelPath: modelPath.path,
                sessionOptions: sessionOptions
            )

            // Get input/output names
            inputNames = try session!.inputNames()
            outputNames = try session!.outputNames()

            isLoaded = true

        } catch {
            throw ONNXBackendError.modelLoadFailed(error.localizedDescription)
        }
    }

    /// Unload the model and free resources.
    public func unload() async throws {
        session = nil
        env = nil
        inputNames = []
        outputNames = []
        isLoaded = false
    }

    /// Process a single input through the model.
    public func process(_ input: ONNXInput) async throws -> ONNXOutput {
        guard isLoaded, let session = session else {
            throw ONNXBackendError.modelNotLoaded
        }

        do {
            // Create input tensors
            let inputs = try createInputTensors(input)

            // Run inference
            let outputSet = Set(outputNames)
            let outputs = try session.run(
                withInputs: inputs,
                outputNames: outputSet,
                runOptions: nil
            )

            // Extract output
            return try extractOutput(from: outputs)

        } catch let error as ONNXBackendError {
            throw error
        } catch {
            throw ONNXBackendError.inferenceError(error.localizedDescription)
        }
    }

    /// Process a batch of inputs.
    public func processBatch(_ inputs: [ONNXInput]) async throws -> [ONNXOutput] {
        var outputs: [ONNXOutput] = []
        outputs.reserveCapacity(inputs.count)

        for input in inputs {
            let output = try await process(input)
            outputs.append(output)
        }

        return outputs
    }

    // MARK: - Model Info

    /// Get the names of model inputs.
    public var modelInputNames: [String] {
        inputNames
    }

    /// Get the names of model outputs.
    public var modelOutputNames: [String] {
        outputNames
    }

    // MARK: - Private Helpers

    private func createInputTensors(_ input: ONNXInput) throws -> [String: ORTValue] {
        var tensors: [String: ORTValue] = [:]
        let seqLen = input.tokenIDs.count

        // Find input names (common patterns for transformer models)
        let tokenInputName = findInputName(candidates: ["input_ids", "input", "tokens"])
        let maskInputName = findInputName(candidates: ["attention_mask", "mask", "input_mask"])
        let typeInputName = findInputName(candidates: ["token_type_ids", "segment_ids", "type_ids"])

        // Create input_ids tensor
        if let name = tokenInputName {
            let tensor = try createInt64Tensor(from: input.tokenIDs, shape: [1, seqLen])
            tensors[name] = tensor
        }

        // Create attention_mask tensor
        if let name = maskInputName {
            let tensor = try createInt64Tensor(from: input.attentionMask, shape: [1, seqLen])
            tensors[name] = tensor
        }

        // Create token_type_ids tensor if needed
        if let name = typeInputName {
            let typeData = input.tokenTypeIDs ?? [Int](repeating: 0, count: seqLen)
            let tensor = try createInt64Tensor(from: typeData, shape: [1, seqLen])
            tensors[name] = tensor
        }

        guard !tensors.isEmpty else {
            throw ONNXBackendError.invalidInput("Could not match any input names")
        }

        return tensors
    }

    private func createInt64Tensor(from values: [Int], shape: [Int]) throws -> ORTValue {
        let int64Values = values.map { Int64($0) }
        let shape = shape.map { NSNumber(value: $0) }

        return try int64Values.withUnsafeBufferPointer { buffer in
            let data = Data(buffer: buffer)
            return try ORTValue(
                tensorData: NSMutableData(data: data),
                elementType: .int64,
                shape: shape
            )
        }
    }

    private func findInputName(candidates: [String]) -> String? {
        for candidate in candidates {
            if inputNames.contains(candidate) {
                return candidate
            }
        }
        // Try case-insensitive match
        for candidate in candidates {
            if let match = inputNames.first(where: { $0.lowercased() == candidate.lowercased() }) {
                return match
            }
        }
        return nil
    }

    private func extractOutput(from outputs: [String: ORTValue]) throws -> ONNXOutput {
        // Find the main output (usually last_hidden_state, embeddings, or sentence_embedding)
        let outputCandidates = [
            "last_hidden_state",
            "sentence_embedding",
            "embeddings",
            "output",
            "pooler_output"
        ]

        var outputValue: ORTValue?

        // Try candidates first
        for candidate in outputCandidates {
            if let value = outputs[candidate] {
                outputValue = value
                break
            }
        }

        // Fall back to first output
        if outputValue == nil, let first = outputs.values.first {
            outputValue = first
        }

        guard let value = outputValue else {
            throw ONNXBackendError.invalidOutput("No output tensor found")
        }

        // Extract tensor info
        let tensorInfo = try value.tensorTypeAndShapeInfo()
        let shape = tensorInfo.shape.map { Int(truncating: $0) }

        // Get float data
        let tensorData = try value.tensorData() as Data
        let floatData: [Float] = tensorData.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Float.self))
        }

        return ONNXOutput(values: floatData, shape: shape)
    }
}

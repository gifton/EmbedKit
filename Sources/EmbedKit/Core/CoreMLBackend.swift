//
//  CoreMLBackend.swift
//  EmbedKit
//
//  CoreML-based implementation of ModelBackend for on-device inference
//

import Foundation
import CoreML
import Accelerate

/// Errors that can occur during CoreML model operations
public enum CoreMLError: LocalizedError {
    case modelLoadingFailed(URL, Error)
    case invalidModelFormat(String)
    case inputShapeMismatch(expected: [Int], actual: [Int])
    case outputShapeMismatch(String)
    case inferenceFailure(Error)
    case unsupportedModelType(String)
    case missingRequiredInput(String)
    case missingRequiredOutput(String)

    public var errorDescription: String? {
        switch self {
        case .modelLoadingFailed(let url, let error):
            return "Failed to load model from \(url): \(error.localizedDescription)"
        case .invalidModelFormat(let details):
            return "Invalid model format: \(details)"
        case .inputShapeMismatch(let expected, let actual):
            return "Input shape mismatch. Expected: \(expected), Actual: \(actual)"
        case .outputShapeMismatch(let details):
            return "Output shape mismatch: \(details)"
        case .inferenceFailure(let error):
            return "Inference failed: \(error.localizedDescription)"
        case .unsupportedModelType(let type):
            return "Unsupported model type: \(type)"
        case .missingRequiredInput(let name):
            return "Model missing required input: \(name)"
        case .missingRequiredOutput(let name):
            return "Model missing required output: \(name)"
        }
    }
}

/// Configuration for CoreML model backend
public struct CoreMLConfiguration: Sendable {
    /// Whether to use GPU acceleration (Neural Engine)
    public let useNeuralEngine: Bool

    /// Whether to allow CPU fallback
    public let allowCPUFallback: Bool

    /// Maximum batch size for inference
    public let maxBatchSize: Int

    /// Input tensor names (for different model formats)
    public let inputNames: InputNames

    /// Output tensor names (for different model formats)
    public let outputNames: OutputNames

    public struct InputNames: Sendable {
        public let tokenIds: String
        public let attentionMask: String
        public let tokenTypeIds: String?

        public init(
            tokenIds: String = "input_ids",
            attentionMask: String = "attention_mask",
            tokenTypeIds: String? = "token_type_ids"
        ) {
            self.tokenIds = tokenIds
            self.attentionMask = attentionMask
            self.tokenTypeIds = tokenTypeIds
        }
    }

    public struct OutputNames: Sendable {
        public let lastHiddenState: String
        public let poolerOutput: String?

        public init(
            lastHiddenState: String = "last_hidden_state",
            poolerOutput: String? = "pooler_output"
        ) {
            self.lastHiddenState = lastHiddenState
            self.poolerOutput = poolerOutput
        }
    }

    public init(
        useNeuralEngine: Bool = true,
        allowCPUFallback: Bool = true,
        maxBatchSize: Int = 32,
        inputNames: InputNames = InputNames(),
        outputNames: OutputNames = OutputNames()
    ) {
        self.useNeuralEngine = useNeuralEngine
        self.allowCPUFallback = allowCPUFallback
        self.maxBatchSize = maxBatchSize
        self.inputNames = inputNames
        self.outputNames = outputNames
    }
}

/// CoreML-based model backend for on-device embedding generation
public actor CoreMLBackend: ModelBackend {
    // MLModel is thread-safe for inference, so we can use nonisolated(unsafe)
    nonisolated(unsafe) private var model: MLModel?
    private let configuration: CoreMLConfiguration
    private var modelMetadata: ModelMetadata?

    /// Unique identifier for this backend
    public let identifier: String = "CoreML"

    /// Whether the model is currently loaded
    public var isLoaded: Bool {
        return model != nil
    }

    /// Model metadata
    public var metadata: ModelMetadata? {
        return modelMetadata
    }

    /// Initialize with configuration
    public init(configuration: CoreMLConfiguration = CoreMLConfiguration()) {
        self.configuration = configuration
    }

    /// Load a CoreML model from the specified URL
    public func loadModel(from url: URL) async throws {
        do {
            // Configure model for optimal performance
            let config = MLModelConfiguration()

            // Set compute units based on configuration
            if configuration.useNeuralEngine {
                config.computeUnits = configuration.allowCPUFallback ? .all : .cpuAndNeuralEngine
            } else {
                config.computeUnits = .cpuOnly
            }

            // Load model
            let loadedModel = try await MLModel.load(
                contentsOf: url,
                configuration: config
            )

            // Validate model structure
            try validateModelStructure(loadedModel)

            // Extract and store metadata
            self.modelMetadata = try extractModelMetadata(loadedModel)

            // Store model
            self.model = loadedModel

        } catch let error as CoreMLError {
            throw error
        } catch {
            throw CoreMLError.modelLoadingFailed(url, error)
        }
    }

    /// Unload the current model
    public func unloadModel() async throws {
        self.model = nil
        self.modelMetadata = nil
    }

    /// Generate embeddings for a single tokenized input
    public func generateEmbeddings(for input: TokenizedInput) async throws -> ModelOutput {
        guard let model = model else {
            throw CoreMLError.unsupportedModelType("No model loaded")
        }

        // Run inference (MLModel.prediction is thread-safe)
        // Creating input and running prediction in nonisolated context
        let prediction = try await runPrediction(model: model, tokenizedInput: input, config: configuration)

        // Extract embeddings from output
        return try extractModelOutput(from: prediction, tokenCount: input.tokenIds.count)
    }

    /// Thread-safe prediction wrapper - creates input and runs prediction
    private nonisolated func runPrediction(model: MLModel, tokenizedInput: TokenizedInput, config: CoreMLConfiguration) async throws -> MLFeatureProvider {
        do {
            // Convert TokenizedInput to CoreML input
            let mlInput = try createModelInput(from: tokenizedInput, configuration: config)
            // Run prediction
            return try await model.prediction(from: mlInput)
        } catch let error as CoreMLError {
            throw error
        } catch {
            throw CoreMLError.inferenceFailure(error)
        }
    }

    /// Generate embeddings for multiple tokenized inputs
    public func generateEmbeddings(for inputs: [TokenizedInput]) async throws -> [ModelOutput] {
        // Process in batches for efficiency
        var results: [ModelOutput] = []

        for batch in inputs.chunked(into: configuration.maxBatchSize) {
            // Process batch concurrently
            let batchResults = try await withThrowingTaskGroup(of: ModelOutput.self) { group in
                for input in batch {
                    group.addTask {
                        try await self.generateEmbeddings(for: input)
                    }
                }

                var batchOutputs: [ModelOutput] = []
                for try await output in group {
                    batchOutputs.append(output)
                }
                return batchOutputs
            }

            results.append(contentsOf: batchResults)
        }

        return results
    }

    /// Get input dimensions of the loaded model
    public func inputDimensions() async -> (sequence: Int, features: Int)? {
        guard let metadata = modelMetadata else { return nil }
        return (sequence: metadata.maxSequenceLength, features: metadata.embeddingDimensions)
    }

    /// Get output dimensions of the loaded model
    public func outputDimensions() async -> Int? {
        return modelMetadata?.embeddingDimensions
    }

    // MARK: - Private Methods

    /// Validate that the model has required inputs and outputs
    private func validateModelStructure(_ model: MLModel) throws {
        let description = model.modelDescription

        // Check for required inputs
        guard description.inputDescriptionsByName[configuration.inputNames.tokenIds] != nil else {
            throw CoreMLError.missingRequiredInput(configuration.inputNames.tokenIds)
        }

        guard description.inputDescriptionsByName[configuration.inputNames.attentionMask] != nil else {
            throw CoreMLError.missingRequiredInput(configuration.inputNames.attentionMask)
        }

        // Check for required outputs
        guard description.outputDescriptionsByName[configuration.outputNames.lastHiddenState] != nil else {
            throw CoreMLError.missingRequiredOutput(configuration.outputNames.lastHiddenState)
        }
    }

    /// Extract metadata from the model
    private func extractModelMetadata(_ model: MLModel) throws -> ModelMetadata {
        let description = model.modelDescription

        // Extract input dimensions
        guard let inputDesc = description.inputDescriptionsByName[configuration.inputNames.tokenIds],
              case .multiArray = inputDesc.type,
              let inputArray = inputDesc.multiArrayConstraint else {
            throw CoreMLError.invalidModelFormat("Cannot extract input dimensions")
        }

        let sequenceLength = inputArray.shape[1].intValue

        // Extract output dimensions
        guard let outputDesc = description.outputDescriptionsByName[configuration.outputNames.lastHiddenState],
              case .multiArray = outputDesc.type,
              let outputArray = outputDesc.multiArrayConstraint else {
            throw CoreMLError.invalidModelFormat("Cannot extract output dimensions")
        }

        let hiddenSize = outputArray.shape.last?.intValue ?? 768

        return ModelMetadata(
            name: "CoreML Model",
            version: "1.0",
            embeddingDimensions: hiddenSize,
            maxSequenceLength: sequenceLength,
            vocabularySize: 30522,  // TODO: Extract from model
            modelType: "BERT",  // TODO: Detect from metadata
            additionalInfo: [
                "sequenceLength": String(sequenceLength),
                "hiddenSize": String(hiddenSize)
            ]
        )
    }

    /// Create MLFeatureProvider input from TokenizedInput
    private nonisolated func createModelInput(from input: TokenizedInput, configuration: CoreMLConfiguration) throws -> MLFeatureProvider {
        var features: [String: MLFeatureValue] = [:]

        // Convert token IDs to MLMultiArray
        let tokenArray = try MLMultiArray(shape: [1, NSNumber(value: input.tokenIds.count)], dataType: .int32)
        for (i, tokenId) in input.tokenIds.enumerated() {
            tokenArray[[0, NSNumber(value: i)]] = NSNumber(value: tokenId)
        }
        features[configuration.inputNames.tokenIds] = MLFeatureValue(multiArray: tokenArray)

        // Convert attention mask to MLMultiArray
        let attentionArray = try MLMultiArray(shape: [1, NSNumber(value: input.attentionMask.count)], dataType: .int32)
        for (i, mask) in input.attentionMask.enumerated() {
            attentionArray[[0, NSNumber(value: i)]] = NSNumber(value: mask)
        }
        features[configuration.inputNames.attentionMask] = MLFeatureValue(multiArray: attentionArray)

        // Add token type IDs if present and required
        if let tokenTypeIds = input.tokenTypeIds,
           let tokenTypeName = configuration.inputNames.tokenTypeIds {
            let typeArray = try MLMultiArray(shape: [1, NSNumber(value: tokenTypeIds.count)], dataType: .int32)
            for (i, typeId) in tokenTypeIds.enumerated() {
                typeArray[[0, NSNumber(value: i)]] = NSNumber(value: typeId)
            }
            features[tokenTypeName] = MLFeatureValue(multiArray: typeArray)
        }

        return try MLDictionaryFeatureProvider(dictionary: features)
    }

    /// Extract ModelOutput from CoreML prediction
    private func extractModelOutput(from prediction: MLFeatureProvider, tokenCount: Int) throws -> ModelOutput {
        // Extract last hidden states
        guard let hiddenStatesValue = prediction.featureValue(for: configuration.outputNames.lastHiddenState),
              let hiddenStates = hiddenStatesValue.multiArrayValue else {
            throw CoreMLError.missingRequiredOutput(configuration.outputNames.lastHiddenState)
        }

        // Convert MLMultiArray to [[Float]]
        let hiddenSize = hiddenStates.shape[2].intValue
        var tokenEmbeddings: [[Float]] = []

        for tokenIdx in 0..<tokenCount {
            var embedding: [Float] = []
            for dimIdx in 0..<hiddenSize {
                let value = hiddenStates[[0, NSNumber(value: tokenIdx), NSNumber(value: dimIdx)]].floatValue
                embedding.append(value)
            }
            tokenEmbeddings.append(embedding)
        }

        // Prepare metadata
        var metadata: [String: String] = [
            "hiddenSize": String(hiddenSize),
            "tokenCount": String(tokenCount)
        ]

        // Extract pooler output if available and add to metadata
        if let poolerName = configuration.outputNames.poolerOutput,
           let poolerValue = prediction.featureValue(for: poolerName),
           let poolerArray = poolerValue.multiArrayValue {
            metadata["hasPoolerOutput"] = "true"
            // Note: We could store pooler output in metadata if needed
        }

        return ModelOutput(
            tokenEmbeddings: tokenEmbeddings,
            attentionWeights: nil,  // Not typically available from CoreML models
            metadata: metadata
        )
    }
}

// MARK: - Array Extension for Chunking

private extension Array {
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }
}
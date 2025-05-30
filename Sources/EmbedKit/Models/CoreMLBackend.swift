import Foundation
import CoreML
import Accelerate
import OSLog

/// Core ML implementation of the model backend
public actor CoreMLBackend: ModelBackend {
    private let logger = Logger(subsystem: "EmbedKit", category: "CoreMLBackend")
    
    public let identifier: String
    private var model: MLModel?
    private var compiledModelURL: URL?
    private var _metadata: ModelMetadata?
    
    public var isLoaded: Bool {
        model != nil
    }
    
    public var metadata: ModelMetadata? {
        _metadata
    }
    
    public init(identifier: String) {
        self.identifier = identifier
    }
    
    public func loadModel(from url: URL) async throws {
        logger.info("Loading Core ML model from: \(url.path)")
        
        let startTime = Date()
        
        // Compile the model if needed
        let compiledURL: URL
        if url.pathExtension == "mlmodelc" {
            compiledURL = url
        } else {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .all
            
            compiledURL = try await Task.detached {
                try MLModel.compileModel(at: url)
            }.value
            
            self.compiledModelURL = compiledURL
        }
        
        // Load the compiled model
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        
        let loadedModel = try MLModel(contentsOf: compiledURL, configuration: configuration)
        
        self.model = loadedModel
        
        // Extract metadata from model description
        let modelDescription = loadedModel.modelDescription
        self._metadata = extractMetadata(from: modelDescription)
        
        let duration = Date().timeIntervalSince(startTime)
        logger.info("Model loaded successfully in \(duration)s with \(self._metadata?.embeddingDimensions ?? 0) dimensions")
    }
    
    public func unloadModel() async throws {
        logger.info("Unloading model")
        
        self.model = nil
        self._metadata = nil
        
        // Clean up compiled model if we created it
        if let compiledURL = compiledModelURL {
            try? FileManager.default.removeItem(at: compiledURL)
            self.compiledModelURL = nil
        }
    }
    
    public func generateEmbeddings(for input: TokenizedInput) async throws -> ModelOutput {
        guard let model = model else {
            throw EmbeddingError.modelNotLoaded
        }
        
        // Prepare input features
        let inputFeatures = try prepareInput(input, for: model)
        
        // Run prediction
        let output = try model.prediction(from: inputFeatures)
        
        // Extract embeddings from output
        let embeddings = try extractEmbeddings(from: output, model: model)
        
        return ModelOutput(
            tokenEmbeddings: embeddings,
            attentionWeights: nil,
            metadata: [:]
        )
    }
    
    public func inputDimensions() async -> (sequence: Int, features: Int)? {
        guard let model = model else { return nil }
        
        // Extract input dimensions from model description
        guard let inputDescription = model.modelDescription.inputDescriptionsByName.values.first,
              let shape = inputDescription.multiArrayConstraint?.shape else {
            return nil
        }
        
        // Assuming shape is [batch, sequence, features] or [sequence, features]
        if shape.count >= 2 {
            let sequence = shape[shape.count - 2].intValue
            let features = shape[shape.count - 1].intValue
            return (sequence, features)
        }
        
        return nil
    }
    
    public func outputDimensions() async -> Int? {
        guard let model = model else { return nil }
        
        // Find the embeddings output
        for (name, description) in model.modelDescription.outputDescriptionsByName {
            if name.lowercased().contains("embedding") || name.lowercased().contains("output") {
                if let shape = description.multiArrayConstraint?.shape,
                   let lastDimension = shape.last {
                    return lastDimension.intValue
                }
            }
        }
        
        return nil
    }
    
    // MARK: - Private Helpers
    
    private func extractMetadata(from description: MLModelDescription) -> ModelMetadata {
        // Extract dimensions from output shape
        var embeddingDimensions = 768 // Default
        var maxSequenceLength = 512 // Default
        
        // Try to find embedding dimensions from outputs
        for (name, output) in description.outputDescriptionsByName {
            if let shape = output.multiArrayConstraint?.shape,
               (name.lowercased().contains("embedding") || name.lowercased().contains("output")) {
                if let lastDim = shape.last {
                    embeddingDimensions = lastDim.intValue
                }
            }
        }
        
        // Try to find sequence length from inputs
        for (name, input) in description.inputDescriptionsByName {
            if let shape = input.multiArrayConstraint?.shape,
               (name.lowercased().contains("input") || name.lowercased().contains("token")) {
                if shape.count >= 2 {
                    maxSequenceLength = shape[shape.count - 2].intValue
                }
            }
        }
        
        let metadata = description.metadata[MLModelMetadataKey.description] as? [String: Any] ?? [:]
        
        return ModelMetadata(
            name: metadata["name"] as? String ?? "CoreML Model",
            version: metadata["version"] as? String ?? "1.0",
            embeddingDimensions: embeddingDimensions,
            maxSequenceLength: maxSequenceLength,
            vocabularySize: 30522, // Default BERT vocab size
            modelType: "coreml",
            additionalInfo: [:]
        )
    }
    
    private func prepareInput(_ input: TokenizedInput, for model: MLModel) throws -> MLFeatureProvider {
        let inputDescription = model.modelDescription.inputDescriptionsByName
        var features: [String: MLFeatureValue] = [:]
        
        // Find and populate token IDs input
        for (name, _) in inputDescription {
            if name.lowercased().contains("input_ids") || name.lowercased().contains("token") {
                let multiArray = try MLMultiArray(input.tokenIds)
                features[name] = MLFeatureValue(multiArray: multiArray)
            } else if name.lowercased().contains("attention_mask") || name.lowercased().contains("mask") {
                let multiArray = try MLMultiArray(input.attentionMask)
                features[name] = MLFeatureValue(multiArray: multiArray)
            } else if name.lowercased().contains("token_type") && input.tokenTypeIds != nil {
                let multiArray = try MLMultiArray(input.tokenTypeIds!)
                features[name] = MLFeatureValue(multiArray: multiArray)
            }
        }
        
        return try MLDictionaryFeatureProvider(dictionary: features)
    }
    
    private func extractEmbeddings(from output: MLFeatureProvider, model: MLModel) throws -> [[Float]] {
        let outputDescription = model.modelDescription.outputDescriptionsByName
        
        // Find the embeddings output
        for (name, _) in outputDescription {
            if let featureValue = output.featureValue(for: name),
               let multiArray = featureValue.multiArrayValue {
                return try convertToFloat2D(multiArray)
            }
        }
        
        throw EmbeddingError.inferenceFailed("Could not find embeddings in model output")
    }
    
    private func convertToFloat2D(_ multiArray: MLMultiArray) throws -> [[Float]] {
        let shape = multiArray.shape
        guard shape.count >= 2 else {
            throw EmbeddingError.inferenceFailed("Expected at least 2D output, got \(shape.count)D")
        }
        
        let sequenceLength = shape[shape.count - 2].intValue
        let embeddingSize = shape[shape.count - 1].intValue
        
        var result: [[Float]] = []
        result.reserveCapacity(sequenceLength)
        
        // Convert to Float32 array
        let totalElements = sequenceLength * embeddingSize
        var floatArray = [Float](repeating: 0, count: totalElements)
        
        switch multiArray.dataType {
        case .float32:
            floatArray.withUnsafeMutableBufferPointer { buffer in
                buffer.baseAddress!.initialize(from: multiArray.dataPointer.bindMemory(to: Float.self, capacity: totalElements), count: totalElements)
            }
            
        case .float64:
            let doublePointer = multiArray.dataPointer.bindMemory(to: Double.self, capacity: totalElements)
            vDSP_vdpsp(doublePointer, 1, &floatArray, 1, vDSP_Length(totalElements))
            
        default:
            throw EmbeddingError.inferenceFailed("Unsupported data type: \(multiArray.dataType)")
        }
        
        // Reshape to 2D
        for i in 0..<sequenceLength {
            let start = i * embeddingSize
            let end = start + embeddingSize
            result.append(Array(floatArray[start..<end]))
        }
        
        return result
    }
}

// MARK: - MLMultiArray Extensions

extension MLMultiArray {
    convenience init(_ array: [Int]) throws {
        try self.init(shape: [NSNumber(value: array.count)], dataType: .int32)
        
        for (index, value) in array.enumerated() {
            self[index] = NSNumber(value: value)
        }
    }
}
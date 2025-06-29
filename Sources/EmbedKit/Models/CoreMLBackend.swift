import Foundation
@preconcurrency import CoreML
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
        
        // Extract metadata using enhanced extractor
        self._metadata = try await CoreMLMetadataExtractor.extractMetadata(
            from: loadedModel,
            modelIdentifier: identifier,
            modelURL: url
        )
        
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
            throw ContextualEmbeddingError.modelNotLoaded(
                context: ErrorContext(
                    operation: .inference,
                    modelIdentifier: try? ModelIdentifier(identifier),
                    sourceLocation: SourceLocation()
                )
            )
        }
        
        // Create helper and run prediction
        let helper = CoreMLPredictionHelper(modelIdentifier: identifier)
        
        // CoreML operations are synchronous, so we just call them directly
        return try helper.predict(input: input, model: model)
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
    
}

// MARK: - Helper for non-isolated predictions

struct CoreMLPredictionHelper {
    let modelIdentifier: String
    
    func predict(input: TokenizedInput, model: MLModel) throws -> ModelOutput {
        // Prepare input features
        let inputFeatures = try prepareInput(input, for: model)
        
        // Run prediction
        let output = try model.prediction(from: inputFeatures)
        
        // Extract embeddings from output
        let embeddings = try extractEmbeddings(from: output, model: model)
        
        // Try to extract attention weights if available
        let attentionWeights = try? extractAttentionWeights(from: output, model: model)
        
        return ModelOutput(
            tokenEmbeddings: embeddings,
            attentionWeights: attentionWeights,
            metadata: [:]
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
        
        throw ContextualEmbeddingError.inferenceFailed(
            context: ErrorContext(
                operation: .inference,
                modelIdentifier: try? ModelIdentifier(modelIdentifier),
                metadata: ErrorMetadata()
                    .with(key: "reason", value: "Could not find embeddings in model output"),
                sourceLocation: SourceLocation()
            )
        )
    }
    
    private func convertToFloat2D(_ multiArray: MLMultiArray) throws -> [[Float]] {
        let shape = multiArray.shape
        guard shape.count >= 2 else {
            throw ContextualEmbeddingError.inferenceFailed(
                context: ErrorContext(
                    operation: .inference,
                    modelIdentifier: try? ModelIdentifier(modelIdentifier),
                    metadata: ErrorMetadata()
                        .with(key: "reason", value: "Expected at least 2D output, got \(shape.count)D"),
                    sourceLocation: SourceLocation()
                )
            )
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
            throw ContextualEmbeddingError.inferenceFailed(
                context: ErrorContext(
                    operation: .inference,
                    modelIdentifier: try? ModelIdentifier(modelIdentifier),
                    metadata: ErrorMetadata()
                        .with(key: "reason", value: "Unsupported data type: \(multiArray.dataType)"),
                    sourceLocation: SourceLocation()
                )
            )
        }
        
        // Reshape to 2D
        for i in 0..<sequenceLength {
            let start = i * embeddingSize
            let end = start + embeddingSize
            result.append(Array(floatArray[start..<end]))
        }
        
        return result
    }
    
    private func extractAttentionWeights(from output: MLFeatureProvider, model: MLModel) throws -> [[Float]]? {
        let outputDescription = model.modelDescription.outputDescriptionsByName
        
        // Look for attention weights in the output
        // Common names: "attentions", "attention_weights", "attention_scores"
        for (name, _) in outputDescription {
            if name.lowercased().contains("attention") && !name.lowercased().contains("mask") {
                if let featureValue = output.featureValue(for: name),
                   let multiArray = featureValue.multiArrayValue {
                    // Extract the last attention layer's weights
                    // Shape is typically [batch, num_heads, seq_len, seq_len]
                    // We want to average across heads and extract diagonal or mean
                    return try extractAttentionFromMultiArray(multiArray)
                }
            }
        }
        
        return nil
    }
    
    private func extractAttentionFromMultiArray(_ multiArray: MLMultiArray) throws -> [[Float]] {
        let shape = multiArray.shape
        
        // Handle different attention tensor shapes
        if shape.count == 4 {
            // [batch, num_heads, seq_len, seq_len]
            let batch = shape[0].intValue
            let numHeads = shape[1].intValue
            let seqLen = shape[2].intValue
            
            // Average attention weights across all heads
            var averagedWeights: [[Float]] = []
            
            for b in 0..<batch {
                var tokenWeights = [Float](repeating: 0, count: seqLen)
                
                // Average across heads
                for h in 0..<numHeads {
                    for i in 0..<seqLen {
                        // Sum attention from all tokens to token i
                        var sum: Float = 0
                        for j in 0..<seqLen {
                            let index = [b, h, i, j] as [NSNumber]
                            sum += multiArray[index].floatValue
                        }
                        tokenWeights[i] += sum / Float(seqLen)
                    }
                }
                
                // Average across heads
                tokenWeights = tokenWeights.map { $0 / Float(numHeads) }
                averagedWeights.append(tokenWeights)
            }
            
            return averagedWeights
        } else if shape.count == 3 {
            // [batch, seq_len, seq_len] - already averaged across heads
            let batch = shape[0].intValue
            let seqLen = shape[1].intValue
            
            var weights: [[Float]] = []
            
            for b in 0..<batch {
                var tokenWeights = [Float](repeating: 0, count: seqLen)
                
                for i in 0..<seqLen {
                    // Average attention from all tokens to token i
                    var sum: Float = 0
                    for j in 0..<seqLen {
                        let index = [b, i, j] as [NSNumber]
                        sum += multiArray[index].floatValue
                    }
                    tokenWeights[i] = sum / Float(seqLen)
                }
                
                weights.append(tokenWeights)
            }
            
            return weights
        } else if shape.count == 2 {
            // [batch, seq_len] - pre-computed attention weights
            return try convertToFloat2D(multiArray)
        }
        
        // Unsupported shape - return empty array
        return []
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

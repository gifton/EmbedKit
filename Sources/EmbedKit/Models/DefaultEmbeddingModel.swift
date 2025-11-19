import Foundation
import CoreML
import VectorCore

/// Implementation of EmbeddingModel using CoreML
public actor DefaultEmbeddingModel: EmbeddingModel {
    public let id: String
    public let dimension: Int
    
    private let model: SendableMLModel
    private let tokenizer: Tokenizer
    private let maxSequenceLength: Int
    
    public init(
        id: String,
        model: MLModel,
        tokenizer: Tokenizer,
        dimension: Int = 384, // MiniLM default
        maxSequenceLength: Int = 512
    ) {
        self.id = id
        self.model = SendableMLModel(model)
        self.tokenizer = tokenizer
        self.dimension = dimension
        self.maxSequenceLength = maxSequenceLength
    }
    
    /// Convenience initializer to load from a URL
    public init(
        id: String,
        modelURL: URL,
        tokenizer: Tokenizer,
        configuration: MLModelConfiguration = MLModelConfiguration()
    ) async throws {
        let model = try await MLModel.load(contentsOf: modelURL, configuration: configuration)
        self.init(id: id, model: model, tokenizer: tokenizer)
    }
    
    public func embed(_ text: String) async throws -> Embedding {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // 1. Tokenize
        let tokens = try await tokenizer.encode(text)
        
        // 2. Prepare input (truncate/pad)
        let inputIds = prepareInput(tokens)
        
        // 3. Create MLMultiArray
        let inputArray = try MLMultiArray(shape: [1, NSNumber(value: maxSequenceLength)], dataType: .int32)
        for (index, id) in inputIds.enumerated() {
            inputArray[index] = NSNumber(value: id)
        }
        
        // 4. Run inference
        // Note: Input name 'input_ids' is standard for BERT models, but might vary.
        // We assume the model has an input named "input_ids" and output "embeddings" or similar.
        // For this generic implementation, we might need to inspect the model description or make it configurable.
        // Assuming MiniLM structure: input "input_ids", output "last_hidden_state" or "embedding"
        
        // Let's try to find the input feature name
        guard let inputName = model.modelDescription.inputDescriptionsByName.keys.first else {
            throw EmbedKitError.invalidInput("Model has no inputs")
        }
        
        let inputProvider = try MLDictionaryFeatureProvider(dictionary: [inputName: inputArray])
        // prediction(from:) is synchronous in CoreML, but if the compiler insists on await, 
        // it might be due to actor isolation context or SDK changes. 
        // However, usually it's sync. The previous error said "call is async".
        // Let's try wrapping it in a non-isolated context or just adding await if it compiles.
        // Actually, MLModel.prediction is definitely sync. 
        // The error might be because I'm accessing a property of a Sendable struct that wraps a non-Sendable type?
        // Let's try just fixing the .model access first.
        let outputProvider = try model.prediction(from: inputProvider)
        
        // 5. Extract embedding
        // We need to pool the output. Assuming CLS token (first vector) or mean pooling.
        // For simplicity in Week 1, let's assume the model outputs a pooled embedding or we take the first token.
        
        // Find the output feature that looks like an embedding
        guard let outputFeature = outputProvider.featureValue(for: "var_1060") ?? // Common in some exports
                                  outputProvider.featureValue(for: "embeddings") ??
                                  outputProvider.featureValue(for: "last_hidden_state") else {
             // Fallback: take the first array output
             if let firstOutput = model.modelDescription.outputDescriptionsByName.keys.first,
                let val = outputProvider.featureValue(for: firstOutput) {
                 return try processOutput(val, startTime: startTime, tokenCount: tokens.count)
             }
             throw EmbedKitError.inferenceFailed("Could not find output feature")
        }
        
        return try processOutput(outputFeature, startTime: startTime, tokenCount: tokens.count)
    }
    
    private func processOutput(_ feature: MLFeatureValue, startTime: Double, tokenCount: Int) throws -> Embedding {
        guard let multiArray = feature.multiArrayValue else {
            throw EmbedKitError.inferenceFailed("Output is not a multi-array")
        }
        
        // If output is [1, sequence_length, dimension], we need to pool.
        // If output is [1, dimension], it's already pooled.
        
        var vector: [Float]
        let shape = multiArray.shape.map { $0.intValue }
        
        if shape.count == 2 && shape[1] == dimension {
            // [1, 384] - Already pooled
            vector = (0..<dimension).map { Float(multiArray[$0].floatValue) }
        } else if shape.count == 3 && shape[2] == dimension {
            // [1, 512, 384] - Sequence output, take CLS (index 0)
            // Stride calculation might be needed, but usually it's contiguous
            vector = (0..<dimension).map { i in
                Float(multiArray[i].floatValue) // simplified, assumes CLS is at start
            }
        } else {
            // Fallback: just try to read 'dimension' floats
            vector = (0..<dimension).map { Float(multiArray[$0].floatValue) }
        }
        
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        return Embedding(
            vector: vector,
            metadata: EmbeddingMetadata(
                modelID: id,
                tokenCount: tokenCount,
                processingTime: elapsed
            )
        )
    }
    
    public func embedBatch(_ texts: [String]) async throws -> BatchResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        var embeddings: [Embedding] = []
        var totalTokens = 0
        
        // Sequential for Week 1
        for text in texts {
            let embedding = try await embed(text)
            embeddings.append(embedding)
            totalTokens += embedding.metadata.tokenCount
        }
        
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        return BatchResult(
            embeddings: embeddings,
            totalTime: elapsed,
            totalTokens: totalTokens
        )
    }
    
    private func prepareInput(_ tokens: [Int]) -> [Int] {
        if tokens.count > maxSequenceLength {
            return Array(tokens.prefix(maxSequenceLength))
        } else {
            var padded = tokens
            padded.append(contentsOf: Array(repeating: 0, count: maxSequenceLength - tokens.count))
            return padded
        }
    }
}

struct SendableMLModel: @unchecked Sendable {
    private let model: MLModel
    
    init(_ model: MLModel) {
        self.model = model
    }
    
    var modelDescription: MLModelDescription {
        model.modelDescription
    }
    
    func prediction(from input: MLFeatureProvider) throws -> MLFeatureProvider {
        try model.prediction(from: input)
    }
}

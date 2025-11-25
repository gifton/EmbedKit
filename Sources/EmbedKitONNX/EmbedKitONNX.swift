// EmbedKitONNX - ONNX Runtime Support for EmbedKit
//
// This module adds ONNX model support to EmbedKit, enabling inference
// on .onnx model files using the ONNX Runtime.
//
// Usage:
// ```swift
// import EmbedKit
// import EmbedKitONNX
//
// let model = LocalONNXModel(
//     modelURL: URL(fileURLWithPath: "model.onnx"),
//     tokenizer: myTokenizer
// )
//
// let embedding = try await model.embed("Hello world")
// ```
//
// Note: This module adds ~50-100MB to your binary size due to the
// ONNX Runtime dependency. Only import it if you need ONNX support.

@_exported import EmbedKit

// Re-export public types
public typealias ONNXModelID = ModelID
public typealias ONNXEmbedding = Embedding
public typealias ONNXEmbeddingMetadata = EmbeddingMetadata

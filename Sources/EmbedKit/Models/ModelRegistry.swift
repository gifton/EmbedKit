//
//  ModelRegistry.swift
//  EmbedKit
//
//  Registry of available embedding models with metadata
//

import Foundation
#if os(iOS) && canImport(UIKit)
import UIKit
#endif

/// Metadata about an embedding model
public struct EmbeddingModelInfo: Codable, Sendable {
    public let identifier: String
    public let name: String
    public let dimensions: Int
    public let maxSequenceLength: Int
    public let modelType: ModelType
    public let size: ModelSize
    public let performance: PerformanceProfile
    public let downloadURL: URL?
    public let localPath: String?
    public let checksums: Checksums?
    public let description: String

    public enum ModelType: String, Codable, Sendable {
        case bert = "BERT"
        case distilbert = "DistilBERT"
        case sentenceTransformer = "SentenceTransformer"
        case mpnet = "MPNet"
        case miniLM = "MiniLM"
        case custom = "Custom"
    }

    public enum ModelSize: String, Codable, Sendable {
        case tiny = "tiny"      // < 50MB
        case small = "small"    // 50-100MB
        case base = "base"      // 100-250MB
        case large = "large"    // 250-500MB
        case xlarge = "xlarge"  // > 500MB
    }

    public struct PerformanceProfile: Codable, Sendable {
        public let speed: Speed
        public let accuracy: Accuracy
        public let memoryUsageMB: Int

        public enum Speed: String, Codable, Sendable {
            case veryFast = "very_fast"    // < 10ms
            case fast = "fast"              // 10-50ms
            case medium = "medium"          // 50-100ms
            case slow = "slow"              // > 100ms
        }

        public enum Accuracy: String, Codable, Sendable {
            case high = "high"
            case medium = "medium"
            case low = "low"
        }

        public init(speed: Speed, accuracy: Accuracy, memoryUsageMB: Int) {
            self.speed = speed
            self.accuracy = accuracy
            self.memoryUsageMB = memoryUsageMB
        }
    }

    public struct Checksums: Codable, Sendable {
        public let sha256: String?
        public let md5: String?

        public init(sha256: String? = nil, md5: String? = nil) {
            self.sha256 = sha256
            self.md5 = md5
        }
    }

    public init(
        identifier: String,
        name: String,
        dimensions: Int,
        maxSequenceLength: Int,
        modelType: ModelType,
        size: ModelSize,
        performance: PerformanceProfile,
        downloadURL: URL? = nil,
        localPath: String? = nil,
        checksums: Checksums? = nil,
        description: String
    ) {
        self.identifier = identifier
        self.name = name
        self.dimensions = dimensions
        self.maxSequenceLength = maxSequenceLength
        self.modelType = modelType
        self.size = size
        self.performance = performance
        self.downloadURL = downloadURL
        self.localPath = localPath
        self.checksums = checksums
        self.description = description
    }
}

/// Predefined models available for use
public enum PretrainedModel: String, CaseIterable, Sendable {
    // Sentence Transformers
    case miniLM_L6_v2 = "sentence-transformers/all-MiniLM-L6-v2"
    case miniLM_L12_v2 = "sentence-transformers/all-MiniLM-L12-v2"
    case mpnetBase_v2 = "sentence-transformers/all-mpnet-base-v2"
    case distilBERT_v1 = "sentence-transformers/all-distilroberta-v1"

    // OpenAI Compatible
    case textEmbedding3Small = "text-embedding-3-small"
    case textEmbedding3Large = "text-embedding-3-large"
    case textEmbeddingAda002 = "text-embedding-ada-002"

    // Custom/Local
    case customBERT = "custom/bert-base-uncased"

    /// Get model information
    public var info: EmbeddingModelInfo {
        switch self {
        case .miniLM_L6_v2:
            return EmbeddingModelInfo(
                identifier: self.rawValue,
                name: "all-MiniLM-L6-v2",
                dimensions: 384,
                maxSequenceLength: 512,
                modelType: .miniLM,
                size: .tiny,
                performance: .init(speed: .veryFast, accuracy: .medium, memoryUsageMB: 45),
                downloadURL: URL(string: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"),
                description: "Fastest model, good for mobile. Maps sentences to 384D vectors. Trained on 1B+ sentence pairs."
            )

        case .miniLM_L12_v2:
            return EmbeddingModelInfo(
                identifier: self.rawValue,
                name: "all-MiniLM-L12-v2",
                dimensions: 384,
                maxSequenceLength: 512,
                modelType: .miniLM,
                size: .small,
                performance: .init(speed: .fast, accuracy: .medium, memoryUsageMB: 85),
                downloadURL: URL(string: "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2"),
                description: "Balanced speed/quality. 384D vectors with better accuracy than L6."
            )

        case .mpnetBase_v2:
            return EmbeddingModelInfo(
                identifier: self.rawValue,
                name: "all-mpnet-base-v2",
                dimensions: 768,
                maxSequenceLength: 512,
                modelType: .mpnet,
                size: .base,
                performance: .init(speed: .medium, accuracy: .high, memoryUsageMB: 180),
                downloadURL: URL(string: "https://huggingface.co/sentence-transformers/all-mpnet-base-v2"),
                description: "Best quality/speed trade-off. 768D vectors with state-of-art performance."
            )

        case .distilBERT_v1:
            return EmbeddingModelInfo(
                identifier: self.rawValue,
                name: "all-distilroberta-v1",
                dimensions: 768,
                maxSequenceLength: 512,
                modelType: .distilbert,
                size: .base,
                performance: .init(speed: .fast, accuracy: .high, memoryUsageMB: 150),
                downloadURL: URL(string: "https://huggingface.co/sentence-transformers/all-distilroberta-v1"),
                description: "DistilRoBERTa-based model. Good for longer texts with 768D output."
            )

        case .textEmbedding3Small:
            return EmbeddingModelInfo(
                identifier: self.rawValue,
                name: "text-embedding-3-small",
                dimensions: 1536,
                maxSequenceLength: 8192,
                modelType: .custom,
                size: .large,
                performance: .init(speed: .slow, accuracy: .high, memoryUsageMB: 350),
                description: "OpenAI's small embedding model. 1536D vectors, requires API key."
            )

        case .textEmbedding3Large:
            return EmbeddingModelInfo(
                identifier: self.rawValue,
                name: "text-embedding-3-large",
                dimensions: 3072,
                maxSequenceLength: 8192,
                modelType: .custom,
                size: .xlarge,
                performance: .init(speed: .slow, accuracy: .high, memoryUsageMB: 700),
                description: "OpenAI's large embedding model. 3072D vectors, highest quality."
            )

        case .textEmbeddingAda002:
            return EmbeddingModelInfo(
                identifier: self.rawValue,
                name: "text-embedding-ada-002",
                dimensions: 1536,
                maxSequenceLength: 8192,
                modelType: .custom,
                size: .large,
                performance: .init(speed: .slow, accuracy: .high, memoryUsageMB: 350),
                description: "OpenAI's ada-002 model. 1536D vectors, legacy model."
            )

        case .customBERT:
            return EmbeddingModelInfo(
                identifier: self.rawValue,
                name: "bert-base-uncased",
                dimensions: 768,
                maxSequenceLength: 512,
                modelType: .bert,
                size: .base,
                performance: .init(speed: .medium, accuracy: .medium, memoryUsageMB: 220),
                localPath: "Models/bert-base-uncased.mlmodel",
                description: "Standard BERT base model. 768D vectors, general purpose."
            )
        }
    }

    /// Recommended model for different use cases
    public static func recommended(for useCase: UseCase) -> PretrainedModel {
        switch useCase {
        case .mobile:
            return .miniLM_L6_v2
        case .desktop:
            return .mpnetBase_v2
        case .server:
            return .mpnetBase_v2
        case .highAccuracy:
            return .textEmbedding3Large
        case .balanced:
            return .miniLM_L12_v2
        }
    }

    public enum UseCase {
        case mobile
        case desktop
        case server
        case highAccuracy
        case balanced
    }
}

/// Registry of available embedding models
public actor ModelRegistry {
    private var models: [String: EmbeddingModelInfo] = [:]
    private var customModels: [String: EmbeddingModelInfo] = [:]

    public init() {
        // Register all predefined models
        for model in PretrainedModel.allCases {
            models[model.rawValue] = model.info
        }
    }

    /// Register a custom model
    public func register(model: EmbeddingModelInfo) {
        customModels[model.identifier] = model
    }

    /// Get model info by identifier
    public func getModel(_ identifier: String) -> EmbeddingModelInfo? {
        return models[identifier] ?? customModels[identifier]
    }

    /// Get model info for a predefined model
    public func getModel(_ model: PretrainedModel) -> EmbeddingModelInfo {
        return model.info
    }

    /// List all available models
    public func listModels() -> [EmbeddingModelInfo] {
        let predefined = Array(models.values)
        let custom = Array(customModels.values)
        return predefined + custom
    }

    /// List models matching criteria
    public func listModels(
        dimensions: Int? = nil,
        maxSize: EmbeddingModelInfo.ModelSize? = nil,
        minSpeed: EmbeddingModelInfo.PerformanceProfile.Speed? = nil
    ) -> [EmbeddingModelInfo] {
        let allModels = listModels()

        return allModels.filter { model in
            // Filter by dimensions
            if let dims = dimensions, model.dimensions != dims {
                return false
            }

            // Filter by size
            if let maxSize = maxSize {
                let sizeOrder: [EmbeddingModelInfo.ModelSize] = [.tiny, .small, .base, .large, .xlarge]
                if let modelIdx = sizeOrder.firstIndex(of: model.size),
                   let maxIdx = sizeOrder.firstIndex(of: maxSize),
                   modelIdx > maxIdx {
                    return false
                }
            }

            // Filter by speed
            if let minSpeed = minSpeed {
                let speedOrder: [EmbeddingModelInfo.PerformanceProfile.Speed] = [.veryFast, .fast, .medium, .slow]
                if let modelIdx = speedOrder.firstIndex(of: model.performance.speed),
                   let minIdx = speedOrder.firstIndex(of: minSpeed),
                   modelIdx > minIdx {
                    return false
                }
            }

            return true
        }
    }

    /// Find best model for device capabilities
    public func recommendModel(
        for device: DeviceProfile,
        useCase: PretrainedModel.UseCase = .balanced
    ) -> EmbeddingModelInfo? {
        // Determine constraints based on device
        let maxSize: EmbeddingModelInfo.ModelSize
        let minSpeed: EmbeddingModelInfo.PerformanceProfile.Speed

        switch device.type {
        case .iPhone:
            maxSize = device.memoryGB >= 6 ? .base : .small
            minSpeed = .fast
        case .iPad:
            maxSize = device.memoryGB >= 8 ? .large : .base
            minSpeed = .medium
        case .mac:
            maxSize = .xlarge
            minSpeed = .slow
        case .watch:
            maxSize = .tiny
            minSpeed = .veryFast
        case .vision:
            maxSize = .base
            minSpeed = .fast
        }

        // Find matching models
        let candidates = listModels(maxSize: maxSize, minSpeed: minSpeed)

        // Return best match based on use case
        if useCase == .highAccuracy {
            return candidates.max { $0.dimensions < $1.dimensions }
        } else if useCase == .mobile {
            return candidates.min { $0.performance.memoryUsageMB < $1.performance.memoryUsageMB }
        } else {
            // Balanced - prefer medium dimensions
            return candidates.first { $0.dimensions == 768 } ?? candidates.first
        }
    }

    public struct DeviceProfile: Sendable {
        public let type: DeviceType
        public let memoryGB: Int
        public let hasNeuralEngine: Bool

        public enum DeviceType: Sendable {
            case iPhone
            case iPad
            case mac
            case watch
            case vision
        }

        public init(type: DeviceType, memoryGB: Int, hasNeuralEngine: Bool) {
            self.type = type
            self.memoryGB = memoryGB
            self.hasNeuralEngine = hasNeuralEngine
        }

        /// Get current device profile
        public static var current: DeviceProfile {
            let memoryGB = Int(ProcessInfo.processInfo.physicalMemory / 1_073_741_824)

            #if os(iOS)
                #if canImport(UIKit)
                let deviceType: DeviceType = UIDevice.current.userInterfaceIdiom == .pad ? .iPad : .iPhone
                #else
                let deviceType: DeviceType = .iPhone
                #endif
                return DeviceProfile(
                    type: deviceType,
                    memoryGB: memoryGB,
                    hasNeuralEngine: true
                )
            #elseif os(macOS)
            return DeviceProfile(
                type: .mac,
                memoryGB: memoryGB,
                hasNeuralEngine: true
            )
            #elseif os(watchOS)
            return DeviceProfile(
                type: .watch,
                memoryGB: 1,
                hasNeuralEngine: false
            )
            #elseif os(visionOS)
            return DeviceProfile(
                type: .vision,
                memoryGB: 8,
                hasNeuralEngine: true
            )
            #else
            return DeviceProfile(
                type: .iPhone,
                memoryGB: 4,
                hasNeuralEngine: true
            )
            #endif
        }
    }
}
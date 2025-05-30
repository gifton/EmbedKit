import Foundation
import PipelineKit
import Logging
import AsyncAlgorithms

/// Handler for text embedding commands
public actor EmbedTextHandler: CommandHandler {
    public typealias CommandType = EmbedTextCommand
    
    private let embedder: any TextEmbedder
    private let logger: Logger
    
    public init(embedder: any TextEmbedder) {
        self.embedder = embedder
        self.logger = Logger(label: "EmbedKit.EmbedTextHandler")
    }
    
    public func handle(_ command: EmbedTextCommand) async throws -> EmbeddingVector {
        logger.debug("Handling embed text command", metadata: [
            "text_length": "\(command.text.count)",
            "model": "\(command.modelIdentifier ?? "default")"
        ])
        
        // Ensure model is loaded
        if await !embedder.isReady {
            logger.info("Loading embedding model")
            try await embedder.loadModel()
        }
        
        // Generate embedding
        let startTime = Date()
        let embedding = try await embedder.embed(command.text)
        let duration = Date().timeIntervalSince(startTime)
        
        logger.debug("Generated embedding", metadata: [
            "dimensions": "\(embedding.dimensions)",
            "duration_ms": "\(Int(duration * 1000))"
        ])
        
        return embedding
    }
}

/// Handler for batch embedding commands
public actor EmbedBatchHandler: CommandHandler {
    public typealias CommandType = EmbedBatchCommand
    
    private let embedder: any TextEmbedder
    private let logger: Logger
    
    public init(embedder: any TextEmbedder) {
        self.embedder = embedder
        self.logger = Logger(label: "EmbedKit.EmbedBatchHandler")
    }
    
    public func handle(_ command: EmbedBatchCommand) async throws -> [EmbeddingVector] {
        logger.debug("Handling embed batch command", metadata: [
            "batch_size": "\(command.texts.count)",
            "model": "\(command.modelIdentifier ?? "default")"
        ])
        
        // Ensure model is loaded
        if await !embedder.isReady {
            logger.info("Loading embedding model")
            try await embedder.loadModel()
        }
        
        // Generate embeddings
        let startTime = Date()
        let embeddings = try await embedder.embed(batch: command.texts)
        let duration = Date().timeIntervalSince(startTime)
        
        logger.debug("Generated batch embeddings", metadata: [
            "batch_size": "\(embeddings.count)",
            "duration_ms": "\(Int(duration * 1000))",
            "embeddings_per_second": "\(Double(embeddings.count) / duration)"
        ])
        
        return embeddings
    }
}

/// Handler for streaming embedding commands
public actor EmbedStreamHandler<S: AsyncSequence & Sendable>: CommandHandler where S.Element == String {
    public typealias CommandType = EmbedStreamCommand<S>
    
    private let embedder: any TextEmbedder
    private let logger: Logger
    
    public init(embedder: any TextEmbedder) {
        self.embedder = embedder
        self.logger = Logger(label: "EmbedKit.EmbedStreamHandler")
    }
    
    public func handle(_ command: EmbedStreamCommand<S>) async throws -> AsyncStream<Result<EmbeddingVector, Error>> {
        logger.debug("Handling embed stream command", metadata: [
            "max_concurrency": "\(command.maxConcurrency)",
            "model": "\(command.modelIdentifier ?? "default")"
        ])
        
        // Ensure model is loaded
        if await !embedder.isReady {
            logger.info("Loading embedding model")
            try await embedder.loadModel()
        }
        
        return AsyncStream { continuation in
            Task {
                var processedCount = 0
                let startTime = Date()
                
                do {
                    // Process texts with limited concurrency
                    try await withThrowingTaskGroup(of: (Int, Result<EmbeddingVector, Error>).self) { group in
                        var index = 0
                        var activeCount = 0
                        
                        for try await text in command.texts {
                            let currentIndex = index
                            index += 1
                            
                            // Wait if we've reached max concurrency
                            while activeCount >= command.maxConcurrency {
                                if let (_, result) = try await group.next() {
                                    continuation.yield(result)
                                    activeCount -= 1
                                    processedCount += 1
                                    
                                    if processedCount % 100 == 0 {
                                        let duration = Date().timeIntervalSince(startTime)
                                        logger.debug("Stream progress", metadata: [
                                            "processed": "\(processedCount)",
                                            "rate": "\(Double(processedCount) / duration)"
                                        ])
                                    }
                                }
                            }
                            
                            // Add new task
                            group.addTask {
                                do {
                                    let embedding = try await self.embedder.embed(text)
                                    return (currentIndex, .success(embedding))
                                } catch {
                                    return (currentIndex, .failure(error))
                                }
                            }
                            activeCount += 1
                        }
                        
                        // Process remaining tasks
                        for try await (_, result) in group {
                            continuation.yield(result)
                            processedCount += 1
                        }
                    }
                } catch {
                    continuation.yield(.failure(error))
                }
                
                let duration = Date().timeIntervalSince(startTime)
                logger.info("Stream completed", metadata: [
                    "total_processed": "\(processedCount)",
                    "total_duration_s": "\(Int(duration))",
                    "average_rate": "\(Double(processedCount) / duration)"
                ])
                
                continuation.finish()
            }
        }
    }
}

/// Handler for model loading commands
public actor LoadEmbeddingModelHandler: CommandHandler {
    public typealias CommandType = LoadEmbeddingModelCommand
    
    private let modelManager: EmbeddingModelManager
    private let logger: Logger
    
    public init(modelManager: EmbeddingModelManager) {
        self.modelManager = modelManager
        self.logger = Logger(label: "EmbedKit.LoadModelHandler")
    }
    
    public func handle(_ command: LoadEmbeddingModelCommand) async throws -> ModelMetadata {
        logger.info("Loading embedding model", metadata: [
            "identifier": "\(command.identifier)",
            "url": "\(command.modelURL)"
        ])
        
        let startTime = Date()
        let metadata = try await modelManager.loadModel(
            from: command.modelURL,
            identifier: command.identifier,
            configuration: command.backendConfiguration
        )
        let duration = Date().timeIntervalSince(startTime)
        
        logger.info("Model loaded successfully", metadata: [
            "identifier": "\(command.identifier)",
            "dimensions": "\(metadata.embeddingDimensions)",
            "duration_s": "\(Int(duration))"
        ])
        
        return metadata
    }
}

/// Handler for model warmup commands
public actor WarmupEmbeddingModelHandler: CommandHandler {
    public typealias CommandType = WarmupEmbeddingModelCommand
    
    private let embedder: any TextEmbedder
    private let logger: Logger
    
    public init(embedder: any TextEmbedder) {
        self.embedder = embedder
        self.logger = Logger(label: "EmbedKit.WarmupHandler")
    }
    
    public func handle(_ command: WarmupEmbeddingModelCommand) async throws -> TimeInterval {
        logger.info("Warming up embedding model", metadata: [
            "iterations": "\(command.iterations)"
        ])
        
        // Ensure model is loaded
        if await !embedder.isReady {
            try await embedder.loadModel()
        }
        
        let testTexts = [
            "This is a warmup sentence.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models need initialization."
        ]
        
        var totalDuration: TimeInterval = 0
        
        for i in 0..<command.iterations {
            let startTime = Date()
            
            for text in testTexts {
                _ = try await embedder.embed(text)
            }
            
            let iterationDuration = Date().timeIntervalSince(startTime)
            totalDuration += iterationDuration
            
            logger.debug("Warmup iteration completed", metadata: [
                "iteration": "\(i + 1)",
                "duration_ms": "\(Int(iterationDuration * 1000))"
            ])
        }
        
        let averageDuration = totalDuration / Double(command.iterations)
        
        logger.info("Model warmup completed", metadata: [
            "average_duration_ms": "\(Int(averageDuration * 1000))",
            "total_duration_s": "\(totalDuration)"
        ])
        
        return averageDuration
    }
}


import Foundation
import AsyncAlgorithms
import OSLog

/// Advanced streaming embedder with backpressure control and resource management
public actor StreamingEmbedder<Embedder: TextEmbedder> {
    private let logger = Logger(subsystem: "EmbedKit", category: "StreamingEmbedder")
    
    private let embedder: Embedder
    private let configuration: StreamingConfiguration
    
    public struct StreamingConfiguration: Sendable {
        /// Maximum number of concurrent embedding operations
        public let maxConcurrency: Int
        
        /// Buffer size for input texts before applying backpressure
        public let inputBufferSize: Int
        
        /// Buffer size for output embeddings
        public let outputBufferSize: Int
        
        /// Timeout for individual embedding operations
        public let operationTimeout: TimeInterval
        
        /// Strategy for handling backpressure
        public let backpressureStrategy: BackpressureStrategy
        
        /// Whether to preserve order of results
        public let preserveOrder: Bool
        
        /// Batch size for processing multiple texts together
        public let batchSize: Int
        
        public init(
            maxConcurrency: Int = 10,
            inputBufferSize: Int = 1000,
            outputBufferSize: Int = 1000,
            operationTimeout: TimeInterval = 30.0,
            backpressureStrategy: BackpressureStrategy = .dropOldest,
            preserveOrder: Bool = true,
            batchSize: Int = 32
        ) {
            self.maxConcurrency = maxConcurrency
            self.inputBufferSize = inputBufferSize
            self.outputBufferSize = outputBufferSize
            self.operationTimeout = operationTimeout
            self.backpressureStrategy = backpressureStrategy
            self.preserveOrder = preserveOrder
            self.batchSize = batchSize
        }
    }
    
    public enum BackpressureStrategy: Sendable {
        case dropOldest
        case dropNewest
        case block
        case reject
    }
    
    public init(embedder: Embedder, configuration: StreamingConfiguration = StreamingConfiguration()) {
        self.embedder = embedder
        self.configuration = configuration
    }
    
    /// Stream embeddings for a sequence of texts with backpressure control
    public func embedTextStream<S: AsyncSequence & Sendable>(
        _ texts: S
    ) -> AsyncThrowingStream<StreamingResult, Error> where S.Element == String {
        AsyncThrowingStream { continuation in
            Task {
                await processStream(texts, continuation: continuation)
            }
        }
    }
    
    /// Stream embeddings with batching for improved performance
    public func embedBatchStream<S: AsyncSequence & Sendable>(
        _ texts: S
    ) -> AsyncThrowingStream<[StreamingResult], Error> where S.Element == String {
        AsyncThrowingStream { continuation in
            Task {
                await processBatchStream(texts, continuation: continuation)
            }
        }
    }
    
    /// Stream embeddings with custom processing pipeline
    public func embedWithPipeline<S: AsyncSequence & Sendable, T>(
        _ texts: S,
        transform: @escaping @Sendable (EmbeddingVector, String) async throws -> T
    ) -> AsyncThrowingStream<T, Error> where S.Element == String, T: Sendable {
        AsyncThrowingStream { continuation in
            Task {
                await processWithPipeline(texts, transform: transform, continuation: continuation)
            }
        }
    }
    
    private func processStream<S: AsyncSequence & Sendable>(
        _ texts: S,
        continuation: AsyncThrowingStream<StreamingResult, Error>.Continuation
    ) async where S.Element == String {
        defer { continuation.finish() }
        
        do {
            // Ensure embedder is ready
            if await !embedder.isReady {
                try await embedder.loadModel()
            }
            
            let results = embedTextStream(texts)
            for try await result in results {
                continuation.yield(result)
            }
        } catch {
            continuation.finish(throwing: error)
        }
    }
    
    private func processBatchStream<S: AsyncSequence & Sendable>(
        _ texts: S,
        continuation: AsyncThrowingStream<[StreamingResult], Error>.Continuation
    ) async where S.Element == String {
        defer { continuation.finish() }
        
        do {
            // Ensure embedder is ready
            if await !embedder.isReady {
                try await embedder.loadModel()
            }
            
            // Batch the input texts
            let batches = texts.chunks(ofCount: configuration.batchSize)
            
            try await withThrowingTaskGroup(of: [StreamingResult].self) { group in
                var activeTasks = 0
                var batchIndex = 0
                
                for try await batch in batches {
                    let currentBatchIndex = batchIndex
                    batchIndex += 1
                    
                    // Apply backpressure control
                    while activeTasks >= configuration.maxConcurrency {
                        if let results = try await group.next() {
                            continuation.yield(results)
                            activeTasks -= 1
                        }
                    }
                    
                    // Process batch
                    group.addTask { [embedder] in
                        let batchTexts = Array(batch)
                        let embeddings = try await embedder.embed(batch: batchTexts)
                        
                        return zip(batchTexts, embeddings).enumerated().map { index, pair in
                            StreamingResult(
                                text: pair.0,
                                embedding: pair.1,
                                index: currentBatchIndex * self.configuration.batchSize + index,
                                timestamp: Date()
                            )
                        }
                    }
                    activeTasks += 1
                }
                
                // Collect remaining results
                while let results = try await group.next() {
                    continuation.yield(results)
                }
            }
        } catch {
            continuation.finish(throwing: error)
        }
    }
    
    private func processWithPipeline<S: AsyncSequence & Sendable, T: Sendable>(
        _ texts: S,
        transform: @escaping @Sendable (EmbeddingVector, String) async throws -> T,
        continuation: AsyncThrowingStream<T, Error>.Continuation
    ) async where S.Element == String {
        defer { continuation.finish() }
        
        do {
            let results = embedTextStream(texts)
            
            try await withThrowingTaskGroup(of: (Int, T).self) { group in
                var activeTasks = 0
                var completedResults: [Int: T] = [:]
                var nextIndex = 0
                
                for try await result in results {
                    // Apply backpressure control
                    while activeTasks >= configuration.maxConcurrency {
                        if let (index, transformed) = try await group.next() {
                            if configuration.preserveOrder {
                                completedResults[index] = transformed
                                // Yield results in order
                                while let orderedResult = completedResults.removeValue(forKey: nextIndex) {
                                    continuation.yield(orderedResult)
                                    nextIndex += 1
                                }
                            } else {
                                continuation.yield(transformed)
                            }
                            activeTasks -= 1
                        }
                    }
                    
                    // Process transformation
                    let currentIndex = result.index
                    group.addTask {
                        let transformed = try await transform(result.embedding, result.text)
                        return (currentIndex, transformed)
                    }
                    activeTasks += 1
                }
                
                // Collect remaining results
                while let (index, transformed) = try await group.next() {
                    if configuration.preserveOrder {
                        completedResults[index] = transformed
                        while let orderedResult = completedResults.removeValue(forKey: nextIndex) {
                            continuation.yield(orderedResult)
                            nextIndex += 1
                        }
                    } else {
                        continuation.yield(transformed)
                    }
                }
            }
        } catch {
            continuation.finish(throwing: error)
        }
    }
    
    /// Core streaming implementation with advanced backpressure control
    private func embedStream<S: AsyncSequence & Sendable>(
        _ texts: S
    ) -> AsyncThrowingStream<StreamingResult, Error> where S.Element == String {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    var textBuffer: [(String, Int)] = []
                    var textIndex = 0
                    var activeTasks = 0
                    
                    try await withThrowingTaskGroup(of: StreamingResult.self) { group in
                        for try await text in texts {
                            // Check buffer size and apply backpressure
                            if textBuffer.count >= configuration.inputBufferSize {
                                switch configuration.backpressureStrategy {
                                case .dropOldest:
                                    textBuffer.removeFirst()
                                case .dropNewest:
                                    continue // Skip this text
                                case .block:
                                    // Process some buffered texts first
                                    if !textBuffer.isEmpty {
                                        let (bufferedText, index) = textBuffer.removeFirst()
                                        try await processText(bufferedText, index: index, group: &group, activeTasks: &activeTasks, continuation: continuation)
                                    }
                                case .reject:
                                    throw StreamingError.bufferOverflow
                                }
                            }
                            
                            textBuffer.append((text, textIndex))
                            textIndex += 1
                            
                            // Process texts when we have enough or hit concurrency limit
                            while !textBuffer.isEmpty && activeTasks < configuration.maxConcurrency {
                                let (textToProcess, index) = textBuffer.removeFirst()
                                try await processText(textToProcess, index: index, group: &group, activeTasks: &activeTasks, continuation: continuation)
                            }
                            
                            // Collect completed results
                            while activeTasks >= configuration.maxConcurrency {
                                if let result = try await group.next() {
                                    continuation.yield(result)
                                    activeTasks -= 1
                                }
                            }
                        }
                        
                        // Process remaining buffered texts
                        while !textBuffer.isEmpty {
                            let (textToProcess, index) = textBuffer.removeFirst()
                            try await processText(textToProcess, index: index, group: &group, activeTasks: &activeTasks, continuation: continuation)
                        }
                        
                        // Collect all remaining results
                        while let result = try await group.next() {
                            continuation.yield(result)
                            activeTasks -= 1
                        }
                    }
                } catch {
                    continuation.finish(throwing: error)
                    return
                }
                
                continuation.finish()
            }
        }
    }
    
    private func processText(
        _ text: String,
        index: Int,
        group: inout ThrowingTaskGroup<StreamingResult, Error>,
        activeTasks: inout Int,
        continuation: AsyncThrowingStream<StreamingResult, Error>.Continuation
    ) async throws {
        group.addTask { [embedder] in
            let startTime = Date()
            
            // Apply timeout
            let embedding = try await withThrowingTaskGroup(of: EmbeddingVector.self) { timeoutGroup in
                timeoutGroup.addTask {
                    try await embedder.embed(text)
                }
                
                timeoutGroup.addTask {
                    try await Task.sleep(nanoseconds: UInt64(self.configuration.operationTimeout * 1_000_000_000))
                    throw StreamingError.operationTimeout
                }
                
                return try await timeoutGroup.next()!
            }
            
            return StreamingResult(
                text: text,
                embedding: embedding,
                index: index,
                timestamp: startTime,
                duration: Date().timeIntervalSince(startTime)
            )
        }
        activeTasks += 1
    }
    
    /// Get streaming statistics
    public func getStatistics() async -> StreamingStatistics {
        StreamingStatistics(
            maxConcurrency: configuration.maxConcurrency,
            inputBufferSize: configuration.inputBufferSize,
            outputBufferSize: configuration.outputBufferSize,
            isEmbedderReady: await embedder.isReady,
            modelIdentifier: await embedder.modelIdentifier
        )
    }
}

/// Result of a streaming embedding operation
public struct StreamingResult: Sendable {
    public let text: String
    public let embedding: EmbeddingVector
    public let index: Int
    public let timestamp: Date
    public let duration: TimeInterval?
    
    public init(
        text: String,
        embedding: EmbeddingVector,
        index: Int,
        timestamp: Date,
        duration: TimeInterval? = nil
    ) {
        self.text = text
        self.embedding = embedding
        self.index = index
        self.timestamp = timestamp
        self.duration = duration
    }
}

/// Statistics about streaming operations
public struct StreamingStatistics: Sendable {
    public let maxConcurrency: Int
    public let inputBufferSize: Int
    public let outputBufferSize: Int
    public let isEmbedderReady: Bool
    public let modelIdentifier: String
    
    public init(
        maxConcurrency: Int,
        inputBufferSize: Int,
        outputBufferSize: Int,
        isEmbedderReady: Bool,
        modelIdentifier: String
    ) {
        self.maxConcurrency = maxConcurrency
        self.inputBufferSize = inputBufferSize
        self.outputBufferSize = outputBufferSize
        self.isEmbedderReady = isEmbedderReady
        self.modelIdentifier = modelIdentifier
    }
}

/// Errors specific to streaming operations
public enum StreamingError: LocalizedError {
    case bufferOverflow
    case operationTimeout
    case invalidConfiguration(String)
    case embedderNotReady
    
    public var errorDescription: String? {
        switch self {
        case .bufferOverflow:
            return "Input buffer overflow - consider adjusting buffer size or backpressure strategy"
        case .operationTimeout:
            return "Embedding operation timed out"
        case .invalidConfiguration(let details):
            return "Invalid streaming configuration: \(details)"
        case .embedderNotReady:
            return "Embedder is not ready for streaming operations"
        }
    }
}

/// Utility for creating mock text sources for testing
public struct MockTextSource: AsyncSequence, Sendable {
    public typealias Element = String
    
    private let texts: [String]
    private let delay: TimeInterval
    
    public init(texts: [String], delay: TimeInterval = 0.1) {
        self.texts = texts
        self.delay = delay
    }
    
    public func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(texts: texts, delay: delay)
    }
    
    public struct AsyncIterator: AsyncIteratorProtocol {
        private let texts: [String]
        private let delay: TimeInterval
        private var index = 0
        
        init(texts: [String], delay: TimeInterval) {
            self.texts = texts
            self.delay = delay
        }
        
        public mutating func next() async -> String? {
            guard index < texts.count else { return nil }
            
            if delay > 0 {
                try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            }
            
            let text = texts[index]
            index += 1
            return text
        }
    }
    
    /// Create a source with sample documents
    public static func sampleDocuments(count: Int = 100, delay: TimeInterval = 0.1) -> MockTextSource {
        let documents = (1...count).map { i in
            "Document \\(i): This is a sample document with some text content for embedding. " +
            "It contains multiple sentences to make it more realistic. " +
            "The content varies slightly to ensure different embeddings."
        }
        return MockTextSource(texts: documents, delay: delay)
    }
    
    /// Create a source that simulates real-time data
    public static func realTimeStream(textsPerSecond: Double = 10, duration: TimeInterval = 60) -> MockTextSource {
        let totalTexts = Int(textsPerSecond * duration)
        let delay = 1.0 / textsPerSecond
        
        let texts = (1...totalTexts).map { i in
            "Real-time message \\(i): timestamp \\(Date().timeIntervalSince1970)"
        }
        
        return MockTextSource(texts: texts, delay: delay)
    }
}
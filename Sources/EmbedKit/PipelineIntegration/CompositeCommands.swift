import Foundation
import PipelineKit

// MARK: - Composite Commands for EmbedKit + VectorStoreKit Integration

/// Command to index a document by embedding its text and storing in vector store
public struct IndexDocumentCommand: Command, ValidatableCommand {
    public typealias Result = IndexResult
    
    public let document: EmbeddingDocument
    public let modelIdentifier: ModelIdentifier?
    public let storeIdentifier: String?
    public let useCache: Bool
    
    public init(
        document: EmbeddingDocument,
        modelIdentifier: ModelIdentifier? = nil,
        storeIdentifier: String? = nil,
        useCache: Bool = true
    ) {
        self.document = document
        self.modelIdentifier = modelIdentifier
        self.storeIdentifier = storeIdentifier
        self.useCache = useCache
    }
    
    public func validate() throws {
        guard !document.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ValidationError.custom("Document text cannot be empty")
        }
        
        guard document.text.count <= 10_000 else {
            throw ValidationError.custom("Document text exceeds maximum length of 10,000 characters")
        }
    }
}

/// Command to index multiple documents in batch
public struct BatchIndexDocumentsCommand: Command, ValidatableCommand {
    public typealias Result = BatchIndexResult
    
    public let documents: [EmbeddingDocument]
    public let configuration: BatchIndexConfiguration
    public let modelIdentifier: ModelIdentifier?
    public let storeIdentifier: String?
    
    public init(
        documents: [EmbeddingDocument],
        configuration: BatchIndexConfiguration = BatchIndexConfiguration(),
        modelIdentifier: ModelIdentifier? = nil,
        storeIdentifier: String? = nil
    ) {
        self.documents = documents
        self.configuration = configuration
        self.modelIdentifier = modelIdentifier
        self.storeIdentifier = storeIdentifier
    }
    
    public func validate() throws {
        guard !documents.isEmpty else {
            throw ValidationError.custom("Documents array cannot be empty")
        }
        
        guard documents.count <= 10_000 else {
            throw ValidationError.custom("Batch size exceeds maximum of 10,000 documents")
        }
        
        for (index, document) in documents.enumerated() {
            guard !document.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw ValidationError.custom("Document at index \(index) has empty text")
            }
            
            guard document.text.count <= 10_000 else {
                throw ValidationError.custom("Document at index \(index) exceeds maximum text length")
            }
        }
    }
}

/// Command to search for similar documents
public struct SearchDocumentsCommand: Command, ValidatableCommand {
    public typealias Result = [SearchResult]
    
    public let query: String
    public let k: Int
    public let filter: MetadataFilter?
    public let modelIdentifier: ModelIdentifier?
    public let storeIdentifier: String?
    
    public init(
        query: String,
        k: Int = 10,
        filter: MetadataFilter? = nil,
        modelIdentifier: ModelIdentifier? = nil,
        storeIdentifier: String? = nil
    ) {
        self.query = query
        self.k = k
        self.filter = filter
        self.modelIdentifier = modelIdentifier
        self.storeIdentifier = storeIdentifier
    }
    
    public func validate() throws {
        guard !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ValidationError.custom("Query cannot be empty")
        }
        
        guard query.count <= 10_000 else {
            throw ValidationError.custom("Query exceeds maximum length of 10,000 characters")
        }
        
        guard k > 0 && k <= 1000 else {
            throw ValidationError.custom("k must be between 1 and 1000")
        }
    }
}

/// Command to find similar documents to a given document
public struct FindSimilarDocumentsCommand: Command, ValidatableCommand {
    public typealias Result = [SearchResult]
    
    public let documentId: String
    public let k: Int
    public let filter: MetadataFilter?
    public let storeIdentifier: String?
    
    public init(
        documentId: String,
        k: Int = 10,
        filter: MetadataFilter? = nil,
        storeIdentifier: String? = nil
    ) {
        self.documentId = documentId
        self.k = k
        self.filter = filter
        self.storeIdentifier = storeIdentifier
    }
    
    public func validate() throws {
        guard !documentId.isEmpty else {
            throw ValidationError.custom("Document ID cannot be empty")
        }
        
        guard k > 0 && k <= 1000 else {
            throw ValidationError.custom("k must be between 1 and 1000")
        }
    }
}

/// Command to update an existing document
public struct UpdateDocumentCommand: Command, ValidatableCommand {
    public typealias Result = IndexResult
    
    public let documentId: String
    public let newText: String?
    public let newMetadata: [String: String]?
    public let modelIdentifier: ModelIdentifier?
    public let storeIdentifier: String?
    
    public init(
        documentId: String,
        newText: String? = nil,
        newMetadata: [String: String]? = nil,
        modelIdentifier: ModelIdentifier? = nil,
        storeIdentifier: String? = nil
    ) {
        self.documentId = documentId
        self.newText = newText
        self.newMetadata = newMetadata
        self.modelIdentifier = modelIdentifier
        self.storeIdentifier = storeIdentifier
    }
    
    public func validate() throws {
        guard !documentId.isEmpty else {
            throw ValidationError.custom("Document ID cannot be empty")
        }
        
        if let text = newText {
            guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw ValidationError.custom("New text cannot be empty")
            }
            
            guard text.count <= 10_000 else {
                throw ValidationError.custom("New text exceeds maximum length of 10,000 characters")
            }
        }
        
        guard newText != nil || newMetadata != nil else {
            throw ValidationError.custom("Must provide either new text or new metadata")
        }
    }
}

/// Command to delete a document from the vector store
public struct DeleteDocumentCommand: Command, ValidatableCommand {
    public typealias Result = Bool
    
    public let documentId: String
    public let storeIdentifier: String?
    
    public init(
        documentId: String,
        storeIdentifier: String? = nil
    ) {
        self.documentId = documentId
        self.storeIdentifier = storeIdentifier
    }
    
    public func validate() throws {
        guard !documentId.isEmpty else {
            throw ValidationError.custom("Document ID cannot be empty")
        }
    }
}

/// Command to create a new vector index
public struct CreateIndexCommand: Command, ValidatableCommand {
    public typealias Result = IndexConfiguration
    
    public let configuration: IndexConfiguration
    public let modelIdentifier: ModelIdentifier?
    
    public init(
        configuration: IndexConfiguration,
        modelIdentifier: ModelIdentifier? = nil
    ) {
        self.configuration = configuration
        self.modelIdentifier = modelIdentifier
    }
    
    public func validate() throws {
        guard configuration.dimension > 0 else {
            throw ValidationError.custom("Index dimension must be positive")
        }
        
        guard configuration.capacity > 0 else {
            throw ValidationError.custom("Index capacity must be positive")
        }
        
        guard !configuration.name.isEmpty else {
            throw ValidationError.custom("Index name cannot be empty")
        }
    }
}

/// Command to stream index documents from an async source
public struct StreamIndexDocumentsCommand: Command {
    public typealias Result = AsyncThrowingStream<IndexResult, Error>
    
    public let documentSource: any AsyncDocumentSource
    public let configuration: BatchIndexConfiguration
    public let modelIdentifier: ModelIdentifier?
    public let storeIdentifier: String?
    
    public init(
        documentSource: any AsyncDocumentSource,
        configuration: BatchIndexConfiguration = BatchIndexConfiguration(),
        modelIdentifier: ModelIdentifier? = nil,
        storeIdentifier: String? = nil
    ) {
        self.documentSource = documentSource
        self.configuration = configuration
        self.modelIdentifier = modelIdentifier
        self.storeIdentifier = storeIdentifier
    }
}

// MARK: - Async Document Source

/// Protocol for async document sources
public protocol AsyncDocumentSource: AsyncSequence, Sendable where Element == EmbeddingDocument, Element: Sendable {}

/// Concrete implementation of AsyncDocumentSource
public struct ArrayDocumentSource: AsyncDocumentSource {
    private let documents: [EmbeddingDocument]
    
    public init(_ documents: [EmbeddingDocument]) {
        self.documents = documents
    }
    
    public func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(documents: documents)
    }
    
    public struct AsyncIterator: AsyncIteratorProtocol {
        private var index = 0
        private let documents: [EmbeddingDocument]
        
        init(documents: [EmbeddingDocument]) {
            self.documents = documents
        }
        
        public mutating func next() async -> EmbeddingDocument? {
            guard index < documents.count else { return nil }
            let document = documents[index]
            index += 1
            return document
        }
    }
}
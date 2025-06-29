import Foundation

// MARK: - Error Context

/// A structured context for errors that provides rich metadata about the operation that failed.
///
/// This replaces string-based error descriptions with a type-safe, structured approach
/// that enables better error tracking, debugging, and recovery strategies.
///
/// ## Example
///
/// ```swift
/// let context = ErrorContext(
///     operation: .embedding,
///     modelIdentifier: .miniLM_L6_v2,
///     metadata: ErrorMetadata()
///         .with(\.inputSize, 1024)
///         .with(\.batchSize, 32)
/// )
/// ```
public struct ErrorContext: Sendable {
    /// The operation that was being performed when the error occurred
    public let operation: Operation
    
    /// The model identifier involved in the operation (if applicable)
    public let modelIdentifier: ModelIdentifier?
    
    /// Timestamp when the error occurred
    public let timestamp: Date
    
    /// Unique identifier for correlating related errors
    public let correlationId: UUID
    
    /// Structured metadata about the error
    public let metadata: ErrorMetadata
    
    /// The source location where the error occurred
    public let sourceLocation: SourceLocation?
    
    // MARK: - Initialization
    
    public init(
        operation: Operation,
        modelIdentifier: ModelIdentifier? = nil,
        metadata: ErrorMetadata = ErrorMetadata(),
        sourceLocation: SourceLocation? = nil,
        timestamp: Date = Date(),
        correlationId: UUID = UUID()
    ) {
        self.operation = operation
        self.modelIdentifier = modelIdentifier
        self.metadata = metadata
        self.sourceLocation = sourceLocation
        self.timestamp = timestamp
        self.correlationId = correlationId
    }
    
    // MARK: - Operations
    
    /// Types of operations that can fail
    public enum Operation: String, Sendable, CaseIterable {
        case initialization = "initialization"
        case modelLoading = "model_loading"
        case embedding = "embedding"
        case batchEmbedding = "batch_embedding"
        case tokenization = "tokenization"
        case inference = "inference"
        case caching = "caching"
        case metalAcceleration = "metal_acceleration"
        case streaming = "streaming"
        case validation = "validation"
        case configuration = "configuration"
        case resourceManagement = "resource_management"
        
        /// Human-readable description of the operation
        public var description: String {
            switch self {
            case .initialization:
                return "System initialization"
            case .modelLoading:
                return "Model loading"
            case .embedding:
                return "Text embedding"
            case .batchEmbedding:
                return "Batch text embedding"
            case .tokenization:
                return "Text tokenization"
            case .inference:
                return "Model inference"
            case .caching:
                return "Cache operation"
            case .metalAcceleration:
                return "Metal GPU acceleration"
            case .streaming:
                return "Streaming operation"
            case .validation:
                return "Input validation"
            case .configuration:
                return "Configuration"
            case .resourceManagement:
                return "Resource management"
            }
        }
    }
}

// MARK: - Error Metadata

/// Type-safe metadata container for error context
@dynamicMemberLookup
public struct ErrorMetadata: Sendable {
    private var storage: [String: String] = [:]
    
    public init() {}
    
    // MARK: - Well-Known Keys
    
    /// The size of the input that caused the error
    public var inputSize: Int? {
        get { storage["inputSize"].flatMap(Int.init) }
        set { storage["inputSize"] = newValue.map(String.init) }
    }
    
    /// The batch size being processed
    public var batchSize: Int? {
        get { storage["batchSize"].flatMap(Int.init) }
        set { storage["batchSize"] = newValue.map(String.init) }
    }
    
    /// The sequence length that caused the error
    public var sequenceLength: Int? {
        get { storage["sequenceLength"].flatMap(Int.init) }
        set { storage["sequenceLength"] = newValue.map(String.init) }
    }
    
    /// Memory usage at the time of error (in bytes)
    public var memoryUsage: Int? {
        get { storage["memoryUsage"].flatMap(Int.init) }
        set { storage["memoryUsage"] = newValue.map(String.init) }
    }
    
    /// Processing duration before error (in seconds)
    public var duration: TimeInterval? {
        get { storage["duration"].flatMap(Double.init) }
        set { storage["duration"] = newValue.map { String($0) } }
    }
    
    /// Retry attempt number
    public var retryAttempt: Int? {
        get { storage["retryAttempt"].flatMap(Int.init) }
        set { storage["retryAttempt"] = newValue.map(String.init) }
    }
    
    /// Error code from underlying system
    public var systemErrorCode: Int? {
        get { storage["systemErrorCode"].flatMap(Int.init) }
        set { storage["systemErrorCode"] = newValue.map(String.init) }
    }
    
    /// Device information
    public var device: String? {
        get { storage["device"] }
        set { storage["device"] = newValue }
    }
    
    // MARK: - Dynamic Member Lookup
    
    public subscript(dynamicMember key: String) -> String? {
        get { storage[key] }
        set { storage[key] = newValue }
    }
    
    // MARK: - Builder Pattern
    
    /// Sets a value using a key path and returns self for chaining
    public func with<T>(_ keyPath: WritableKeyPath<ErrorMetadata, T?>, _ value: T) -> ErrorMetadata {
        var copy = self
        copy[keyPath: keyPath] = value
        return copy
    }
    
    /// Sets a custom key-value pair
    public func with(key: String, value: String) -> ErrorMetadata {
        var copy = self
        copy.storage[key] = value
        return copy
    }
    
    // MARK: - Conversion
    
    /// Converts to a dictionary for compatibility with existing error handling
    public var dictionary: [String: String] {
        storage
    }
}

// MARK: - Source Location

/// Information about where in the code an error occurred
public struct SourceLocation: Sendable {
    public let file: String
    public let function: String
    public let line: Int
    
    public init(
        file: String = #file,
        function: String = #function,
        line: Int = #line
    ) {
        self.file = file
        self.function = function
        self.line = line
    }
    
    /// A formatted string representation of the source location
    public var description: String {
        let fileName = URL(fileURLWithPath: file).lastPathComponent
        return "\(fileName):\(line) in \(function)"
    }
}

// MARK: - Enhanced Error Protocol

/// Protocol for errors that carry structured context
public protocol ContextualError: Error {
    /// The error context
    var context: ErrorContext { get }
    
    /// The underlying error (if any)
    var underlyingError: Error? { get }
}

// MARK: - Enhanced Embedding Error

/// Enhanced version of EmbeddingError with structured context
public enum ContextualEmbeddingError: ContextualError, LocalizedError {
    case modelNotLoaded(context: ErrorContext, underlyingError: Error? = nil)
    case tokenizationFailed(context: ErrorContext, underlyingError: Error? = nil)
    case inferenceFailed(context: ErrorContext, underlyingError: Error? = nil)
    case invalidInput(context: ErrorContext, reason: InvalidInputReason, underlyingError: Error? = nil)
    case dimensionMismatch(context: ErrorContext, expected: Int, actual: Int, underlyingError: Error? = nil)
    case resourceUnavailable(context: ErrorContext, resource: ResourceType, underlyingError: Error? = nil)
    case configurationError(context: ErrorContext, issue: ConfigurationIssue, underlyingError: Error? = nil)
    case networkError(context: ErrorContext, statusCode: Int, underlyingError: Error? = nil)
    case validationFailed(context: ErrorContext, reason: String, underlyingError: Error? = nil)
    
    // MARK: - Context
    
    public var context: ErrorContext {
        switch self {
        case .modelNotLoaded(let context, _),
             .tokenizationFailed(let context, _),
             .inferenceFailed(let context, _),
             .invalidInput(let context, _, _),
             .dimensionMismatch(let context, _, _, _),
             .resourceUnavailable(let context, _, _),
             .configurationError(let context, _, _),
             .networkError(let context, _, _),
             .validationFailed(let context, _, _):
            return context
        }
    }
    
    public var underlyingError: Error? {
        switch self {
        case .modelNotLoaded(_, let error),
             .tokenizationFailed(_, let error),
             .inferenceFailed(_, let error),
             .invalidInput(_, _, let error),
             .dimensionMismatch(_, _, _, let error),
             .resourceUnavailable(_, _, let error),
             .configurationError(_, _, let error),
             .networkError(_, _, let error),
             .validationFailed(_, _, let error):
            return error
        }
    }
    
    // MARK: - Reasons
    
    public enum InvalidInputReason: String, Sendable {
        case empty = "empty_input"
        case tooLong = "input_too_long"
        case invalidCharacters = "invalid_characters"
        case unsupportedLanguage = "unsupported_language"
        case malformed = "malformed_input"
    }
    
    public enum ResourceType: String, Sendable {
        case model = "model"
        case memory = "memory"
        case gpu = "gpu"
        case cache = "cache"
        case network = "network"
    }
    
    public enum ConfigurationIssue: String, Sendable {
        case invalid = "invalid_configuration"
        case missing = "missing_configuration"
        case incompatible = "incompatible_configuration"
        case outOfRange = "configuration_out_of_range"
    }
    
    // MARK: - LocalizedError
    
    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded(let context, _):
            return "Model not loaded for \(context.operation.description)"
        case .tokenizationFailed(let context, _):
            return "Tokenization failed during \(context.operation.description)"
        case .inferenceFailed(let context, _):
            return "Inference failed during \(context.operation.description)"
        case .invalidInput(let context, let reason, _):
            return "Invalid input (\(reason.rawValue)) during \(context.operation.description)"
        case .dimensionMismatch(let context, let expected, let actual, _):
            return "Dimension mismatch (expected \(expected), got \(actual)) during \(context.operation.description)"
        case .resourceUnavailable(let context, let resource, _):
            return "\(resource.rawValue) unavailable during \(context.operation.description)"
        case .configurationError(let context, let issue, _):
            return "Configuration error (\(issue.rawValue)) during \(context.operation.description)"
        case .networkError(let context, let statusCode, _):
            return "Network error (status code: \(statusCode)) during \(context.operation.description)"
        case .validationFailed(let context, let reason, _):
            return "Validation failed (\(reason)) during \(context.operation.description)"
        }
    }
    
    public var failureReason: String? {
        switch self {
        case .invalidInput(_, let reason, _):
            switch reason {
            case .empty: return "The input text was empty"
            case .tooLong: return "The input text exceeded maximum length"
            case .invalidCharacters: return "The input contained invalid characters"
            case .unsupportedLanguage: return "The input language is not supported"
            case .malformed: return "The input was malformed"
            }
        case .dimensionMismatch(_, let expected, let actual, _):
            return "Expected embedding dimensions of \(expected) but got \(actual)"
        case .resourceUnavailable(_, let resource, _):
            switch resource {
            case .model: return "The requested model is not available"
            case .memory: return "Insufficient memory available"
            case .gpu: return "GPU acceleration is not available"
            case .cache: return "Cache system is not available"
            case .network: return "Network resource is not available"
            }
        case .configurationError(_, let issue, _):
            switch issue {
            case .invalid: return "The configuration contains invalid values"
            case .missing: return "Required configuration is missing"
            case .incompatible: return "The configuration is incompatible"
            case .outOfRange: return "Configuration values are out of valid range"
            }
        case .networkError(_, let statusCode, _):
            if statusCode == 404 {
                return "The requested resource was not found"
            } else if statusCode >= 500 {
                return "Server error occurred"
            } else if statusCode >= 400 {
                return "Client request error"
            } else {
                return "Network request failed"
            }
        case .validationFailed(_, let reason, _):
            return reason
        default:
            return nil
        }
    }
}

// MARK: - Error Context Builder

/// Convenience functions for building error contexts
public extension ErrorContext {
    /// Creates a context for model loading operations
    static func modelLoading(
        _ modelIdentifier: ModelIdentifier,
        metadata: ErrorMetadata = ErrorMetadata(),
        location: SourceLocation = SourceLocation()
    ) -> ErrorContext {
        ErrorContext(
            operation: .modelLoading,
            modelIdentifier: modelIdentifier,
            metadata: metadata,
            sourceLocation: location
        )
    }
    
    /// Creates a context for embedding operations
    static func embedding(
        modelIdentifier: ModelIdentifier? = nil,
        inputSize: Int? = nil,
        location: SourceLocation = SourceLocation()
    ) -> ErrorContext {
        var metadata = ErrorMetadata()
        if let inputSize = inputSize {
            metadata = metadata.with(\.inputSize, inputSize)
        }
        
        return ErrorContext(
            operation: .embedding,
            modelIdentifier: modelIdentifier,
            metadata: metadata,
            sourceLocation: location
        )
    }
    
    /// Creates a context for batch operations
    static func batchEmbedding(
        modelIdentifier: ModelIdentifier? = nil,
        batchSize: Int,
        location: SourceLocation = SourceLocation()
    ) -> ErrorContext {
        ErrorContext(
            operation: .batchEmbedding,
            modelIdentifier: modelIdentifier,
            metadata: ErrorMetadata().with(\.batchSize, batchSize),
            sourceLocation: location
        )
    }
}


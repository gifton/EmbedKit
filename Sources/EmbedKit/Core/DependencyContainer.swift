import Foundation
import CoreML

/// Dependency injection container for EmbedKit
/// 
/// Provides centralized dependency management with proper Swift 6 concurrency
public actor DependencyContainer {
    /// Singleton instance
    public static let shared = DependencyContainer()
    
    /// Registered dependencies
    private var dependencies: [ObjectIdentifier: any Sendable] = [:]
    
    /// Factories for lazy creation
    private var factories: [ObjectIdentifier: (@Sendable () async throws -> any Sendable)] = [:]
    
    /// Singleton instances
    private var singletons: [ObjectIdentifier: any Sendable] = [:]
    
    private init() {}
    
    // MARK: - Registration
    
    /// Register a dependency instance
    public func register<T: Sendable>(_ type: T.Type, instance: T) {
        let key = ObjectIdentifier(type)
        dependencies[key] = instance
    }
    
    /// Register a factory for lazy creation
    public func register<T: Sendable>(
        _ type: T.Type,
        factory: @escaping @Sendable () async throws -> T
    ) {
        let key = ObjectIdentifier(type)
        factories[key] = factory
    }
    
    /// Register a singleton factory
    public func registerSingleton<T: Sendable>(
        _ type: T.Type,
        factory: @escaping @Sendable () async throws -> T
    ) {
        let key = ObjectIdentifier(type)
        factories[key] = { [weak self] in
            guard let self = self else {
                throw DependencyError.containerDeallocated
            }
            
            return try await self.getOrCreateSingleton(key: key, factory: factory)
        }
    }
    
    /// Helper to get or create singleton (actor-isolated)
    private func getOrCreateSingleton<T: Sendable>(
        key: ObjectIdentifier,
        factory: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        // Check if singleton already exists
        if let existing = singletons[key] as? T {
            return existing
        }
        
        // Create and store singleton
        let instance = try await factory()
        singletons[key] = instance
        return instance
    }
    
    /// Register protocol implementation
    public func register<P, T: Sendable>(
        _ protocolType: P.Type,
        implementation: T.Type,
        factory: @escaping @Sendable () async throws -> T
    ) {
        let key = ObjectIdentifier(protocolType)
        factories[key] = factory
    }
    
    // MARK: - Resolution
    
    /// Resolve a dependency
    public func resolve<T: Sendable>(_ type: T.Type) async throws -> T {
        let key = ObjectIdentifier(type)
        
        // Check registered instances
        if let instance = dependencies[key] as? T {
            return instance
        }
        
        // Check singletons
        if let singleton = singletons[key] as? T {
            return singleton
        }
        
        // Check factories
        if let factory = factories[key] {
            let instance = try await factory()
            guard let typedInstance = instance as? T else {
                throw DependencyError.typeMismatch(
                    expected: String(describing: T.self),
                    actual: String(describing: Swift.type(of: instance))
                )
            }
            return typedInstance
        }
        
        throw DependencyError.notRegistered(String(describing: T.self))
    }
    
    /// Resolve optional dependency
    public func resolveOptional<T: Sendable>(_ type: T.Type) async -> T? {
        try? await resolve(type)
    }
    
    /// Check if dependency is registered
    public func isRegistered<T>(_ type: T.Type) -> Bool {
        let key = ObjectIdentifier(type)
        return dependencies[key] != nil || factories[key] != nil
    }
    
    // MARK: - Management
    
    /// Clear all dependencies
    public func clear() {
        dependencies.removeAll()
        factories.removeAll()
        singletons.removeAll()
    }
    
    /// Clear specific dependency
    public func clear<T>(_ type: T.Type) {
        let key = ObjectIdentifier(type)
        dependencies.removeValue(forKey: key)
        factories.removeValue(forKey: key)
        singletons.removeValue(forKey: key)
    }
    
    /// Get registration statistics
    public func getStatistics() -> ContainerStatistics {
        ContainerStatistics(
            registeredInstances: dependencies.count,
            registeredFactories: factories.count,
            activeSingletons: singletons.count
        )
    }
}

// MARK: - Dependency Builder

/// Builder for configuring dependencies
public final class DependencyBuilder: Sendable {
    private let registrations: [@Sendable (DependencyContainer) async -> Void]
    
    private init(registrations: [@Sendable (DependencyContainer) async -> Void]) {
        self.registrations = registrations
    }
    
    /// Create a new builder
    public static func builder() -> DependencyBuilder {
        DependencyBuilder(registrations: [])
    }
    
    /// Register a dependency
    public func register<T: Sendable>(
        _ type: T.Type,
        instance: T
    ) -> DependencyBuilder {
        let newRegistrations = registrations + [{ container in
            await container.register(type, instance: instance)
        }]
        return DependencyBuilder(registrations: newRegistrations)
    }
    
    /// Register a factory
    public func register<T: Sendable>(
        _ type: T.Type,
        factory: @escaping @Sendable () async throws -> T
    ) -> DependencyBuilder {
        let newRegistrations = registrations + [{ container in
            await container.register(type, factory: factory)
        }]
        return DependencyBuilder(registrations: newRegistrations)
    }
    
    /// Register a singleton
    public func singleton<T: Sendable>(
        _ type: T.Type,
        factory: @escaping @Sendable () async throws -> T
    ) -> DependencyBuilder {
        let newRegistrations = registrations + [{ container in
            await container.registerSingleton(type, factory: factory)
        }]
        return DependencyBuilder(registrations: newRegistrations)
    }
    
    /// Build and configure the container
    public func build() async -> DependencyContainer {
        let container = DependencyContainer.shared
        for registration in registrations {
            await registration(container)
        }
        return container
    }
}

// MARK: - Default Registrations

extension DependencyContainer {
    /// Register default EmbedKit dependencies
    public func registerDefaults(configuration: EmbedKitConfig) async {
        // Register configuration
        await register(EmbedKitConfig.self, instance: configuration)
        
        // Register telemetry
        await registerSingleton(TelemetryProtocol.self) {
            TelemetryManager.shared.system
        }
        
        // Register model registry
        await registerSingleton(HuggingFaceModelRegistry.self) {
            HuggingFaceModelRegistry()
        }
        
        // Register persistent storage
        await registerSingleton(PersistentModelRegistry.self) {
            try await PersistentModelRegistry(
                storageDirectory: configuration.storageDirectory
            )
        }
        
        // Model loader registration - TODO: implement ModelLoaderProtocol adapter
        
        // Register tokenizer factory
        await register(TokenizerFactoryProtocol.self) {
            DefaultTokenizerFactory()
        }
        
        // Register pooling strategy factory
        await register(PoolingStrategyFactoryProtocol.self) {
            DefaultPoolingStrategyFactory()
        }
        
        // Register Metal accelerator if enabled
        if configuration.useMetalAcceleration {
            await registerSingleton(MetalAcceleratorProtocol.self) {
                guard let device = MTLCreateSystemDefaultDevice() else {
                    throw MetalError.deviceNotAvailable
                }
                return try MetalAccelerator(device: device)
            }
        }
        
        // Register cache if enabled
        if configuration.cacheEnabled {
            await registerSingleton(EmbeddingCache.self) {
                EmbeddingCache(maxSize: configuration.maxCacheSize)
            }
        }
    }
}

// MARK: - Property Wrappers

/// Helper for async dependency resolution
public struct AsyncDependency<T: Sendable>: Sendable {
    private let type: T.Type
    
    public init(_ type: T.Type) {
        self.type = type
    }
    
    public func resolve() async throws -> T {
        try await DependencyContainer.shared.resolve(type)
    }
    
    public func resolveOptional() async -> T? {
        await DependencyContainer.shared.resolveOptional(type)
    }
}

// MARK: - Supporting Types

/// Container statistics
public struct ContainerStatistics: Sendable {
    public let registeredInstances: Int
    public let registeredFactories: Int
    public let activeSingletons: Int
}

/// Dependency errors
public enum DependencyError: Error, Sendable {
    case notRegistered(String)
    case typeMismatch(expected: String, actual: String)
    case containerDeallocated
    case circularDependency([String])
}

extension DependencyError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .notRegistered(let type):
            return "Dependency not registered: \(type)"
        case .typeMismatch(let expected, let actual):
            return "Type mismatch - expected: \(expected), actual: \(actual)"
        case .containerDeallocated:
            return "Dependency container was deallocated"
        case .circularDependency(let types):
            return "Circular dependency detected: \(types.joined(separator: " -> "))"
        }
    }
}

// MARK: - Default Implementations

/// Adapter to bridge between Tokenizer and TokenizerProtocol
struct TokenizerAdapter: TokenizerProtocol {
    private let tokenizer: any Tokenizer
    
    init(_ tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }
    
    var maxSequenceLength: Int {
        tokenizer.maxSequenceLength
    }
    
    var vocabularySize: Int {
        tokenizer.vocabularySize
    }
    
    func tokenize(_ text: String) async throws -> [Int] {
        let tokenized = try await tokenizer.tokenize(text)
        return tokenized.tokenIds
    }
    
    func tokenizeBatch(_ texts: [String]) async throws -> [[Int]] {
        let tokenized = try await tokenizer.tokenize(batch: texts)
        return tokenized.map { $0.tokenIds }
    }
    
    func detokenize(_ tokens: [Int]) async throws -> String {
        // This is a simplified implementation
        // Real implementation would need inverse vocabulary lookup
        return ""
    }
}

/// Default tokenizer factory
struct DefaultTokenizerFactory: TokenizerFactoryProtocol {
    func createTokenizer(for modelIdentifier: ModelIdentifier) async throws -> any TokenizerProtocol {
        switch modelIdentifier {
        case .miniLM_L6_v2:
            // Create a wrapper that adapts BERTTokenizer to TokenizerProtocol
            let bertTokenizer = try await BERTTokenizer(
                vocabularyPath: Bundle.main.path(forResource: "vocab", ofType: "txt"),
                maxSequenceLength: 512
            )
            return TokenizerAdapter(bertTokenizer)
        default:
            throw TokenizerError.unsupportedModel(modelIdentifier)
        }
    }
}

/// Default pooling strategy factory
struct DefaultPoolingStrategyFactory: PoolingStrategyFactoryProtocol {
    func createPoolingStrategy(_ type: PoolingStrategy) -> any PoolingStrategyProtocol {
        switch type {
        case .mean:
            return MeanPoolingStrategy()
        case .max:
            return MaxPoolingStrategy()
        case .cls:
            return CLSPoolingStrategy()
        case .attentionWeighted:
            return AttentionWeightedPoolingStrategy()
        }
    }
}

/// Embedding cache type alias
typealias EmbeddingCache = OptimizedLRUCache<String, Embedding>

/// Tokenizer error
enum TokenizerError: Error, Sendable {
    case unsupportedModel(ModelIdentifier)
}

// MARK: - Concrete Pooling Implementations

struct MeanPoolingStrategy: PoolingStrategyProtocol {
    let type = PoolingStrategy.mean
    
    func pool(tokenEmbeddings: [[Float]], attentionMask: [Float]?) async throws -> [Float] {
        // Implementation would go here
        []
    }
}

struct MaxPoolingStrategy: PoolingStrategyProtocol {
    let type = PoolingStrategy.max
    
    func pool(tokenEmbeddings: [[Float]], attentionMask: [Float]?) async throws -> [Float] {
        // Implementation would go here
        []
    }
}

struct CLSPoolingStrategy: PoolingStrategyProtocol {
    let type = PoolingStrategy.cls
    
    func pool(tokenEmbeddings: [[Float]], attentionMask: [Float]?) async throws -> [Float] {
        // Implementation would go here
        []
    }
}

struct AttentionWeightedPoolingStrategy: PoolingStrategyProtocol {
    let type = PoolingStrategy.attentionWeighted
    
    func pool(tokenEmbeddings: [[Float]], attentionMask: [Float]?) async throws -> [Float] {
        // Implementation would go here
        []
    }
}


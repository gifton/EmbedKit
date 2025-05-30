import Foundation
import OSLog

/// Model version information for tracking and management
public struct ModelVersion: Sendable, Codable, Hashable {
    public let identifier: String
    public let version: String
    public let buildNumber: Int
    public let createdAt: Date
    public let metadata: [String: String]
    
    public init(
        identifier: String,
        version: String,
        buildNumber: Int = 1,
        createdAt: Date = Date(),
        metadata: [String: String] = [:]
    ) {
        self.identifier = identifier
        self.version = version
        self.buildNumber = buildNumber
        self.createdAt = createdAt
        self.metadata = metadata
    }
    
    /// Create a semantic version string
    public var semanticVersion: String {
        "\(version).\(buildNumber)"
    }
    
    /// Check if this version is newer than another
    public func isNewer(than other: ModelVersion) -> Bool {
        if version != other.version {
            return version.compare(other.version, options: .numeric) == .orderedDescending
        }
        return buildNumber > other.buildNumber
    }
}

/// Registry for managing multiple model versions
public actor ModelVersionRegistry {
    private let logger = Logger(subsystem: "EmbedKit", category: "ModelVersionRegistry")
    
    private var versions: [String: [ModelVersion]] = [:]
    private var activeVersions: [String: ModelVersion] = [:]
    private var modelFiles: [ModelVersion: URL] = [:]
    
    public init() {}
    
    /// Register a new model version
    public func register(
        version: ModelVersion,
        modelURL: URL
    ) async throws {
        // Verify the model file exists
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw ModelVersionError.modelFileNotFound(modelURL.path)
        }
        
        // Add to versions list
        versions[version.identifier, default: []].append(version)
        versions[version.identifier]?.sort { $0.isNewer(than: $1) }
        
        // Store model file path
        modelFiles[version] = modelURL
        
        // If this is the first version or newer than current active, make it active
        if activeVersions[version.identifier] == nil ||
           version.isNewer(than: activeVersions[version.identifier]!) {
            activeVersions[version.identifier] = version
        }
        
        logger.info("Registered model version \\(version.semanticVersion) for \\(version.identifier)")
    }
    
    /// Get the active version for a model
    public func getActiveVersion(for modelId: String) async -> ModelVersion? {
        activeVersions[modelId]
    }
    
    /// Set the active version for a model
    public func setActiveVersion(_ version: ModelVersion) async throws {
        guard versions[version.identifier]?.contains(version) == true else {
            throw ModelVersionError.versionNotFound(version.semanticVersion)
        }
        
        activeVersions[version.identifier] = version
        logger.info("Set active version to \\(version.semanticVersion) for \\(version.identifier)")
    }
    
    /// Get all versions for a model
    public func getVersions(for modelId: String) async -> [ModelVersion] {
        versions[modelId] ?? []
    }
    
    /// Get the model file URL for a version
    public func getModelURL(for version: ModelVersion) async -> URL? {
        modelFiles[version]
    }
    
    /// Remove a model version
    public func removeVersion(_ version: ModelVersion) async throws {
        versions[version.identifier]?.removeAll { $0 == version }
        modelFiles.removeValue(forKey: version)
        
        // If this was the active version, choose the latest remaining version
        if activeVersions[version.identifier] == version {
            activeVersions[version.identifier] = versions[version.identifier]?.first
        }
        
        logger.info("Removed model version \\(version.semanticVersion) for \\(version.identifier)")
    }
    
    /// Get registry statistics
    public func getStatistics() async -> RegistryStatistics {
        let totalModels = Set(versions.keys).count
        let totalVersions = versions.values.flatMap { $0 }.count
        let activeCount = activeVersions.count
        
        return RegistryStatistics(
            totalModels: totalModels,
            totalVersions: totalVersions,
            activeVersions: activeCount
        )
    }
}

public struct RegistryStatistics: Sendable {
    public let totalModels: Int
    public let totalVersions: Int
    public let activeVersions: Int
}

/// Hot-swappable model manager that can switch models at runtime
public actor HotSwappableModelManager: EmbeddingModelManager {
    private let logger = Logger(subsystem: "EmbedKit", category: "HotSwappableModelManager")
    
    private let registry: ModelVersionRegistry
    private var loadedModels: [String: any TextEmbedder] = [:]
    private var modelConfigurations: [String: EmbeddingConfiguration] = [:]
    private let maxConcurrentModels: Int
    
    public init(
        registry: ModelVersionRegistry,
        maxConcurrentModels: Int = 3
    ) {
        self.registry = registry
        self.maxConcurrentModels = maxConcurrentModels
    }
    
    public func loadModel(
        from url: URL,
        identifier: String,
        configuration: ModelBackendConfiguration? = nil
    ) async throws -> ModelMetadata {
        // Create version if not exists
        let version = ModelVersion(identifier: identifier, version: "1.0")
        try await registry.register(version: version, modelURL: url)
        
        return try await loadVersion(version, configuration: configuration)
    }
    
    /// Load a specific model version
    public func loadVersion(
        _ version: ModelVersion,
        configuration: ModelBackendConfiguration? = nil
    ) async throws -> ModelMetadata {
        logger.info("Loading model version \\(version.semanticVersion)")
        
        guard let modelURL = await registry.getModelURL(for: version) else {
            throw ModelVersionError.modelFileNotFound("Version \\(version.semanticVersion)")
        }
        
        // Check if we need to unload old models due to memory constraints
        if loadedModels.count >= maxConcurrentModels {
            try await evictOldestModel()
        }
        
        // Create and load the embedder
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: version.identifier,
            configuration: EmbeddingConfiguration()
        )
        
        try await embedder.loadModel()
        
        // Store the loaded model
        loadedModels[version.identifier] = embedder
        
        // Set as active version
        try await registry.setActiveVersion(version)
        
        return ModelMetadata(
            name: version.identifier,
            version: version.semanticVersion,
            embeddingDimensions: await embedder.dimensions,
            maxSequenceLength: await embedder.configuration.maxSequenceLength,
            vocabularySize: 30522,
            modelType: "coreml",
            additionalInfo: version.metadata
        )
    }
    
    /// Hot-swap to a different model version
    public func swapToVersion(_ version: ModelVersion) async throws {
        logger.info("Hot-swapping to version \\(version.semanticVersion)")
        
        // Load the new version if not already loaded
        if loadedModels[version.identifier] == nil {
            _ = try await loadVersion(version)
        } else {
            // Just set as active
            try await registry.setActiveVersion(version)
        }
        
        logger.info("Successfully swapped to version \\(version.semanticVersion)")
    }
    
    public func unloadModel(identifier: String) async throws {
        if let embedder = loadedModels[identifier] {
            try await embedder.unloadModel()
            loadedModels.removeValue(forKey: identifier)
            logger.info("Unloaded model \\(identifier)")
        }
    }
    
    public func getModel(identifier: String?) async -> (any TextEmbedder)? {
        if let id = identifier {
            return loadedModels[id]
        } else {
            // Return any active model or the first available
            for (_, embedder) in loadedModels {
                return embedder
            }
            return nil
        }
    }
    
    /// Get information about all loaded models
    public func getLoadedModels() async -> [(version: ModelVersion, isActive: Bool)] {
        var result: [(version: ModelVersion, isActive: Bool)] = []
        
        for modelId in loadedModels.keys {
            if let activeVersion = await registry.getActiveVersion(for: modelId) {
                result.append((version: activeVersion, isActive: true))
            }
        }
        
        return result
    }
    
    /// Preload multiple model versions for fast switching
    public func preloadVersions(_ versions: [ModelVersion]) async throws {
        for version in versions {
            if loadedModels[version.identifier] == nil {
                do {
                    _ = try await loadVersion(version)
                } catch {
                    logger.error("Failed to preload version \\(version.semanticVersion): \\(error)")
                }
            }
        }
    }
    
    private func evictOldestModel() async throws {
        // Simple LRU: remove the first model (oldest)
        if let (identifier, embedder) = loadedModels.first {
            try await embedder.unloadModel()
            loadedModels.removeValue(forKey: identifier)
            logger.info("Evicted model \\(identifier) due to memory constraints")
        }
    }
}

/// A/B testing manager for comparing model versions
public actor ModelABTestManager {
    private let logger = Logger(subsystem: "EmbedKit", category: "ModelABTestManager")
    
    private let modelManager: HotSwappableModelManager
    private var testConfigurations: [String: ABTestConfiguration] = [:]
    private var testResults: [String: ABTestResults] = [:]
    
    public struct ABTestConfiguration: Sendable {
        public let testId: String
        public let controlVersion: ModelVersion
        public let treatmentVersion: ModelVersion
        public let trafficSplit: Double // 0.0 to 1.0
        public let duration: TimeInterval
        public let startTime: Date
        
        public init(
            testId: String,
            controlVersion: ModelVersion,
            treatmentVersion: ModelVersion,
            trafficSplit: Double = 0.5,
            duration: TimeInterval = 3600, // 1 hour
            startTime: Date = Date()
        ) {
            self.testId = testId
            self.controlVersion = controlVersion
            self.treatmentVersion = treatmentVersion
            self.trafficSplit = trafficSplit
            self.duration = duration
            self.startTime = startTime
        }
        
        public var isActive: Bool {
            let elapsed = Date().timeIntervalSince(startTime)
            return elapsed < duration
        }
    }
    
    public struct ABTestResults: Sendable {
        public let testId: String
        public var controlMetrics: PerformanceMetrics
        public var treatmentMetrics: PerformanceMetrics
        public let startTime: Date
        public let endTime: Date?
        
        public var isComplete: Bool { endTime != nil }
        
        public var winner: String? {
            guard isComplete else { return nil }
            
            if treatmentMetrics.averageLatency < controlMetrics.averageLatency &&
               treatmentMetrics.errorRate <= controlMetrics.errorRate {
                return "treatment"
            } else if controlMetrics.averageLatency < treatmentMetrics.averageLatency &&
                      controlMetrics.errorRate <= treatmentMetrics.errorRate {
                return "control"
            }
            return "inconclusive"
        }
    }
    
    public struct PerformanceMetrics: Sendable {
        public var totalRequests: Int = 0
        public var totalLatency: TimeInterval = 0
        public var errorCount: Int = 0
        
        public var averageLatency: TimeInterval {
            totalRequests > 0 ? totalLatency / Double(totalRequests) : 0
        }
        
        public var errorRate: Double {
            totalRequests > 0 ? Double(errorCount) / Double(totalRequests) : 0
        }
        
        public mutating func recordRequest(latency: TimeInterval, error: Bool = false) {
            totalRequests += 1
            totalLatency += latency
            if error {
                errorCount += 1
            }
        }
    }
    
    public init(modelManager: HotSwappableModelManager) {
        self.modelManager = modelManager
    }
    
    /// Start an A/B test between two model versions
    public func startTest(_ configuration: ABTestConfiguration) async throws {
        // Preload both versions
        try await modelManager.preloadVersions([
            configuration.controlVersion,
            configuration.treatmentVersion
        ])
        
        testConfigurations[configuration.testId] = configuration
        testResults[configuration.testId] = ABTestResults(
            testId: configuration.testId,
            controlMetrics: PerformanceMetrics(),
            treatmentMetrics: PerformanceMetrics(),
            startTime: configuration.startTime,
            endTime: nil
        )
        
        logger.info("Started A/B test \\(configuration.testId)")
    }
    
    /// Select which model version to use for a request
    public func selectVersionForRequest(testId: String) async -> ModelVersion? {
        guard let config = testConfigurations[testId],
              config.isActive else {
            return nil
        }
        
        // Simple random selection based on traffic split
        let random = Double.random(in: 0...1)
        return random < config.trafficSplit ? config.treatmentVersion : config.controlVersion
    }
    
    /// Record the result of a request for A/B testing
    public func recordTestResult(
        testId: String,
        version: ModelVersion,
        latency: TimeInterval,
        error: Bool = false
    ) async {
        guard var results = testResults[testId],
              let config = testConfigurations[testId] else {
            return
        }
        
        if version == config.controlVersion {
            results.controlMetrics.recordRequest(latency: latency, error: error)
        } else if version == config.treatmentVersion {
            results.treatmentMetrics.recordRequest(latency: latency, error: error)
        }
        
        testResults[testId] = results
    }
    
    /// Get the results of an A/B test
    public func getTestResults(testId: String) async -> ABTestResults? {
        testResults[testId]
    }
    
    /// Stop an A/B test and get final results
    public func stopTest(testId: String) async -> ABTestResults? {
        guard var results = testResults[testId] else { return nil }
        
        results = ABTestResults(
            testId: results.testId,
            controlMetrics: results.controlMetrics,
            treatmentMetrics: results.treatmentMetrics,
            startTime: results.startTime,
            endTime: Date()
        )
        
        testResults[testId] = results
        testConfigurations.removeValue(forKey: testId)
        
        logger.info("Stopped A/B test \(testId), winner: \(results.winner ?? "inconclusive")")
        return results
    }
}

/// Errors related to model versioning
public enum ModelVersionError: LocalizedError {
    case versionNotFound(String)
    case modelFileNotFound(String)
    case versionConflict(String)
    case invalidVersion(String)
    
    public var errorDescription: String? {
        switch self {
        case .versionNotFound(let version):
            return "Model version not found: \(version)"
        case .modelFileNotFound(let path):
            return "Model file not found: \(path)"
        case .versionConflict(let details):
            return "Version conflict: \(details)"
        case .invalidVersion(let version):
            return "Invalid version format: \(version)"
        }
    }
}
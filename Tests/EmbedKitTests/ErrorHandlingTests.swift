import Foundation
import Testing
@testable import EmbedKit

@Suite("Error Handling Components Tests")
struct ErrorHandlingTests {
    
    // MARK: - ErrorContext Tests
    
    @Test("Create error context with builder pattern")
    func testErrorContextBuilder() {
        let context = ErrorContext(
            operation: .embedding,
            modelIdentifier: .miniLM_L6_v2,
            metadata: ErrorMetadata()
                .with(\.inputSize, 1024)
                .with(\.batchSize, 32)
                .with(key: "custom", value: "test")
        )
        
        #expect(context.operation == .embedding)
        #expect(context.modelIdentifier == .miniLM_L6_v2)
        #expect(context.metadata.inputSize == 1024)
        #expect(context.metadata.batchSize == 32)
        #expect(context.metadata.dictionary["custom"] == "test")
        #expect(context.timestamp.timeIntervalSinceNow < 1) // Recently created
        #expect(context.correlationId != UUID())
    }
    
    @Test("Error metadata dynamic member lookup")
    func testErrorMetadataDynamicMemberLookup() {
        var metadata = ErrorMetadata()
        
        // Test well-known keys
        metadata.inputSize = 512
        metadata.batchSize = 16
        metadata.sequenceLength = 128
        metadata.memoryUsage = 1_048_576
        metadata.duration = 1.5
        metadata.retryAttempt = 2
        metadata.systemErrorCode = -1001
        metadata.device = "GPU"
        
        #expect(metadata.inputSize == 512)
        #expect(metadata.batchSize == 16)
        #expect(metadata.sequenceLength == 128)
        #expect(metadata.memoryUsage == 1_048_576)
        #expect(metadata.duration == 1.5)
        #expect(metadata.retryAttempt == 2)
        #expect(metadata.systemErrorCode == -1001)
        #expect(metadata.device == "GPU")
        
        // Test dynamic member lookup
        metadata[dynamicMember: "customKey"] = "customValue"
        #expect(metadata[dynamicMember: "customKey"] == "customValue")
    }
    
    @Test("Source location formatting")
    func testSourceLocationFormatting() {
        let location = SourceLocation(
            file: "/Users/test/project/Sources/Test.swift",
            function: "testFunction()",
            line: 42
        )
        
        #expect(location.description == "Test.swift:42 in testFunction()")
    }
    
    @Test("Error context convenience builders")
    func testErrorContextConvenienceBuilders() {
        // Model loading context
        let modelContext = ErrorContext.modelLoading(
            .miniLM_L6_v2,
            metadata: ErrorMetadata().with(\.memoryUsage, 1_000_000)
        )
        
        #expect(modelContext.operation == .modelLoading)
        #expect(modelContext.modelIdentifier == .miniLM_L6_v2)
        #expect(modelContext.metadata.memoryUsage == 1_000_000)
        
        // Embedding context
        let embedContext = ErrorContext.embedding(
            modelIdentifier: .miniLM_L6_v2,
            inputSize: 256
        )
        
        #expect(embedContext.operation == .embedding)
        #expect(embedContext.modelIdentifier == .miniLM_L6_v2)
        #expect(embedContext.metadata.inputSize == 256)
        
        // Batch embedding context
        let batchContext = ErrorContext.batchEmbedding(
            modelIdentifier: .miniLM_L6_v2,
            batchSize: 64
        )
        
        #expect(batchContext.operation == .batchEmbedding)
        #expect(batchContext.modelIdentifier == .miniLM_L6_v2)
        #expect(batchContext.metadata.batchSize == 64)
    }
    
    @Test("ContextualEmbeddingError creation and properties")
    func testContextualEmbeddingError() {
        let context = ErrorContext.embedding(inputSize: 1024)
        let underlyingError = NSError(domain: "TestDomain", code: 42, userInfo: nil)
        
        // Test different error types
        let errors: [ContextualEmbeddingError] = [
            .modelNotLoaded(context: context, underlyingError: underlyingError),
            .tokenizationFailed(context: context),
            .inferenceFailed(context: context),
            .invalidInput(context: context, reason: .tooLong),
            .dimensionMismatch(context: context, expected: 768, actual: 512),
            .resourceUnavailable(context: context, resource: .gpu),
            .configurationError(context: context, issue: .incompatible)
        ]
        
        for error in errors {
            #expect(error.context.operation == .embedding)
            #expect(error.errorDescription != nil)
            
            switch error {
            case .modelNotLoaded:
                #expect(error.underlyingError as? NSError == underlyingError)
            case .invalidInput(_, let reason, _):
                #expect(reason == .tooLong)
                #expect(error.failureReason != nil)
            case .dimensionMismatch(_, let expected, let actual, _):
                #expect(expected == 768)
                #expect(actual == 512)
            case .resourceUnavailable(_, let resource, _):
                #expect(resource == .gpu)
            case .configurationError(_, let issue, _):
                #expect(issue == .incompatible)
            default:
                break
            }
        }
    }
    
    @Test("All error operation types")
    func testAllErrorOperationTypes() {
        let operations = ErrorContext.Operation.allCases
        
        #expect(operations.count == 12)
        
        for operation in operations {
            #expect(!operation.description.isEmpty)
            #expect(!operation.rawValue.isEmpty)
        }
    }
    
    // MARK: - CircuitBreaker Tests
    
    @Test("Circuit breaker state transitions")
    func testCircuitBreakerStateTransitions() async {
        let breaker = CircuitBreaker(failureThreshold: 3, recoveryTimeout: 0.5)
        
        // Initial state should be closed
        #expect(await breaker.currentState == .closed)
        #expect(await breaker.canExecute())
        
        // Record failures below threshold
        await breaker.recordFailure()
        await breaker.recordFailure()
        #expect(await breaker.currentState == .closed)
        #expect(await breaker.canExecute())
        
        // Third failure should open the circuit
        await breaker.recordFailure()
        #expect(await breaker.currentState == .open)
        #expect(await !breaker.canExecute())
        
        // Wait for recovery timeout
        try? await Task.sleep(nanoseconds: 600_000_000) // 0.6 seconds
        
        // Should transition to half-open
        #expect(await breaker.canExecute())
        #expect(await breaker.currentState == .halfOpen)
        
        // Success should close the circuit
        await breaker.recordSuccess()
        #expect(await breaker.currentState == .closed)
        #expect(await breaker.canExecute())
    }
    
    @Test("Circuit breaker recovery after success")
    func testCircuitBreakerRecoveryAfterSuccess() async {
        let breaker = CircuitBreaker(failureThreshold: 2, recoveryTimeout: 1.0)
        
        // Open the circuit
        await breaker.recordFailure()
        await breaker.recordFailure()
        #expect(await breaker.currentState == .open)
        
        // Record success - should reset and close
        await breaker.recordSuccess()
        #expect(await breaker.currentState == .closed)
        
        // Should be able to execute again
        #expect(await breaker.canExecute())
    }
    
    // MARK: - GracefulDegradationManager Tests
    
    @Test("Graceful degradation level assessment")
    func testGracefulDegradationLevelAssessment() async {
        let manager = GracefulDegradationManager()
        
        // Initial assessment should be normal
        let level = await manager.assessDegradationLevel(operation: "test")
        #expect(level == .normal)
        
        // Record some errors
        await manager.recordError(for: "test", error: NSError(domain: "Test", code: 1))
        await manager.recordError(for: "test", error: NSError(domain: "Test", code: 2))
        
        // With high error rate, should degrade
        let newLevel = await manager.assessDegradationLevel(operation: "test")
        #expect(newLevel.rawValue >= GracefulDegradationManager.DegradationLevel.normal.rawValue)
    }
    
    @Test("Graceful degradation configuration adjustment")
    func testGracefulDegradationConfigurationAdjustment() async {
        let manager = GracefulDegradationManager()
        
        let originalConfig = Configuration.default(for: .miniLM_L6_v2)
        
        // Test normal level - no changes
        var adjustedConfig = await manager.applyDegradation(level: .normal, configuration: originalConfig)
        #expect(adjustedConfig.model.maxSequenceLength == originalConfig.model.maxSequenceLength)
        #expect(adjustedConfig.resources.batchSize == originalConfig.resources.batchSize)
        
        // Test reduced level
        adjustedConfig = await manager.applyDegradation(level: .reduced, configuration: originalConfig)
        #expect(adjustedConfig.model.maxSequenceLength <= 256)
        #expect(adjustedConfig.resources.batchSize <= 16)
        
        // Test minimal level
        adjustedConfig = await manager.applyDegradation(level: .minimal, configuration: originalConfig)
        #expect(adjustedConfig.model.maxSequenceLength <= 128)
        #expect(adjustedConfig.resources.batchSize <= 8)
        
        // Test emergency level
        adjustedConfig = await manager.applyDegradation(level: .emergency, configuration: originalConfig)
        #expect(adjustedConfig.model.maxSequenceLength <= 64)
        #expect(adjustedConfig.resources.batchSize == 1)
        #expect(adjustedConfig.model.poolingStrategy == .cls)
    }
    
    @Test("Graceful degradation status tracking")
    func testGracefulDegradationStatusTracking() async {
        let manager = GracefulDegradationManager()
        
        // Assess multiple operations
        _ = await manager.assessDegradationLevel(operation: "embedding")
        _ = await manager.assessDegradationLevel(operation: "tokenization")
        
        let status = await manager.getDegradationStatus()
        #expect(status["embedding"] != nil)
        #expect(status["tokenization"] != nil)
    }
    
    @Test("Degradation level comparison")
    func testDegradationLevelComparison() {
        let levels = GracefulDegradationManager.DegradationLevel.allCases
        
        #expect(levels[0] < levels[1])
        #expect(levels[1] < levels[2])
        #expect(levels[2] < levels[3])
        
        #expect(GracefulDegradationManager.DegradationLevel.normal < .reduced)
        #expect(GracefulDegradationManager.DegradationLevel.reduced < .minimal)
        #expect(GracefulDegradationManager.DegradationLevel.minimal < .emergency)
    }
    
    @Test("Error info creation")
    func testErrorInfoCreation() {
        let error = NSError(domain: "TestDomain", code: 42)
        let info = ErrorInfo(
            error: error,
            operation: "test_operation",
            timestamp: Date(),
            errorCount: 5
        )
        
        #expect(info.error as NSError == error)
        #expect(info.operation == "test_operation")
        #expect(info.errorCount == 5)
        #expect(info.timestamp.timeIntervalSinceNow < 1)
    }
    
    @Test("Error handling result creation")
    func testErrorHandlingResultCreation() {
        let error = NSError(domain: "TestDomain", code: 42)
        let errorInfo = ErrorInfo(
            error: error,
            operation: "test",
            timestamp: Date(),
            errorCount: 1
        )
        
        let result = ErrorHandlingResult(
            strategy: .retryWithBackoff,
            shouldRetry: true,
            fallbackValue: "default",
            errorInfo: errorInfo
        )
        
        #expect(result.strategy == .retryWithBackoff)
        #expect(result.shouldRetry)
        #expect(result.fallbackValue as? String == "default")
        #expect(result.errorInfo.operation == "test")
    }
    
    @Test("All error handling strategies")
    func testAllErrorHandlingStrategies() {
        let strategies = ErrorHandlingStrategy.allCases
        
        #expect(strategies.count == 6)
        #expect(strategies.contains(.retryWithBackoff))
        #expect(strategies.contains(.retryAfterRecovery))
        #expect(strategies.contains(.useFallback))
        #expect(strategies.contains(.degradeGracefully))
        #expect(strategies.contains(.failFast))
        #expect(strategies.contains(.ignore))
        
        for strategy in strategies {
            #expect(!strategy.rawValue.isEmpty)
        }
    }
    
    @Test("Resource usage tracking")
    func testResourceUsageTracking() {
        var usage = ResourceUsage()
        
        usage.memoryUsagePercent = 0.75
        usage.cpuUsagePercent = 0.50
        usage.diskUsagePercent = 0.60
        usage.diskSpaceAvailable = 10_737_418_240 // 10 GB
        
        #expect(usage.memoryUsagePercent == 0.75)
        #expect(usage.cpuUsagePercent == 0.50)
        #expect(usage.diskUsagePercent == 0.60)
        #expect(usage.diskSpaceAvailable == 10_737_418_240)
    }
    
    @Test("Concurrent error recording")
    func testConcurrentErrorRecording() async {
        let manager = GracefulDegradationManager()
        
        // Record errors and successes concurrently
        await withTaskGroup(of: Void.self) { group in
            // Record errors
            for i in 0..<10 {
                group.addTask {
                    await manager.recordError(
                        for: "concurrent_test",
                        error: NSError(domain: "Test", code: i)
                    )
                }
            }
            
            // Record successes
            for _ in 0..<5 {
                group.addTask {
                    await manager.recordSuccess(for: "concurrent_test")
                }
            }
            
            // Wait for all tasks to complete
            await group.waitForAll()
        }
        
        // Small delay to ensure all state updates are complete
        try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
        
        // Assess level after concurrent operations
        let level = await manager.assessDegradationLevel(operation: "concurrent_test")
        // With 10 errors and 5 successes (error rate = 66.7%), level should be emergency
        // Error rate thresholds: normal < 5%, reduced < 15%, minimal < 30%, emergency >= 30%
        #expect(level == .emergency) // 66.7% error rate is well above 30% threshold
    }
    
    @Test("Error context metadata chaining")
    func testErrorContextMetadataChaining() {
        let metadata = ErrorMetadata()
            .with(\.inputSize, 1024)
            .with(\.batchSize, 32)
            .with(\.sequenceLength, 512)
            .with(\.memoryUsage, 1_048_576)
            .with(\.duration, 2.5)
            .with(\.retryAttempt, 3)
            .with(\.systemErrorCode, -1001)
            .with(\.device, "Metal")
            .with(key: "custom1", value: "value1")
            .with(key: "custom2", value: "value2")
        
        #expect(metadata.inputSize == 1024)
        #expect(metadata.batchSize == 32)
        #expect(metadata.sequenceLength == 512)
        #expect(metadata.memoryUsage == 1_048_576)
        #expect(metadata.duration == 2.5)
        #expect(metadata.retryAttempt == 3)
        #expect(metadata.systemErrorCode == -1001)
        #expect(metadata.device == "Metal")
        #expect(metadata.dictionary["custom1"] == "value1")
        #expect(metadata.dictionary["custom2"] == "value2")
    }
    
    @Test("Invalid input reason descriptions")
    func testInvalidInputReasonDescriptions() {
        let reasons: [ContextualEmbeddingError.InvalidInputReason] = [
            .empty, .tooLong, .invalidCharacters, .unsupportedLanguage, .malformed
        ]
        
        for reason in reasons {
            let error = ContextualEmbeddingError.invalidInput(
                context: ErrorContext.embedding(),
                reason: reason
            )
            
            #expect(error.errorDescription != nil)
            #expect(error.failureReason != nil)
        }
    }
    
    @Test("Resource type descriptions")
    func testResourceTypeDescriptions() {
        let resources: [ContextualEmbeddingError.ResourceType] = [
            .model, .memory, .gpu, .cache, .network
        ]
        
        for resource in resources {
            let error = ContextualEmbeddingError.resourceUnavailable(
                context: ErrorContext.embedding(),
                resource: resource
            )
            
            #expect(error.errorDescription != nil)
            #expect(error.failureReason != nil)
        }
    }
    
    @Test("Configuration issue descriptions")
    func testConfigurationIssueDescriptions() {
        let issues: [ContextualEmbeddingError.ConfigurationIssue] = [
            .invalid, .missing, .incompatible, .outOfRange
        ]
        
        for issue in issues {
            let error = ContextualEmbeddingError.configurationError(
                context: ErrorContext.embedding(),
                issue: issue
            )
            
            #expect(error.errorDescription != nil)
            #expect(error.failureReason != nil)
        }
    }
}

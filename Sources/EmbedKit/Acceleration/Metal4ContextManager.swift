// EmbedKit - Metal4ContextManager
// Shared GPU context management using VectorAccelerate's Metal4Context

import Foundation
import VectorAccelerate

// MARK: - Metal4ContextManager

/// Manages a shared Metal4Context for GPU operations across EmbedKit.
///
/// `Metal4ContextManager` provides centralized access to VectorAccelerate's
/// `Metal4Context`, ensuring efficient GPU resource sharing and avoiding
/// redundant context initialization.
///
/// ## Example Usage
/// ```swift
/// let context = try await Metal4ContextManager.shared()
/// let kernel = try await L2DistanceKernel(context: context)
/// ```
public actor Metal4ContextManager {

    // MARK: - Singleton

    /// The shared manager instance.
    private static let manager = Metal4ContextManager()

    // MARK: - State

    /// The cached Metal4Context instance.
    private var cachedContext: Metal4Context?

    /// Whether initialization has been attempted.
    private var initializationAttempted = false

    /// Cached initialization error if any.
    private var initializationError: Error?

    // MARK: - Initialization

    private init() {}

    // MARK: - Public Access

    /// Get the shared Metal4Context instance.
    ///
    /// Creates the context lazily on first access. Subsequent calls return
    /// the same instance. Throws if Metal 4 is not available on this device.
    ///
    /// - Returns: Shared Metal4Context
    /// - Throws: Metal4ContextError if context cannot be created
    public static func shared() async throws -> Metal4Context {
        try await manager.getContext()
    }

    /// Get or create the Metal4Context.
    private func getContext() async throws -> Metal4Context {
        // Fast path - already initialized
        if let context = cachedContext {
            return context
        }

        // Check if we already tried and failed
        if initializationAttempted, let error = initializationError {
            throw error
        }

        // Initialize
        initializationAttempted = true

        do {
            let context = try await Metal4Context()
            cachedContext = context
            return context
        } catch {
            let wrappedError = Metal4ContextError.initializationFailed(error)
            initializationError = wrappedError
            throw wrappedError
        }
    }

    /// Check if Metal4Context is available without initializing.
    ///
    /// This is a lightweight check that doesn't create the context.
    public static var isAvailable: Bool {
        #if canImport(Metal)
        return true  // Actual capability check happens during initialization
        #else
        return false
        #endif
    }

    /// Check if a shared context has been successfully initialized.
    public static func isInitialized() async -> Bool {
        await manager.cachedContext != nil
    }

    // MARK: - Lifecycle

    /// Release the shared context and free GPU resources.
    ///
    /// Call this when your application is terminating or when you need
    /// to free GPU memory. The context will be recreated on next access.
    public static func releaseSharedContext() async {
        await manager.release()
    }

    /// Release internal resources.
    private func release() {
        cachedContext = nil
        initializationAttempted = false
        initializationError = nil
    }

    /// Reset the manager for testing purposes.
    ///
    /// Releases all resources and resets internal state.
    @available(*, deprecated, message: "For testing only")
    public static func resetForTesting() async {
        await releaseSharedContext()
    }
}

// MARK: - Errors

/// Errors from Metal4ContextManager operations.
public enum Metal4ContextError: Error, LocalizedError, Sendable {
    /// Failed to initialize Metal4Context.
    case initializationFailed(Error)

    /// Metal 4 is not available on this device.
    case metal4NotAvailable

    public var errorDescription: String? {
        switch self {
        case .initializationFailed(let underlying):
            return "Failed to initialize Metal4Context: \(underlying.localizedDescription)"
        case .metal4NotAvailable:
            return "Metal 4 is not available on this device"
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .initializationFailed:
            return "Ensure you are running on a device with Metal 4 support (Apple Silicon Mac or modern iOS device)."
        case .metal4NotAvailable:
            return "Metal 4 requires Apple Silicon. This device does not support Metal 4 features."
        }
    }
}

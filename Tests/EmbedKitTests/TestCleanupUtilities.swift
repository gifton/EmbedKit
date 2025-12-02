// EmbedKit - Test Cleanup Utilities
//
// Provides cleanup functions for shared resources to prevent test hangs
// from accumulated singleton state.

import Foundation
@testable import EmbedKit

// MARK: - Test Cleanup Functions

/// Clean up all shared EmbedKit resources.
///
/// Call this at the end of test suites that use Metal resources or
/// memory monitoring to prevent resource accumulation.
@available(*, deprecated, message: "For testing only")
func cleanupEmbedKitTestResources() async {
    // Reset SharedMetalContextManager
    await SharedMetalContextManager.shared.resetForTesting()

    // Reset MemoryMonitor
    MemoryMonitor.shared.forceResetForTesting()
}

/// Clean up only Metal resources.
///
/// Use when you only need to release GPU buffers and contexts.
@available(*, deprecated, message: "For testing only")
func cleanupMetalTestResources() async {
    await SharedMetalContextManager.shared.resetForTesting()
}

/// Clean up only memory monitoring resources.
///
/// Use when you only need to stop memory pressure monitoring.
@available(*, deprecated, message: "For testing only")
func cleanupMemoryMonitorTestResources() {
    MemoryMonitor.shared.forceResetForTesting()
}

// MARK: - Test Trait for Automatic Cleanup

/// A test trait that cleans up Metal resources after each test.
///
/// Usage:
/// ```swift
/// @Suite("MyTests", .serialized)
/// @MetalTestCleanup
/// struct MyMetalTests {
///     // tests...
/// }
/// ```
///
/// Note: Since Swift Testing doesn't have built-in teardown,
/// this is primarily documentation. Tests should call cleanup
/// functions explicitly when needed.
@available(*, deprecated, message: "For documentation only - call cleanup functions explicitly")
struct MetalTestCleanup {}

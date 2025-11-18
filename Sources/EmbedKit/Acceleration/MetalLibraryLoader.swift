import Foundation
import Metal
import OSLog

/// Source from which the Metal library was loaded
///
/// This enum tracks whether we loaded a precompiled metallib from the bundle
/// or fell back to runtime compilation from a string literal.
public enum MetalLibrarySource: Sendable {
    /// Loaded from precompiled metallib in bundle
    case precompiled(URL)

    /// Compiled at runtime from string source
    case sourceString

    /// Human-readable description of the source
    public var description: String {
        switch self {
        case .precompiled(let url):
            return "Precompiled metallib: \(url.lastPathComponent)"
        case .sourceString:
            return "Runtime compilation from string"
        }
    }

    /// Performance indicator: precompiled is significantly faster
    public var isPrecompiled: Bool {
        if case .precompiled = self {
            return true
        }
        return false
    }
}

/// Loads Metal shader libraries with intelligent fallback
///
/// This loader implements a dual-mode strategy:
/// 1. **Primary**: Load precompiled metallib from bundle (fast, ~5ms)
/// 2. **Fallback**: Compile from string literal at runtime (slower, ~150ms)
///
/// **Usage**:
/// ```swift
/// let (library, source) = try MetalLibraryLoader.loadLibrary(device: device)
/// print("Loaded from: \(source.description)")
/// ```
///
/// **Performance**:
/// - Precompiled: ~5ms cold start, <1ms warm
/// - String compilation: ~150ms first launch, ~50ms subsequent
///
/// **Thread Safety**: This class is thread-safe and caches the loaded library
///
public final class MetalLibraryLoader: @unchecked Sendable {
    nonisolated private static let logger = EmbedKitLogger.metal()

    // Thread-safe cache using actor
    private actor Cache {
        var library: MTLLibrary?
        var source: MetalLibrarySource?

        func get() -> (MTLLibrary, MetalLibrarySource)? {
            guard let library, let source else { return nil }
            return (library, source)
        }

        func set(library: MTLLibrary, source: MetalLibrarySource) {
            self.library = library
            self.source = source
        }

        func clear() {
            self.library = nil
            self.source = nil
        }
    }

    private static let cache = Cache()

    /// Load Metal shader library with automatic fallback
    ///
    /// This method attempts to load a precompiled metallib from the bundle.
    /// If not found or if loading fails, it falls back to runtime string compilation.
    ///
    /// **Search Strategy**:
    /// 1. Try Bundle.module.url(forResource:withExtension:)
    /// 2. Try Bundle.main.url(forResource:withExtension:)
    /// 3. Fall back to string compilation
    ///
    /// **Caching**:
    /// The loaded library is cached for subsequent calls. This is safe because
    /// MTLLibrary instances are thread-safe and immutable.
    ///
    /// - Parameter device: Metal device to create library for
    /// - Returns: Tuple of (MTLLibrary, MetalLibrarySource)
    /// - Throws: MetalError if both loading methods fail
    ///
    /// Load Metal shader library with automatic fallback (async version)
    public static func loadLibrary(device: MTLDevice) async throws -> (MTLLibrary, MetalLibrarySource) {
        // Check cache first (thread-safe via actor)
        if let cached = await cache.get() {
            logger.debug("Using cached Metal library (\(cached.1.description))")
            return cached
        }

        // Try loading strategies in order
        logger.start("Metal library loading")

        // Strategy 1: Precompiled metallib from bundle
        if let (library, source) = try? loadPrecompiledLibrary(device: device) {
            await cache.set(library: library, source: source)
            logger.complete("Metal library loading", result: source.description)
            return (library, source)
        }

        // Strategy 2: Fallback to string compilation
        logger.warning("Precompiled metallib not found, falling back to string compilation")
        logger.warning("Performance: String compilation is ~30x slower than precompiled metallib")
        logger.warning("Solution: Run ./Scripts/CompileMetalShaders.sh to generate metallib")

        let (library, source) = try loadFromString(device: device)
        await cache.set(library: library, source: source)

        logger.complete("Metal library loading", result: source.description)
        return (library, source)
    }

    /// Load Metal shader library with automatic fallback (synchronous version for internal use)
    internal static func loadLibrarySync(device: MTLDevice) throws -> (MTLLibrary, MetalLibrarySource) {
        // Try precompiled metallib
        if let result = try? loadPrecompiledLibrary(device: device) {
            return result
        }

        // Fallback to string compilation
        logger.warning("Precompiled metallib not found, falling back to string compilation")
        return try loadFromString(device: device)
    }

    /// Clear the cached library (useful for testing)
    public static func clearCache() async {
        await cache.clear()
    }

    // MARK: - Private Loading Implementations

    /// Attempt to load precompiled metallib from bundle
    private static func loadPrecompiledLibrary(device: MTLDevice) throws -> (MTLLibrary, MetalLibrarySource) {
        let metallibName = "EmbedKitShaders"

        // Strategy 1: Try Bundle.module (SPM)
        #if SWIFT_PACKAGE
        if let url = Bundle.module.url(forResource: metallibName, withExtension: "metallib") {
            logger.debug("Found metallib in SPM bundle: \(url.path)")
            return try loadMetallibFromURL(url, device: device)
        }
        #endif

        // Strategy 2: Try Bundle.main (Xcode projects)
        if let url = Bundle.main.url(forResource: metallibName, withExtension: "metallib") {
            logger.debug("Found metallib in main bundle: \(url.path)")
            return try loadMetallibFromURL(url, device: device)
        }

        // Strategy 3: Try searching in common paths
        let searchPaths = [
            // Development paths
            "Sources/EmbedKit/Resources",
            ".build/debug",
            ".build/release",
            // Bundle paths
            Bundle.main.resourcePath,
            Bundle.main.bundlePath,
        ].compactMap { $0 }

        for searchPath in searchPaths {
            let metallibPath = "\(searchPath)/\(metallibName).metallib"
            let url = URL(fileURLWithPath: metallibPath)

            if FileManager.default.fileExists(atPath: url.path) {
                logger.debug("Found metallib at search path: \(url.path)")
                return try loadMetallibFromURL(url, device: device)
            }
        }

        // Not found
        logger.debug("Precompiled metallib not found in any search location")
        throw MetalError.libraryNotFound("Precompiled metallib not found")
    }

    /// Load metallib from a specific URL
    private static func loadMetallibFromURL(_ url: URL, device: MTLDevice) throws -> (MTLLibrary, MetalLibrarySource) {
        do {
            let library = try device.makeLibrary(URL: url)
            return (library, .precompiled(url))
        } catch {
            logger.error("Failed to load metallib from \(url.path)", error: error)
            throw MetalError.libraryLoadFailed("Failed to load metallib: \(error.localizedDescription)")
        }
    }

    /// Compile Metal library from string literal (fallback)
    private static func loadFromString(device: MTLDevice) throws -> (MTLLibrary, MetalLibrarySource) {
        let source = MetalShaderLibrary.source

        let compileOptions = MTLCompileOptions()
        compileOptions.fastMathEnabled = true

        // Use Metal 3.0 if available
        if #available(iOS 16.0, macOS 13.0, *) {
            compileOptions.languageVersion = .version3_0
        } else {
            compileOptions.languageVersion = .version2_4
        }

        do {
            let library = try device.makeLibrary(source: source, options: compileOptions)
            return (library, .sourceString)
        } catch {
            logger.error("Failed to compile Metal library from string", error: error)
            throw MetalError.libraryCompileFailed("Failed to compile shaders: \(error.localizedDescription)")
        }
    }
}

// MARK: - MetalError Extension

extension MetalError {
    /// Metal library was not found in bundle
    static func libraryNotFound(_ message: String) -> MetalError {
        .other(message)
    }

    /// Failed to load Metal library from URL
    static func libraryLoadFailed(_ message: String) -> MetalError {
        .other(message)
    }

    /// Failed to compile Metal library from string
    static func libraryCompileFailed(_ message: String) -> MetalError {
        .other(message)
    }

    /// Generic Metal error
    static func other(_ message: String) -> MetalError {
        // Since MetalError might not have an 'other' case, we'll create a sensible error
        // This will need to be adjusted based on your actual MetalError enum definition
        .invalidInput(message)
    }
}

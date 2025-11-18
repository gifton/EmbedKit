import Foundation

/// High-precision benchmark runner with warmup, statistical analysis, and memory tracking
public actor BenchmarkRunner {

    /// Configuration for benchmark execution
    public struct Configuration {
        /// Number of warmup iterations to stabilize caches/JIT
        public var warmupIterations: Int

        /// Number of measurement iterations for statistics
        public var measurementIterations: Int

        /// Minimum benchmark duration (will run more iterations if needed)
        public var minimumDuration: TimeInterval

        /// Whether to collect memory metrics
        public var collectMemory: Bool

        /// Whether to print progress during benchmark
        public var verbose: Bool

        public init(
            warmupIterations: Int = 10,
            measurementIterations: Int = 100,
            minimumDuration: TimeInterval = 1.0,
            collectMemory: Bool = true,
            verbose: Bool = true
        ) {
            self.warmupIterations = warmupIterations
            self.measurementIterations = measurementIterations
            self.minimumDuration = minimumDuration
            self.collectMemory = collectMemory
            self.verbose = verbose
        }

        /// Configuration optimized for quick smoke tests
        public static var quick: Configuration {
            Configuration(
                warmupIterations: 3,
                measurementIterations: 20,
                minimumDuration: 0.5,
                collectMemory: false,
                verbose: false
            )
        }

        /// Configuration for comprehensive benchmarks
        public static var comprehensive: Configuration {
            Configuration(
                warmupIterations: 20,
                measurementIterations: 200,
                minimumDuration: 2.0,
                collectMemory: true,
                verbose: true
            )
        }

        /// Configuration for long-running benchmarks
        public static var longRunning: Configuration {
            Configuration(
                warmupIterations: 5,
                measurementIterations: 20,
                minimumDuration: 1.0,
                collectMemory: true,
                verbose: true
            )
        }
    }

    public let configuration: Configuration

    public init(configuration: Configuration = Configuration()) {
        self.configuration = configuration
    }

    /// Run a benchmark and return performance statistics
    /// - Parameters:
    ///   - name: Descriptive name for the benchmark
    ///   - operation: The async operation to benchmark
    /// - Returns: Complete benchmark result with timing and memory stats
    public func measure<T>(
        name: String,
        operation: @Sendable @escaping () async throws -> T
    ) async throws -> BenchmarkResult {

        if configuration.verbose {
            print("📊 Benchmarking: \(name)")
        }

        // Warmup phase - stabilize caches and JIT
        if configuration.verbose {
            print("  🔥 Warming up (\(configuration.warmupIterations) iterations)...")
        }

        for _ in 0..<configuration.warmupIterations {
            _ = try await operation()
        }

        // Measurement phase
        if configuration.verbose {
            print("  ⏱️  Measuring (\(configuration.measurementIterations) iterations)...")
        }

        var timings: [TimeInterval] = []
        var memoryDeltas: [Int64] = []

        let startMemory = configuration.collectMemory ? MemoryProfiler.currentUsage() : 0

        for iteration in 0..<configuration.measurementIterations {
            let memBefore = configuration.collectMemory ? MemoryProfiler.currentUsage() : 0

            // High-precision timing
            let startTime = CFAbsoluteTimeGetCurrent()
            _ = try await operation()
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime

            let memAfter = configuration.collectMemory ? MemoryProfiler.currentUsage() : 0

            timings.append(elapsed)

            if configuration.collectMemory {
                memoryDeltas.append(memAfter - memBefore)
            }

            // Progress indicator every 25%
            if configuration.verbose {
                let progressPoints = [25, 50, 75]
                let progress = Int((Double(iteration + 1) / Double(configuration.measurementIterations)) * 100)

                if progressPoints.contains(progress) {
                    print("    \(progress)% complete")
                }
            }
        }

        let endMemory = configuration.collectMemory ? MemoryProfiler.currentUsage() : 0

        // Compute statistics
        let stats = PerformanceStatistics(samples: timings)
        let memStats = configuration.collectMemory ?
            MemoryStatistics(
                deltas: memoryDeltas,
                total: endMemory - startMemory
            ) : nil

        let result = BenchmarkResult(
            name: name,
            timing: stats,
            memory: memStats,
            iterations: configuration.measurementIterations
        )

        if configuration.verbose {
            printResult(result)
        }

        return result
    }

    /// Print benchmark result summary
    private func printResult(_ result: BenchmarkResult) {
        print("  ✅ Complete:")
        print("     p50: \(result.timing.formatMs(result.timing.p50))")
        print("     p95: \(result.timing.formatMs(result.timing.p95))")
        print("     p99: \(result.timing.formatMs(result.timing.p99))")

        if let mem = result.memory {
            print("     Memory: \(mem.formatBytes(mem.averageDelta)) per operation")
        }

        // Warn if measurements are unstable
        if result.timing.coefficientOfVariation > 0.15 {
            print("     ⚠️  High variability (CV: \(String(format: "%.2f", result.timing.coefficientOfVariation)))")
        }

        print()
    }

    /// Run multiple benchmarks sequentially
    /// - Parameter benchmarks: Array of (name, operation) tuples
    /// - Returns: Array of benchmark results
    public func measureAll<T>(
        benchmarks: [(name: String, operation: @Sendable () async throws -> T)]
    ) async throws -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []

        for benchmark in benchmarks {
            let result = try await measure(name: benchmark.name, operation: benchmark.operation)
            results.append(result)
        }

        return results
    }
}

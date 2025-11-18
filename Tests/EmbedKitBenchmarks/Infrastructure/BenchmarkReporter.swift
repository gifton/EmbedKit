import Foundation

/// Collects and exports benchmark results to various formats
public actor BenchmarkReporter {

    private var results: [BenchmarkResult] = []

    public init() {}

    /// Record a benchmark result
    public func record(_ result: BenchmarkResult) {
        results.append(result)
    }

    /// Record multiple benchmark results
    public func recordAll(_ newResults: [BenchmarkResult]) {
        results.append(contentsOf: newResults)
    }

    /// Get all recorded results
    public func allResults() -> [BenchmarkResult] {
        results
    }

    /// Clear all recorded results
    public func clear() {
        results.removeAll()
    }

    // MARK: - JSON Export

    /// Export results as JSON for programmatic analysis
    /// - Parameter url: File URL to write JSON to
    public func exportJSON(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        let data = try encoder.encode(results)
        try data.write(to: url)
    }

    // MARK: - Markdown Export

    /// Export results as Markdown table for documentation
    /// - Returns: Markdown-formatted string
    public func exportMarkdown() -> String {
        var markdown = """
        # EmbedKit Performance Benchmarks

        **Date:** \(ISO8601DateFormatter().string(from: Date()))
        **Hardware:** \(HardwareInfo.current.model) (\(HardwareInfo.current.processorCount) cores, \(HardwareInfo.current.memoryGB)GB RAM)

        ## Summary

        | Benchmark | p50 | p95 | p99 | Iterations | Memory |
        |-----------|-----|-----|-----|------------|--------|

        """

        for result in results {
            let p50Ms = result.timing.formatMs(result.timing.p50)
            let p95Ms = result.timing.formatMs(result.timing.p95)
            let p99Ms = result.timing.formatMs(result.timing.p99)

            let memStr = result.memory.map { $0.formatBytes($0.averageDelta) } ?? "N/A"

            markdown += "| \(result.name) | \(p50Ms) | \(p95Ms) | \(p99Ms) | \(result.iterations) | \(memStr) |\n"
        }

        markdown += "\n## Detailed Statistics\n\n"

        for result in results {
            markdown += """
            ### \(result.name)

            **Timing:**
            - Mean: \(result.timing.formatMs(result.timing.mean))
            - Median: \(result.timing.formatMs(result.timing.median))
            - Min: \(result.timing.formatMs(result.timing.min))
            - Max: \(result.timing.formatMs(result.timing.max))
            - Std Dev: \(result.timing.formatMs(result.timing.standardDeviation))
            - CV: \(String(format: "%.2f", result.timing.coefficientOfVariation))

            **Percentiles:**
            - p50: \(result.timing.formatMs(result.timing.p50))
            - p90: \(result.timing.formatMs(result.timing.p90))
            - p95: \(result.timing.formatMs(result.timing.p95))
            - p99: \(result.timing.formatMs(result.timing.p99))

            """

            if let mem = result.memory {
                markdown += """
                **Memory:**
                - Average: \(mem.formatBytes(mem.averageDelta))
                - Min: \(mem.formatBytes(mem.minDelta))
                - Max: \(mem.formatBytes(mem.maxDelta))
                - Total: \(mem.formatBytes(mem.total))

                """
            }

            markdown += "\n---\n\n"
        }

        return markdown
    }

    // MARK: - Console Output

    /// Print summary to console
    public func printSummary() {
        print("\n" + String(repeating: "=", count: 80))
        print("BENCHMARK SUMMARY")
        print(String(repeating: "=", count: 80))
        print()

        for result in results {
            print("📊 \(result.name)")
            print("   p50: \(result.timing.formatMs(result.timing.p50))")
            print("   p95: \(result.timing.formatMs(result.timing.p95))")
            print("   p99: \(result.timing.formatMs(result.timing.p99))")

            if let mem = result.memory {
                print("   Memory: \(mem.formatBytes(mem.averageDelta))")
            }

            // Warn about high variability
            if result.timing.coefficientOfVariation > 0.15 {
                print("   ⚠️  High variability (CV: \(String(format: "%.2f", result.timing.coefficientOfVariation)))")
            }

            print()
        }

        print(String(repeating: "=", count: 80))
        print()
    }

    /// Print a compact one-line summary
    public func printCompact() {
        for result in results {
            let status = result.timing.coefficientOfVariation < 0.10 ? "✅" : "⚠️"
            print("\(status) \(result.name): \(result.timing.formatMs(result.timing.p50)) (p50)")
        }
    }

    // MARK: - CSV Export

    /// Export results as CSV for spreadsheet analysis
    /// - Returns: CSV-formatted string
    public func exportCSV() -> String {
        var csv = "Name,p50 (ms),p95 (ms),p99 (ms),Mean (ms),Std Dev (ms),CV,Iterations,Memory (bytes)\n"

        for result in results {
            let p50 = result.timing.p50 * 1000
            let p95 = result.timing.p95 * 1000
            let p99 = result.timing.p99 * 1000
            let mean = result.timing.mean * 1000
            let stdDev = result.timing.standardDeviation * 1000
            let cv = result.timing.coefficientOfVariation
            let mem = result.memory?.averageDelta ?? 0

            csv += "\(result.name),\(p50),\(p95),\(p99),\(mean),\(stdDev),\(cv),\(result.iterations),\(mem)\n"
        }

        return csv
    }

    /// Export CSV to file
    /// - Parameter url: File URL to write CSV to
    public func exportCSV(to url: URL) throws {
        let csv = exportCSV()
        try csv.write(to: url, atomically: true, encoding: .utf8)
    }
}

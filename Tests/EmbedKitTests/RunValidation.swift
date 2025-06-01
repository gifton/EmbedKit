#!/usr/bin/env swift

import Foundation

// Simple script to run EmbedKit validation tests

print("""
╔═══════════════════════════════════════════════════════╗
║             🚀 EmbedKit Validation Runner 🚀           ║
╚═══════════════════════════════════════════════════════╝

This will validate:
  ✓ Model loading and unloading
  ✓ Embedding generation (single & batch)
  ✓ Similarity calculations
  ✓ PipelineKit integration
  ✓ Performance benchmarks
  ✓ Memory usage
  ✓ Error handling

""")

print("Starting validation tests...\n")

// Run the tests using Swift's test runner
let process = Process()
process.executableURL = URL(fileURLWithPath: "/usr/bin/swift")
process.arguments = ["test", "--filter", "ValidationTestRunner"]
process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

do {
    try process.run()
    process.waitUntilExit()
    
    let exitCode = process.terminationStatus
    
    if exitCode == 0 {
        print("\n✅ Validation completed successfully!")
    } else {
        print("\n❌ Validation failed with exit code: \(exitCode)")
    }
    
    exit(exitCode)
} catch {
    print("❌ Failed to run tests: \(error)")
    exit(1)
}
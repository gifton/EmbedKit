#!/usr/bin/env swift

import Foundation
import CoreML

func findModelURL() -> URL? {
    // Resolve repo root relative to this script (Scripts/ModelTools → Scripts → repo root)
    let scriptURL = URL(fileURLWithPath: #file).standardizedFileURL
    let repoRoot = scriptURL
        .deletingLastPathComponent() // ModelTools
        .deletingLastPathComponent() // Scripts
        .deletingLastPathComponent() // repo root

    let candidates = [
        repoRoot.appendingPathComponent("MiniLM-L12-v2.mlpackage"),
        repoRoot.appendingPathComponent("MiniLM-L12-v2-quantized8.mlpackage"),
        repoRoot.appendingPathComponent("MiniLM-L12-v2-quantized4.mlpackage"),
    ]
    for url in candidates where FileManager.default.fileExists(atPath: url.path) {
        return url
    }
    return nil
}

print(String(repeating: "=", count: 60))
print("CoreML Model Compiler (Swift)")
print(String(repeating: "=", count: 60))

guard let modelPath = findModelURL() else {
    print("\n❌ Error: Could not find MiniLM-L12-v2 .mlpackage at repository root")
    print("\n💡 Generate it with:\n   python Scripts/ModelTools/convert_minilm_l12.py")
    exit(1)
}

print("\n📦 Model path: \(modelPath.path)")
print("✅ Model package found")

print("\n⚙️  Compiling model...")
do {
    let compiledURL = try MLModel.compileModel(at: modelPath)
    print("✅ Model compiled successfully!")
    print("\n📍 Compiled model location:")
    print("   \(compiledURL.path)")

    if FileManager.default.fileExists(atPath: compiledURL.path) {
        print("\n✅ Compiled model verified at:")
        print("   \(compiledURL.path)")

        print("\n🔄 Testing model load...")
        let model = try MLModel(contentsOf: compiledURL)
        print("✅ Model loads successfully!")

        let description = model.modelDescription
        print("\n📋 Model Information:")
        print("   Input names: \(description.inputDescriptionsByName.keys.sorted())")
        print("   Output names: \(description.outputDescriptionsByName.keys.sorted())")
    }

    print("\n" + String(repeating: "=", count: 60))
    print("✨ Success! Model is ready for testing")
    print(String(repeating: "=", count: 60))
    print("\n🧪 You can now run tests with:")
    print("   swift test --filter CoreMLBackendTests")

} catch {
    print("\n❌ Compilation failed: \(error)")
    print("\nError details:")
    print(String(describing: error))
    exit(1)
}

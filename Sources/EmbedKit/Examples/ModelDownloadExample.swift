import Foundation

/// Example demonstrating model downloading and management
public func modelDownloadExample() async throws {
    print("=== EmbedKit Model Download Example ===\n")
    
    // Create a model manager with download capability
    let downloadDelegate = ConsoleDownloadDelegate()
    let modelManager = ModelManager(
        options: ModelManager.ManagerLoadingOptions(
            allowDownload: true,
            registry: HuggingFaceModelRegistry()
        ),
        downloadDelegate: downloadDelegate
    )
    
    // List available models
    print("📋 Checking available models...")
    let availableModels = try await modelManager.listAvailableModels()
    
    print("\nCurrently available models:")
    for model in availableModels {
        let location = model.location == DownloadedModelInfo.ModelLocation.bundled ? "📦 Bundled" : "💾 Downloaded"
        let size = model.size.map { " (\(ByteCountFormatter.string(fromByteCount: $0, countStyle: .file)))" } ?? ""
        print("  \(location) \(model.identifier.rawValue)\(size)")
    }
    
    // Try to load a model (will download if needed)
    let modelToLoad = ModelIdentifier.miniLM_L6_v2
    print("\n🔄 Loading model: \(modelToLoad.rawValue)")
    print("   This will download the model if not found locally...")
    
    do {
        // Create embedder (will download model if needed)
        let embedder = try await modelManager.createEmbedder(
            identifier: modelToLoad
        )
        
        print("\n✅ Model loaded successfully!")
        print("   Dimensions: \(embedder.dimensions)")
        
        // Test the embedder
        print("\n🧪 Testing embedder...")
        let testText = "EmbedKit makes it easy to use embedding models in Swift!"
        let embedding = try await embedder.embed(testText)
        
        print("   Generated embedding with \(embedding.dimensions) dimensions")
        print("   First 5 values: \(embedding.prefix(5).map { String(format: "%.4f", $0) })")
        
        // Test similarity
        let similarText = "Swift embedding models are simple with EmbedKit"
        let differentText = "The weather is nice today"
        
        let embedding2 = try await embedder.embed(similarText)
        let embedding3 = try await embedder.embed(differentText)
        
        let similarity1 = embedding.cosineSimilarity(with: embedding2)
        let similarity2 = embedding.cosineSimilarity(with: embedding3)
        
        print("\n📊 Similarity scores:")
        print("   Similar text: \(String(format: "%.4f", similarity1))")
        print("   Different text: \(String(format: "%.4f", similarity2))")
        
    } catch {
        print("\n❌ Error: \(error)")
    }
    
    // Demonstrate model search
    print("\n🔍 Searching for embedding models...")
    let registry = HuggingFaceModelRegistry()
    
    do {
        let searchResults = try await registry.searchModels(query: "embedding", limit: 5)
        
        print("\nTop 5 embedding models from Hugging Face:")
        for (index, result) in searchResults.enumerated() {
            print("  \(index + 1). \(result.modelId)")
            print("     Downloads: \(result.downloads.formatted()) | Likes: \(result.likes)")
        }
    } catch {
        print("   Search failed: \(error)")
    }
    
    // Show how to manage downloaded models
    print("\n📂 Model Management:")
    print("   Downloaded models are stored in: ~/Documents/EmbedKitModels/")
    
    let downloadedModels = try await modelManager.listAvailableModels()
        .filter { $0.location == DownloadedModelInfo.ModelLocation.downloaded }
    
    if !downloadedModels.isEmpty {
        print("\n   Downloaded models:")
        for model in downloadedModels {
            let size = model.size.map { ByteCountFormatter.string(fromByteCount: $0, countStyle: .file) } ?? "Unknown size"
            print("     - \(model.identifier.rawValue) (\(size))")
        }
        
        // Example of deleting a model (commented out to not actually delete)
        // try await modelManager.deleteModel(downloadedModels[0].identifier)
        // print("   ✅ Deleted model: \(downloadedModels[0].identifier.rawValue)")
    }
    
    print("\n✨ Example complete!")
}

// MARK: - Advanced Usage Example

public func advancedModelManagementExample() async throws {
    print("=== Advanced Model Management Example ===\n")
    
    // Create a custom download configuration
    let downloadConfig = ModelDownloader.DownloadConfiguration(
        maxRetries: 5,
        timeoutInterval: 600, // 10 minutes
        verifyChecksum: true
    )
    
    // Create model manager with custom configuration
    let modelManager = ModelManager(
        options: ModelManager.ManagerLoadingOptions(
            allowDownload: true,
            registry: HuggingFaceModelRegistry(),
            downloadConfiguration: downloadConfig,
            verifySignature: true
        )
    )
    
    // Download multiple models concurrently
    let modelsToDownload: [ModelIdentifier] = [
        .miniLM_L6_v2,
        ModelIdentifier(family: "bge", variant: "small-en", version: "v1.5"),
        ModelIdentifier(family: "gte", variant: "small", version: "v1")
    ]
    
    print("📥 Downloading multiple models concurrently...")
    
    await withTaskGroup(of: Result<String, Error>.self) { group in
        for modelId in modelsToDownload {
            group.addTask {
                do {
                    _ = try await modelManager.loadModel(modelId)
                    return .success("✅ \(modelId.rawValue)")
                } catch {
                    return .failure(error)
                }
            }
        }
        
        for await result in group {
            switch result {
            case .success(let message):
                print(message)
            case .failure(let error):
                print("❌ Failed: \(error)")
            }
        }
    }
    
    // Create embedders for all downloaded models
    print("\n🚀 Creating embedders for all models...")
    
    var embedders: [ModelIdentifier: CoreMLTextEmbedder] = [:]
    
    for modelId in modelsToDownload {
        do {
            let embedder = try await modelManager.createEmbedder(identifier: modelId)
            embedders[modelId] = embedder
            print("   ✅ Created embedder for \(modelId.rawValue)")
        } catch {
            print("   ❌ Failed to create embedder for \(modelId.rawValue): \(error)")
        }
    }
    
    // Compare embeddings from different models
    if embedders.count > 1 {
        print("\n🔬 Comparing embeddings from different models...")
        
        let testText = "Machine learning models for natural language processing"
        
        var embeddings: [(ModelIdentifier, EmbeddingVector)] = []
        
        for (modelId, embedder) in embedders {
            let embedding = try await embedder.embed(testText)
            embeddings.append((modelId, embedding))
            print("   \(modelId.rawValue): \(embedding.dimensions) dimensions")
        }
        
        // Compare similarities between models
        print("\n📊 Cross-model similarity matrix:")
        print("   (How similar are embeddings from different models for the same text)")
        
        for i in 0..<embeddings.count {
            for j in i+1..<embeddings.count {
                let (model1, emb1) = embeddings[i]
                let (model2, emb2) = embeddings[j]
                
                // Can only compare if dimensions match
                if emb1.dimensions == emb2.dimensions {
                    let similarity = emb1.cosineSimilarity(with: emb2)
                    print("   \(model1.family) ↔ \(model2.family): \(String(format: "%.4f", similarity))")
                } else {
                    print("   \(model1.family) ↔ \(model2.family): Different dimensions (\(emb1.dimensions) vs \(emb2.dimensions))")
                }
            }
        }
    }
    
    print("\n✨ Advanced example complete!")
}

// MARK: - Offline Mode Example

public func offlineModelUsageExample() async throws {
    print("=== Offline Model Usage Example ===\n")
    
    // Create model manager in offline mode (no downloads)
    let modelManager = ModelManager(
        options: ModelManager.ManagerLoadingOptions(
            allowDownload: false
        )
    )
    
    print("🔌 Running in offline mode (downloads disabled)")
    
    // List only locally available models
    let localModels = try await modelManager.listAvailableModels()
    
    if localModels.isEmpty {
        print("\n⚠️  No models found locally.")
        print("   In offline mode, you need to:")
        print("   1. Bundle models with your app, or")
        print("   2. Download models while online for offline use")
        return
    }
    
    print("\n📦 Found \(localModels.count) local model(s):")
    for model in localModels {
        let type = model.location == DownloadedModelInfo.ModelLocation.bundled ? "Bundled" : "Cached"
        print("   - \(model.identifier.rawValue) (\(type))")
    }
    
    // Use the first available model
    let modelToUse = localModels[0].identifier
    print("\n🎯 Using model: \(modelToUse.rawValue)")
    
    let embedder = try await modelManager.createEmbedder(
        identifier: modelToUse
    )
    
    // Demonstrate offline embedding generation
    let texts = [
        "Offline embedding generation",
        "No internet connection required",
        "Fast local inference"
    ]
    
    print("\n⚡ Generating embeddings offline...")
    
    for text in texts {
        let embedding = try await embedder.embed(text)
        print("   '\(text)' → \(embedding.dimensions)D vector")
    }
    
    print("\n✨ Offline usage complete!")
}
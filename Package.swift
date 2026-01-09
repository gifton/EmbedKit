// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

/*
 EmbedKit - High-Performance Text Embedding Framework for Apple Platforms

 Part of the Vector Suite Kit (VSK) ecosystem, EmbedKit provides:
 - On-device text embedding generation using CoreML
 - GPU-first vector storage and similarity search via VectorAccelerate
 - Metal4 GPU acceleration for all vector operations
 - Complete tokenization system (BERT, BPE, SentencePiece)
 - Production-ready with Swift 6 actor-based concurrency

 Products:
 - EmbedKit: Core library with CoreML + VectorAccelerate integration
 - EmbedKitONNX: Optional ONNX Runtime support for .onnx models

 Copyright (c) 2024-2025 Vector Suite Kit Contributors
 Licensed under MIT License
 */

import PackageDescription

let package = Package(
    name: "EmbedKit",
    platforms: [
        .macOS(.v26),
        .iOS(.v26),
        .tvOS(.v26),
        .custom("watchOS", versionString: "13.0"),
        .custom("visionOS", versionString: "3.0")
        // Requires macOS 26+/iOS 26+ for Metal 4 APIs
    ],
    products: [
        // Core library - CoreML only, minimal dependencies
        .library(
            name: "EmbedKit",
            targets: ["EmbedKit"]
        ),
        // Optional ONNX support - adds ~50-100MB binary size
        .library(
            name: "EmbedKitONNX",
            targets: ["EmbedKitONNX"]
        )
    ],
    dependencies: [
        // VSK dependencies - VectorAccelerate provides GPU-first indexing (v0.3.0+)
        .package(url: "https://github.com/gifton/VectorCore.git", from: "0.1.6"),
        // Using branch:main to get v0.3.4+ features with MetalCompilerPlugin support
        // (SPM doesn't allow stable versions to depend on branch-based packages)
        .package(url: "https://github.com/gifton/VectorAccelerate.git", branch: "main"),

        // System dependencies
        .package(url: "https://github.com/apple/swift-log.git", from: "1.5.0"),

        // MetalCompilerPlugin - generates debug.metallib with source for Xcode Metal Debugger
        .package(url: "https://github.com/schwa/MetalCompilerPlugin", branch: "main"),

        // ONNX Runtime - only used by EmbedKitONNX target
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager", from: "1.19.0"),
    ],
    targets: [
        // MARK: - Core EmbedKit (with VSK integration)
        .target(
            name: "EmbedKit",
            dependencies: [
                .product(name: "VectorCore", package: "VectorCore"),
                .product(name: "VectorAccelerate", package: "VectorAccelerate"),
                .product(name: "Logging", package: "swift-log")
            ],
            path: "Sources/EmbedKit",
            exclude: [
                // Metal sources: release via Scripts/CompileMetalShaders.sh, debug via MetalCompilerPlugin
                "Shaders",
                // Plugin config file not a resource
                "metal-compiler-plugin.json"
            ],
            resources: [
                .copy("Resources/vocab.txt"),
                .copy("Resources/MiniLM-L12-v2.mlpackage"),
                .copy("Resources/EmbedKitShaders.metallib")
            ],
            swiftSettings: [
                // Swift 6+ has strict concurrency enabled by default
            ],
            linkerSettings: [
                // Metal 4 tensor operations framework (iOS 26+ / macOS 26+)
                // Required for MTLTensor and tensor_ops in Metal shaders
                .linkedFramework("MetalPerformancePrimitives"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders")
            ],
            plugins: [
                // Generates debug.metallib with embedded source for Xcode Metal Debugger
                .plugin(name: "MetalCompilerPlugin", package: "MetalCompilerPlugin")
            ]
        ),

        // MARK: - EmbedKitONNX (optional ONNX support)
        .target(
            name: "EmbedKitONNX",
            dependencies: [
                "EmbedKit",
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
            ],
            path: "Sources/EmbedKitONNX",
            swiftSettings: [
                // Swift 6+ has strict concurrency enabled by default
            ]
        ),

        // MARK: - Tests
        .testTarget(
            name: "EmbedKitTests",
            dependencies: ["EmbedKit"],
            linkerSettings: [
                // Metal 4 frameworks for tensor tests
                .linkedFramework("MetalPerformancePrimitives"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders")
            ]
        ),
        .testTarget(
            name: "EmbedKitONNXTests",
            dependencies: ["EmbedKitONNX", "EmbedKit"]
        ),
    ]
)

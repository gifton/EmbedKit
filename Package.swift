// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

/*
 EmbedKit - High-Performance Text Embedding Framework for Apple Platforms

 Part of the Vector Suite Kit (VSK) ecosystem, EmbedKit provides:
 - On-device text embedding generation using CoreML
 - Type-safe embedding types with compile-time dimension verification
 - GPU acceleration via Metal compute shaders
 - Complete tokenization system (BERT, BPE, SentencePiece)
 - Production-ready with Swift 6 actor-based concurrency

 Copyright (c) 2024 Vector Suite Kit Contributors
 Licensed under MIT License
 */

import PackageDescription

let package = Package(
    name: "EmbedKit",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .tvOS(.v17),
        .watchOS(.v10),
        .visionOS(.v1)
    ],
    products: [
        .library(
            name: "EmbedKit",
            targets: ["EmbedKit"]
        ),
    ],
    dependencies: [
        // VSK dependencies - Official releases
        .package(url: "https://github.com/gifton/VectorCore.git", from: "0.1.4"),
        .package(url: "https://github.com/gifton/VectorIndex.git", from: "0.1.1"),
        // .package(path: "../VectorAccelerate"),  // Temporarily disabled due to Swift 6 concurrency issues

        // System dependencies
        .package(url: "https://github.com/apple/swift-log.git", from: "1.5.0"),
    ],
    targets: [
        .target(
            name: "EmbedKit",
            dependencies: [
                .product(name: "VectorCore", package: "VectorCore"),
                .product(name: "VectorIndex", package: "VectorIndex"),
                // .product(name: "VectorAccelerate", package: "VectorAccelerate"),  // Temporarily disabled
                .product(name: "Logging", package: "swift-log"),
            ],
            exclude: [
                // Exclude Metal shader source files (using precompiled metallib instead)
                "Shaders/Kernels",
                "Shaders/README.md"
            ],
            resources: [
                // Precompiled Metal shader library
                // Compile shaders with: ./Scripts/CompileMetalShaders.sh
                .process("Resources/EmbedKitShaders.metallib")
            ],
            swiftSettings: [
                .enableUpcomingFeature("BareSlashRegexLiterals"),
                .enableUpcomingFeature("ConciseMagicFile"),
                .enableUpcomingFeature("ForwardTrailingClosures"),
                .enableUpcomingFeature("ImplicitOpenExistentials"),
                .enableUpcomingFeature("DisableOutwardActorInference"),
                .enableExperimentalFeature("StrictConcurrency"),
            ]
        ),
        .testTarget(
            name: "EmbedKitTests",
            dependencies: ["EmbedKit"]
        ),
        .testTarget(
            name: "EmbedKitBenchmarks",
            dependencies: ["EmbedKit"],
            path: "Tests/EmbedKitBenchmarks",
            swiftSettings: [
                .enableUpcomingFeature("BareSlashRegexLiterals"),
                .enableUpcomingFeature("ConciseMagicFile"),
                .enableUpcomingFeature("ForwardTrailingClosures"),
                .enableUpcomingFeature("ImplicitOpenExistentials"),
                .enableUpcomingFeature("DisableOutwardActorInference"),
                .enableExperimentalFeature("StrictConcurrency"),
            ]
        ),
    ]
)

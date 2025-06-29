// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "EmbedKit",
    platforms: [
        .iOS(.v17),      // iOS 17+ for latest features
        .macOS(.v13),    // macOS 13+ for Metal 3 support
        .tvOS(.v17),     // tvOS 17+ for consistency with iOS
        .watchOS(.v10),  // watchOS 10+ for consistency
        .visionOS(.v1)   // visionOS for future AR/VR embeddings
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "EmbedKit",
            targets: ["EmbedKit"]),
    ],
    dependencies: [
        // Local dependency on PipelineKit
        .package(path: "../PipelineKit"),
        // For structured logging
        .package(url: "https://github.com/apple/swift-log.git", from: "1.5.0"),
        // For collections utilities
        .package(url: "https://github.com/apple/swift-collections.git", from: "1.0.0"),
        // For async algorithms
        .package(url: "https://github.com/apple/swift-async-algorithms.git", from: "1.0.0"),
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "EmbedKit",
            dependencies: [
                .product(name: "PipelineKit", package: "PipelineKit"),
                .product(name: "Logging", package: "swift-log"),
                .product(name: "Collections", package: "swift-collections"),
                .product(name: "AsyncAlgorithms", package: "swift-async-algorithms"),
            ],
            exclude: ["PipelineIntegration/README.md", "Examples/README.md"]
        ),
        .testTarget(
            name: "EmbedKitTests",
            dependencies: ["EmbedKit"]
        ),
    ]
)

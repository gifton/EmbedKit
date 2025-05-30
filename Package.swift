// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "EmbedKit",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
        .tvOS(.v16),
        .watchOS(.v9)
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "EmbedKit",
            targets: ["EmbedKit"]),
    ],
    dependencies: [
        // Local dependency on PipelineKit - temporarily disabled
        // .package(path: "../PipelineKit"),
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
                // .product(name: "PipelineKit", package: "PipelineKit"), // Temporarily disabled
                .product(name: "Logging", package: "swift-log"),
                .product(name: "Collections", package: "swift-collections"),
                .product(name: "AsyncAlgorithms", package: "swift-async-algorithms"),
            ]
        ),
        .testTarget(
            name: "EmbedKitTests",
            dependencies: ["EmbedKit"]
        ),
    ]
)

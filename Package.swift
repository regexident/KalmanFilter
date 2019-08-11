// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "KalmanFilter",
    products: [
        // Products define the executables and libraries produced by a package, and make them visible to other packages.
        .library(
            name: "KalmanFilter",
            targets: [
                "KalmanFilter",
            ]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/regexident/BayesFilter", .branch("master")),
        .package(url: "https://github.com/regexident/Surge.git", .branch("development")),
        // .package(url: "https://github.com/mattt/Surge.git", .upToNextMajor(from: "2.0.0")),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages which this package depends on.
        .target(
            name: "KalmanFilter",
            dependencies: [
                "BayesFilter",
                "Surge",
            ]
        ),
        .testTarget(
            name: "KalmanFilterTests",
            dependencies: [
                "KalmanFilter",
            ]
        ),
    ]
)

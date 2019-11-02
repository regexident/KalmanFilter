// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "KalmanFilter",
    products: [
        .library(
            name: "KalmanFilter",
            targets: [
                "KalmanFilter",
            ]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/regexident/BayesFilter", .branch("master")),
        .package(url: "https://github.com/regexident/StateSpaceModel", .branch("master")),
        .package(url: "https://github.com/jounce/Surge", from: "2.3.0"),
    ],
    targets: [
        .target(
            name: "KalmanFilter",
            dependencies: [
                "BayesFilter",
                "StateSpaceModel",
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

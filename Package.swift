// swift-tools-version: 6.2
// flux2-klein-swift — Swift/MLX port of Black Forest Labs FLUX.2-klein-4B (Apache-2.0):
// a compact rectified-flow MMDiT (5 double-stream + 20 single-stream blocks) + Qwen3-4B
// 3-layer-tap conditioner + FLUX.2 VAE. Ships MLXEngine `textToImage` with unified
// multi-reference editing. Oracle = diffusers (PyTorch) goldens + mflux (Python-MLX) render.
// See PORTING-SPEC.md — phases gate on fp32/CPU goldens; never advance on a red gate.

import PackageDescription

let package = Package(
    name: "Klein",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "Klein", targets: ["Klein"]),
        .library(name: "MLXKlein", targets: ["MLXKlein"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "3.31.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        // FLUX.2 VAE — neutral in-house package shared with Lens/ERNIE; net dep, not re-ported.
        .package(url: "https://github.com/xocialize/flux2-vae-mlx-swift", from: "0.1.0"),
        .package(url: "https://github.com/xocialize/mlx-profiling.git", from: "0.1.0"),
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.21.0"),
    ],
    targets: [
        .target(
            name: "Klein",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Flux2VAE", package: "flux2-vae-mlx-swift"),
                .product(name: "MLXProfiling", package: "mlx-profiling"),
            ],
            path: "Sources/Klein"
        ),
        .target(
            name: "MLXKlein",
            dependencies: [
                "Klein",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXProfiling", package: "mlx-profiling"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/MLXKlein"
        ),
        .executableTarget(
            name: "klein-cli",
            dependencies: [
                "Klein", "MLXKlein",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            ],
            path: "Sources/KleinCLI"
        ),
        .testTarget(
            name: "KleinTests",
            dependencies: ["Klein"],
            path: "Tests/KleinTests"
        ),
        .testTarget(
            name: "MLXKleinTests",
            dependencies: [
                "MLXKlein",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ],
            path: "Tests/MLXKleinTests"
        ),
    ]
)

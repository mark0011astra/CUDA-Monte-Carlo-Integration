# Monte Carlo Integration with GPU Acceleration

This repository provides an implementation of Monte Carlo integration using GPU acceleration via CUDA and the `numba` library. The code is designed to integrate arbitrary functions over specified multidimensional bounds, and includes both CPU and GPU kernels for performing computations. The visualization of the sample points and the computed integral value is also supported using `matplotlib`.

## Overview

Monte Carlo integration is a method for approximating the integral of a function by random sampling. This method is especially useful in high-dimensional spaces where traditional numerical integration techniques may be inefficient. In this project, we accelerate the integration process using CUDA-enabled GPUs.

### Key Features
- **GPU Acceleration**: Offloads computationally expensive tasks to the GPU using `numba.cuda`.
- **Flexible Dimensionality**: Supports integration over arbitrary-dimensional spaces.
- **Error Handling**: Includes boundary and dimension checks to ensure valid input.
- **Visualization**: Provides functions for visualizing the distribution of sample points and the corresponding integration results.

## Requirements

To use this code, you will need the following:
- Python 3.x
- CUDA-enabled GPU and the CUDA toolkit
- Numba library (`numba`)
- Matplotlib for visualization (`matplotlib`)
- Numpy for numerical operations (`numpy`)

### Installation
You can install the required packages using the following command:

```
pip install numba matplotlib numpy
```

Ensure that your system has the CUDA toolkit installed and correctly configured for GPU operations. You can verify CUDA installation using:

```
nvcc --version
```

## Code Structure

### `monte_carlo_integration(func, dim, bounds, total_samples, batch_size)`

This is the main function to compute the Monte Carlo integration over a specified function. It accepts the following arguments:

- `func`: The function to be integrated, defined in a GPU-compatible form using `@cuda.jit(device=True)`.
- `dim`: The number of dimensions over which to integrate.
- `bounds`: A list of tuples defining the integration bounds for each dimension.
- `total_samples`: Total number of random samples to generate for the integration.
- `batch_size`: Number of samples to process in each GPU batch.

This function returns the following:

- `integral`: The computed integral value.
- `samples_list`: A list of sample points generated during the integration process.
- `results_list`: A list of the corresponding function evaluations at the sampled points.

### `visualize_samples(samples_list, results_list, bounds, sample_fraction, max_samples)`

This function visualizes the samples generated during the Monte Carlo integration, providing insights into how the random points are distributed over the integration space.

- `samples_list`: The list of sample points.
- `results_list`: The corresponding function evaluations at the sampled points.
- `bounds`: The integration bounds for each dimension.
- `sample_fraction`: A fraction of the total samples to visualize. Default is `None`, which adjusts based on `max_samples`.
- `max_samples`: Maximum number of samples to display.

### Example Usage

The following code demonstrates how to use the Monte Carlo integration function with a Gaussian function over a two-dimensional space:

```python
@cuda.jit(device=True)
def gaussian_function(x):
    total = 0.0
    for i in range(len(x)):
        total += x[i] ** 2
    return math.exp(-total)

dim = 2
bounds = [(-5, 5)] * dim

result, samples_list, results_list = monte_carlo_integration(gaussian_function, dim, bounds, total_samples=10**6, batch_size=10**5)

print(f"Calculated integral: {result}")
visualize_samples(samples_list, results_list, bounds, sample_fraction=None, max_samples=100000)
```

This example performs Monte Carlo integration of the Gaussian function over a two-dimensional space with boundaries \([-5, 5]\). The result of the integral is displayed, and a plot of the sampled points is generated for visualization.

## Performance Considerations

The code leverages CUDA to perform parallel computations on the GPU, significantly speeding up the integration process, especially in high-dimensional spaces. The batch processing mechanism allows for efficient handling of large sample sizes, ensuring optimal GPU utilization. Users can adjust the `batch_size` to match their system's memory capacity.

## Error Handling

- **Dimension Check**: The dimension (`dim`) must be greater than 0.
- **Boundary Check**: The bounds must be provided as tuples with two elements for each dimension.
- **CUDA Compatibility**: Ensure that the provided function is compatible with CUDA by using the `@cuda.jit(device=True)` decorator.

## Visualization

The `visualize_samples` function allows users to visualize the distribution of the sample points in the integration space. This is useful for verifying that the Monte Carlo method has sampled the space uniformly and that the function evaluations align with expected values.

## Future Work

Possible future enhancements include:
- **Higher-dimensional visualization**: Extending visualization to work in more than two dimensions.
- **Adaptive sampling**: Implementing techniques to adaptively sample regions with higher variance for improved accuracy.
- **Error estimation**: Including error estimation methods to quantify the uncertainty in the Monte Carlo results.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

# GPUアクセラレーションによるモンテカルロ積分

このリポジトリは、CUDAと`numba`ライブラリを用いたGPUアクセラレーションによるモンテカルロ積分の実装を提供します。このコードは、指定された多次元の範囲にわたって任意の関数を積分するもので、計算を行うためにCPUおよびGPUカーネルの両方をサポートしています。また、サンプル点の分布と計算された積分結果の可視化もサポートしています。

## 概要

モンテカルロ積分は、ランダムサンプリングによって関数の積分を近似する手法です。この手法は、特に高次元空間において伝統的な数値積分法が効率的に機能しない場合に有効です。本プロジェクトでは、CUDA対応のGPUを使用して積分プロセスを加速させます。

### 主な特徴
- **GPUアクセラレーション**: 計算負荷の高い処理を`numba.cuda`を用いてGPUにオフロードします。
- **柔軟な次元対応**: 任意の次元空間における積分をサポートします。
- **エラーハンドリング**: 入力の有効性を確保するための境界チェックと次元チェックを含んでいます。
- **可視化機能**: サンプル点の分布とそれに対応する積分結果を可視化する機能を提供します。

## 必要条件

このコードを使用するには、以下の環境が必要です：
- Python 3.x
- CUDA対応のGPUとCUDAツールキット
- Numbaライブラリ (`numba`)
- 可視化用のMatplotlib (`matplotlib`)
- 数値計算用のNumpy (`numpy`)

### インストール
必要なパッケージは、以下のコマンドでインストールできます。

```
pip install numba matplotlib numpy
```

システムにCUDAツールキットがインストールされ、GPU操作用に正しく設定されていることを確認してください。CUDAのインストール状況は、次のコマンドで確認できます。

```
nvcc --version
```

## コード構成

### `monte_carlo_integration(func, dim, bounds, total_samples, batch_size)`

これは、指定された関数のモンテカルロ積分を計算するためのメイン関数です。以下の引数を受け取ります：

- `func`: GPU互換形式で定義された、積分対象の関数 (`@cuda.jit(device=True)` デコレータを使用)。
- `dim`: 積分を行う空間の次元数。
- `bounds`: 各次元における積分範囲を定義するタプルのリスト。
- `total_samples`: 積分のために生成されるランダムサンプルの総数。
- `batch_size`: GPUで1バッチごとに処理されるサンプル数。

この関数は次の結果を返します：

- `integral`: 計算された積分値。
- `samples_list`: 積分処理中に生成されたサンプル点のリスト。
- `results_list`: 各サンプル点で評価された関数の結果のリスト。

### `visualize_samples(samples_list, results_list, bounds, sample_fraction, max_samples)`

モンテカルロ積分中に生成されたサンプル点を可視化する関数で、ランダムにサンプルされた点が積分空間内にどのように分布しているかを確認できます。

- `samples_list`: サンプル点のリスト。
- `results_list`: 各サンプル点で評価された関数の結果。
- `bounds`: 各次元における積分範囲。
- `sample_fraction`: 可視化するサンプルの割合。デフォルトは`None`で、`max_samples`に基づいて調整されます。
- `max_samples`: 表示するサンプルの最大数。

### 使用例

以下は、2次元空間におけるガウス関数を用いたモンテカルロ積分の使用例です。

```python
@cuda.jit(device=True)
def gaussian_function(x):
    total = 0.0
    for i in range(len(x)):
        total += x[i] ** 2
    return math.exp(-total)

dim = 2
bounds = [(-5, 5)] * dim

result, samples_list, results_list = monte_carlo_integration(gaussian_function, dim, bounds, total_samples=10**6, batch_size=10**5)

print(f"Calculated integral: {result}")
visualize_samples(samples_list, results_list, bounds, sample_fraction=None, max_samples=100000)
```

この例では、境界\([-5, 5]\)の2次元空間におけるガウス関数のモンテカルロ積分を行います。積分結果が表示され、サンプル点の可視化も行われます。

## パフォーマンスの考慮事項

このコードはCUDAを使用して計算を並列化し、特に高次元空間における積分プロセスを大幅に高速化します。バッチ処理メカニズムにより、大量のサンプル数を効率的に処理し、GPUの最適な利用を実現しています。`batch_size`を調整することで、システムのメモリ容量に合わせた最適化が可能です。

## エラーハンドリング

- **次元チェック**: 次元数(`dim`)は1以上である必要があります。
- **境界チェック**: 各境界は2要素を持つタプル形式で指定する必要があります。
- **CUDA互換性**: 提供される関数は`@cuda.jit(device=True)`デコレータを用いてCUDA互換形式である必要があります。

## 可視化

`visualize_samples`関数を使用することで、積分空間におけるサンプル点の分布を視覚的に確認できます。これにより、モンテカルロ法が空間全体を均等にサンプリングできているか、関数の評価が予想通りであるかを確認するのに役立ちます。

## 今後の課題

今後の拡張の可能性として、以下が考えられます：
- **高次元の可視化**: 2次元を超える次元での可視化のサポート。
- **適応的サンプリング**: 分散の大きい領域を適応的にサンプリングする手法の導入。
- **誤差推定**: モンテカルロ法の結果の不確実性を定量化する誤差推定方法の追加。

## ライセンス

本プロジェクトはMITライセンスの下で公開されています。詳細については、[LICENSE](LICENSE)ファイルを参照してください。

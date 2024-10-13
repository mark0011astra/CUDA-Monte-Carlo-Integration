import numpy as np
from numba import cuda, float64, jit
import math
import matplotlib.pyplot as plt

# 任意の関数をGPU上で使える形式に事前に変換する (日本語コメント)
# Convert any function into a format usable on the GPU (English comment)
def monte_carlo_integration(func, dim, bounds, total_samples=10**6, batch_size=10**5):
    # 境界チェックとエラー処理 (Boundary check and error handling)
    if dim <= 0:
        raise ValueError("次元は1以上でなければなりません (Dimension must be greater than 0)")
    if not all(isinstance(b, tuple) and len(b) == 2 for b in bounds):
        raise ValueError("各境界は2つの要素を持つタプルでなければなりません (Each bound must be a tuple of two values)")

    n_batches = total_samples // batch_size
    vol = np.prod([b[1] - b[0] for b in bounds])  # 積分範囲の体積 (Volume of the integration range)

    cumulative_result = 0.0  # 累積結果 (Cumulative result)
    samples_list = []  # 可視化用サンプルリスト (Sample list for visualization)
    results_list = []  # 可視化用結果リスト (Results list for visualization)

    # GPUカーネルで使うための関数を定義する (Define function for GPU kernel)
    @cuda.jit(device=True)
    def gpu_func(x):
        return func(x)

    # サンプルの生成と積分計算 (Generate samples and compute integration)
    for batch in range(n_batches):
        # 各次元でランダムサンプルを生成 (Generate random samples within the bounds)
        samples = np.array([np.random.uniform(b[0], b[1], batch_size) for b in bounds]).T
        samples_list.append(samples)

        # GPUに転送 (Transfer to GPU)
        samples_device = cuda.to_device(samples)
        results_device = cuda.device_array(batch_size, dtype=np.float64)

        # GPUカーネル (GPU kernel to evaluate the function)
        @cuda.jit
        def kernel(samples, results, dim):
            idx = cuda.grid(1)
            if idx < samples.shape[0]:
                x = samples[idx]
                results[idx] = gpu_func(x)

        # ブロックとグリッドの設定 (Setting up threads and blocks)
        threads_per_block = 256
        blocks_per_grid = (batch_size + (threads_per_block - 1)) // threads_per_block

        # カーネルの実行 (Launch kernel)
        kernel[blocks_per_grid, threads_per_block](samples_device, results_device, dim)

        # 結果をCPUにコピー (Copy result to CPU)
        results = results_device.copy_to_host()

        # 結果を累積 (Accumulate the result)
        cumulative_result += np.sum(results)
        results_list.append(results)

    # 総サンプル数に基づいて積分値を計算 (Compute the final integral based on total samples)
    integral = vol * (cumulative_result / total_samples)
    
    return integral, samples_list, results_list

# 可視化関数: 各サンプルの状態と積分範囲をプロット (Visualization function: Plot the samples and integration range)
def visualize_samples(samples_list, results_list, bounds, sample_fraction=None, max_samples=50000):
    plt.figure(figsize=(8, 8))

    # 各バッチのサンプルをプロット (Plot samples for each batch)
    total_samples = sum(len(samples) for samples in samples_list)
    if sample_fraction is None:
        sample_fraction = min(1.0, max_samples / total_samples)

    combined_samples = np.vstack(samples_list)
    combined_results = np.hstack(results_list)

    n_samples = len(combined_samples)
    indices = np.random.choice(n_samples, int(n_samples * sample_fraction), replace=False)
    
    plt.scatter(combined_samples[indices, 0], combined_samples[indices, 1], c=combined_results[indices], cmap='coolwarm', s=0.5, alpha=0.5)

    plt.xlim(bounds[0][0], bounds[0][1])
    plt.ylim(bounds[1][0], bounds[1][1])

    plt.colorbar(label='Integral value')
    plt.title('Monte Carlo Integration: Sample Distribution and Integral Results')
    plt.xlabel('X dimension')
    plt.ylabel('Y dimension')
    plt.grid(True)
    plt.show()

# 使い方: 例としてガウス関数を定義して積分に使用 (Example usage: Define Gaussian function for integration)
@cuda.jit(device=True)
def gaussian_function(x):
    total = 0.0
    for i in range(len(x)):
        total += x[i] ** 2
    return math.exp(-total)

# 次元数と積分範囲の設定 (Define dimensions and bounds for integration)
dim = 2
bounds = [(-5, 5)] * dim

# 積分の実行とサンプルデータ取得 (Execute integration and retrieve sample data)
result, samples_list, results_list = monte_carlo_integration(gaussian_function, dim, bounds, total_samples=10**6, batch_size=10**5)

# 結果を表示 (Display results)
print(f"積分値：{result} (Calculated integral: {result})")

# サンプルの可視化 (Visualize the samples)
visualize_samples(samples_list, results_list, bounds, sample_fraction=None, max_samples=100000)

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

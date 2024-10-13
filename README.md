# Monte Carlo Integration using CUDA with Python

This repository contains a Python script to perform Monte Carlo integration on a GPU using CUDA. It demonstrates how to integrate a multidimensional function using random sampling, leveraging GPU acceleration with Numba's CUDA interface. The example script includes a Gaussian function integration and visualizes the results.

## Requirements

To run this script, you need the following libraries:

- `numpy`
- `numba`
- `matplotlib`
- `cuda-enabled GPU`
- `CUDA Toolkit`

You can install the required Python libraries with:

```
pip install numpy numba matplotlib
```

## How to Use

The main function `monte_carlo_integration` performs the integration. It takes a user-defined function, dimension, integration bounds, and number of samples, then evaluates the function on a GPU.

Here’s how you can use the script:

1. Define the function you want to integrate, making sure it's compatible with CUDA (for example, the Gaussian function in the script).
2. Set the dimensions and bounds of the integration.
3. Call the `monte_carlo_integration` function to compute the integral and visualize the sample distribution.

### Example

The script provides an example using a Gaussian function in 2 dimensions:

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
print(f"積分値：{result} (Calculated integral: {result})")
```

### Visualization

You can also visualize the sample distribution and the function values over the integration space using the `visualize_samples` function:

```python
visualize_samples(samples_list, results_list, bounds, sample_fraction=None, max_samples=100000)
```

This will generate a scatter plot of the sampled points, with colors representing the computed function values.

## Customization

You can customize the following parameters in the `monte_carlo_integration` function:

- **`dim`**: The number of dimensions for the integration.
- **`bounds`**: A list of tuples specifying the lower and upper bounds for each dimension.
- **`total_samples`**: The total number of random samples to use for integration.
- **`batch_size`**: Number of samples processed per batch on the GPU.

The `visualize_samples` function can also be customized by adjusting the number of samples displayed and the fraction of total samples to visualize.

## License

This project is licensed under the MIT License.

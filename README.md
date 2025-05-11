# Plavchan-GPU

A GPU-accelerated implementation of the Plavchan periodogram for time series analysis in astronomy.

## Installation

```bash
pip install plavchan_gpu
```

### Requirements

- CUDA-capable GPU and CUDA toolkit > 11.0
- Python 3.6+

## Usage

```python
from plavchan_gpu import plavchan_periodogram

# Example data
mags = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # List of magnitude lists for each object
times = [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]  # List of time lists for each object
trial_periods = [0.1, 0.2, 0.3, 0.4, 0.5]   # List of trial periods

# Calculate periodogram
result = plavchan_periodogram(mags, times, trial_periods, width=0.1)

# result will be a list of periodogram values for each object and trial period
```

## License

MIT
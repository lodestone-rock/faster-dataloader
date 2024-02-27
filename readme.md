## Readme for Threading-based Dataloader

This `threading_dataloader`, which utilizes multithreading to improve data loading performance compared to the standard PyTorch dataloader on IO heavy workload.

### Usage

```python
from threading_dataloader import threading_dataloader

# Example usage
dataset = # Your dataset object
batch_size = 32
num_workers = 4

dataloader = threading_dataloader(dataset, batch_size=batch_size, num_workers=num_workers)

for batch in dataloader:
    # Process your batch data here
    # ...
```

### Parameters:

- **dataset (iterable):** The dataset object to load data from.
- **batch_size (int, optional):** (Default: 1) The number of samples per batch.
- **num_workers (int, optional):** (Default: 10) The number of worker threads to use for parallel loading.
- **collate_fn (callable, optional):** A function to collate individual samples into a batch. If None, the default collate function is used.
- **shuffle (bool, optional):** (Default: False) Whether to shuffle the dataset before loading.
- **prefetch_factor (int, optional):** (Default: 4) The number of batches to prefetch in advance.
- **seed (int, optional):** (Default: 0) The seed for the random number generator used for shuffling.
- **timeout (int, optional):** The maximum number of seconds to wait for a batch when using a prefetch queue. 

### Notes:

- This implementation utilizes `ThreadPoolExecutor` from the `concurrent.futures` module for multithreading.
- It provides an option for prefetching batches asynchronously using a queue, which can improve performance for slow data loading operations.
- Be mindful of setting an appropriate `num_workers` value based on your hardware and dataset size. Too many workers can lead to overhead and decreased performance.
- Consider using `timeout` to avoid blocking the main thread indefinitely if data loading encounters issues.

Please refer to the code comments for further details and clarifications.
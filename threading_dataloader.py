from concurrent.futures import ThreadPoolExecutor
import random
from queue import Queue

def threading_dataloader(dataset, batch_size=1, num_workers=10, collate_fn=None, shuffle=False, prefetch_factor=4, seed=0, timeout=None):
    """
    A function to load data using multiple threads. This function can be used to speed up the data loading process. 
    
    Parameters:
    dataset (iterable): The dataset to load.
    batch_size (int, optional): The number of samples per batch. Defaults to 1.
    num_workers (int, optional): The number of worker threads to use. Defaults to 10.
    collate_fn (callable, optional): A function to collate samples into a batch. If None, the default collate_fn is used.
    shuffle (bool, optional): Whether to shuffle the dataset before loading. Defaults to False.
    prefetch_factor (int, optional): The number of batches to prefetch. Defaults to 4.
    seed (int, optional): The seed for the random number generator. Defaults to 0.
    timeout (int, optional): The maximum number of seconds to wait for a batch. If None, there is no timeout.

    Yields:
    object: A batch of data.
    """
    # Initialize a random number generator with the given seed
    random.seed(seed)

    # Create a ThreadPoolExecutor with the specified number of workers
    workers = ThreadPoolExecutor(max_workers=num_workers)

    # Generate batches of indices based on the dataset size and batch size
    num_samples = len(dataset)
    batch_indices = [list(range(i, min(i + batch_size, num_samples))) for i in range(0, num_samples, batch_size)]
    if shuffle:
        indices = list(range(num_samples))
        random.shuffle(indices)
        batch_indices = [indices[i:i+batch_size] for i in range(0, num_samples, batch_size)]

    # Create a queue to store prefetched batches
    prefetch_queue = Queue(maxsize=prefetch_factor * num_workers)
    
    # Function to prefetch batches of samples
    def batch_to_queue(indices):
        """
        Function to load a batch of data and put it into the prefetch queue.

        Parameters:
        indices (list): The indices of the samples in the batch.
        """
        batch = [dataset[i] for i in indices]
        if collate_fn is not None:
            batch = collate_fn(batch)
        prefetch_queue.put(batch)
    
    # Submit the prefetch tasks to the worker threads
    for indices in batch_indices:
        workers.submit(batch_to_queue, indices)

    # Yield the prefetched batches
    for _ in range(len(batch_indices)):
        yield prefetch_queue.get(timeout=timeout)

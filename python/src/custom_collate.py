import torch


def custom_collate_fn(batch):
    """
    Custom collate function to handle batches with varying tensor sizes and filter out invalid samples.

    Args:
        batch (list): List of tensors.

    Returns:
        torch.Tensor: Batch of tensors with uniform dimensions.
    """
    # Filter out None values from the batch
    batch = [item for item in batch if item is not None]
    
    # If the batch is empty after filtering, return an empty tensor
    if len(batch) == 0:
        return torch.empty(0)

    # Find the minimum number of examples across all items in the batch to ensure uniform size
    min_size = min([b.shape[0] for b in batch])
    
    # Truncate each item in the batch to the minimum size
    batch = [b[:min_size] for b in batch]

    # Stack the truncated items into a single tensor
    return torch.stack(batch)

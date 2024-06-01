import torch


def custom_collate_fn(batch):
    """
    Custom collate function to handle batches with varying tensor sizes and filter out invalid samples.

    Args:
        batch (list): List of tensors.

    Returns:
        torch.Tensor: Batch of tensors with uniform dimensions.
    """
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return torch.empty(0)

    min_size = min([b.shape[0] for b in batch])
    batch = [b[:min_size] for b in batch]

    return torch.stack(batch)

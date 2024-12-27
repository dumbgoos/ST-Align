import torch

def get_positive_and_negatives(index, cluster_test):
    """
    Given an index, randomly samples a positive example from the same class (excluding itself),
    and randomly samples negative examples from each of the other classes.

    Args:
        index (int): The index of the original sample.
        cluster_test (list or Tensor): List or tensor of labels for each data point.

    Returns:
        pos_index (int): The index of the positive example.
        negative_indices (list): List of indices for negative examples.
    """
    # Convert inputs to tensors if they aren't already
    if not isinstance(cluster_test, torch.Tensor):
        cluster_test = torch.tensor(cluster_test)

    current_label = cluster_test[index]

    # Get unique labels excluding the current label
    labels = torch.unique(cluster_test)
    labels = labels[labels != current_label]

    negative_indices = []
    for label in labels:
        # Get all indices of the current label
        label_indices = torch.where(cluster_test == label)[0]
        # Randomly select one index from label_indices
        rand_idx = torch.randint(high=len(label_indices), size=(1,)).item()
        neg_index = label_indices[rand_idx].item()
        negative_indices.append(neg_index)

    return negative_indices

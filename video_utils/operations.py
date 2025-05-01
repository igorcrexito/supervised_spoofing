
def normalize_array(array):
    """
    Normalize the array values to be in the range [0, 1] for each channel.
    """
    # Calculate the min and max along the channel dimension
    min_val = array.min(axis=(0, 1), keepdims=True)
    max_val = array.max(axis=(0, 1), keepdims=True)
    # Normalize values to [0, 1]
    normalized = (array - min_val) / (max_val - min_val + 1e-8)  # Add small epsilon to avoid division by zero
    return normalized


import numpy as np

# Quantize the continuous features
# Make the mapping function
def quantize(columns: np.array, bin_width: int, n_bins: int):
    """
    Quantize into bins but keep some notion of locality sensitivity
    """
    offsets = np.arange(n_bins) * (bin_width // n_bins)
    columns = np.reshape(columns, columns.shape + (1,))
    # print(np.repeat(columns, n_bins, axis=2) + offsets)
    quantized = (np.repeat(columns, n_bins, axis=2) + offsets) // bin_width
    
    return np.reshape(quantized, (columns.shape[0], -1))

def separate_domains(columns: np.array):
    """
    Separate domains by interleaving them (instead of shifting by domain ranges).
    This makes it range-agnostic
    """
    assert(len(columns.shape) == 2)
    n_domains = columns.shape[1]
    idx_shifts = np.arange(n_domains)
    return columns * n_domains + idx_shifts

print(quantize(np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]), 6, 6))
print(separate_domains(quantize(np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]), np.array([2, 2]), 2)))
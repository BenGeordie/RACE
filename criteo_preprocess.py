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

def separate_domains_by_range(columns: np.array):
    """
    Separate domains by shifting columns by the cumulative range of previous columns.
    """
    shifts = np.cumsum(np.max(columns, axis=0) - np.min(columns, axis=0))
    return columns + shifts

def separate_domains(columns: np.array):
    """
    Separate domains by interleaving them (instead of shifting by domain ranges).
    This makes it range-agnostic
    """
    assert(len(columns.shape) == 2)
    n_domains = columns.shape[1]
    idx_shifts = np.arange(n_domains)
    return columns * n_domains + idx_shifts

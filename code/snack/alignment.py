import numpy as np
import torch
from functools import lru_cache

try:
    # Try to import numba for performance optimization
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print(
        "Numba not found. The Needleman-Wunsch algorithm will run in native Python mode."
    )


def needleman_wunsch(seq1, seq2, metric_func):
    """Optimized Needleman-Wunsch algorithm for sequence alignment"""
    len1 = len(seq1)
    len2 = len(seq2)

    # Pre-compute metric values to avoid redundant calculations
    # This significantly reduces computational overhead for sequence alignment
    @lru_cache(maxsize=None)
    def cached_metric(a, b):
        # Convert tensor result to scalar for numpy operations
        result = metric_func(a, b)
        if isinstance(result, torch.Tensor):
            return result.item()
        return result

    # Pre-compute all needed metric values
    metric_values = {}

    # Cache gap penalties
    gap_penalty = cached_metric("-", "-")

    # For seq1 to gap
    for i in range(len1):
        key = (seq1[i], "-")
        metric_values[key] = cached_metric(seq1[i], "-")

    # For gap to seq2
    for j in range(len2):
        key = ("-", seq2[j])
        metric_values[key] = cached_metric("-", seq2[j])

    # For all pairs of characters
    for i in range(len1):
        for j in range(len2):
            key = (seq1[i], seq2[j])
            metric_values[key] = cached_metric(seq1[i], seq2[j])

    # Use optimized implementation based on available libraries
    if NUMBA_AVAILABLE:
        return _nw_numba(seq1, seq2, metric_values)
    else:
        return _nw_numpy(seq1, seq2, metric_values)


def _nw_numpy(seq1, seq2, metric_values):
    """NumPy implementation of Needleman-Wunsch"""
    len1 = len(seq1)
    len2 = len(seq2)

    # Initialization
    dp = np.zeros((len1 + 1, len2 + 1))

    # Fill first row and column
    for i in range(1, len1 + 1):
        dp[i][0] = dp[i - 1][0] + metric_values.get((seq1[i - 1], "-"), 0)

    for j in range(1, len2 + 1):
        dp[0][j] = dp[0][j - 1] + metric_values.get(("-", seq2[j - 1]), 0)

    # Fill the DP matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            match = dp[i - 1][j - 1] + metric_values.get((seq1[i - 1], seq2[j - 1]), 0)
            delete = dp[i - 1][j] + metric_values.get((seq1[i - 1], "-"), 0)
            insert = dp[i][j - 1] + metric_values.get(("-", seq2[j - 1]), 0)
            dp[i][j] = min(match, delete, insert)

    # Traceback
    aligned_seq1 = []
    aligned_seq2 = []
    i, j = len1, len2

    while i > 0 and j > 0:
        match_score = metric_values.get((seq1[i - 1], seq2[j - 1]), 0)
        delete_score = metric_values.get((seq1[i - 1], "-"), 0)
        insert_score = metric_values.get(("-", seq2[j - 1]), 0)

        if dp[i][j] == dp[i - 1][j - 1] + match_score:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + delete_score:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append("-")
            i -= 1
        else:
            aligned_seq1.append("-")
            aligned_seq2.append(seq2[j - 1])
            j -= 1

    # Handle remaining characters
    while i > 0:
        aligned_seq1.append(seq1[i - 1])
        aligned_seq2.append("-")
        i -= 1

    while j > 0:
        aligned_seq1.append("-")
        aligned_seq2.append(seq2[j - 1])
        j -= 1

    return "".join(reversed(aligned_seq1)), "".join(reversed(aligned_seq2))


if NUMBA_AVAILABLE:

    @njit
    def _nw_core(len1, len2, dp, match_matrix, delete_matrix, insert_matrix):
        """JIT-compiled core of Needleman-Wunsch algorithm"""
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                match = dp[i - 1][j - 1] + match_matrix[i - 1][j - 1]
                delete = dp[i - 1][j] + delete_matrix[i - 1]
                insert = dp[i][j - 1] + insert_matrix[j - 1]
                dp[i][j] = min(match, delete, insert)
        return dp

    def _nw_numba(seq1, seq2, metric_values):
        """Numba-accelerated implementation of Needleman-Wunsch"""
        len1 = len(seq1)
        len2 = len(seq2)

        # Initialization
        dp = np.zeros((len1 + 1, len2 + 1))

        # Fill first row and column
        for i in range(1, len1 + 1):
            dp[i][0] = dp[i - 1][0] + metric_values.get((seq1[i - 1], "-"), 0)

        for j in range(1, len2 + 1):
            dp[0][j] = dp[0][j - 1] + metric_values.get(("-", seq2[j - 1]), 0)

        # Prepare matrices for numba JIT
        match_matrix = np.zeros((len1, len2))
        delete_matrix = np.zeros(len1)
        insert_matrix = np.zeros(len2)

        for i in range(len1):
            delete_matrix[i] = metric_values.get((seq1[i], "-"), 0)
            for j in range(len2):
                match_matrix[i][j] = metric_values.get((seq1[i], seq2[j]), 0)
                if i == 0:
                    insert_matrix[j] = metric_values.get(("-", seq2[j]), 0)

        # Run JIT-compiled core
        dp = _nw_core(len1, len2, dp, match_matrix, delete_matrix, insert_matrix)

        # Traceback (same as _nw_numpy)
        aligned_seq1 = []
        aligned_seq2 = []
        i, j = len1, len2

        while i > 0 and j > 0:
            match_score = metric_values.get((seq1[i - 1], seq2[j - 1]), 0)
            delete_score = metric_values.get((seq1[i - 1], "-"), 0)
            insert_score = metric_values.get(("-", seq2[j - 1]), 0)

            if abs(dp[i][j] - (dp[i - 1][j - 1] + match_score)) < 1e-6:
                aligned_seq1.append(seq1[i - 1])
                aligned_seq2.append(seq2[j - 1])
                i -= 1
                j -= 1
            elif abs(dp[i][j] - (dp[i - 1][j] + delete_score)) < 1e-6:
                aligned_seq1.append(seq1[i - 1])
                aligned_seq2.append("-")
                i -= 1
            else:
                aligned_seq1.append("-")
                aligned_seq2.append(seq2[j - 1])
                j -= 1

        # Handle remaining characters
        while i > 0:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append("-")
            i -= 1

        while j > 0:
            aligned_seq1.append("-")
            aligned_seq2.append(seq2[j - 1])
            j -= 1

        return "".join(reversed(aligned_seq1)), "".join(reversed(aligned_seq2))

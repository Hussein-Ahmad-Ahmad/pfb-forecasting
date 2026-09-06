"""
Missing data masking for robustness evaluation.

Two strategies:
1. Random missingness: Uniformly random timesteps
2. Block missingness: Contiguous blocks (simulates sensor failures)

Both return data + mask indicator channel for missing-aware learning.
"""

import torch
import numpy as np


def mask_random(x, rate=0.1, seed=None):
    """
    Apply random missingness to input sequences.
    
    Args:
        x: Input tensor [batch, seq_len, features]
        rate: Fraction of values to mask (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        x_masked: Data with missing values set to 0
        mask: Binary indicator [1=present, 0=missing]
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Create random mask [batch, seq_len, features]
    mask = (torch.rand_like(x) > rate).float()
    x_masked = x * mask
    
    return x_masked, mask


def mask_block(x, rate=0.1, block_size=5, seed=None):
    """
    Apply block missingness to input sequences.
    Simulates sensor failures or communication outages.
    
    Args:
        x: Input tensor [batch, seq_len, features]
        rate: Fraction of sequence length to mask (in blocks)
        block_size: Size of each contiguous block to mask
        seed: Random seed for reproducibility
        
    Returns:
        x_masked: Data with missing blocks set to 0
        mask: Binary indicator [1=present, 0=missing]
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    batch_size, seq_len, n_features = x.shape
    mask = torch.ones_like(x)
    
    # Calculate number of blocks to mask
    total_to_mask = int(seq_len * rate)
    n_blocks = max(1, total_to_mask // block_size)
    
    # Mask random blocks for each batch sample
    for b in range(batch_size):
        for _ in range(n_blocks):
            # Random start position
            start_idx = np.random.randint(0, max(1, seq_len - block_size))
            end_idx = min(start_idx + block_size, seq_len)
            
            # Mask this block across all features
            mask[b, start_idx:end_idx, :] = 0
    
    x_masked = x * mask
    return x_masked, mask


def apply_missingness(x, missing_type='random', rate=0.1, block_size=5, seed=None):
    """
    Unified interface for applying missingness.
    
    Args:
        x: Input tensor
        missing_type: 'random' or 'block'
        rate: Missingness rate
        block_size: Block size for block missingness
        seed: Random seed
        
    Returns:
        x_with_mask: Concatenated [x_masked, mask] with doubled feature dim
    """
    if missing_type == 'random':
        x_masked, mask = mask_random(x, rate, seed)
    elif missing_type == 'block':
        x_masked, mask = mask_block(x, rate, block_size, seed)
    else:
        raise ValueError(f"Unknown missing_type: {missing_type}")
    
    # Concatenate data and mask indicator
    # Features: 7 → 14 (original + mask channel)
    x_with_mask = torch.cat([x_masked, mask], dim=-1)
    
    return x_with_mask


def get_effective_features(original_features, use_mask_channel=True):
    """
    Calculate effective feature dimension after masking.
    
    Args:
        original_features: Original number of features (e.g., 7 for ETT)
        use_mask_channel: Whether mask indicator is concatenated
        
    Returns:
        Effective feature count (14 if mask channel added, 7 otherwise)
    """
    return original_features * 2 if use_mask_channel else original_features


# Usage example for paper methodology section:
"""
Robustness Evaluation Protocol:
--------------------------------
Missing values are masked to zero with a binary indicator channel appended,
allowing models to explicitly learn missing-aware representations.

Random missingness:  10%, 20%, 30% of timesteps uniformly masked
Block missingness:   Contiguous 5-timestep blocks totaling 10%, 20%, 30%

Models evaluated: PFB-Direct, PatchTST, DLinear
Datasets: ETTm1, Weather
Horizon: H=192
Seed: 2021 (for reproducibility)

Example usage:
--------------
from utils.missing_data import apply_missingness

# During data loading for robustness experiments:
x_masked = apply_missingness(
    x, 
    missing_type='random',  # or 'block'
    rate=0.2,              # 20% missing
    seed=2021
)

# Model receives concatenated [data, mask]:
# Input shape: [batch, seq_len, 14] where 14 = 7 features + 7 mask indicators
"""

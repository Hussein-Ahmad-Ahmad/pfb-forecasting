"""
Deterministic seed setting for reproducible experiments.
Used across multi-seed validation and ablation studies.
"""

import random
import numpy as np
import torch


def set_seed(seed):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value (e.g., 2021, 2022, 2023)
    
    Usage:
        from utils.set_seed import set_seed
        set_seed(args.seed)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f'Random seed set to: {seed}')

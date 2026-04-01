"""랜덤 시드 고정"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """모든 랜덤 시드를 고정하여 실험 재현성 확보"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

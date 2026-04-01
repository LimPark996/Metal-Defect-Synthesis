"""
학습률 스케줄러
- Warmup + Cosine Annealing
"""

import math

import torch.optim as optim


def get_lr_scheduler(optimizer: optim.Optimizer, num_epochs: int, warmup_epochs: int):
    """
    Warmup + Cosine Annealing 스케줄러

    Args:
        optimizer: 옵티마이저
        num_epochs: 전체 에폭 수
        warmup_epochs: 워밍업 에폭 수

    Returns:
        LambdaLR 스케줄러
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

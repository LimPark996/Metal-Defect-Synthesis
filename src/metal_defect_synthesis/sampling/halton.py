"""
halton.py — Halton 시퀀스 및 2D 마스크 생성

이 모듈이 하는 일:
  1. 1D 저불일치(low-discrepancy) Halton 시퀀스 계산
  2. 16x16 토큰 격자를 공간적으로 균일하게 덮는 2D 좌표 순서 생성
"""

from typing import List

import torch


def halton_sequence(base: int, n_samples: int) -> List[float]:
    """
    Halton sequence 생성 (1D)

    Args:
        base: 기저 (소수 사용: 2, 3, 5, ...)
        n_samples: 생성할 샘플 수

    Returns:
        list of floats in [0, 1)
    """
    result = []
    for i in range(1, n_samples + 1):
        f = 1.0
        r = 0.0
        n = i
        while n > 0:
            f /= base
            r += f * (n % base)
            n //= base
        result.append(r)
    return result


def build_halton_mask(input_size: int = 16, n_points: int = 10000) -> torch.Tensor:
    """
    2D Halton mask 생성

    Args:
        input_size: 16 (16x16 = 256 토큰)
        n_points: 충분히 큰 수 (중복 제거 후 256개 남음)

    Returns:
        mask: [256, 2] 텐서, 각 행은 (row, col) 좌표
    """
    # 2D Halton sequence (base 2, 3)
    x = halton_sequence(2, n_points)
    y = halton_sequence(3, n_points)

    # [0, 1) → [0, input_size) 스케일링
    coords = []
    seen = set()

    for xi, yi in zip(x, y):
        row = int(xi * input_size)
        col = int(yi * input_size)

        row = min(row, input_size - 1)
        col = min(col, input_size - 1)

        # 중복 제거
        if (row, col) not in seen:
            seen.add((row, col))
            coords.append([row, col])

        # 256개 모이면 종료
        if len(coords) >= input_size ** 2:
            break

    return torch.tensor(coords, dtype=torch.long)

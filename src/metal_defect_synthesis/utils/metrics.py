"""
평가 지표
- PSNR, SSIM, Edge IoU
"""

from typing import Dict

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def compute_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """
    이미지 품질 평가 메트릭 계산

    Args:
        original: [H, W, C] numpy 배열, 값 범위 [0, 1]
        reconstructed: [H, W, C] numpy 배열, 값 범위 [0, 1]

    Returns:
        dict: {psnr, ssim, edge_iou}
    """
    original = np.clip(original, 0, 1)
    reconstructed = np.clip(reconstructed, 0, 1)

    # PSNR
    psnr_value = psnr(original, reconstructed, data_range=1.0)

    # SSIM
    ssim_value = ssim(original, reconstructed, data_range=1.0, channel_axis=2)

    # Edge IoU
    orig_gray = (original.mean(axis=2) * 255).astype(np.uint8)
    rec_gray = (reconstructed.mean(axis=2) * 255).astype(np.uint8)

    edges_orig = cv2.Canny(orig_gray, 50, 150)
    edges_rec = cv2.Canny(rec_gray, 50, 150)

    intersection = (edges_orig & edges_rec).sum()
    union = (edges_orig | edges_rec).sum()
    edge_iou = intersection / (union + 1e-6)

    return {
        'psnr': float(psnr_value),
        'ssim': float(ssim_value),
        'edge_iou': float(edge_iou),
    }

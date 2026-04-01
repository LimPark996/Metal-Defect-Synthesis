"""
이미지 변환 유틸리티
- 전처리, 후처리, 시각화 함수
"""

from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    텐서를 다시 이미지로 바꾸기 위한 역변환 함수
    [-1, 1] → [0, 1] → numpy 배열
    """
    tensor = tensor.clone()
    tensor = tensor * 0.5 + 0.5
    tensor = tensor.clamp(0, 1)

    if tensor.dim() == 4:
        tensor = tensor[0]

    return tensor.permute(1, 2, 0).cpu().numpy()


def preprocess_image(
    pil_image: Image.Image,
    image_size: int = 256,
    device: str = "cuda",
) -> torch.Tensor:
    """PIL Image → 모델 입력 텐서"""
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    tensor = transform(pil_image).unsqueeze(0)
    return tensor.to(device)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """텐서 → PIL Image"""
    tensor = tensor.cpu().detach()

    if tensor.dim() == 4:
        tensor = tensor[0]

    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)

    numpy_img = tensor.permute(1, 2, 0).numpy()
    numpy_img = (numpy_img * 255).astype(np.uint8)

    return Image.fromarray(numpy_img)


def visualize_mask_on_image(
    pil_image: Image.Image,
    mask_indices: List[int],
    image_size: int = 256,
    latent_size: int = 16,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.4,
) -> Image.Image:
    """이미지 위에 마스크 영역 시각화"""
    img_np = np.array(pil_image.resize((image_size, image_size))).astype(np.float32)

    mask = np.zeros((latent_size, latent_size), dtype=np.float32)
    for idx in mask_indices:
        y = idx // latent_size
        x = idx % latent_size
        mask[y, x] = 1.0

    scale = image_size // latent_size
    mask_upscaled = np.kron(mask, np.ones((scale, scale)))
    mask_3d = np.stack([mask_upscaled] * 3, axis=-1)

    color_overlay = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    overlay = img_np * (1 - mask_3d * alpha) + color_overlay * mask_3d * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)


def get_mask_preset(preset_name: str) -> List[int]:
    """마스크 프리셋 반환"""
    presets = {
        "center_small": [],
        "center_large": [],
        "top_left": [],
        "bottom_right": []
    }

    for y in range(5, 11):
        for x in range(5, 11):
            presets["center_small"].append(y * 16 + x)

    for y in range(4, 12):
        for x in range(4, 12):
            presets["center_large"].append(y * 16 + x)

    for y in range(0, 6):
        for x in range(0, 6):
            presets["top_left"].append(y * 16 + x)

    for y in range(10, 16):
        for x in range(10, 16):
            presets["bottom_right"].append(y * 16 + x)

    return presets.get(preset_name, presets["center_small"])

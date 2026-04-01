"""
인페인팅 (결함 합성)
- 원본 이미지의 일부 영역을 특정 결함으로 채움
"""

from typing import List, Union

import torch
import torch.nn.functional as F

from ..models.vqgan_wrapper import encode_to_tokens, decode_from_tokens


def cosine_schedule(t: torch.Tensor) -> torch.Tensor:
    """Cosine masking schedule"""
    return torch.cos(t * torch.pi / 2)


@torch.no_grad()
def inpaint_image(
    original_image: torch.Tensor,
    mask_region: Union[List[int], torch.Tensor],
    target_class: int,
    maskgit_model,
    vqgan_model,
    num_steps: int = 8,
    temperature: float = 1.0,
    latent_size: int = 16,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    원본 이미지의 일부 영역을 특정 결함으로 채움 (Inpainting)

    Args:
        original_image: [1, 3, 256, 256] 원본 이미지 (값 범위 [-1, 1])
        mask_region: 마스킹할 영역의 인덱스 리스트 (0~255)
        target_class: 어떤 결함으로 채울지 (0~5)
        maskgit_model: MaskGIT 모델
        vqgan_model: VQGAN 모델
        num_steps: 디코딩 스텝 수
        temperature: 샘플링 온도
        latent_size: 잠재 공간 크기
        device: 디바이스

    Returns:
        result_image: Inpainting 결과 이미지
    """
    maskgit_model.eval()

    if original_image.dim() == 3:
        original_image = original_image.unsqueeze(0)
    original_image = original_image.to(device)

    mask_token_id = maskgit_model.mask_token_id
    seq_len = maskgit_model.seq_len

    if isinstance(mask_region, list):
        mask_region = torch.tensor(mask_region, device=device)

    num_mask_tokens = len(mask_region)

    # 1. 원본 이미지 → 토큰
    original_tokens = encode_to_tokens(original_image, vqgan_model, latent_size)

    # 2. 마스킹할 영역만 [MASK]로 교체
    tokens = original_tokens.clone().view(1, -1)
    tokens[:, mask_region] = mask_token_id

    # 3. 클래스 라벨
    labels = torch.tensor([target_class], device=device)
    drop_label = torch.zeros(1, dtype=torch.bool, device=device)

    # 4. Iterative decoding (cosine schedule)
    for step in range(num_steps):
        is_masked = (tokens == mask_token_id)
        num_masked = is_masked.sum().item()

        if num_masked == 0:
            break

        t = torch.tensor(step / num_steps, device=device)
        ratio = cosine_schedule(t)
        num_to_keep_masked = (ratio * num_mask_tokens).long().item()
        num_to_unmask = max(1, num_masked - num_to_keep_masked)

        logits = maskgit_model(tokens, labels, drop_label)
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        max_probs, predicted_tokens = probs.max(dim=-1)

        max_probs = max_probs * is_masked.float()

        _, top_k = max_probs.view(-1).topk(num_to_unmask)
        tokens.view(-1)[top_k] = predicted_tokens.view(-1)[top_k]

    # 남은 [MASK] 강제로 채우기
    is_masked = (tokens == mask_token_id)
    if is_masked.any():
        logits = maskgit_model(tokens, labels, drop_label)
        probs = F.softmax(logits / temperature, dim=-1)
        _, predicted_tokens = probs.max(dim=-1)
        tokens[is_masked] = predicted_tokens[is_masked]

    # 5. 토큰 → 이미지
    tokens = tokens.view(1, latent_size, latent_size)
    result_image = decode_from_tokens(tokens, vqgan_model, latent_size)

    return result_image

"""
VQGAN 래퍼 - LlamaGen VQGAN 모델 로드/인코딩/디코딩
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def load_vqgan(
    ckpt_path: str,
    device: torch.device,
    codebook_size: int = 16384,
    codebook_dim: int = 8,
):
    """
    Fine-tuned LlamaGen VQGAN 로드

    Args:
        ckpt_path: 체크포인트 경로
        device: cuda or cpu
        codebook_size: 코드북 크기 (기본 16384)
        codebook_dim: 코드북 임베딩 차원 (기본 8)

    Returns:
        vqgan: 로드된 VQGAN 모델 (eval 모드)
    """
    from tokenizer.tokenizer_image.vq_model import VQ_models

    # 1. 모델 생성
    vqgan = VQ_models["VQ-16"](
        codebook_size=codebook_size,
        codebook_embed_dim=codebook_dim
    )

    # 2. 체크포인트 로드
    checkpoint = torch.load(ckpt_path, map_location="cuda")

    if "vqmodel" in checkpoint:
        state_dict = checkpoint["vqmodel"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # 3. 가중치 로드
    vqgan.load_state_dict(state_dict, strict=True)

    # 4. 디바이스 이동 + eval 모드
    vqgan = vqgan.to(device)
    vqgan.eval()

    # 5. 파라미터 고정 (학습 안 함)
    for param in vqgan.parameters():
        param.requires_grad = False

    logger.info(f"VQGAN 로드 완료 (파라미터: {sum(p.numel() for p in vqgan.parameters()):,})")
    return vqgan


@torch.no_grad()
def encode_to_tokens(
    images: torch.Tensor,
    vqgan_model,
    latent_size: int = 16,
) -> torch.Tensor:
    """
    이미지를 VQGAN으로 인코딩하여 토큰 인덱스 반환

    Args:
        images: [B, 3, 256, 256] 텐서, 값 범위 [-1, 1]
        vqgan_model: LlamaGen VQGAN 모델
        latent_size: 잠재 공간 크기 (기본 16)

    Returns:
        tokens: [B, 16, 16] 텐서, 각 값은 0 ~ 16383
    """
    _, _, (_, _, indices) = vqgan_model.encode(images)
    batch_size = images.shape[0]
    tokens = indices.view(batch_size, latent_size, latent_size)
    return tokens


@torch.no_grad()
def decode_from_tokens(
    tokens: torch.Tensor,
    vqgan_model,
    latent_size: int = 16,
    codebook_size: int = 16384,
) -> torch.Tensor:
    """
    토큰 인덱스를 VQGAN으로 디코딩하여 이미지 반환

    Args:
        tokens: [B, 16, 16] 토큰 텐서
        vqgan_model: LlamaGen VQGAN 모델
        latent_size: 잠재 공간 크기 (기본 16)
        codebook_size: 코드북 크기 (기본 16384)

    Returns:
        images: [B, 3, 256, 256] 이미지 텐서
    """
    batch_size = tokens.shape[0]
    tokens_flat = tokens.view(-1)
    tokens_flat = torch.clamp(tokens_flat, 0, codebook_size - 1)

    embed_dim = vqgan_model.quantize.embedding.weight.shape[1]

    images = vqgan_model.decode_code(
        tokens_flat,
        shape=(batch_size, embed_dim, latent_size, latent_size)
    )
    return images

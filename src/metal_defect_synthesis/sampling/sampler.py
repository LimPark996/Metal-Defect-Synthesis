"""
Halton Sampler - 이미지 생성기
- Halton sequence로 토큰 순서 결정 (공간적 균일성)
- CFG (Classifier-Free Guidance) 지원
"""

import math

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .halton import build_halton_mask
from ..models.vqgan_wrapper import decode_from_tokens


class HaltonSampler:
    """
    Halton-MaskGIT 스타일 샘플러

    특징:
    - Halton sequence로 토큰 순서 결정 (공간적 균일성)
    - CFG (Classifier-Free Guidance) 지원
    - Temperature scheduling
    """

    def __init__(
        self,
        num_steps: int = 32,
        cfg_weight: float = 2.0,
        temperature: float = 1.0,
        randomize: bool = True,
        latent_size: int = 16,
        codebook_size: int = 16384,
        mask_token_id: int = 16384,
    ):
        self.num_steps = num_steps
        self.cfg_weight = cfg_weight
        self.temperature = temperature
        self.randomize = randomize
        self.latent_size = latent_size
        self.codebook_size = codebook_size
        self.mask_token_id = mask_token_id
        self.halton_mask = build_halton_mask(latent_size)

    @torch.no_grad()
    def sample(self, model, vqgan, num_samples: int, class_labels: torch.Tensor, device: torch.device):
        """
        이미지 생성

        Args:
            model: MaskGITTransformer
            vqgan: VQGAN 디코더
            num_samples: 생성할 이미지 수
            class_labels: [num_samples] 클래스 라벨
            device: cuda/cpu

        Returns:
            images: [num_samples, 3, 256, 256] 생성된 이미지
            codes: [num_samples, 16, 16] 생성된 토큰
        """
        model.eval()
        B = num_samples
        H = W = self.latent_size
        seq_len = H * W

        # 1. 전체 [MASK]로 초기화
        code = torch.full((B, H, W), self.mask_token_id, device=device)
        labels = class_labels.to(device)

        # 2. CFG용 라벨
        drop_label_cond = torch.zeros(B, dtype=torch.bool, device=device)
        drop_label_uncond = torch.ones(B, dtype=torch.bool, device=device)

        # 3. Halton mask 준비 (샘플마다 랜덤 오프셋)
        if self.randomize:
            offsets = torch.randint(0, seq_len, (B,))
            halton_masks = torch.stack([
                torch.roll(self.halton_mask, shifts=offset.item(), dims=0)
                for offset in offsets
            ])
        else:
            halton_masks = self.halton_mask.unsqueeze(0).expand(B, -1, -1)

        halton_masks = halton_masks.to(device)

        # 4. 스텝별 생성
        prev_idx = 0

        for step in tqdm(range(self.num_steps), desc="Sampling", leave=False):
            ratio = (step + 1) / self.num_steps
            r = 1 - (math.acos(ratio) / (math.pi * 0.5))
            curr_idx = max(step + 1, int(r * seq_len))
            curr_idx = min(curr_idx, seq_len)

            positions = halton_masks[:, prev_idx:curr_idx]

            if positions.shape[1] == 0:
                continue

            # Forward (CFG)
            if self.cfg_weight > 0:
                code_double = torch.cat([code, code], dim=0)
                labels_double = torch.cat([labels, labels], dim=0)
                drop_double = torch.cat([drop_label_cond, drop_label_uncond], dim=0)

                logits_double = model(code_double, labels_double, drop_double)
                logits_cond, logits_uncond = logits_double.chunk(2, dim=0)

                # CFG 공식
                logits = (1 + self.cfg_weight) * logits_cond - self.cfg_weight * logits_uncond
            else:
                logits = model(code, labels, drop_label_cond)

            # Softmax + 샘플링
            logits = logits.view(B, H, W, -1)
            probs = F.softmax(logits / self.temperature, dim=-1)

            for b in range(B):
                for pos in positions[b]:
                    row, col = pos[0].item(), pos[1].item()
                    prob = probs[b, row, col, :self.codebook_size]
                    prob = prob / prob.sum()

                    sampled_token = torch.multinomial(prob, 1).item()
                    code[b, row, col] = sampled_token

            prev_idx = curr_idx

        # 5. VQGAN 디코딩
        code = torch.clamp(code, 0, self.codebook_size - 1)
        images = decode_from_tokens(code, vqgan, self.latent_size, self.codebook_size)
        images = torch.clamp(images, -1, 1)

        model.train()
        return images, code

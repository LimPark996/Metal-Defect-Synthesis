"""
MaskGIT 학습
- Halton-MaskGIT 스타일 마스킹 + Cross-Entropy 학습
- Arccos 마스킹 스케줄, CFG, Gradient Clipping
"""

import os
import math
import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# 마스킹 함수
# =============================================================================

def get_mask_schedule(r: torch.Tensor, mode: str = 'arccos') -> torch.Tensor:
    """
    마스킹 스케줄 계산

    Args:
        r: [B] 텐서, 0~1 사이 랜덤 값
        mode: 'arccos' | 'cosine' | 'linear' | 'square'

    Returns:
        mask_ratio: [B] 텐서, 마스킹할 비율 (0~1)
    """
    if mode == 'arccos':  # Halton-MaskGIT 기본
        return torch.arccos(r) / (math.pi * 0.5)
    elif mode == 'cosine':
        return torch.cos(r * math.pi * 0.5)
    elif mode == 'linear':
        return 1 - r
    elif mode == 'square':
        return 1 - (r ** 2)
    else:
        return 1 - r


def mask_tokens(
    tokens: torch.Tensor,
    mask_token_id: int,
    mode: str = 'arccos'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    토큰의 일부를 [MASK]로 교체

    Args:
        tokens: [B, H, W] 원본 토큰
        mask_token_id: [MASK] 토큰 ID (16384)
        mode: 마스킹 스케줄

    Returns:
        masked_tokens: [B, H, W] 마스킹된 토큰
        mask: [B, H, W] bool, 마스킹 위치
    """
    device = tokens.device
    B, H, W = tokens.shape
    seq_len = H * W

    r = torch.rand(B, device=device)
    mask_ratios = get_mask_schedule(r, mode)

    num_to_mask = (mask_ratios * seq_len).long()
    num_to_mask = torch.clamp(num_to_mask, min=1, max=seq_len-1)

    tokens_flat = tokens.view(B, -1)
    masked_tokens = tokens_flat.clone()
    mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)

    for i in range(B):
        num_mask = num_to_mask[i].item()
        perm = torch.randperm(seq_len, device=device)
        mask_indices = perm[:num_mask]

        masked_tokens[i, mask_indices] = mask_token_id
        mask[i, mask_indices] = True

    masked_tokens = masked_tokens.view(B, H, W)
    mask = mask.view(B, H, W)

    return masked_tokens, mask


# =============================================================================
# MaskGIT Trainer
# =============================================================================

class MaskGITTrainer:
    """MaskGIT 학습 Trainer"""

    def __init__(
        self,
        model,
        dataloader,
        device: torch.device,
        mask_token_id: int = 16384,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.03,
        warmup_epochs: int = 5,
        grad_clip: float = 1.0,
        drop_label_prob: float = 0.1,
        save_every: int = 10,
        save_dir: str = "checkpoints",
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.mask_token_id = mask_token_id
        self.num_epochs = num_epochs
        self.grad_clip = grad_clip
        self.drop_label_prob = drop_label_prob
        self.save_every = save_every
        self.save_dir = save_dir

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=weight_decay
        )

        from .scheduler import get_lr_scheduler
        self.scheduler = get_lr_scheduler(self.optimizer, num_epochs, warmup_epochs)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.train_losses: List[float] = []
        self.train_accs: List[float] = []

    def train(self) -> Tuple[List[float], List[float]]:
        """전체 학습 실행"""
        os.makedirs(self.save_dir, exist_ok=True)

        for epoch in range(self.num_epochs):
            avg_loss, avg_acc = self.train_one_epoch(epoch)

            self.scheduler.step()
            self.train_losses.append(avg_loss)
            self.train_accs.append(avg_acc)

            current_lr = self.scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} | "
                f"Loss: {avg_loss:.4f} | Acc: {avg_acc*100:.1f}% | LR: {current_lr:.6f}"
            )

            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch + 1)

        return self.train_losses, self.train_accs

    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

        for batch in pbar:
            code = batch['code'].to(self.device)
            y = batch['y'].to(self.device)
            B = code.shape[0]

            # CFG용 라벨 드롭
            drop_label = (torch.rand(B, device=self.device) < self.drop_label_prob)

            # 마스킹 적용
            masked_code, mask = mask_tokens(code, mask_token_id=self.mask_token_id, mode='arccos')

            # Forward
            logits = self.model(masked_code, y, drop_label)

            # Loss (마스킹된 위치에서만)
            B, H, W = code.shape
            logits_flat = logits.view(B * H * W, -1)
            target_flat = code.view(-1)
            mask_flat = mask.view(-1)

            target_masked = target_flat.clone()
            target_masked[~mask_flat] = -100

            loss = self.criterion(logits_flat, target_masked)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            # 정확도 (마스킹 위치)
            with torch.no_grad():
                pred = logits_flat.argmax(dim=-1)
                correct = (pred == target_flat) & mask_flat
                acc = correct.sum().float() / mask_flat.sum().float()

            total_loss += loss.item()
            total_acc += acc.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc.item()*100:.1f}%'
            })

        return total_loss / num_batches, total_acc / num_batches

    def _save_checkpoint(self, epoch: int):
        ckpt_path = os.path.join(self.save_dir, f"maskgit_epoch{epoch:03d}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
        }, ckpt_path)
        logger.info(f"체크포인트 저장: {ckpt_path}")

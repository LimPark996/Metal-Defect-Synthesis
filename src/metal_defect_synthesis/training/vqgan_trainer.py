"""
VQGAN Fine-tuning 학습
- taming-transformers 기반 VQGAN 파인튜닝
- Adaptive GAN loss, Hinge loss, LPIPS perceptual loss
"""

import os
import logging
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# Loss 함수들
# =============================================================================

def adopt_weight(weight: float, global_step: int, threshold: int = 0, value: float = 0.) -> float:
    """
    학습 초반에는 GAN loss를 0으로 만들어서 reconstruction만 학습하고,
    threshold 스텝 이후부터 GAN loss를 활성화
    """
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """
    Discriminator용 Hinge Loss

    진짜는 1보다 크게, 가짜는 -1보다 작게 만들려고 합니다.
    """
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def calculate_adaptive_weight(
    nll_loss: torch.Tensor,
    g_loss: torch.Tensor,
    last_layer: torch.Tensor,
    disc_weight: float = 1.0
) -> torch.Tensor:
    """
    Adaptive weight 계산 - reconstruction loss와 GAN loss의
    gradient 크기를 비교해서 두 loss가 비슷한 영향력을 갖도록 가중치를 자동 조절

    Parameters:
    - nll_loss: reconstruction loss (L1 + perceptual)
    - g_loss: generator의 GAN loss
    - last_layer: decoder의 마지막 레이어 weight
    - disc_weight: 기본 discriminator weight
    """
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    d_weight = d_weight * disc_weight

    return d_weight


# =============================================================================
# VQGAN Trainer
# =============================================================================

class VQGANTrainer:
    """VQGAN Fine-tuning Trainer"""

    def __init__(
        self,
        model,
        discriminator,
        perceptual_loss,
        dataloader,
        device: torch.device,
        num_epochs: int = 50,
        save_every: int = 5,
        save_dir: str = "checkpoints",
        codebook_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        disc_weight: float = 0.8,
        disc_factor: float = 1.0,
        disc_start: int = 1000,
    ):
        """모델, 판별자, 손실, 옵티마이저 및 학습 하이퍼파라미터 초기화."""
        self.model = model
        self.discriminator = discriminator
        self.perceptual_loss = perceptual_loss
        self.dataloader = dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.save_every = save_every
        self.save_dir = save_dir
        self.codebook_weight = codebook_weight
        self.perceptual_weight = perceptual_weight
        self.disc_weight = disc_weight
        self.disc_factor = disc_factor
        self.disc_start = disc_start

        # Optimizers
        self.optimizer_g = optim.Adam([
            {'params': model.encoder.parameters(), 'lr': 1e-7},
            {'params': model.quant_conv.parameters(), 'lr': 1e-7},
            {'params': model.quantize.parameters(), 'lr': 4.5e-6},
            {'params': model.post_quant_conv.parameters(), 'lr': 4.5e-6},
            {'params': model.decoder.parameters(), 'lr': 4.5e-6},
        ], betas=(0.5, 0.9))

        self.optimizer_d = optim.Adam(
            discriminator.parameters(),
            lr=4.5e-6,
            betas=(0.5, 0.9)
        )

        self.train_logs: List[Dict] = []
        self.global_step = 0

    def train(self) -> List[Dict]:
        """전체 학습 실행"""
        os.makedirs(self.save_dir, exist_ok=True)

        for epoch in range(self.num_epochs):
            epoch_logs = self.train_epoch(epoch)
            self.train_logs.append(epoch_logs)

            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch + 1)

        return self.train_logs

    def train_epoch(self, epoch: int) -> Dict:
        """한 에폭 학습"""
        self.model.train()
        self.discriminator.train()

        epoch_logs = {
            'total_loss': 0, 'nll_loss': 0, 'codebook_loss': 0,
            'g_loss': 0, 'd_loss': 0, 'd_weight': 0
        }

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)
            self.global_step += 1

            # Forward: Encode -> Quantize -> Decode
            z = self.model.encoder(images)
            z = self.model.quant_conv(z)
            z_q, codebook_loss, _ = self.model.quantize(z)
            z_q = self.model.post_quant_conv(z_q)
            reconstructions = self.model.decoder(z_q)

            # Generator 업데이트
            rec_loss = torch.abs(images.contiguous() - reconstructions.contiguous())

            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(images.contiguous(), reconstructions.contiguous())
                rec_loss = rec_loss + self.perceptual_weight * p_loss

            nll_loss = torch.mean(rec_loss)

            logits_fake = self.discriminator(reconstructions.contiguous())
            g_loss = -torch.mean(logits_fake)

            current_disc_factor = adopt_weight(self.disc_factor, self.global_step, threshold=self.disc_start)

            if current_disc_factor > 0:
                try:
                    d_weight = calculate_adaptive_weight(
                        nll_loss, g_loss,
                        last_layer=self.model.decoder.conv_out.weight,
                        disc_weight=self.disc_weight
                    )
                except RuntimeError:
                    d_weight = torch.tensor(0.0).to(self.device)
            else:
                d_weight = torch.tensor(0.0).to(self.device)

            loss_g = (nll_loss +
                      d_weight * current_disc_factor * g_loss +
                      self.codebook_weight * codebook_loss.mean())

            self.optimizer_g.zero_grad()
            loss_g.backward()
            self.optimizer_g.step()

            # Discriminator 업데이트
            if current_disc_factor > 0:
                logits_real = self.discriminator(images.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
                d_loss = current_disc_factor * hinge_d_loss(logits_real, logits_fake)

                self.optimizer_d.zero_grad()
                d_loss.backward()
                self.optimizer_d.step()
            else:
                d_loss = torch.tensor(0.0).to(self.device)

            # 로깅
            epoch_logs['total_loss'] += loss_g.item()
            epoch_logs['nll_loss'] += nll_loss.item()
            epoch_logs['codebook_loss'] += codebook_loss.mean().item()
            epoch_logs['g_loss'] += g_loss.item()
            epoch_logs['d_loss'] += d_loss.item()
            epoch_logs['d_weight'] += d_weight.item() if torch.is_tensor(d_weight) else d_weight

            pbar.set_postfix({
                'loss': f'{loss_g.item():.4f}',
                'nll': f'{nll_loss.item():.4f}',
                'd_w': f'{d_weight.item() if torch.is_tensor(d_weight) else d_weight:.2f}',
            })

        n_batches = len(self.dataloader)
        for key in epoch_logs:
            epoch_logs[key] /= n_batches

        logger.info(f"Epoch {epoch+1} | Loss: {epoch_logs['total_loss']:.4f} | NLL: {epoch_logs['nll_loss']:.4f}")
        return epoch_logs

    def _save_checkpoint(self, epoch: int):
        """현재 에폭의 모델/판별자/옵티마이저 상태를 체크포인트 파일로 저장."""
        save_path = os.path.join(self.save_dir, f"vqgan_finetune_up_epoch{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'vqmodel': self.model.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
        }, save_path)
        logger.info(f"체크포인트 저장: {save_path}")

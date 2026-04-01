"""
MaskGIT Transformer 모델
- Halton-MaskGIT 스타일 Bidirectional Transformer
- AdaLayerNorm, SwiGLU, QK Normalization, Weight Tying
"""

from typing import Optional, Dict

import torch
import torch.nn as nn

from .layers import TransformerBlock, AdaNorm


class MaskGITTransformer(nn.Module):
    """
    Halton-MaskGIT 스타일 Bidirectional Transformer

    특징:
    - AdaLayerNorm으로 클래스 조건 주입
    - SwiGLU FFN
    - QK Normalization
    - Weight Tying (tok_emb ↔ head)
    """

    def __init__(
        self,
        vocab_size: int = 16384,
        seq_len: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.,
        dropout: float = 0.1,
        num_classes: int = 6,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.mask_token_id = vocab_size  # [MASK] = 16384

        # ===== Embeddings =====
        # 토큰 임베딩: 0~16383 (codebook) + 16384 ([MASK])
        self.tok_emb = nn.Embedding(vocab_size + 1, hidden_dim)

        # 위치 임베딩: 0~255 (16x16 위치)
        self.pos_emb = nn.Embedding(seq_len, hidden_dim)

        # 클래스 임베딩: 0~5 (결함 클래스) + 6 (CFG용 null class)
        self.cls_emb = nn.Embedding(num_classes + 1, hidden_dim)

        # ===== Transformer Blocks =====
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # ===== Output =====
        self.final_norm = AdaNorm(hidden_dim)

        # 출력 헤드 (Weight Tying)
        self.head = nn.Linear(hidden_dim, vocab_size + 1, bias=False)
        self.head.weight = self.tok_emb.weight  # Weight Tying

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.cls_emb.weight, std=0.02)

        # AdaLN modulation 레이어 zero 초기화
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[1].weight)
            nn.init.zeros_(block.adaLN_modulation[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        drop_label: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        순전파

        Args:
            x: [B, H, W] 또는 [B, N] - 마스킹된 토큰 (일부가 16384)
            y: [B] - 클래스 라벨 (0~5)
            drop_label: [B] bool - True면 해당 샘플의 클래스 조건 무시 (CFG용)

        Returns:
            logits: [B, N, vocab_size+1] - 각 위치의 예측 확률
        """
        B = x.shape[0]

        # [B, H, W] → [B, N] 변환
        if x.dim() == 3:
            x = x.view(B, -1)

        # ===== 클래스 조건 처리 =====
        if drop_label is not None:
            y = torch.where(drop_label, torch.full_like(y, self.num_classes), y)

        cond = self.cls_emb(y)

        # ===== 토큰 + 위치 임베딩 =====
        pos = torch.arange(self.seq_len, device=x.device)
        x = self.tok_emb(x) + self.pos_emb(pos)

        # ===== Transformer 블록 통과 =====
        for block in self.blocks:
            x = block(x, cond)

        # ===== 출력 =====
        x = self.final_norm(x, cond)
        logits = self.head(x)

        return logits


def get_model_config(size: str = 'small') -> Dict[str, int]:
    """
    모델 크기별 설정 반환

    Halton-MaskGIT 논문 기준:
    - Tiny:  23M params (hidden=384, depth=6, heads=6)
    - Small: 69M params (hidden=512, depth=12, heads=8)
    - Base:  142M params (hidden=768, depth=12, heads=12)
    - Large: 480M params (hidden=1024, depth=24, heads=16)
    """
    configs = {
        'tiny': {'hidden_dim': 384, 'num_layers': 6, 'num_heads': 6},
        'small': {'hidden_dim': 512, 'num_layers': 12, 'num_heads': 8},
        'base': {'hidden_dim': 768, 'num_layers': 12, 'num_heads': 12},
        'large': {'hidden_dim': 1024, 'num_layers': 24, 'num_heads': 16},
    }
    return configs[size]

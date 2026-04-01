"""
트랜스포머 기본 블록들
- RMSNorm, SwiGLU, QKNorm, Attention, TransformerBlock, AdaNorm
- Halton-MaskGIT 논문의 아키텍처를 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLayerNorm의 modulation 연산"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm


class SwiGLU(nn.Module):
    """SwiGLU FFN (LLaMA style)"""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        # hidden_dim 조정 (SwiGLU는 2/3배 사용)
        hidden_dim = int(2 * hidden_dim / 3)
        # 256의 배수로 맞춤
        hidden_dim = 256 * ((hidden_dim + 255) // 256)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: SiLU(W1(x)) * W3(x)
        x = F.silu(self.w1(x)) * self.w3(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x


class QKNorm(nn.Module):
    """Query/Key Normalization for stable attention"""
    def __init__(self, dim: int):
        super().__init__()
        self.q_norm = RMSNorm(dim)
        self.k_norm = RMSNorm(dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        return self.q_norm(q), self.k_norm(k)


class Attention(nn.Module):
    """Multi-Head Self-Attention with QK Norm and Flash Attention"""
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.qk_norm = QKNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Q, K, V projection
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        # QK Normalization
        q, k = self.qk_norm(q, k)

        # Reshape for multi-head
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Flash Attention (PyTorch 2.0+)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.
        )

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.wo(out)

        return out


class TransformerBlock(nn.Module):
    """Transformer Block with AdaLayerNorm"""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., dropout: float = 0.):
        super().__init__()

        # AdaLN modulation MLP
        # 6개 파라미터: gamma1, beta1, alpha1, gamma2, beta2, alpha2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6)
        )

        # Attention
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)

        # FFN (SwiGLU)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] 토큰 임베딩
            cond: [B, C] 클래스 조건 임베딩
        """
        # AdaLN modulation 파라미터 계산
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = \
            self.adaLN_modulation(cond).chunk(6, dim=-1)

        # Attention with AdaLN
        x = x + alpha1.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), beta1, gamma1)
        )

        # FFN with AdaLN
        x = x + alpha2.unsqueeze(1) * self.ffn(
            modulate(self.norm2(x), beta2, gamma2)
        )

        return x


class AdaNorm(nn.Module):
    """Final AdaLayerNorm before output"""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 2)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(cond).chunk(2, dim=-1)
        return modulate(self.norm(x), shift, scale)

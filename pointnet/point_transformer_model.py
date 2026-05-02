import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class PointTransformerRegressor(nn.Module):
    """
    Point-cloud regressor with token selection + transformer encoder.

    Input:
      pts: (B, 3, N)
    Output:
      y_hat: (B, output_dim)
    """

    def __init__(
        self,
        output_dim: int,
        embed_dim: int = 128,
        depth: int = 6,
        heads: int = 4,
        token_count: int = 256,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_count = token_count

        self.stem = nn.Sequential(
            nn.Conv1d(3, embed_dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(embed_dim // 2),
            nn.GELU(),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

        self.token_score = nn.Conv1d(embed_dim, 1, kernel_size=1)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    heads=heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )

    @staticmethod
    def _gather_tokens(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N), idx: (B, K) -> (B, C, K)
        idx_expand = idx.unsqueeze(1).expand(-1, x.size(1), -1)
        return torch.gather(x, dim=2, index=idx_expand)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        # pts: (B, 3, N)
        x = self.stem(pts)  # (B, C, N)

        n = x.size(2)
        k = min(self.token_count, n)

        scores = self.token_score(x).squeeze(1)  # (B, N)
        idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=False).indices  # (B, K)

        x_tok = self._gather_tokens(x, idx)  # (B, C, K)
        xyz_tok = self._gather_tokens(pts, idx).transpose(1, 2)  # (B, K, 3)

        x_tok = x_tok.transpose(1, 2)  # (B, K, C)
        x_tok = x_tok + self.pos_mlp(xyz_tok)

        for block in self.blocks:
            x_tok = block(x_tok)

        x_tok = self.norm(x_tok)
        x_max = x_tok.max(dim=1).values
        x_avg = x_tok.mean(dim=1)
        feat = torch.cat([x_max, x_avg], dim=1)

        return self.head(feat)

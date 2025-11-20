from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from flax import linen as nn


class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, *, train: bool) -> Any:
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            dropout_rate=self.dropout_rate,
            deterministic=not train,
            decode=False,
            kernel_init=nn.initializers.xavier_uniform(),
        )(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = x + residual

        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.mlp_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = nn.Dense(self.embed_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        return x + residual


class TransformerLM(nn.Module):
    vocab_size: int
    max_length: int
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 8
    mlp_dim: int = 2048
    dropout_rate: float = 0.1

    def setup(self) -> None:
        self.token_embed = nn.Embed(self.vocab_size, self.embed_dim)
        self.pos_embed = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.max_length, self.embed_dim),
        )
        self.blocks = [
            TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.num_layers)
        ]
        self.norm = nn.LayerNorm()
        self.lm_head = nn.Dense(self.vocab_size, kernel_init=nn.initializers.xavier_uniform())

    def __call__(self, token_ids, *, train: bool) -> jnp.ndarray:
        x = self.token_embed(token_ids)
        seq_len = x.shape[1]
        x = x + self.pos_embed[:seq_len, :]
        for block in self.blocks:
            x = block(x, train=train)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

from __future__ import annotations

import math
from typing import Dict, Literal

import torch
from torch import nn

from .mlp import MLPRegressor


class SelfAttentionBlock(nn.Module):
    """Scaled dot-product self-attention block with residual normalization.

    This module performs multi-head self-attention using explicit Query/Key/Value
    projections and applies a residual connection followed by layer normalization.

    Args:
        embed_dim: Embedding size for each token.
        num_heads: Number of attention heads.
        attention_dropout: Dropout probability applied to attention weights.

    Shape:
        Input: [batch_size, sequence_length, embed_dim]
        Output: [batch_size, sequence_length, embed_dim]
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)

        self.attention_dropout = nn.Dropout(p=attention_dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """Apply scaled dot-product self-attention.

        Args:
            input_tokens: Embedded tabular tokens with shape
                [batch_size, num_features, embed_dim].

        Returns:
            Contextualized tokens with shape
            [batch_size, num_features, embed_dim].
        """

        batch_size, sequence_length, _ = input_tokens.shape

        # [batch_size, num_features, embed_dim]
        query = self.query_projection(input_tokens)
        key = self.key_projection(input_tokens)
        value = self.value_projection(input_tokens)

        # [batch_size, num_heads, num_features, head_dim]
        query = query.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        # [batch_size, num_heads, num_features, num_features]
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        # [batch_size, num_heads, num_features, head_dim]
        context_vector = torch.matmul(attention_weights, value)

        # [batch_size, num_features, embed_dim]
        context_vector = context_vector.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            self.embed_dim,
        )

        # [batch_size, num_features, embed_dim]
        attention_output = self.output_projection(context_vector)

        # Residual Add + Norm, preserving [batch_size, num_features, embed_dim]
        residual_sum = input_tokens + attention_output
        normalized_output = self.layer_norm(residual_sum)
        return normalized_output


class AttentionMLPRegressor(nn.Module):
    """Tabular regressor with feature embedding, self-attention, and MLP head.

    Pipeline:
    1) Embed each scalar feature into an embedding vector.
    2) Apply self-attention across feature tokens.
    3) Aggregate contextualized tokens to a single vector.
    4) Predict outputs with the existing MLP regressor.

    Args:
        input_dim: Number of scalar input features.
        output_dim: Number of regression targets.
        model_cfg: Model configuration dictionary. Expected keys include:
            - embed_dim: Embedding size per feature token.
            - num_heads: Number of attention heads.
            - attention_dropout: Dropout for attention weights.
            - aggregation: "mean" or "flatten".
            - hidden_layers/activation/dropout: Reused by the MLP head.

    Shape:
        Input: [batch_size, input_dim]
        Output: [batch_size, output_dim]
    """

    def __init__(self, input_dim: int, output_dim: int, model_cfg: Dict) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = int(model_cfg.get("embed_dim", 32))
        num_heads = int(model_cfg.get("num_heads", 4))
        attention_dropout = float(model_cfg.get("attention_dropout", 0.0))
        aggregation = str(model_cfg.get("aggregation", "mean")).lower()

        if aggregation not in {"mean", "flatten"}:
            raise ValueError("aggregation must be one of: mean, flatten")
        self.aggregation: Literal["mean", "flatten"] = aggregation

        self.feature_embedding = nn.Linear(1, self.embed_dim)
        self.self_attention_block = SelfAttentionBlock(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
        )

        mlp_input_dim = self.embed_dim if self.aggregation == "mean" else (self.input_dim * self.embed_dim)
        self.mlp_base = MLPRegressor(
            input_dim=mlp_input_dim,
            output_dim=output_dim,
            model_cfg=model_cfg,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict outputs from tabular input features.

        Args:
            features: Input tensor with shape [batch_size, num_features].

        Returns:
            Predicted tensor with shape [batch_size, output_dim].
        """

        # [batch_size, num_features] -> [batch_size, num_features, 1]
        tokenized_features = features.unsqueeze(-1)

        # [batch_size, num_features, 1] -> [batch_size, num_features, embed_dim]
        embedded_features = self.feature_embedding(tokenized_features)

        # [batch_size, num_features, embed_dim] -> [batch_size, num_features, embed_dim]
        contextualized_features = self.self_attention_block(embedded_features)

        if self.aggregation == "mean":
            # Global average pooling over feature tokens: [batch_size, embed_dim]
            aggregated_features = contextualized_features.mean(dim=1)
        else:
            # Flatten feature tokens: [batch_size, num_features * embed_dim]
            aggregated_features = contextualized_features.reshape(features.shape[0], -1)

        # Existing MLP head: [batch_size, mlp_input_dim] -> [batch_size, output_dim]
        return self.mlp_base(aggregated_features)
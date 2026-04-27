from typing import Dict

from .attention_mlp import AttentionMLPRegressor
from .mlp import MLPRegressor


def build_regressor(input_dim: int, output_dim: int, model_cfg: Dict):
	model_name = str(model_cfg.get("model_name", "mlp_regressor")).lower()

	if model_name in {"mlp", "mlp_regressor"}:
		return MLPRegressor(input_dim=input_dim, output_dim=output_dim, model_cfg=model_cfg)
	if model_name in {"attention_mlp", "attention_mlp_regressor", "self_attention_mlp"}:
		return AttentionMLPRegressor(input_dim=input_dim, output_dim=output_dim, model_cfg=model_cfg)

	raise ValueError(
		"Unsupported model_name '{}' . Available: ['mlp_regressor', 'attention_mlp_regressor']".format(
			model_name
		)
	)

__all__ = ["MLPRegressor", "AttentionMLPRegressor", "build_regressor"]

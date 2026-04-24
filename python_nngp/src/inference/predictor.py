from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from src.models import MLPRegressor


class SurrogatePredictor:
    def __init__(
        self,
        model: torch.nn.Module,
        feature_scaler=None,
        target_scaler=None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        artifact_path: Path,
        input_dim: int,
        output_dim: int,
        model_cfg: Dict,
        device: Optional[str] = None,
    ) -> "SurrogatePredictor":
        with artifact_path.open("rb") as stream:
            payload = pickle.load(stream)

        model = MLPRegressor(input_dim=input_dim, output_dim=output_dim, model_cfg=model_cfg)
        resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state_dict = torch.load(checkpoint_path, map_location=resolved_device)
        model.load_state_dict(state_dict)

        return cls(
            model=model,
            feature_scaler=payload.get("feature_scaler"),
            target_scaler=payload.get("target_scaler"),
            device=resolved_device,
        )

    @torch.no_grad()
    def predict(self, x_physical: np.ndarray) -> np.ndarray:
        x_model = self.feature_scaler.transform(x_physical) if self.feature_scaler is not None else x_physical
        inputs = torch.tensor(x_model, dtype=torch.float32, device=self.device)
        predictions = self.model(inputs).cpu().numpy()
        if self.target_scaler is not None:
            predictions = self.target_scaler.inverse_transform(predictions)
        return predictions

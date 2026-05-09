"""Neural network with learned fighter embeddings for style interaction capture."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class FightDataset(Dataset):
    """PyTorch dataset for fight prediction."""

    def __init__(self, X: np.ndarray, fighter_a_ids: np.ndarray,
                 fighter_b_ids: np.ndarray, y: np.ndarray | None = None,
                 task: str = "winner"):
        self.X = torch.FloatTensor(np.array(X, copy=True))
        self.fighter_a_ids = torch.LongTensor(np.array(fighter_a_ids, copy=True))
        self.fighter_b_ids = torch.LongTensor(np.array(fighter_b_ids, copy=True))
        self.task = task
        if y is not None:
            y_copy = np.array(y, copy=True)
            if task == "duration":
                self.y = torch.FloatTensor(y_copy)
            else:
                self.y = torch.LongTensor(y_copy)
        else:
            self.y = None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.X[idx], self.fighter_a_ids[idx], self.fighter_b_ids[idx], self.y[idx]
        return self.X[idx], self.fighter_a_ids[idx], self.fighter_b_ids[idx]


class FightNet(nn.Module):
    """Neural network architecture with fighter embeddings."""

    def __init__(self, n_features: int, n_fighters: int, embedding_dim: int = 64,
                 hidden_dims: list[int] | None = None, dropout: float = 0.3,
                 n_outputs: int = 1, task: str = "winner"):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128, 64]
        self.task = task

        self.fighter_embedding = nn.Embedding(n_fighters + 1, embedding_dim, padding_idx=0)

        # Input: features + 2 fighter embeddings + embedding difference
        input_dim = n_features + embedding_dim * 3

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim

        self.feature_net = nn.Sequential(*layers)

        if task == "duration":
            self.output = nn.Linear(prev_dim, 1)
        elif task == "winner":
            self.output = nn.Linear(prev_dim, 1)
        else:
            self.output = nn.Linear(prev_dim, n_outputs)

    def forward(self, x: torch.Tensor, fighter_a_id: torch.Tensor,
                fighter_b_id: torch.Tensor) -> torch.Tensor:
        emb_a = self.fighter_embedding(fighter_a_id)
        emb_b = self.fighter_embedding(fighter_b_id)
        emb_diff = emb_a - emb_b

        combined = torch.cat([x, emb_a, emb_b, emb_diff], dim=1)
        hidden = self.feature_net(combined)
        out = self.output(hidden)

        if self.task == "winner":
            return torch.sigmoid(out).squeeze(-1)
        elif self.task == "duration":
            return out.squeeze(-1)
        else:
            return torch.softmax(out, dim=1)


class FighterNeuralNet:
    """Wrapper for training and prediction with the neural fighter model."""

    def __init__(self, task: str = "winner", params: dict[str, Any] | None = None):
        self.task = task
        self.model: FightNet | None = None
        self.calibrator = None
        self.fighter_to_id: dict[str, int] = {}
        self.feature_names: list[str] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        defaults = {
            "fighter_embedding_dim": 64,
            "hidden_dims": [256, 128, 64],
            "dropout": 0.3,
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 100,
            "patience": 15,
        }
        self.params = {**defaults, **(params or {})}

    def _get_fighter_id(self, fighter: str) -> int:
        if fighter not in self.fighter_to_id:
            self.fighter_to_id[fighter] = len(self.fighter_to_id) + 1
        return self.fighter_to_id[fighter]

    def _encode_fighters(self, fighters_a: pd.Series, fighters_b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        ids_a = np.array([self._get_fighter_id(f) for f in fighters_a])
        ids_b = np.array([self._get_fighter_id(f) for f in fighters_b])
        return ids_a, ids_b

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            fighters_a_train: pd.Series, fighters_b_train: pd.Series,
            X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None,
            fighters_a_val: pd.Series | None = None,
            fighters_b_val: pd.Series | None = None) -> None:
        """Train the neural network."""
        self.feature_names = list(X_train.columns)

        # Encode ALL fighters (train + val) before creating the model
        # so the embedding table is large enough
        ids_a_train, ids_b_train = self._encode_fighters(fighters_a_train, fighters_b_train)
        if fighters_a_val is not None:
            self._encode_fighters(fighters_a_val, fighters_b_val)

        n_outputs = 1 if self.task in ("winner", "duration") else y_train.nunique()
        self.model = FightNet(
            n_features=X_train.shape[1],
            n_fighters=len(self.fighter_to_id),
            embedding_dim=self.params["fighter_embedding_dim"],
            hidden_dims=self.params["hidden_dims"],
            dropout=self.params["dropout"],
            n_outputs=n_outputs,
            task=self.task,
        ).to(self.device)

        train_dataset = FightDataset(
            X_train.values, ids_a_train, ids_b_train,
            y_train.values, self.task
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.params["batch_size"],
            shuffle=True, drop_last=True
        )

        if self.task == "duration":
            criterion = nn.MSELoss()
        elif self.task == "winner":
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.params["learning_rate"], weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.params["epochs"]):
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                x, fa, fb, y = [b.to(self.device) for b in batch]
                optimizer.zero_grad()
                pred = self.model(x, fa, fb)
                if self.task == "winner":
                    loss = criterion(pred, y.float())
                else:
                    loss = criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            if X_val is not None:
                val_loss = self._validate(X_val, y_val, fighters_a_val, fighters_b_val, criterion)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= self.params["patience"]:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")

        if best_state:
            self.model.load_state_dict(best_state)
        self.model.eval()
        logger.info(f"Neural net {self.task} trained. Best val loss: {best_val_loss:.4f}")

    def _validate(self, X_val, y_val, fighters_a_val, fighters_b_val, criterion) -> float:
        self.model.eval()
        ids_a, ids_b = self._encode_fighters(fighters_a_val, fighters_b_val)
        dataset = FightDataset(X_val.values, ids_a, ids_b, y_val.values, self.task)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)

        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                x, fa, fb, y = [b.to(self.device) for b in batch]
                pred = self.model(x, fa, fb)
                if self.task == "winner":
                    loss = criterion(pred, y.float())
                else:
                    loss = criterion(pred, y)
                total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    def predict_proba(self, X: pd.DataFrame, fighters_a: pd.Series,
                      fighters_b: pd.Series) -> np.ndarray:
        """Get predicted probabilities."""
        self.model.eval()
        ids_a = np.array([self.fighter_to_id.get(f, 0) for f in fighters_a])
        ids_b = np.array([self.fighter_to_id.get(f, 0) for f in fighters_b])

        dataset = FightDataset(X.values, ids_a, ids_b, task=self.task)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)

        predictions = []
        with torch.no_grad():
            for batch in loader:
                x, fa, fb = [b.to(self.device) for b in batch[:3]]
                pred = self.model(x, fa, fb)
                predictions.append(pred.cpu().numpy())

        return np.concatenate(predictions)

    def calibrate(self, X_val: pd.DataFrame, y_val: pd.Series,
                  fighters_a: pd.Series, fighters_b: pd.Series) -> None:
        """Calibrate with isotonic regression."""
        if self.task == "duration":
            return
        raw = self.predict_proba(X_val, fighters_a, fighters_b)
        if self.task == "winner":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(raw, y_val)
        else:
            self.calibrator = []
            for c in range(raw.shape[1]):
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(raw[:, c], (y_val == c).astype(int))
                self.calibrator.append(iso)

    def predict(self, X: pd.DataFrame, fighters_a: pd.Series,
                fighters_b: pd.Series) -> np.ndarray:
        raw = self.predict_proba(X, fighters_a, fighters_b)
        if self.calibrator is None:
            return raw
        if self.task == "winner":
            return self.calibrator.transform(raw)
        calibrated = np.zeros_like(raw)
        for c, iso in enumerate(self.calibrator):
            calibrated[:, c] = iso.transform(raw[:, c])
        row_sums = calibrated.sum(axis=1, keepdims=True)
        return calibrated / np.maximum(row_sums, 1e-8)

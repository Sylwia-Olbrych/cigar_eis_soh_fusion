"""Shared convolutional BiLSTM model for SOH training and inference."""

from __future__ import annotations

import torch
from torch import nn


class SOHRegressor(nn.Module):
    """The convolutional BiLSTM configuration used by the CIGAR models."""

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 128,
        num_layers: int = 5,
        target_seq_length: int = 41,
    ) -> None:
        super().__init__()
        self.target_seq_length = target_seq_length
        self.conv = nn.Conv1d(input_size, input_size, kernel_size=1)
        self.conv_adjust = nn.Conv1d(
            input_size, input_size, kernel_size=20, stride=1, padding=0
        )
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sequence_length: int | None = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.permute(0, 2, 1)
        features = self.conv(features)
        if features.size(2) != self.target_seq_length:
            features = self.conv_adjust(features)
        if features.size(2) != self.target_seq_length:
            raise ValueError(
                f"Model expects a sequence that becomes {self.target_seq_length} steps "
                "after convolution"
            )
        features = features.permute(0, 2, 1)
        state_shape = (
            self.lstm.num_layers * 2,
            features.size(0),
            self.lstm.hidden_size,
        )
        if self.training:
            hidden = torch.randn(state_shape, device=features.device, dtype=features.dtype)
            cell = torch.randn(state_shape, device=features.device, dtype=features.dtype)
        else:
            hidden = torch.zeros(state_shape, device=features.device, dtype=features.dtype)
            cell = torch.zeros(state_shape, device=features.device, dtype=features.dtype)
        output, _ = self.lstm(features, (hidden, cell))
        output = self.dropout(output)
        return self.fc(output[:, -1, :])

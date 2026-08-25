"""Core UIAS network without participant metadata or dataset indexing."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


def _resnet18_single_channel() -> nn.Module:
    network = resnet18(weights=None)
    network.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    network.fc = nn.Identity()
    return network


class ResidualBlock1D(nn.Module):
    """Residual block used by the acoustic encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, 3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, 1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.skip(values)
        values = F.relu(self.bn1(self.conv1(values)), inplace=True)
        values = self.bn2(self.conv2(values))
        return F.relu(values + residual, inplace=True)


class AcousticEncoder(nn.Module):
    """Encode the aligned acoustic CIR into a fixed-size representation."""

    def __init__(self, output_dim: int = 2048) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.blocks = nn.Sequential(
            ResidualBlock1D(64, 64),
            ResidualBlock1D(64, 128, stride=2),
            ResidualBlock1D(128, 256, stride=2),
            ResidualBlock1D(256, 512, stride=2),
        )
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(self.stem(values.unsqueeze(1))))


class FeatureCalibration(nn.Module):
    """Bidirectional feature-level cross-modal calibration."""

    def __init__(self, dimension: int = 256) -> None:
        super().__init__()
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.output = nn.Linear(dimension, dimension)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(dimension)

    def forward(self, primary: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.query(primary) * self.key(context), dim=-1)
        calibrated = self.output(weights * self.value(context))
        return self.norm(primary + self.dropout(calibrated))


def _modality_projection(input_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(4096, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(2048, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(1024, 256),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.Dropout(0.3),
    )


def _factor_encoder() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(256, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(1024, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
    )


def alignment_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Barlow-Twins-style cross-modal alignment loss."""

    first = (first - first.mean(0)) / (first.std(0, unbiased=False) + 1e-5)
    second = (second - second.mean(0)) / (second.std(0, unbiased=False) + 1e-5)
    cross = first.T @ second / max(first.shape[0], 1)
    diagonal = (torch.diagonal(cross) - 1.0).square().sum()
    off_diagonal = cross - torch.diag_embed(torch.diagonal(cross))
    return diagonal + 5e-3 * off_diagonal.square().sum()


def orthogonality_loss(
    liveness: torch.Tensor, geometry: torch.Tensor
) -> torch.Tensor:
    return F.cosine_similarity(liveness, geometry, dim=1).abs().mean()


def anchor_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    condition_ids: torch.Tensor,
    anchor: torch.Tensor,
    anchor_condition_id: int,
    temperature: float,
) -> torch.Tensor:
    """Pull affected live samples toward the anchor and push spoofs away."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    normalized = F.normalize(features, dim=1)
    normalized_anchor = F.normalize(anchor.detach().reshape(1, -1), dim=1)
    labels = labels.reshape(-1)
    condition_ids = condition_ids.reshape(-1)
    affected_live = normalized[
        (labels > 0.5) & (condition_ids != int(anchor_condition_id))
    ]
    spoofs = normalized[labels <= 0.5]
    if affected_live.numel() == 0 or spoofs.numel() == 0:
        return features.sum() * 0.0
    positive = affected_live @ normalized_anchor.T / temperature
    negative = affected_live @ spoofs.T / temperature
    live_loss = -F.log_softmax(torch.cat((positive, negative), dim=1), dim=1)[
        :, 0
    ].mean()
    anchor_to_live = (normalized_anchor @ affected_live.T).flatten() / temperature
    anchor_to_spoof = (normalized_anchor @ spoofs.T).flatten() / temperature
    anchor_loss = torch.logsumexp(
        torch.cat((anchor_to_live, anchor_to_spoof)), dim=0
    ) - torch.logsumexp(anchor_to_live, dim=0)
    return 0.5 * (live_loss + anchor_loss)


class UIASNet(nn.Module):
    """The released multimodal architecture used by the training utilities."""

    def __init__(self) -> None:
        super().__init__()
        self.mmwave_2d_encoder = _resnet18_single_channel()
        self.mmwave_ra_encoder = _resnet18_single_channel()
        self.acoustic_encoder = AcousticEncoder(output_dim=2048)
        self.mmwave_projection = _modality_projection(1024)
        self.acoustic_projection = _modality_projection(2048)
        self.mmwave_calibration = FeatureCalibration()
        self.acoustic_calibration = FeatureCalibration()
        self.liveness_encoder = _factor_encoder()
        self.geometry_encoder = _factor_encoder()
        self.project_head = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.reconstruction = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.Dropout(0.3),
        )
        self.skin_anchor = nn.Parameter(torch.zeros(512))
        self.raw_skin_scale = nn.Parameter(torch.tensor(math.log(math.expm1(10.0))))
        self.raw_skin_threshold = nn.Parameter(torch.tensor(math.atanh(0.5)))

    @torch.no_grad()
    def initialize_skin_anchor(self, value: torch.Tensor) -> None:
        value = value.reshape(-1).to(self.skin_anchor)
        if value.shape != self.skin_anchor.shape:
            raise ValueError("skin-anchor dimension does not match ProjectHead")
        self.skin_anchor.copy_(value)

    def skin_logits(self, features: torch.Tensor) -> torch.Tensor:
        similarity = F.cosine_similarity(
            features, self.skin_anchor.reshape(1, -1), dim=1
        )
        scale = F.softplus(self.raw_skin_scale) + 1e-6
        threshold = torch.tanh(self.raw_skin_threshold)
        return scale * (similarity - threshold)

    def forward(
        self,
        input_2d: torch.Tensor,
        input_ra: torch.Tensor,
        acoustic: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        condition_ids: torch.Tensor | None = None,
        stage: int = 1,
        anchor_condition_id: int = 0,
        temperature: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        mmwave = self.mmwave_projection(
            torch.cat(
                (
                    self.mmwave_2d_encoder(input_2d.unsqueeze(1)),
                    self.mmwave_ra_encoder(input_ra.unsqueeze(1)),
                ),
                dim=1,
            )
        )
        acoustic_feature = self.acoustic_projection(
            self.acoustic_encoder(acoustic)
        )
        calibrated_mmwave = self.mmwave_calibration(mmwave, acoustic_feature)
        calibrated_acoustic = self.acoustic_calibration(acoustic_feature, mmwave)
        liveness = self.liveness_encoder(calibrated_mmwave - calibrated_acoustic)
        geometry = self.geometry_encoder(calibrated_mmwave + calibrated_acoustic)
        projected = self.project_head(liveness)
        losses = {
            "alignment": alignment_loss(mmwave, acoustic_feature),
            "orthogonality": orthogonality_loss(liveness, geometry),
            "reconstruction": F.mse_loss(
                self.reconstruction(liveness + geometry),
                calibrated_mmwave + calibrated_acoustic,
            ),
            "contrastive": projected.sum() * 0.0,
        }
        if stage == 2:
            if labels is None or condition_ids is None:
                raise ValueError("Stage II requires labels and condition_ids")
            losses["contrastive"] = anchor_contrastive_loss(
                projected,
                labels,
                condition_ids,
                self.skin_anchor,
                anchor_condition_id,
                temperature,
            )
        elif stage != 1:
            raise ValueError(f"unsupported stage: {stage}")
        return {
            "logits": self.skin_logits(projected),
            "projected": projected,
            "liveness": liveness,
            "geometry": geometry,
            **losses,
        }

"""RQ-PAD model and its ablation variants.

The model has two branches whose outputs are concatenated and classified by
an MLP:

* RGB branch: an ImageNet-pretrained ResNet50, fully fine-tuned, projected
  to a 512-d embedding.
* Reconstruction-quality branch: a frozen quality encoder applied to the
  real face and to its e4e reconstruction. The difference between the two
  embeddings (real - reconstruction) is projected to 512-d. Bona fide faces
  are reconstructed faithfully, so this gap is small for them and larger
  for presentation attacks.

Variants (used for the ablation study):

* ``full``      RGB branch + FIQA(real) - FIQA(reconstruction)
* ``wo_gen``    RGB branch + FIQA(real)              (no reconstruction)
* ``wo_fiqa``   RGB branch + ResNet50(real) - ResNet50(reconstruction)
* ``rgb_only``  RGB branch only
"""
from typing import Dict, Optional

import torch
from torch import nn
from torchvision import models

from .fiqa import load_fiqa

VARIANTS = ("full", "wo_gen", "wo_fiqa", "rgb_only")


def _resnet50_trunk() -> nn.Sequential:
    """ImageNet ResNet50 without its classification layer -> [B, 2048, 1, 1]."""
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    return nn.Sequential(*list(backbone.children())[:-1])


class RQPAD(nn.Module):
    """Reconstruction-quality guided face presentation attack detector.

    ``forward(rgb, real_q, recon_q)`` takes the 224x224 RGB face, and the
    112x112 real and reconstructed faces for the quality branch, and returns
    classification logits of shape ``[B, num_outputs]``.
    """

    def __init__(
        self,
        variant: str = "full",
        fiqa_weights: Optional[str] = None,
        num_outputs: int = 2,
        embed_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant '{variant}', expected one of {VARIANTS}")
        self.variant = variant

        self.rgb_encoder = _resnet50_trunk()
        self.rgb_proj = nn.Linear(2048, embed_dim)

        if variant == "rgb_only":
            self.quality_encoder = None
            fused_dim = embed_dim
        else:
            if variant == "wo_fiqa":
                self.quality_encoder, quality_dim = _resnet50_trunk(), 2048
            else:
                if fiqa_weights is None:
                    raise ValueError(f"Variant '{variant}' needs the FIQA weights (--fiqa_weights)")
                self.quality_encoder, quality_dim = load_fiqa(fiqa_weights), 512
            # The encoder weights are frozen. As in the original experiments, the
            # encoder still follows the model's train/eval mode, so its BatchNorm
            # running statistics adapt to the PAD data during training.
            self.quality_encoder.requires_grad_(False)
            self.quality_proj = nn.Linear(quality_dim, embed_dim)
            fused_dim = 2 * embed_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_outputs),
        )

    def _encode_quality(self, x: torch.Tensor) -> torch.Tensor:
        out = self.quality_encoder(x)
        # FIQAModel returns (embedding, quality score); the ResNet50 trunk returns a feature map.
        return out[0] if isinstance(out, tuple) else torch.flatten(out, 1)

    def forward(self, rgb: torch.Tensor, real_q: torch.Tensor, recon_q: torch.Tensor) -> torch.Tensor:
        features = self.rgb_proj(torch.flatten(self.rgb_encoder(rgb), 1))

        if self.quality_encoder is not None:
            with torch.no_grad():
                quality = self._encode_quality(real_q)
                if self.variant != "wo_gen":
                    quality = quality - self._encode_quality(recon_q)
            features = torch.cat([features, self.quality_proj(quality)], dim=1)

        return self.classifier(features)


# Parameter prefixes used by the original research scripts -> names used here.
_LEGACY_PREFIXES = {
    "resnet.": "rgb_encoder.",
    "resnet_proj.": "rgb_proj.",
    "fiqa_model.": "quality_encoder.",
    "gen_resnet.": "quality_encoder.",
    "fiqa_proj.": "quality_proj.",
    "gen_proj.": "quality_proj.",
    "mlp_glob.": "classifier.",
}


def _convert_legacy_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Rename keys of an original-script checkpoint.

    The auxiliary heads ``mlp_resnet`` / ``mlp_gen`` were never part of the
    loss and are dropped.
    """
    converted = {}
    for key, value in state.items():
        for old, new in _LEGACY_PREFIXES.items():
            if key.startswith(old):
                converted[new + key[len(old):]] = value
                break
    return converted


def load_checkpoint(model: RQPAD, path: str) -> dict:
    """Load weights saved by ``train.py`` or by the original research scripts."""
    checkpoint = torch.load(path, map_location="cpu")
    if "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state, checkpoint = _convert_legacy_state_dict(checkpoint), {}
    model.load_state_dict(state)
    return checkpoint

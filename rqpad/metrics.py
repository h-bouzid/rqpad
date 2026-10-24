"""PAD metrics (ISO/IEC 30107-3 terminology).

Scores are attack probabilities: a sample is classified as an attack when
``score >= threshold``.

* APCER: proportion of attacks classified as bona fide.
* BPCER: proportion of bona fide samples classified as attacks.
* ACER:  (APCER + BPCER) / 2. Computed at the threshold where the
  validation set reaches its EER, this is the usual HTER.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .datasets import ATTACK


@dataclass
class Scores:
    labels: np.ndarray
    scores: np.ndarray
    video_index: np.ndarray

    def per_video(self) -> "Scores":
        """Average frame scores over each video."""
        videos, inverse = np.unique(self.video_index, return_inverse=True)
        mean_scores = np.bincount(inverse, weights=self.scores) / np.bincount(inverse)
        first = np.unique(inverse, return_index=True)[1]
        return Scores(self.labels[first], mean_scores, videos)


def equal_error_rate(labels: np.ndarray, scores: np.ndarray):
    """EER and the threshold at which it is reached."""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=ATTACK)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(min(thresholds[i], 1.0))


def error_rates(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    is_attack = labels == ATTACK
    predicted_attack = scores >= threshold
    apcer = float(np.mean(~predicted_attack[is_attack])) if is_attack.any() else float("nan")
    bpcer = float(np.mean(predicted_attack[~is_attack])) if (~is_attack).any() else float("nan")
    return {
        "threshold": float(threshold),
        "accuracy": float(np.mean(predicted_attack == is_attack)),
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": (apcer + bpcer) / 2,
    }


def _report(val: Scores, test: Scores) -> dict:
    _, dev_threshold = equal_error_rate(val.labels, val.scores)
    both_classes = len(np.unique(test.labels)) == 2
    return {
        "auc": float(roc_auc_score(test.labels, test.scores)) if both_classes else float("nan"),
        "eer": equal_error_rate(test.labels, test.scores)[0] if both_classes else float("nan"),
        "dev_threshold": error_rates(test.labels, test.scores, dev_threshold),
        "fixed_threshold": error_rates(test.labels, test.scores, 0.5),
    }


def pad_report(val: Scores, test: Scores) -> dict:
    """Frame- and video-level test metrics; thresholds are chosen on ``val``."""
    return {
        "frame": _report(val, test),
        "video": _report(val.per_video(), test.per_video()),
    }

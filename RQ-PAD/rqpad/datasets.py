"""Video lists, train/val/test splits and the frame-level dataset.

Both datasets are handled as folders of frames named ``img_<i>.jpg``, one
folder per video, mirrored by a folder of e4e reconstructions:

HQ-WMCA (frames extracted with ``scripts/extract_frames.py``)::

    <hqwmca_root>/HQ-WMCA-frames/Bonafide/<video>/img_<i>.jpg
    <hqwmca_root>/HQ-WMCA-frames/{Impersonation,Obfuscation}/<attack>/<video>/img_<i>.jpg
    <hqwmca_root>/HQ-WMCA_e4e/...            (same layout)
    <hqwmca_root>/PROTOCOL-*.csv

SiW-Mv2::

    <siwmv2_root>/SiW-Mv2-crop/Live/<video>/img_<i>.jpg
    <siwmv2_root>/SiW-Mv2-crop/Spoof/<attack>/<video>/img_<i>.jpg
    <siwmv2_root>/SiW-Mv2-e4e/...            (same layout)
    <siwmv2_root>/{trainlist,testlist}_live.txt
"""
import csv
import logging
import math
import os
import random
import re
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

DATASETS = ("hqwmca", "siwmv2")
BONAFIDE, ATTACK = 0, 1

HQWMCA_FRAMES, HQWMCA_RECON = "HQ-WMCA-frames", "HQ-WMCA_e4e"
SIWMV2_FRAMES, SIWMV2_RECON = "SiW-Mv2-crop", "SiW-Mv2-e4e"

# Frames used per video unless overridden: every frame of the short HQ-WMCA
# clips, 100 evenly spaced frames for SiW-Mv2.
DEFAULT_FRAMES_PER_VIDEO = {"hqwmca": 0, "siwmv2": 100}

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
FIQA_MEAN, FIQA_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
FIQA_SIZE = 112

_FRAME_RE = re.compile(r"^img_(\d+)\.(?:jpe?g|png)$", re.IGNORECASE)


@dataclass(frozen=True)
class Video:
    frames_dir: str
    recon_dir: str
    label: int
    attack: str  # "bonafide" or the attack type


@dataclass
class Splits:
    train: List[Video]
    val: List[Video]
    test: List[Video]


# --------------------------------------------------------------------------- #
# Video discovery
# --------------------------------------------------------------------------- #
def _subdirs(path: str) -> List[str]:
    return sorted(e.path for e in os.scandir(path) if e.is_dir())


def _recon_dir(frames_dir: str, frames_root: str, recon_root: str) -> str:
    rel = os.path.relpath(frames_dir, frames_root)
    candidate = os.path.join(recon_root, rel)
    if os.path.isdir(candidate):
        return candidate
    # Some reconstruction folders drop the video extension ("clip.mov" -> "clip").
    return os.path.join(recon_root, os.path.splitext(rel)[0])


def _scan(frames_root: str, recon_root: str, bonafide_dir: str) -> List[Video]:
    """List videos under ``frames_root``.

    Bona fide videos live in ``<bonafide_dir>/<video>``, attacks in
    ``<category>/<attack>/<video>``.
    """
    if not os.path.isdir(frames_root):
        raise FileNotFoundError(f"Frame folder not found: {frames_root}")
    videos = []
    for category in _subdirs(frames_root):
        if os.path.basename(category) == bonafide_dir:
            entries = [(d, "bonafide", BONAFIDE) for d in _subdirs(category)]
        else:
            entries = [(d, os.path.basename(a), ATTACK) for a in _subdirs(category) for d in _subdirs(a)]
        videos += [Video(d, _recon_dir(d, frames_root, recon_root), label, attack) for d, attack, label in entries]
    return videos


def hqwmca_videos(root: str) -> List[Video]:
    return _scan(os.path.join(root, HQWMCA_FRAMES), os.path.join(root, HQWMCA_RECON), "Bonafide")


def siwmv2_videos(root: str) -> List[Video]:
    return _scan(os.path.join(root, SIWMV2_FRAMES), os.path.join(root, SIWMV2_RECON), "Live")


def _name(video: Video) -> str:
    return os.path.basename(video.frames_dir)


# --------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------- #
def hqwmca_protocol(root: str, protocol: str) -> Tuple[List[Video], List[Video], List[Video]]:
    """Official HQ-WMCA protocol CSV -> (train, dev, eval) videos.

    Each CSV row is ``<path>/<video>, <label>, <attack>, <subset>`` with
    label 0 for bona fide and subset in {train, dev, eval}.
    """
    path = protocol if os.path.isfile(protocol) else os.path.join(root, protocol)
    index = {_name(v): v for v in hqwmca_videos(root)}
    subsets = {"train": [], "dev": [], "eval": []}
    missing = 0
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 4 or not row[1].strip().isdigit():
                continue  # header or blank line
            subset = row[3].strip().lower()
            if subset not in subsets:
                continue
            video = index.get(row[0].strip().split("/")[-1])
            if video is None:
                missing += 1
                continue
            label = BONAFIDE if int(row[1]) == 0 else ATTACK
            subsets[subset].append(replace(video, label=label))
    if missing:
        logger.warning("%d videos listed in %s were not found under %s", missing, path, HQWMCA_FRAMES)
    return subsets["train"], subsets["dev"], subsets["eval"]


def siwmv2_leave_one_out(root: str, unseen_attack: str) -> Tuple[List[Video], List[Video]]:
    """SiW-Mv2 leave-one-attack-out protocol -> (train, test) videos.

    Bona fide videos are split with the official ``trainlist_live.txt`` /
    ``testlist_live.txt``; every attack type is used for training except
    ``unseen_attack``, which is held out for testing.
    """
    videos = siwmv2_videos(root)
    attacks = sorted({v.attack for v in videos if v.label == ATTACK})
    if unseen_attack not in attacks:
        raise ValueError(f"Unknown SiW-Mv2 attack '{unseen_attack}'. Available: {', '.join(attacks)}")

    live = {_name(v): v for v in videos if v.label == BONAFIDE}
    spoof = [v for v in videos if v.label == ATTACK]

    def read_ids(filename: str) -> List[Video]:
        with open(os.path.join(root, filename)) as f:
            ids = sorted({line.strip() for line in f if line.strip()})
        missing = sum(i not in live for i in ids)
        if missing:
            logger.warning("%d ids of %s have no folder under %s/Live", missing, filename, SIWMV2_FRAMES)
        return [live[i] for i in ids if i in live]

    train = read_ids("trainlist_live.txt") + [v for v in spoof if v.attack != unseen_attack]
    test = read_ids("testlist_live.txt") + [v for v in spoof if v.attack == unseen_attack]
    return train, test


def _stratified_split(videos: List[Video], val_fraction: float, seed: int) -> Tuple[List[Video], List[Video]]:
    train, val = train_test_split(
        videos, test_size=val_fraction, stratify=[v.label for v in videos], random_state=seed
    )
    return list(train), list(val)


def balance(videos: List[Video], seed: int) -> List[Video]:
    """Oversample the minority class by repeating its videos up to a 1:1 ratio."""
    groups = [[v for v in videos if v.label == c] for c in (BONAFIDE, ATTACK)]
    if not all(groups):
        raise ValueError("The training split needs both bona fide and attack videos.")
    target = max(len(g) for g in groups)
    balanced = []
    for group in groups:
        balanced += (group * math.ceil(target / len(group)))[:target]
    random.Random(seed).shuffle(balanced)
    return balanced


def subsample(videos: List[Video], max_videos: int, seed: int) -> List[Video]:
    """Keep at most ``max_videos`` videos, half from each class (for smoke tests)."""
    if max_videos <= 0 or len(videos) <= max_videos:
        return videos
    rng = random.Random(seed)
    kept = []
    for c in (BONAFIDE, ATTACK):
        group = [v for v in videos if v.label == c]
        kept += rng.sample(group, min(len(group), max(1, max_videos // 2)))
    return kept


def build_splits(
    train_set: str,
    test_set: str,
    roots: dict,
    protocol: Optional[str] = None,
    unseen_attack: Optional[str] = None,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Splits:
    """Build train/val/test video lists.

    Source dataset (``train_set``):
      * HQ-WMCA with ``protocol``: train / dev / eval subsets of the CSV.
      * SiW-Mv2 with ``unseen_attack``: leave-one-attack-out; validation is
        a stratified ``val_fraction`` of the training videos.
      * Otherwise: every video, with a stratified ``val_fraction`` held out
        for validation (cross-dataset training).

    Test set: the held-out protocol split when ``test_set == train_set``,
    every video of ``test_set`` otherwise. The training split is balanced
    by oversampling; validation and test are left untouched.
    """
    root = roots[train_set]
    held_out = None
    if train_set == "hqwmca" and protocol:
        train, val, held_out = hqwmca_protocol(root, protocol)
    elif train_set == "siwmv2" and unseen_attack:
        train, held_out = siwmv2_leave_one_out(root, unseen_attack)
        train, val = _stratified_split(train, val_fraction, seed)
    else:
        videos = hqwmca_videos(root) if train_set == "hqwmca" else siwmv2_videos(root)
        train, val = _stratified_split(videos, val_fraction, seed)

    if test_set == train_set:
        if held_out is None:
            raise ValueError("Intra-dataset evaluation needs --protocol (HQ-WMCA) or --unseen_attack (SiW-Mv2).")
        test = held_out
    else:
        test = hqwmca_videos(roots[test_set]) if test_set == "hqwmca" else siwmv2_videos(roots[test_set])

    return Splits(train=balance(train, seed), val=val, test=test)


# --------------------------------------------------------------------------- #
# Frame-level dataset
# --------------------------------------------------------------------------- #
def build_transforms(train: bool, image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Transforms for the RGB branch (ImageNet stats) and the quality branch ([-1, 1], 112x112)."""
    def make(size, mean, std):
        if train:
            steps = [transforms.RandomResizedCrop(size, scale=(0.85, 1.0)),
                     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)]
        else:
            steps = [transforms.Resize(size), transforms.CenterCrop(size)]
        return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(mean, std)])

    return make(image_size, IMAGENET_MEAN, IMAGENET_STD), make(FIQA_SIZE, FIQA_MEAN, FIQA_STD)


def _frame_names(folder: str) -> List[str]:
    """Frame file names of a video folder, sorted by frame index."""
    if not os.path.isdir(folder):
        return []
    matches = [(int(m.group(1)), name) for name in os.listdir(folder) if (m := _FRAME_RE.match(name))]
    return [name for _, name in sorted(matches)]


def _sample(names: List[str], k: int) -> List[str]:
    """``k`` evenly spaced frames (all frames when ``k <= 0``; repeated when the video is shorter)."""
    if k <= 0 or not names:
        return names
    if len(names) >= k:
        return [names[i] for i in np.linspace(0, len(names) - 1, k, dtype=int)]
    return (names * math.ceil(k / len(names)))[:k]


class PairedFrameDataset(Dataset):
    """Frames of a list of videos, each paired with its e4e reconstruction.

    An item is ``(rgb, real_q, recon_q, label, video_index)``: the RGB face
    for the CNN branch, the real and reconstructed faces for the quality
    branch, the label (0 bona fide, 1 attack) and the index of the video in
    ``self.videos`` (used for video-level scores).
    """

    def __init__(self, videos: List[Video], train: bool, image_size: int = 224, frames_per_video: int = 0):
        self.videos = list(videos)
        self.train = train
        self.image_size = image_size
        self.rgb_transform, self.quality_transform = build_transforms(train, image_size)

        self.samples = []
        self.num_missing = 0  # sampled frames skipped because their reconstruction is missing
        for index, video in enumerate(self.videos):
            recon_names = set(_frame_names(video.recon_dir))
            for name in _sample(_frame_names(video.frames_dir), frames_per_video):
                if name not in recon_names:
                    self.num_missing += 1
                    continue
                self.samples.append((os.path.join(video.frames_dir, name),
                                     os.path.join(video.recon_dir, name), video.label, index))
        if not self.samples:
            raise RuntimeError("No frames found: check the dataset roots and the frame/reconstruction folders.")

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: str) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB").resize((self.image_size, self.image_size))

    def __getitem__(self, i: int):
        frame_path, recon_path, label, video_index = self.samples[i]
        frame, recon = self._load(frame_path), self._load(recon_path)
        if self.train and torch.rand(1).item() < 0.5:  # same flip for the frame and its reconstruction
            frame, recon = ImageOps.mirror(frame), ImageOps.mirror(recon)
        return (self.rgb_transform(frame), self.quality_transform(frame),
                self.quality_transform(recon), label, video_index)

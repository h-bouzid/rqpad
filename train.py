"""Train and evaluate RQ-PAD.

Examples
--------
HQ-WMCA, official protocol (intra-dataset)::

    python train.py --train_set hqwmca --hqwmca_root data/HQ-WMCA \
        --protocol PROTOCOL-grand_test-curated.csv

SiW-Mv2, leave-one-attack-out (intra-dataset)::

    python train.py --train_set siwmv2 --siwmv2_root data/SiW-Mv2 --unseen_attack Paper

Cross-dataset (train on HQ-WMCA, test on all of SiW-Mv2)::

    python train.py --train_set hqwmca --test_set siwmv2 \
        --hqwmca_root data/HQ-WMCA --siwmv2_root data/SiW-Mv2

Evaluate a trained model without training::

    python train.py ... --checkpoint results/<run>/best.pt --eval_only
"""
import argparse
import csv
import json
import logging
import os
import random
import sys
import time

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from rqpad.datasets import DATASETS, DEFAULT_FRAMES_PER_VIDEO, PairedFrameDataset, build_splits, subsample
from rqpad.metrics import Scores, pad_report
from rqpad.models import RQPAD, VARIANTS, load_checkpoint

matplotlib.use("Agg")  # write figures without a display
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger("rqpad")

# Progress bars only in an interactive terminal, so SLURM log files stay readable.
PROGRESS = {"leave": False, "disable": not sys.stderr.isatty()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate RQ-PAD for face presentation attack detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument("--train_set", choices=DATASETS, required=True, help="dataset used for training")
    data.add_argument("--test_set", choices=DATASETS, default=None,
                      help="dataset used for testing; None = same as --train_set")
    data.add_argument("--hqwmca_root", default=None, help="HQ-WMCA root folder")
    data.add_argument("--siwmv2_root", default=None, help="SiW-Mv2 root folder")
    data.add_argument("--protocol", default=None,
                      help="HQ-WMCA protocol CSV, e.g. PROTOCOL-grand_test-curated.csv (file in --hqwmca_root or a path)")
    data.add_argument("--unseen_attack", default=None,
                      help="SiW-Mv2 attack type held out for testing (leave-one-attack-out), e.g. Paper")
    data.add_argument("--frames_per_video", type=int, default=None,
                      help="frames sampled per video, 0 = all; None = all for HQ-WMCA, 100 for SiW-Mv2")
    data.add_argument("--val_fraction", type=float, default=0.2,
                      help="fraction of training videos used for validation when no official split exists")
    data.add_argument("--image_size", type=int, default=224, help="input size of the RGB branch")
    data.add_argument("--max_videos", type=int, default=0,
                      help="cap on videos per split, for quick smoke tests (0 = no cap)")

    model = parser.add_argument_group("model")
    model.add_argument("--variant", choices=VARIANTS, default="full", help="model variant (see rqpad/models.py)")
    model.add_argument("--loss", choices=("ce", "bce"), default="ce",
                       help="ce: 2-logit cross-entropy, bce: 1-logit binary cross-entropy")
    model.add_argument("--fiqa_weights", default="pretrained/aikd_diffiqar_model.pth",
                       help="AI-KD DifFIQA(R) weights")
    model.add_argument("--dropout", type=float, default=0.2)

    optim = parser.add_argument_group("optimisation")
    optim.add_argument("--epochs", type=int, default=20)
    optim.add_argument("--batch_size", type=int, default=128)
    optim.add_argument("--lr", type=float, default=5e-7)
    optim.add_argument("--weight_decay", type=float, default=0.0)
    optim.add_argument("--lr_step", type=int, default=2, help="epochs between learning-rate decays")
    optim.add_argument("--gamma", type=float, default=0.5, help="learning-rate decay factor")

    run = parser.add_argument_group("run")
    run.add_argument("--output_dir", default="results")
    run.add_argument("--run_name", default=None, help="None = built from the experiment settings")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--num_workers", type=int, default=8)
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run.add_argument("--checkpoint", default=None, help="weights to start from (or to evaluate with --eval_only)")
    run.add_argument("--eval_only", action="store_true", help="skip training and evaluate --checkpoint")

    args = parser.parse_args()
    args.test_set = args.test_set or args.train_set
    for name in {args.train_set, args.test_set}:
        if getattr(args, f"{name}_root") is None:
            parser.error(f"--{name}_root is required when using {name}")
    if args.eval_only and args.checkpoint is None:
        parser.error("--eval_only requires --checkpoint")
    if args.epochs < 1 and not args.eval_only:
        parser.error("--epochs must be at least 1")
    return args


def default_run_name(args: argparse.Namespace) -> str:
    if args.train_set == "hqwmca" and args.protocol:
        split = os.path.splitext(os.path.basename(args.protocol))[0].replace("PROTOCOL-", "")
    elif args.train_set == "siwmv2" and args.unseen_attack:
        split = f"LOO-{args.unseen_attack}"
    else:
        split = "all"
    name = f"{args.train_set}-to-{args.test_set}_{split}_{args.variant}"
    return f"{name}_eval" if args.eval_only else name  # never overwrite the training run's outputs


def setup_logging(run_dir: str) -> None:
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(os.path.join(run_dir, "log.txt"))):
        handler.setFormatter(formatter)
        logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Objective:
    """Loss and attack score for the 2-logit (CE) or 1-logit (BCE) head."""

    def __init__(self, kind: str):
        self.kind = kind
        self.num_outputs = 2 if kind == "ce" else 1
        self._loss = nn.CrossEntropyLoss() if kind == "ce" else nn.BCEWithLogitsLoss()

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.kind == "ce":
            return self._loss(logits, labels.long())
        return self._loss(logits.squeeze(1), labels.float())

    def attack_score(self, logits: torch.Tensor) -> torch.Tensor:
        if self.kind == "ce":
            return logits.softmax(dim=1)[:, 1]
        return logits.squeeze(1).sigmoid()


def make_loader(videos, train: bool, dataset_name: str, args: argparse.Namespace) -> DataLoader:
    frames = args.frames_per_video
    if frames is None:
        frames = DEFAULT_FRAMES_PER_VIDEO[dataset_name]
    dataset = PairedFrameDataset(videos, train=train, image_size=args.image_size, frames_per_video=frames)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=train, num_workers=args.num_workers,
                      pin_memory=args.device.startswith("cuda"), persistent_workers=args.num_workers > 0)


def run_epoch(model, loader, objective, device, optimizer=None):
    """One pass over ``loader``; trains when an optimizer is given. Returns (loss, accuracy)."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for rgb, real_q, recon_q, labels, _ in tqdm(loader, desc="train" if training else "val", **PROGRESS):
            rgb, real_q, recon_q, labels = (t.to(device, non_blocking=True) for t in (rgb, real_q, recon_q, labels))
            logits = model(rgb, real_q, recon_q)
            loss = objective.loss(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += ((objective.attack_score(logits) >= 0.5).long() == labels).sum().item()
            seen += labels.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def predict(model, loader, objective, device) -> Scores:
    model.eval()
    labels, scores, videos = [], [], []
    for rgb, real_q, recon_q, y, video_index in tqdm(loader, desc="predict", **PROGRESS):
        logits = model(rgb.to(device), real_q.to(device), recon_q.to(device))
        scores.append(objective.attack_score(logits).cpu())
        labels.append(y)
        videos.append(video_index)
    return Scores(torch.cat(labels).numpy(), torch.cat(scores).numpy(), torch.cat(videos).numpy())


def fit(model, loaders, objective, args, run_dir: str) -> None:
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.gamma)
    logger.info("Trainable parameters: %.2fM", sum(p.numel() for p in trainable) / 1e6)

    best_path = os.path.join(run_dir, "best.pt")
    best_val_acc, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, loaders["train"], objective, args.device, optimizer)
        scheduler.step()
        val_loss, val_acc = run_epoch(model, loaders["val"], objective, args.device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})
        logger.info("Epoch %2d/%d | train loss %.4f acc %.4f | val loss %.4f acc %.4f | %.0fs",
                    epoch, args.epochs, train_loss, train_acc, val_loss, val_acc, time.time() - start)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val_acc": val_acc},
                       best_path)

    with open(os.path.join(run_dir, "history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    save_curves(history, os.path.join(run_dir, "curves.png"))

    checkpoint = load_checkpoint(model, best_path)
    logger.info("Best epoch: %d (val acc %.4f)", checkpoint["epoch"], checkpoint["val_acc"])


def save_curves(history, path: str) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ("loss", "acc")):
        ax.plot(epochs, [h[f"train_{metric}"] for h in history], marker="o", label="train")
        ax.plot(epochs, [h[f"val_{metric}"] for h in history], marker="o", label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy" if metric == "acc" else "loss")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_video_scores(scores: Scores, videos, path: str) -> None:
    per_video = scores.per_video()
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "attack", "label", "score"])
        for index, label, score in zip(per_video.video_index, per_video.labels, per_video.scores):
            writer.writerow([videos[index].frames_dir, videos[index].attack, int(label), f"{score:.6f}"])


def log_report(report: dict) -> None:
    logger.info("Test results (thresholds chosen on the validation set):")
    logger.info("  %-6s %7s %7s | %7s %7s %7s | %7s", "level", "AUC", "EER", "APCER", "BPCER", "HTER", "ACER@.5")
    for level, r in report.items():
        dev, fixed = r["dev_threshold"], r["fixed_threshold"]
        logger.info("  %-6s %7.4f %7.4f | %7.4f %7.4f %7.4f | %7.4f",
                    level, r["auc"], r["eer"], dev["apcer"], dev["bpcer"], dev["acer"], fixed["acer"])


def main() -> None:
    args = parse_args()
    run_dir = os.path.join(args.output_dir, args.run_name or default_run_name(args))
    os.makedirs(run_dir, exist_ok=True)
    setup_logging(run_dir)
    seed_everything(args.seed)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    device_name = torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "CPU"
    logger.info("Run directory: %s | device: %s", run_dir, device_name)

    roots = {"hqwmca": args.hqwmca_root, "siwmv2": args.siwmv2_root}
    splits = build_splits(args.train_set, args.test_set, roots, protocol=args.protocol,
                          unseen_attack=args.unseen_attack, val_fraction=args.val_fraction, seed=args.seed)

    loaders = {}
    for name, dataset_name in (("train", args.train_set), ("val", args.train_set), ("test", args.test_set)):
        if name == "train" and args.eval_only:
            continue
        videos = subsample(getattr(splits, name), args.max_videos, args.seed)
        loaders[name] = make_loader(videos, name == "train", dataset_name, args)
        dataset = loaders[name].dataset
        labels = [label for _, _, label, _ in dataset.samples]
        logger.info("%-5s %5d videos | %7d frames (%d bona fide, %d attack)", name,
                    len(dataset.videos), len(labels), labels.count(0), labels.count(1))
        if dataset.num_missing:
            logger.warning("%-5s %d frames skipped: no matching reconstruction", name, dataset.num_missing)

    objective = Objective(args.loss)
    model = RQPAD(args.variant, fiqa_weights=args.fiqa_weights,
                  num_outputs=objective.num_outputs, dropout=args.dropout)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)
        logger.info("Loaded weights from %s", args.checkpoint)
    model.to(args.device)

    if not args.eval_only:
        fit(model, loaders, objective, args, run_dir)

    val_scores = predict(model, loaders["val"], objective, args.device)
    test_scores = predict(model, loaders["test"], objective, args.device)
    report = pad_report(val_scores, test_scores)

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    save_video_scores(test_scores, loaders["test"].dataset.videos, os.path.join(run_dir, "test_scores.csv"))
    log_report(report)


if __name__ == "__main__":
    main()

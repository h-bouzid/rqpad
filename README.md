# RQ-PAD: Reconstruction Quality for Robust Face Presentation Attack Detection

<p align="center">
  <img src="https://img.shields.io/badge/ICPR-2026-blue" alt="ICPR 2026">
  <img src="https://img.shields.io/badge/Face%20PAD-Anti--Spoofing-purple" alt="Face PAD">
  <img src="https://img.shields.io/badge/PyTorch-%3E%3D1.13-orange" alt="PyTorch">
</p>

Official implementation of **RQ-PAD**, a face presentation attack detection
(PAD) framework that uses the *reconstruction behaviour* of a generative
face model as a complementary spoofing cue.

## Overview

Face anti-spoofing protects face recognition systems against presentation
attacks such as printed photos, replays and masks. Detecting sophisticated
and unseen attacks remains difficult for existing approaches.

RQ-PAD relies on a generative model trained **only on bona fide faces**.
Such a model reconstructs a realistic face whatever its input: for bona fide
samples the reconstruction stays close to the original, while presentation
attacks produce characteristic **discrepancies between the input and its
reconstruction**. RQ-PAD measures this discrepancy in the embedding space of
a face image quality model and uses it as an additional discriminative signal.

```
                ┌──────────────────────────┐
 face (224²) ──►│ ResNet50 (fine-tuned)    │──► 512-d ─────────────┐
                └──────────────────────────┘                       │
                ┌──────────────────────────┐                       ▼
 face (112²) ──►│ FIQA (frozen)            │──┐                ┌─────────┐
                └──────────────────────────┘  ├─ real − recon ►│ concat  │──► MLP ──► bona fide / attack
                ┌──────────────────────────┐  │    (512-d)     └─────────┘
 e4e recon ────►│ FIQA (frozen, shared)    │──┘
                └──────────────────────────┘
```

## Repository structure

```
RQ-PAD/
├── train.py                     # single entry point: training + evaluation
├── rqpad/
│   ├── datasets.py              # HQ-WMCA / SiW-Mv2 protocols, splits, frame dataset
│   ├── models.py                # RQ-PAD model and ablation variants
│   ├── fiqa.py                  # frozen FIQA network (AI-KD DifFIQA(R))
│   └── metrics.py               # AUC, EER, APCER, BPCER, ACER / HTER
├── scripts/
│   ├── extract_frames.py        # HQ-WMCA videos -> frame folders
│   ├── hqwmca_protocols.sh      # HQ-WMCA: grand test + 8 leave-one-out protocols
│   ├── siwmv2_leave_one_out.sh  # SiW-Mv2: 14 leave-one-attack-out runs
│   ├── cross_dataset.sh         # HQ-WMCA <-> SiW-Mv2
│   ├── ablation.sh              # ablation study on HQ-WMCA
│   └── slurm_example.sl         # SLURM job template
├── pretrained/                  # put the FIQA weights here
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/<user>/RQ-PAD.git
cd RQ-PAD
conda create -n rqpad python=3.10 -y
conda activate rqpad
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version first if needed
(see [pytorch.org](https://pytorch.org/get-started/locally/)). The ImageNet
ResNet50 weights are downloaded automatically by `torchvision` on first use.

## Data preparation

### 1. Datasets

Request access to the datasets from their owners:

- **HQ-WMCA**: Idiap Research Institute (RGB stream and protocol files `PROTOCOL-*.csv`)
- **SiW-Mv2**: Michigan State University (face crops and `trainlist_live.txt` / `testlist_live.txt`)

### 2. Extract HQ-WMCA frames

SiW-Mv2 is distributed as frame folders. HQ-WMCA ships videos, so decode
them once:

```bash
python scripts/extract_frames.py --hqwmca_root data/HQ-WMCA
# HQ-WMCA-RGB/<category>/<attack>/<video>.mov -> HQ-WMCA-frames/<category>/<attack>/<video>/img_<i>.jpg
```

### 3. Reconstruct every frame with e4e

We use the pretrained **FFHQ e4e encoder** from
[encoder4editing](https://github.com/omertov/encoder4editing). Download the
*FFHQ Inversion* checkpoint from its
[pretrained models](https://github.com/omertov/encoder4editing#pretrained-models)
section and run the official inference script on each frame folder:

```bash
python scripts/inference.py \
    --images_dir=/path/to/input/frames \
    --save_dir=/path/to/reconstructed/frames \
    /path/to/e4e_ffhq_encode.pt
```

Store the reconstructions in a tree that mirrors the frame tree, **with the
same file name as the frame they reconstruct** (`img_<i>.jpg`), in
`HQ-WMCA_e4e/` and `SiW-Mv2-e4e/`. Frames without a reconstruction are
skipped with a warning.

> Other face inversion or generative reconstruction models can be plugged in
> the same way, as long as they follow this folder layout. We encourage
> researchers to study their impact on RQ-PAD.

### 4. FIQA weights

Download the AI-KD DifFIQA(R) weights from the official
[AI-KD repository](https://github.com/LSIbabnikz/AI-KD) and place them at
`pretrained/aikd_diffiqar_model.pth`.

### Expected layout

```
data/
├── HQ-WMCA/
│   ├── HQ-WMCA-frames/
│   │   ├── Bonafide/<video>/img_<i>.jpg
│   │   ├── Impersonation/<attack>/<video>/img_<i>.jpg
│   │   └── Obfuscation/<attack>/<video>/img_<i>.jpg
│   ├── HQ-WMCA_e4e/                  # same layout as HQ-WMCA-frames
│   ├── PROTOCOL-grand_test-curated.csv
│   └── PROTOCOL-LOO_<attack>.csv
└── SiW-Mv2/
    ├── SiW-Mv2-crop/
    │   ├── Live/<video>/img_<i>.jpg
    │   └── Spoof/<attack>/<video>/img_<i>.jpg
    ├── SiW-Mv2-e4e/                  # same layout as SiW-Mv2-crop
    ├── trainlist_live.txt
    └── testlist_live.txt
```

## Training and evaluation

Everything goes through `train.py`. A run trains the model, keeps the
checkpoint with the best validation accuracy, then evaluates it on the test
set. The defaults are the settings used in the paper (20 epochs, batch size
128, Adam, learning rate 5e-7 halved every 2 epochs, seed 42), so only the
experiment needs to be specified.

**Smoke test.** Before launching a full run, check your data layout with a
few videos per split:

```bash
python train.py --train_set hqwmca --hqwmca_root data/HQ-WMCA \
    --protocol PROTOCOL-grand_test-curated.csv \
    --max_videos 8 --epochs 1 --batch_size 16 --num_workers 0
```

### HQ-WMCA (intra-dataset)

The protocol CSV defines the train, dev (validation) and eval (test) subsets.

```bash
python train.py --train_set hqwmca --hqwmca_root data/HQ-WMCA \
    --protocol PROTOCOL-grand_test-curated.csv

# grand test + the 8 leave-one-out protocols
HQWMCA_ROOT=data/HQ-WMCA bash scripts/hqwmca_protocols.sh
```

### SiW-Mv2 (intra-dataset, leave-one-attack-out)

Bona fide videos follow the official train/test lists. All attack types are
used for training except `--unseen_attack`, which is reserved for testing.
20% of the training videos are held out for validation.

```bash
python train.py --train_set siwmv2 --siwmv2_root data/SiW-Mv2 --unseen_attack Paper

# all 14 attack types
SIWMV2_ROOT=data/SiW-Mv2 bash scripts/siwmv2_leave_one_out.sh
```

### Cross-dataset

The model is trained on all videos of one dataset (with a stratified
validation split) and tested on **every** video of the other.

```bash
python train.py --train_set hqwmca --test_set siwmv2 \
    --hqwmca_root data/HQ-WMCA --siwmv2_root data/SiW-Mv2

python train.py --train_set siwmv2 --test_set hqwmca \
    --hqwmca_root data/HQ-WMCA --siwmv2_root data/SiW-Mv2

# both directions
HQWMCA_ROOT=data/HQ-WMCA SIWMV2_ROOT=data/SiW-Mv2 bash scripts/cross_dataset.sh
```

### Ablation study

`--variant` selects which parts of the model are used:

| `--variant` | RGB branch | Quality branch input | Quality encoder |
|---|---|---|---|
| `full` (default) | ✓ | real − reconstruction | FIQA |
| `wo_gen` | ✓ | real only | FIQA |
| `wo_fiqa` | ✓ | real − reconstruction | ImageNet ResNet50 (frozen) |
| `rgb_only` | ✓ | — | — |

```bash
python train.py --train_set hqwmca --hqwmca_root data/HQ-WMCA \
    --protocol PROTOCOL-grand_test-curated.csv --variant wo_gen

# all reduced variants on all 9 HQ-WMCA protocols
HQWMCA_ROOT=data/HQ-WMCA bash scripts/ablation.sh
```

### Evaluating a trained model

```bash
python train.py --train_set hqwmca --hqwmca_root data/HQ-WMCA \
    --protocol PROTOCOL-grand_test-curated.csv \
    --checkpoint results/hqwmca-to-hqwmca_grand_test-curated_full/best.pt --eval_only
```

Evaluation-only results go to a separate `<run_name>_eval/` folder, so the
training run's outputs are never overwritten.

The scripts forward extra arguments to `train.py`, e.g.
`bash scripts/hqwmca_protocols.sh --batch_size 64`. On a SLURM cluster, see
`scripts/slurm_example.sl`.

### Main arguments

| Argument | Default | Description |
|---|---|---|
| `--train_set` / `--test_set` | — / same as train | `hqwmca` or `siwmv2` |
| `--hqwmca_root`, `--siwmv2_root` | — | dataset root folders |
| `--protocol` | — | HQ-WMCA protocol CSV |
| `--unseen_attack` | — | SiW-Mv2 attack held out for testing |
| `--variant` | `full` | `full`, `wo_gen`, `wo_fiqa`, `rgb_only` |
| `--loss` | `ce` | `ce` (2 logits, cross-entropy) or `bce` (1 logit, binary cross-entropy) |
| `--frames_per_video` | all (HQ-WMCA), 100 (SiW-Mv2) | frames sampled per video, evenly spaced |
| `--epochs`, `--batch_size`, `--lr` | 20, 128, 5e-7 | optimisation |
| `--lr_step`, `--gamma`, `--weight_decay` | 2, 0.5, 0 | StepLR schedule and Adam weight decay |
| `--fiqa_weights` | `pretrained/aikd_diffiqar_model.pth` | FIQA checkpoint |
| `--output_dir`, `--run_name` | `results`, auto | where the run is saved |
| `--checkpoint`, `--eval_only` | — | resume from / evaluate weights |

Run `python train.py --help` for the full list.

## Outputs

Each run writes to `results/<run_name>/`:

| File | Content |
|---|---|
| `config.json` | all arguments of the run |
| `log.txt` | training log and final results table |
| `best.pt` | checkpoint with the best validation accuracy |
| `history.csv`, `curves.png` | loss and accuracy per epoch |
| `metrics.json` | frame- and video-level test metrics |
| `test_scores.csv` | per-video attack score, label and attack type |

## Evaluation protocol

Scores are attack probabilities. Metrics are reported at the **frame level**
and at the **video level**, where frame scores are averaged per video.

| Metric | Definition |
|---|---|
| AUC | area under the ROC curve on the test set |
| EER | equal error rate on the test set |
| APCER | attacks classified as bona fide |
| BPCER | bona fide samples classified as attacks |
| HTER | (APCER + BPCER) / 2 at the threshold where the **validation** set reaches its EER |
| ACER@0.5 | (APCER + BPCER) / 2 at a fixed 0.5 threshold |

All thresholds are chosen on the validation set only; the test set is never
used for model or threshold selection.

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{bouzid2026rq,
  title={RQ-PAD: Reconstruction Quality for Robust Face Presentation Attack Detection},
  author={Bouzid, Hamza and L{\'e}zoray, Olivier and Rosenberger, Christophe},
  booktitle={International Conference on Pattern Recognition},
  pages={651--666},
  year={2026},
  organization={Springer}
}
```

## Acknowledgements

This work builds on [encoder4editing](https://github.com/omertov/encoder4editing)
for face reconstruction, [AI-KD](https://github.com/LSIbabnikz/AI-KD) for face
image quality assessment and the IResNet backbone from
[InsightFace](https://github.com/deepinsight/insightface).

## License

This code is released under the **PolyForm Noncommercial License 1.0.0**
(see [LICENSE](LICENSE)). You may use it for research and other
non-commercial purposes only.

Copyright (c) 2026 &lt;copyright holder&gt;.

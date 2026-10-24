"""Decode HQ-WMCA videos into per-frame images.

Mirrors the video tree into a frame tree, one folder per video:

    <root>/HQ-WMCA-RGB/Impersonation/Print/<video>.mov
 -> <root>/HQ-WMCA-frames/Impersonation/Print/<video>/img_0.jpg, img_1.jpg, ...

These frames are the input of both RQ-PAD and the e4e reconstruction step.

Usage:
    python scripts/extract_frames.py --hqwmca_root data/HQ-WMCA
"""
import argparse
import os
import shutil
import sys

import cv2
from tqdm import tqdm

VIDEO_EXTENSIONS = (".mov", ".mp4", ".avi")


def extract(video_path: str, out_dir: str, quality: int) -> int:
    os.makedirs(out_dir, exist_ok=True)
    capture = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(out_dir, f"img_{count}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        count += 1
    capture.release()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hqwmca_root", required=True)
    parser.add_argument("--src", default="HQ-WMCA-RGB", help="video folder inside --hqwmca_root")
    parser.add_argument("--dst", default="HQ-WMCA-frames", help="output frame folder inside --hqwmca_root")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality")
    parser.add_argument("--overwrite", action="store_true", help="re-extract videos that already have frames")
    args = parser.parse_args()

    src_root = os.path.join(args.hqwmca_root, args.src)
    dst_root = os.path.join(args.hqwmca_root, args.dst)
    videos = sorted(
        os.path.join(folder, name)
        for folder, _, files in os.walk(src_root)
        for name in files if name.lower().endswith(VIDEO_EXTENSIONS)
    )
    if not videos:
        raise SystemExit(f"No videos found under {src_root}")

    failed = []
    for video in tqdm(videos, desc="Extracting", disable=not sys.stderr.isatty()):
        out_dir = os.path.join(dst_root, os.path.splitext(os.path.relpath(video, src_root))[0])
        if os.path.isdir(out_dir) and os.listdir(out_dir):
            if not args.overwrite:
                continue
            shutil.rmtree(out_dir)  # drop stale frames before re-extracting
        if extract(video, out_dir, args.quality) == 0:
            failed.append(video)

    print(f"Frames written to {dst_root}")
    if failed:
        print(f"{len(failed)} videos could not be decoded:", *failed, sep="\n  ")


if __name__ == "__main__":
    main()

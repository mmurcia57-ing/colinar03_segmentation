"""Create four-panel visual review sheets for AI mask proposals."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("artifacts/ai_proposals"))
    p.add_argument("--output", type=Path, default=Path("artifacts/review_sheets"))
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    for mask_path in sorted(args.input.glob("*_mobile_sam_proposal.png")):
        stem = mask_path.name.replace("_mobile_sam_proposal.png", "")
        image_path = root / "data/real/images" / f"{stem}.jpg"
        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        overlay = image.copy()
        overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([255, 70, 20])).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outlined = image.copy()
        cv2.drawContours(outlined, contours, -1, (0, 255, 255), 3)
        fig, ax = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        panels = [(image, "1. Original"), (mask, "2. MobileSAM candidate"), (overlay, "3. Semitransparent overlay"), (outlined, "4. Candidate contours")]
        for axis, (panel, title) in zip(ax.ravel(), panels):
            axis.imshow(panel, cmap="gray", vmin=0, vmax=1 if panel is mask else None)
            axis.set_title(title)
            axis.axis("off")
        fig.suptitle(f"{stem} — urban_built_surface — PROPOSAL, NOT GROUND TRUTH", fontsize=14)
        fig.savefig(args.output / f"{stem}_review.png", dpi=160)
        plt.close(fig)
        print(args.output / f"{stem}_review.png")


if __name__ == "__main__":
    main()

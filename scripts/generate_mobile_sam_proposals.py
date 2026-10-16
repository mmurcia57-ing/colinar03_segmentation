"""Generate editable MobileSAM proposals without touching canonical masks.

The existing mask is used only as an interaction seed: candidate MobileSAM
segments are selected when they overlap the seed. The output is explicitly a
proposal, never a human-validated ground truth.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


LABEL = "urban_built_surface"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, help="One image; default: all five")
    p.add_argument("--points-per-side", type=int, default=12)
    p.add_argument("--min-area", type=int, default=300)
    p.add_argument("--overlap", type=float, default=0.10)
    p.add_argument("--max-side", type=int, default=1024, help="Inference resize; output returns to original size")
    p.add_argument("--output", type=Path, default=Path("artifacts/ai_proposals"))
    return p.parse_args()


def polygon_shapes(mask: np.ndarray) -> list[dict]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    h, w = mask.shape
    for contour in contours:
        if cv2.contourArea(contour) < 20:
            continue
        points = contour[:, 0, :].astype(float)
        shapes.append({
            "label": LABEL,
            "points": [[float(x), float(y)] for x, y in points],
            "group_id": None,
            "description": "MobileSAM candidate; pending visual review",
            "shape_type": "polygon",
            "flags": {},
        })
    return shapes


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    images_dir = root / "data/real/images"
    seeds_dir = root / "data/real/masks"
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)

    import torch
    from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry

    checkpoint = root / "models/mobile_sam.pt"
    if not checkpoint.exists():
        raise SystemExit(f"Missing {checkpoint}; download the official MobileSAM checkpoint first.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = sam_model_registry["vit_t"](checkpoint=str(checkpoint))
    model.to(device=device)
    model.eval()
    generator = SamAutomaticMaskGenerator(
        model,
        points_per_side=args.points_per_side,
        pred_iou_thresh=0.70,
        stability_score_thresh=0.70,
        min_mask_region_area=args.min_area,
    )

    selected = [args.image] if args.image else sorted(images_dir.glob("*.jpg"))
    provenance = {
        "schema": "ai-proposal-provenance-v1",
        "created": date.today().isoformat(),
        "model": "MobileSAM vit_t",
        "checkpoint": "models/mobile_sam.pt",
        "device": device,
        "source_class": LABEL,
        "status": "proposal_pending_visual_review",
        "selection": {
            "method": "automatic masks intersecting existing seed mask",
            "points_per_side": args.points_per_side,
            "min_area": args.min_area,
            "minimum_seed_overlap": args.overlap,
            "max_side": args.max_side,
        },
        "warning": "Not human-validated ground truth; do not use for final QG1 until approved.",
        "images": {},
    }

    for image_path in selected:
        image_path = image_path if image_path.is_absolute() else images_dir / image_path.name
        seed_path = seeds_dir / f"{image_path.stem}_mask.png"
        image = np.asarray(Image.open(image_path).convert("RGB"))
        seed = np.asarray(Image.open(seed_path).convert("L")) > 0
        if seed.shape != image.shape[:2]:
            raise ValueError(f"Seed/image size mismatch for {image_path.name}")
        scale = min(1.0, args.max_side / max(image.shape[:2]))
        if scale < 1.0:
            infer_w = max(1, round(image.shape[1] * scale))
            infer_h = max(1, round(image.shape[0] * scale))
            infer_image = cv2.resize(image, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
            infer_seed = cv2.resize(seed.astype(np.uint8), (infer_w, infer_h), interpolation=cv2.INTER_NEAREST) > 0
        else:
            infer_image, infer_seed = image, seed
        candidates = generator.generate(infer_image)
        proposal_small = np.zeros(infer_seed.shape, dtype=bool)
        kept = 0
        for candidate in candidates:
            cmask = candidate["segmentation"]
            intersection = np.logical_and(cmask, infer_seed).sum()
            overlap = intersection / max(1, cmask.sum())
            seed_recall = intersection / max(1, infer_seed.sum())
            if overlap >= args.overlap and seed_recall >= 0.002:
                proposal_small |= cmask
                kept += 1
        proposal = cv2.resize(proposal_small.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        # Preserve seed pixels only as a conservative interaction hint; the
        # output remains a MobileSAM-selected proposal and is not canonical.
        proposal_u8 = (proposal * 255).astype(np.uint8)
        stem = image_path.stem
        out_mask = output / f"{stem}_mobile_sam_proposal.png"
        out_json = output / f"{stem}_mobile_sam_proposal.json"
        Image.fromarray(proposal_u8, mode="L").save(out_mask)
        labelme = {
            "version": "5.8.1",
            "flags": {},
            "shapes": polygon_shapes(proposal),
            "imagePath": image_path.name,
            "imageData": None,
            "imageHeight": int(image.shape[0]),
            "imageWidth": int(image.shape[1]),
            "description": "MobileSAM proposal pending human visual review",
        }
        out_json.write_text(json.dumps(labelme, ensure_ascii=False, indent=2), encoding="utf-8")
        provenance["images"][image_path.name] = {
            "seed_mask": str(seed_path.relative_to(root)),
            "proposal_mask": str(out_mask.relative_to(root)),
            "editable_annotation": str(out_json.relative_to(root)),
            "candidate_count": len(candidates),
            "kept_candidate_count": kept,
            "foreground_pixels": int(proposal.sum()),
        }
        print(f"{image_path.name}: {len(candidates)} candidates, {kept} kept -> {out_mask.name}")
    (output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

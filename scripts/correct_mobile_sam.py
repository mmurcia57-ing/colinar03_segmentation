"""Prompt-based, non-destructive MobileSAM correction interface."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

LABEL = "urban_built_surface"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("data/real/images/01_calle_mural.jpg"))
    parser.add_argument("--proposal-dir", type=Path, default=Path("artifacts/ai_proposals"), help="Context only; never used as truth")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ai_corrections"))
    parser.add_argument("--self-test", action="store_true", help="Run one positive/negative prompt inference without opening a window")
    return parser.parse_args()


def contours_to_shapes(mask: np.ndarray) -> list[dict]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [{
        "label": LABEL,
        "points": contour[:, 0, :].astype(float).tolist(),
        "group_id": None,
        "description": "Interactive MobileSAM correction",
        "shape_type": "polygon",
        "flags": {},
    } for contour in contours if cv2.contourArea(contour) >= 20]


def labelme_json(image_name: str, mask: np.ndarray) -> dict:
    return {
        "version": "5.8.1", "flags": {}, "shapes": contours_to_shapes(mask),
        "imagePath": image_name, "imageData": None,
        "imageHeight": int(mask.shape[0]), "imageWidth": int(mask.shape[1]),
        "description": "Interactive MobileSAM prompts; pending QG1 visual review",
    }


def main() -> None:
    parsed = parse_args()
    root = Path(__file__).resolve().parents[1]
    image_path = parsed.image if parsed.image.is_absolute() else root / parsed.image
    proposal_dir = parsed.proposal_dir if parsed.proposal_dir.is_absolute() else root / parsed.proposal_dir
    output_dir = parsed.output_dir if parsed.output_dir.is_absolute() else root / parsed.output_dir
    proposal_path = proposal_dir / f"{image_path.stem}_mobile_sam_proposal.png"
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")
    image = np.asarray(Image.open(image_path).convert("RGB"))
    context_proposal = np.asarray(Image.open(proposal_path).convert("L")) > 0 if proposal_path.exists() else None

    import torch
    from mobile_sam import SamPredictor, sam_model_registry

    checkpoint = root / "models/mobile_sam.pt"
    if not checkpoint.exists():
        raise SystemExit(f"Missing MobileSAM checkpoint: {checkpoint}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = sam_model_registry["vit_t"](checkpoint=str(checkpoint))
    model.to(device=device)
    model.eval()
    predictor = SamPredictor(model)
    predictor.set_image(image)

    def predict(points: list[tuple[float, float, int]]) -> tuple[np.ndarray, float, float]:
        coords = np.asarray([[x, y] for x, y, _ in points], dtype=np.float32)
        labels = np.asarray([label for _, _, label in points], dtype=np.int32)
        started = time.perf_counter()
        masks, scores, _ = predictor.predict(point_coords=coords, point_labels=labels, multimask_output=True)
        elapsed = time.perf_counter() - started
        best = int(np.argmax(scores))
        return masks[best].astype(bool), float(scores[best]), elapsed

    if parsed.self_test:
        h, w = image.shape[:2]
        test_points = [(w * 0.50, h * 0.45, 1), (w * 0.50, h * 0.05, 0)]
        mask, score, elapsed = predict(test_points)
        print(f"self_test=ok image={image_path.name} device={device} mask_pixels={int(mask.sum())} score={score:.4f} inference_seconds={elapsed:.3f}")
        return

    current_points: list[tuple[float, float, int]] = []
    current_mask: np.ndarray | None = None
    current_score = 0.0
    current_time = 0.0
    confirmed: list[dict] = []
    undo_stack: list[tuple] = []

    def snapshot() -> None:
        undo_stack.append((list(current_points), None if current_mask is None else current_mask.copy(), current_score, current_time, [dict(item, mask=item["mask"].copy(), points=list(item["points"])) for item in confirmed]))

    def accumulated() -> np.ndarray:
        result = np.zeros(image.shape[:2], dtype=bool)
        for item in confirmed:
            result |= item["mask"]
        return result

    def save() -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        final_mask = accumulated()
        stem = image_path.stem
        Image.fromarray((final_mask * 255).astype(np.uint8), mode="L").save(output_dir / f"{stem}_corrected.png")
        (output_dir / f"{stem}_corrected.json").write_text(json.dumps(labelme_json(image_path.name, final_mask), indent=2), encoding="utf-8")
        decisions = {
            "schema": "mobile-sam-prompt-correction-v2",
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "image": str(image_path.relative_to(root)), "class": LABEL,
            "model": "MobileSAM vit_t", "checkpoint": "models/mobile_sam.pt", "device": device,
            "inference": "SamPredictor with independent positive/negative prompts per confirmed region",
            "context_proposal": str(proposal_path.relative_to(root)) if context_proposal is not None else None,
            "regions": [{
                "region_id": index + 1,
                "positive_points": [[float(x), float(y)] for x, y, label in item["points"] if label == 1],
                "negative_points": [[float(x), float(y)] for x, y, label in item["points"] if label == 0],
                "score": item["score"], "inference_seconds": item["inference_seconds"],
            } for index, item in enumerate(confirmed)],
            "status": "correction_pending_visual_review",
            "warning": "Not approved ground truth until reviewed and explicitly accepted.",
        }
        (output_dir / f"{stem}_decisions.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")
        print(f"saved={output_dir} regions={len(confirmed)} pixels={int(final_mask.sum())}")

    fig, axes = plt.subplots(1, 4, figsize=(21, 6))
    try:
        fig.canvas.manager.set_window_title(f"MobileSAM prompts — {image_path.name}")
    except Exception:
        pass

    def redraw() -> None:
        axes[0].clear(); axes[1].clear(); axes[2].clear(); axes[3].clear()
        axes[0].imshow(image)
        axes[1].imshow(np.zeros(image.shape[:2], dtype=bool) if current_mask is None else current_mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Propuesta actual: agrega puntos" if current_mask is None else f"Actual score={current_score:.3f}, {current_time:.2f}s")
        axes[2].imshow(accumulated(), cmap="gray", vmin=0, vmax=1); axes[2].set_title(f"Acumulada: {len(confirmed)} regiones")
        if context_proposal is not None: axes[3].imshow(context_proposal, cmap="gray", vmin=0, vmax=1)
        axes[3].set_title("Contexto rechazado (no criterio)")
        for x, y, label in current_points:
            axes[0].scatter([x], [y], c="lime" if label == 1 else "red", s=70, edgecolors="black")
        for item in confirmed:
            for x, y, label in item["points"]:
                axes[0].scatter([x], [y], c="cyan" if label == 1 else "orange", s=45, edgecolors="black")
        axes[0].set_title("Izq=positivo | Der=negativo")
        axes[1].text(0.01, 0.01, "c confirmar | d eliminar última | u deshacer | r reiniciar | s guardar | q salir", transform=axes[1].transAxes, color="yellow", fontsize=9, bbox={"facecolor": "black", "alpha": 0.65})
        for axis in axes: axis.axis("off")
        fig.canvas.draw_idle()

    def click(event) -> None:
        nonlocal current_mask, current_score, current_time
        if event.inaxes is not axes[0] or event.xdata is None or event.ydata is None or event.button not in (1, 3): return
        snapshot(); label = 1 if event.button == 1 else 0
        current_points.append((float(event.xdata), float(event.ydata), label))
        current_mask, current_score, current_time = predict(current_points)
        print(f"prompt={'positive' if label == 1 else 'negative'} x={event.xdata:.1f} y={event.ydata:.1f} inference_seconds={current_time:.3f} score={current_score:.4f}")
        redraw()

    def key(event) -> None:
        nonlocal current_points, current_mask, current_score, current_time, confirmed
        if event.key == "c" and current_mask is not None:
            snapshot(); confirmed.append({"points": list(current_points), "mask": current_mask.copy(), "score": current_score, "inference_seconds": current_time})
            current_points = []; current_mask = None; current_score = 0.0; current_time = 0.0; redraw()
        elif event.key == "d" and confirmed:
            snapshot(); confirmed.pop(); redraw()
        elif event.key == "u" and undo_stack:
            current_points, current_mask, current_score, current_time, confirmed = undo_stack.pop(); redraw()
        elif event.key == "r":
            snapshot(); current_points = []; current_mask = None; current_score = 0.0; current_time = 0.0; confirmed = []; redraw()
        elif event.key == "s": save()
        elif event.key == "q": plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", click)
    fig.canvas.mpl_connect("key_press_event", key)
    redraw(); plt.show()


if __name__ == "__main__":
    main()

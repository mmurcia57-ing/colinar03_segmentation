"""Generate independent ADE20K semantic-segmentation proposals.

This script reads only the five original JPEGs. It does not read canonical
masks, MobileSAM proposals, Otsu results, or K-means results.
"""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation, __version__ as transformers_version


MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
MODEL_REVISION = "main"
LABEL = "urban_built_surface"
# Conservative structural/outdoor surface mapping from ADE20K's 150 classes.
INCLUDE_IDS = {
    0: "wall",
    1: "building",
    6: "road",
    11: "sidewalk",
    25: "house",
    48: "skyscraper",
    52: "path",
    53: "stairs",
    61: "bridge",
}
EXCLUDE_EXAMPLES = {
    2: "sky", 4: "tree", 9: "grass", 12: "person", 17: "plant",
    20: "car", 72: "palm", 80: "bus", 83: "truck", 102: "van",
    104: "fountain", 116: "minibike", 127: "bicycle",
}


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.astype(np.float32).copy()
    result[mask] = 0.55 * result[mask] + 0.45 * np.array([40, 180, 255], dtype=np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    images_dir = root / "data/real/images"
    output = root / "artifacts/semantic_proposals"
    output.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model.to(device)
    model.eval()
    id2label = {int(key): value for key, value in model.config.id2label.items()}
    missing = {idx: name for idx, name in INCLUDE_IDS.items() if id2label.get(idx) != name}
    if missing:
        raise RuntimeError(f"Model taxonomy changed; expected labels do not match: {missing}")

    provenance = {
        "schema": "semantic-proposal-provenance-v1",
        "created": date.today().isoformat(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transformers_version": transformers_version,
        "torch_version": torch.__version__,
        "device": device,
        "torch_num_threads": torch.get_num_threads(),
        "license": "NVIDIA Source Code License for SegFormer (non-commercial research/evaluation)",
        "license_url": "https://github.com/NVlabs/SegFormer/blob/master/LICENSE",
        "dataset_taxonomy": "ADE20K / scene_parse_150",
        "source_images_only": True,
        "forbidden_inputs": ["data/real/masks", "artifacts/ai_proposals", "Otsu", "K-means"],
        "target_class": LABEL,
        "included_ade20k_classes": INCLUDE_IDS,
        "excluded_examples": EXCLUDE_EXAMPLES,
        "limitations": [
            "ADE20K has no explicit facade or roof class.",
            "The mapping is a proposal rule, not human-validated ground truth.",
            "CPU inference is expected to be slow and domain shift may affect Colombian street scenes.",
        ],
        "images": {},
    }

    for image_path in sorted(images_dir.glob("*.jpg")):
        image = np.asarray(Image.open(image_path).convert("RGB")).copy()
        # Resize before preprocessing so CPU inference never receives the
        # multi-megapixel source image. The output is restored to source size.
        model_image = np.asarray(Image.fromarray(image).resize((512, 512), Image.Resampling.BILINEAR)).copy()
        inputs = processor(images=model_image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        start = time.perf_counter()
        with torch.inference_mode():
            logits = model(**inputs).logits
            # Argmax at model resolution first; upsampling all 150 logits to
            # multi-megapixel size would require several GB of RAM on CPU.
            class_map_small = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        class_map = np.asarray(Image.fromarray(class_map_small).resize((image.shape[1], image.shape[0]), Image.Resampling.NEAREST), dtype=np.uint8)
        inference_seconds = time.perf_counter() - start
        binary = np.isin(class_map, list(INCLUDE_IDS)).astype(np.uint8)
        stem = image_path.stem
        mask_path = output / f"{stem}_semantic_proposal.png"
        labels_path = output / f"{stem}_ade20k_labels.npy"
        sheet_path = output / f"{stem}_comparison.png"
        Image.fromarray(binary * 255, mode="L").save(mask_path)
        np.save(labels_path, class_map)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image); axes[0].set_title("Original")
        axes[1].imshow(binary, cmap="gray", vmin=0, vmax=1); axes[1].set_title("SegFormer ADE20K → urban_built_surface")
        axes[2].imshow(overlay(image, binary.astype(bool))); axes[2].set_title(f"Superposición | CPU {inference_seconds:.2f}s")
        for axis in axes: axis.axis("off")
        fig.tight_layout()
        fig.savefig(sheet_path, dpi=140)
        plt.close(fig)
        counts = {id2label[int(class_id)]: int((class_map == class_id).sum()) for class_id in np.unique(class_map)}
        provenance["images"][image_path.name] = {
            "proposal_mask": str(mask_path.relative_to(root)),
            "class_map": str(labels_path.relative_to(root)),
            "comparison": str(sheet_path.relative_to(root)),
            "inference_seconds": inference_seconds,
            "foreground_pixels": int(binary.sum()),
            "class_pixel_counts": counts,
        }
        print(f"{image_path.name}: {inference_seconds:.3f}s -> {mask_path.name}")

    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

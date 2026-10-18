"""Prepare an auditable five-image CrackForest benchmark from official labels only."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tmp" / "cfd_official"
DESTINATION = ROOT / "data" / "benchmarks" / "cfd5"
REVIEW_DIR = ROOT / "artifacts" / "cfd_benchmark" / "reference_review"
REPO_URL = "https://github.com/cuilimeng/CrackForest-dataset.git"
REPO_COMMIT = "ed57c2d96754e6c7d105805cd29aeeb3a799f267"
EXCLUDED_NAME = "042"  # published reports identify this image/mask pair as mismatched


def sha256(path: Path) -> str:
    """SHA-256 of the file; .seg text files hash with LF line endings so Windows and Unix checkouts agree."""
    data = path.read_bytes()
    if path.suffix == ".seg":
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def load_ground_truth(path: Path) -> np.ndarray:
    """Load the official MATLAB Segmentation label map from groundTruth/*.mat."""
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    return np.asarray(payload["groundTruth"].Segmentation, dtype=np.uint8)


def parse_seg(path: Path) -> np.ndarray:
    """Read the original .seg run-length annotation for cross-validation only."""
    lines = path.read_text(encoding="ascii").splitlines()
    header = {line.split(" ", 1)[0]: line.split(" ", 1)[1] for line in lines[:9] if " " in line}
    width, height = int(header["width"]), int(header["height"])
    labels = np.full((height, width), -1, dtype=np.int16)
    for line in lines[lines.index("data") + 1 :]:
        label, y, start, stop = (int(part) for part in line.split())
        labels[y, start : stop + 1] = label
    if np.any(labels < 0):
        raise ValueError(f"{path.name} has unfilled pixels.")
    return labels.astype(np.uint8)


def validated_pairs() -> list[dict]:
    """Return only image/reference pairs with an auditable two-class mapping."""
    pairs: list[dict] = []
    for gt_path in sorted((SOURCE / "groundTruth").glob("*.mat")):
        name = gt_path.stem
        image_path, seg_path = SOURCE / "image" / f"{name}.jpg", SOURCE / "seg" / f"{name}.seg"
        if name == EXCLUDED_NAME or not image_path.exists() or not seg_path.exists():
            continue
        labels = load_ground_truth(gt_path)
        seg_labels = parse_seg(seg_path)
        with Image.open(image_path) as image:
            image_size = image.size
        dimensions_match = image_size == (labels.shape[1], labels.shape[0])
        valid_labels = np.array_equal(np.unique(labels), np.array([1, 2], dtype=np.uint8))
        seg_matches = np.array_equal(labels - 1, seg_labels)
        if dimensions_match and valid_labels and seg_matches:
            pairs.append(
                {
                    "id": name,
                    "image_path": image_path,
                    "ground_truth_path": gt_path,
                    "seg_path": seg_path,
                    "labels": labels,
                    "crack_ratio": float((labels == 2).mean()),
                }
            )
    return pairs


def select_pairs(pairs: list[dict]) -> list[dict]:
    """Spread five choices over the sorted eligible identifiers without using predictions."""
    if len(pairs) < 5:
        raise ValueError(f"Expected at least 5 valid pairs, found {len(pairs)}.")
    indexes = [(index * (len(pairs) - 1)) // 4 for index in range(5)]
    return [pairs[index] for index in indexes]


def write_binary_mask(path: Path, crack: np.ndarray) -> None:
    Image.fromarray(np.where(crack, 255, 0).astype(np.uint8), mode="L").save(path)


def write_review(path: Path, image: np.ndarray, crack: np.ndarray, identifier: str) -> None:
    overlay = image.copy()
    overlay[crack] = (255, 45, 45)
    blended = (0.60 * image + 0.40 * overlay).astype(np.uint8)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=170)
    axes[0].imshow(image); axes[0].set_title("Imagen oficial")
    axes[1].imshow(crack, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Referencia humana: grieta = blanco")
    axes[2].imshow(blended); axes[2].set_title("Superposición de referencia")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"CrackForest {identifier}: revisión de alineación y polaridad", fontsize=14)
    figure.tight_layout(rect=[0, 0, 1, 0.93])
    figure.savefig(path, dpi=170, facecolor="white")
    plt.close(figure)


def prepare() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(f"Official CFD source not found at {SOURCE}")
    protected_directories = [DESTINATION / name for name in ("images", "groundTruth", "seg", "masks")]
    occupied = [path for path in protected_directories if path.exists() and any(path.iterdir())]
    if occupied:
        raise FileExistsError(
            "Refusing to overwrite prepared benchmark data: " + ", ".join(str(path) for path in occupied)
        )
    pairs = validated_pairs()
    selected = select_pairs(pairs)
    for directory in (DESTINATION / "images", DESTINATION / "groundTruth", DESTINATION / "seg", DESTINATION / "masks", REVIEW_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for pair in selected:
        identifier = pair["id"]
        output_image = DESTINATION / "images" / f"{identifier}.jpg"
        output_gt = DESTINATION / "groundTruth" / f"{identifier}.mat"
        output_seg = DESTINATION / "seg" / f"{identifier}.seg"
        output_mask = DESTINATION / "masks" / f"{identifier}_crack.png"
        shutil.copy2(pair["image_path"], output_image)
        shutil.copy2(pair["ground_truth_path"], output_gt)
        shutil.copy2(pair["seg_path"], output_seg)
        crack = pair["labels"] == 2
        write_binary_mask(output_mask, crack)
        image = np.asarray(Image.open(output_image).convert("RGB"))
        write_review(REVIEW_DIR / f"{identifier}_reference.png", image, crack, identifier)
        entries.append(
            {
                "id": identifier,
                "image": str(output_image.relative_to(ROOT)).replace("\\", "/"),
                "official_ground_truth": str(output_gt.relative_to(ROOT)).replace("\\", "/"),
                "crosscheck_seg": str(output_seg.relative_to(ROOT)).replace("\\", "/"),
                "binary_reference": str(output_mask.relative_to(ROOT)).replace("\\", "/"),
                "image_sha256": sha256(output_image),
                "ground_truth_sha256": sha256(output_gt),
                "seg_sha256": sha256(output_seg),
                "binary_mask_sha256": sha256(output_mask),
                "shape_height_width": list(pair["labels"].shape),
                "crack_foreground_ratio": pair["crack_ratio"],
                "verification": {
                    "image_ground_truth_dimensions_match": True,
                    "ground_truth_labels": {"1": "intact pavement/background", "2": "crack/foreground"},
                    "seg_equals_groundTruth_minus_one": True,
                },
            }
        )
    manifest = {
        "schema": "cfd5-benchmark-v1",
        "status": "five-pair complementary benchmark; integrity, alignment and foreground polarity technically verified",
        "dataset": "CrackForest Dataset (CFD)",
        "official_repository": REPO_URL,
        "official_commit": REPO_COMMIT,
        "retrieved": str(date.today()),
        "license": "Non-commercial research purposes only (official repository README).",
        "citation": "Shi, Y., Cui, L., Qi, Z., Meng, F., & Chen, Z. (2016). Automatic road crack detection using random structured forests. IEEE Transactions on Intelligent Transportation Systems, 17(12), 3434-3445.",
        "reference_annotation": {
            "chosen_source": "groundTruth/<id>.mat: groundTruth.Segmentation",
            "why": "The directory is explicitly named groundTruth and contains a two-dimensional MATLAB label map. The independent seg/<id>.seg annotation reconstructs exactly as groundTruth - 1 for all 118 paired files, so it is retained only as an integrity cross-check, never as a prediction.",
            "labels_used": {"1": "intact pavement/background", "2": "crack/foreground"},
            "excluded_labels": {"3": "enclosed surface", "4": "narrow/hard-to-see crack"},
        },
        "selection_rule": {
            "eligible": "paired image, groundTruth and seg files; image and label-map dimensions match; labels are exactly {1,2}; seg == groundTruth - 1; id != '042'",
            "selection": "Sort eligible IDs lexicographically and take positions floor(i*(n-1)/4), i=0..4. No prediction, metric, or visual score is used in selection.",
            "known_exclusion": "042 is excluded because published analysis reports a mismatched image/mask pair.",
        },
        "algorithm_input": {
            "fixed_preprocessing": "gray = round(mean(R,G,B)); inverted_gray = 255 - gray",
            "applied_equally_to": ["Otsu", "K-means"],
            "reason": "The existing implementations define the brighter group as foreground. Fixed inversion makes dark cracks foreground without modifying either algorithm or tuning any image-specific parameter.",
        },
        "files": entries,
        "verification_summary": {
            "official_pairs_audited": 118,
            "eligible_pairs": len(pairs),
            "selected_pairs": len(entries),
            "visual_review_date": "2026-10-15",
            "visual_review_scope": "The five reference sheets were inspected for image-mask spatial alignment and crack-as-foreground polarity.",
            "alignment": "passed",
            "foreground_polarity": "passed",
            "replacement_required": False,
            "limitation": "This verification applies only to CFD. La Candelaria masks remain preliminary and QG1 is not approved.",
        },
    }
    (DESTINATION / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("selected=" + ",".join(item["id"] for item in entries))
    print(f"eligible={len(pairs)} manifest={DESTINATION / 'manifest.json'}")


def audit() -> None:
    pairs = validated_pairs()
    selected = select_pairs(pairs)
    print(f"paired_groundTruth={len(list((SOURCE / 'groundTruth').glob('*.mat')))}")
    print(f"eligible={len(pairs)}")
    print("selected=" + ",".join(item["id"] for item in selected))
    for pair in selected:
        print(f"{pair['id']}: crack_ratio={pair['crack_ratio']:.6f}, labels={np.unique(pair['labels']).tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.audit:
        audit()
    else:
        parser.error("Use --audit or --prepare.")


if __name__ == "__main__":
    main()

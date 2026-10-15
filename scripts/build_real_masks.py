"""Rasteriza anotaciones manuales; no consulta Otsu, K-means ni otra segmentacion."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data" / "real" / "annotations.json"
IMAGE_DIR = ROOT / "data" / "real" / "images"
MASK_DIR = ROOT / "data" / "real" / "masks"


def rasterize(image_path: Path, polygons: list[list[list[float]]]) -> Image.Image:
    with Image.open(image_path) as source:
        width, height = source.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        points = [(round(x * (width - 1)), round(y * (height - 1))) for x, y in polygon]
        draw.polygon(points, fill=255)
    return mask


def main() -> None:
    manifest = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    for image_name, annotation in manifest["images"].items():
        output = MASK_DIR / f"{Path(image_name).stem}_mask.png"
        rasterize(IMAGE_DIR / image_name, annotation["polygons"]).save(output)
        print(f"created {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

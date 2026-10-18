"""Exporta anotaciones LabelMe revisadas a PNG binario determinista.

Este conversor no crea anotaciones ni decide la semántica: solo convierte
polígonos etiquetados explícitamente como ``urban_built_surface``. Las
anotaciones deben ser revisadas visualmente antes de exportarlas.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

try:
    from scripts.build_real_masks import rasterize, write_gray_png
except ModuleNotFoundError:  # ejecución directa: python scripts/export_labelme_masks.py
    from build_real_masks import rasterize, write_gray_png


def export_one(annotation_path: Path, image_path: Path, output_path: Path) -> None:
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    with Image.open(image_path) as image:
        width, height = image.size
    polygons: list[list[list[float]]] = []
    for shape in annotation.get("shapes", []):
        if shape.get("label") != "urban_built_surface":
            continue
        if shape.get("shape_type", "polygon") != "polygon":
            raise ValueError(f"Solo se admiten polígonos: {annotation_path}")
        points = shape.get("points", [])
        if len(points) < 3:
            raise ValueError(f"Polígono insuficiente: {annotation_path}")
        polygons.append([[x / (width - 1), y / (height - 1)] for x, y in points])
    if not polygons:
        raise ValueError(f"No hay polígonos urban_built_surface: {annotation_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_gray_png(output_path, rasterize(image_path, polygons))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=Path("data/real/labelme"))
    parser.add_argument("--images", type=Path, default=Path("data/real/images"))
    parser.add_argument("--output", type=Path, default=Path("data/real/masks_reviewed"))
    args = parser.parse_args()
    files = sorted(args.annotations.glob("*.json"))
    if not files:
        raise SystemExit(f"No se encontraron anotaciones editables en {args.annotations}")
    for annotation_path in files:
        image_name = f"{annotation_path.stem}.jpg"
        image_path = args.images / image_name
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        output_path = args.output / f"{annotation_path.stem}_mask.png"
        export_one(annotation_path, image_path, output_path)
        print(f"created {output_path}")


if __name__ == "__main__":
    main()

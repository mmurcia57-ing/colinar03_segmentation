"""Rasteriza anotaciones manuales; no consulta Otsu, K-means ni otra segmentacion."""
from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data" / "real" / "annotations.json"
IMAGE_DIR = ROOT / "data" / "real" / "images"
MASK_DIR = ROOT / "data" / "real" / "masks"


def _fill_polygon(mask: np.ndarray, polygon: list[list[float]]) -> None:
    """Rellena un polígono con regla de centro de píxel y aritmética determinista."""
    height, width = mask.shape
    vertices = [(float(x) * (width - 1), float(y) * (height - 1)) for x, y in polygon]
    for row in range(height):
        y = row + 0.5
        intersections = []
        for (x1, y1), (x2, y2) in zip(vertices, vertices[1:] + vertices[:1]):
            if (y1 <= y < y2) or (y2 <= y < y1):
                intersections.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
        intersections.sort()
        for left, right in zip(intersections[0::2], intersections[1::2]):
            first = max(0, int(np.ceil(left - 0.5)))
            last = min(width - 1, int(np.floor(right - 0.5)))
            if first <= last:
                mask[row, first : last + 1] = 255


def rasterize(image_path: Path, polygons: list[list[list[float]]]) -> np.ndarray:
    """Rasteriza manualmente sin depender de ImageDraw ni de su versión instalada."""
    with Image.open(image_path) as source:
        width, height = source.size
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        _fill_polygon(mask, polygon)
    return mask


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def write_gray_png(path: Path, mask: np.ndarray) -> None:
    """Escribe PNG grayscale con bytes canónicos, independientemente de Pillow."""
    height, width = mask.shape
    rows = b"".join(b"\x00" + bytes(row.tolist()) for row in mask)
    payload = zlib.compress(rows, level=9)
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    png += _chunk(b"IDAT", payload) + _chunk(b"IEND", b"")
    path.write_bytes(png)


def main() -> None:
    manifest = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    for image_name, annotation in manifest["images"].items():
        output = MASK_DIR / f"{Path(image_name).stem}_mask.png"
        write_gray_png(output, rasterize(IMAGE_DIR / image_name, annotation["polygons"]))
        print(f"created {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

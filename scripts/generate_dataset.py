"""Genera tres escenas sintéticas y sus máscaras ground truth determinísticas."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

def make_scene(index: int, size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    image = Image.new("L", (size, size), 25); mask = Image.new("1", (size, size), 0)
    di, dm = ImageDraw.Draw(image), ImageDraw.Draw(mask)
    if index == 1:
        di.ellipse((35, 45, 150, 180), fill=205); dm.ellipse((35, 45, 150, 180), fill=1)
        di.rectangle((145, 110, 225, 225), fill=150); dm.rectangle((145, 110, 225, 225), fill=1)
    elif index == 2:
        di.polygon([(35, 215), (125, 35), (225, 215)], fill=190); dm.polygon([(35, 215), (125, 35), (225, 215)], fill=1)
        di.ellipse((65, 90, 185, 205), fill=105); dm.ellipse((65, 90, 185, 205), fill=1)
    else:
        di.rounded_rectangle((30, 35, 225, 220), radius=22, fill=180); dm.rounded_rectangle((30, 35, 225, 220), radius=22, fill=1)
        di.line((30, 130, 225, 130), fill=80, width=8); dm.line((30, 130, 225, 130), fill=1, width=8)
    return np.asarray(image, dtype=np.uint8), np.asarray(mask, dtype=np.uint8) * 255

def main() -> None:
    image_dir, mask_dir = ROOT / "data" / "images", ROOT / "data" / "masks"
    image_dir.mkdir(parents=True, exist_ok=True); mask_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, 4):
        image, mask = make_scene(index)
        Image.fromarray(image).save(image_dir / f"scene_{index}.png")
        Image.fromarray(mask).save(mask_dir / f"scene_{index}_mask.png")
    print(f"Generated {len(list(image_dir.glob('*.png')))} image/mask pairs in {ROOT / 'data'}")

if __name__ == "__main__":
    main()

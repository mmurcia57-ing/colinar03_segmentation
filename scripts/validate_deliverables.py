"""Validate CFD integrity and coherence among JSON, executed notebook and PDF."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import nbformat
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    """SHA-256 of the file; .seg text files hash with LF line endings so Windows and Unix checkouts agree."""
    data = path.read_bytes()
    if path.suffix == ".seg":
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def compact(text: str) -> str:
    return "".join(text.split())


def main() -> None:
    manifest = json.loads((ROOT / "data" / "benchmarks" / "cfd5" / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        for field, digest_field in (
            ("image", "image_sha256"),
            ("official_ground_truth", "ground_truth_sha256"),
            ("crosscheck_seg", "seg_sha256"),
            ("binary_reference", "binary_mask_sha256"),
        ):
            path = ROOT / item[field]
            assert path.exists(), path
            assert sha256(path) == item[digest_field], f"Hash mismatch: {path}"
        with Image.open(ROOT / item["image"]) as image, Image.open(ROOT / item["binary_reference"]) as mask:
            assert image.size == mask.size, f"Dimension mismatch: {item['id']}"

    results = json.loads((ROOT / "deliverables" / "results.json").read_text(encoding="utf-8"))
    notebook = nbformat.read(ROOT / "notebooks" / "evaluacion_segmentacion.ipynb", as_version=4)
    output_text = "\n".join(
        str(output.get("text", ""))
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.output_type == "stream"
    )
    assert "NOTEBOOK_JSON_MATCH=PASS" in output_text, "Executed notebook did not verify JSON coherence."

    pdf_path = ROOT / "deliverables" / "informe_academico.pdf"
    info = subprocess.run(["pdfinfo", str(pdf_path)], check=True, capture_output=True, text=True).stdout
    page_line = next(line for line in info.splitlines() if line.startswith("Pages:"))
    page_count = int(page_line.split(":", 1)[1].strip())
    assert page_count <= 6, f"La memoria supera el límite de 6 páginas: {page_count}."
    with tempfile.TemporaryDirectory() as temp_dir:
        text_path = Path(temp_dir) / "report.txt"
        subprocess.run(["pdftotext", "-layout", str(pdf_path), str(text_path)], check=True)
        pdf_text = text_path.read_text(encoding="utf-8", errors="replace")
    compact_pdf = compact(pdf_text)
    assert compact(results["title"]) in compact_pdf, "PDF title differs from JSON title."

    cfd = results["cfd"]
    for row in cfd["rows"]:
        for field in ("iou", "dice", "precision", "recall"):
            assert f"{row[field]:.4f}" in pdf_text, f"Missing CFD {row['image']} {row['method']} {field}"
    for method, item in cfd["aggregate_pixel_weighted"].items():
        assert compact(str(item["confusion_matrix"])) in compact_pdf, f"Missing CFD aggregate matrix {method}"

    for row in results["la_candelaria"]["rows"]:
        for field in ("iou", "dice"):
            assert f"{row[field]:.4f}" in pdf_text, f"Missing La Candelaria {row['image']} {row['method']} {field}"

    print("CFD_INTEGRITY=PASS")
    print("NOTEBOOK_JSON_MATCH=PASS")
    print("PDF_JSON_MATCH=PASS")
    print(f"PDF_PAGES={page_count}")


if __name__ == "__main__":
    main()

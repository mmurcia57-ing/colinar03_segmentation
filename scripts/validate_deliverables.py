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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    assert page_count == 6, f"Expected exactly six designed pages, found {page_count}."
    with tempfile.TemporaryDirectory() as temp_dir:
        text_path = Path(temp_dir) / "report.txt"
        subprocess.run(["pdftotext", "-layout", str(pdf_path), str(text_path)], check=True)
        pdf_text = text_path.read_text(encoding="utf-8", errors="replace")
    compact_pdf = compact(pdf_text)
    assert compact(results["title"]) in compact_pdf, "PDF title differs from JSON title."

    cfd = results["complementary_benchmark"]
    for row in cfd["rows"]:
        for field in ("iou", "dice", "precision", "recall"):
            assert f"{row[field]:.4f}" in pdf_text, f"Missing CFD {row['image']} {row['method']} {field}"
        matrix = compact(str(cfd["confusion_matrices"][row["image"]][row["method"]]))
        assert matrix in compact_pdf, f"Missing CFD matrix {row['image']} {row['method']}"

    case = results["case_study"]
    for row in case["rows"]:
        for field in ("iou", "dice", "precision", "recall"):
            assert f"{row[field]:.4f}" in pdf_text, f"Missing La Candelaria {row['image']} {row['method']} {field}"
    for method in ("Otsu", "K-means"):
        matrix = compact(str(case["aggregate_pixel_weighted"][method]["confusion_matrix"]))
        assert matrix in compact_pdf, f"Missing aggregate La Candelaria matrix {method}"

    print("CFD_INTEGRITY=PASS")
    print("NOTEBOOK_JSON_MATCH=PASS")
    print("PDF_JSON_MATCH=PASS")
    print(f"PDF_PAGES={page_count}")


if __name__ == "__main__":
    main()

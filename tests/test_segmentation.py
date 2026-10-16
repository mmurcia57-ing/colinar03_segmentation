import numpy as np
from pathlib import Path
from PIL import Image
from segmentation_lab.algorithms import kmeans_segment, otsu_threshold
from segmentation_lab.metrics import binary_metrics, confusion_matrix
from scripts.build_real_masks import rasterize, write_gray_png
from scripts.export_labelme_masks import export_one

ROOT = Path(__file__).resolve().parents[1]

def test_otsu_separates_two_intensity_groups():
    image = np.array([[20, 20, 20, 220, 220, 220]], dtype=np.uint8)
    threshold, prediction = otsu_threshold(image)
    assert 20 <= threshold < 220
    assert prediction.tolist() == [[False, False, False, True, True, True]]

def test_kmeans_is_reproducible_and_labels_bright_group():
    image = np.array([[10, 10, 12, 245, 250, 248]], dtype=np.uint8)
    centroids, prediction = kmeans_segment(image, seed=11)
    assert np.allclose(np.sort(centroids), [10.6666667, 247.6666667])
    assert prediction.tolist() == [[False, False, False, True, True, True]]

def test_metrics_and_confusion_matrix_have_expected_values():
    truth = np.array([[1, 1, 0, 0]], dtype=bool)
    prediction = np.array([[1, 0, 1, 0]], dtype=bool)
    assert confusion_matrix(truth, prediction).tolist() == [[1, 1], [1, 1]]
    metrics = binary_metrics(truth, prediction)
    assert metrics["iou"] == 1 / 3; assert metrics["dice"] == 0.5
    assert metrics["precision"] == 0.5; assert metrics["recall"] == 0.5


def test_real_dataset_has_five_independent_manual_mask_pairs():
    annotations = __import__("json").loads((ROOT / "data/real/annotations.json").read_text(encoding="utf-8"))
    image_dir = ROOT / "data/real/images"
    mask_dir = ROOT / "data/real/masks"
    assert len(annotations["images"]) == 5
    for image_name, annotation in annotations["images"].items():
        with Image.open(image_dir / image_name) as image, Image.open(mask_dir / f"{Path(image_name).stem}_mask.png") as mask:
            assert image.size == mask.size
            assert set(np.asarray(mask.convert("L")).ravel()).issubset({0, 255})
        assert annotation["polygons"]


def test_real_annotations_use_one_nontrivial_target_class():
    annotations = __import__("json").loads((ROOT / "data/real/annotations.json").read_text(encoding="utf-8"))
    assert annotations["label"] == "urban_built_surface"
    for image_name in annotations["images"]:
        mask = np.asarray(Image.open(ROOT / "data/real/masks" / f"{Path(image_name).stem}_mask.png").convert("L"))
        foreground_ratio = float((mask == 255).mean())
        assert 0.01 < foreground_ratio < 0.99, (image_name, foreground_ratio)


def test_real_sources_cover_all_images_with_traceable_licenses():
    annotations = __import__("json").loads((ROOT / "data/real/annotations.json").read_text(encoding="utf-8"))
    sources = __import__("json").loads((ROOT / "data/real/sources.json").read_text(encoding="utf-8"))
    assert {entry["local_file"] for entry in sources["images"]} == set(annotations["images"])
    for entry in sources["images"]:
        assert entry["author"] and entry["license"]
        assert entry["source_page"].startswith("https://commons.wikimedia.org/wiki/File:")


def test_real_masks_regenerate_byte_for_byte(tmp_path):
    annotations = __import__("json").loads((ROOT / "data/real/annotations.json").read_text(encoding="utf-8"))
    for image_name, annotation in annotations["images"].items():
        generated = rasterize(ROOT / "data/real/images" / image_name, annotation["polygons"])
        output = tmp_path / f"{Path(image_name).stem}_mask.png"
        write_gray_png(output, generated)
        tracked = ROOT / "data/real/masks" / output.name
        assert output.read_bytes() == tracked.read_bytes(), image_name


def test_labelme_exporter_converts_only_target_polygons(tmp_path):
    annotation = tmp_path / "01_calle_mural.json"
    annotation.write_text(__import__("json").dumps({"shapes": [
        {"label": "urban_built_surface", "shape_type": "polygon", "points": [[0, 0], [100, 0], [100, 100]]},
        {"label": "person", "shape_type": "polygon", "points": [[0, 0], [10, 0], [10, 10]]},
    ]}), encoding="utf-8")
    output = tmp_path / "mask.png"
    export_one(annotation, ROOT / "data/real/images/01_calle_mural.jpg", output)
    values = set(np.asarray(Image.open(output).convert("L")).ravel())
    assert values.issubset({0, 255})
    assert 255 in values

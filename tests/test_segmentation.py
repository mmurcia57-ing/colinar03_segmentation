import numpy as np
from pathlib import Path
from PIL import Image
from segmentation_lab.algorithms import kmeans_segment, otsu_threshold
from segmentation_lab.metrics import binary_metrics, confusion_matrix

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

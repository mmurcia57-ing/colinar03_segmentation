"""Métricas binarias implementadas explícitamente para hacer auditable la evaluación."""
from __future__ import annotations
import numpy as np

def confusion_matrix(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Devuelve [[TN, FP], [FN, TP]] para máscaras booleanas."""
    truth = np.asarray(reference, dtype=bool).ravel()
    pred = np.asarray(prediction, dtype=bool).ravel()
    if truth.shape != pred.shape:
        raise ValueError("Referencia y predicción deben tener la misma forma.")
    tn = np.sum(~truth & ~pred); fp = np.sum(~truth & pred)
    fn = np.sum(truth & ~pred); tp = np.sum(truth & pred)
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)

def binary_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Calcula IoU, Dice, precisión, recall y exactitud."""
    tn, fp, fn, tp = confusion_matrix(reference, prediction).ravel()
    def safe(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 1.0
    return {"iou": safe(tp, tp + fp + fn), "dice": safe(2 * tp, 2 * tp + fp + fn),
            "precision": safe(tp, tp + fp), "recall": safe(tp, tp + fn),
            "accuracy": safe(tp + tn, tp + tn + fp + fn)}

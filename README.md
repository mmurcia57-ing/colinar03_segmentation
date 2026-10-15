# Laboratorio UNIR: Evaluación de la segmentación

Implementación reproducible de segmentación binaria mediante **Otsu** y **K-means**, con evaluación propia mediante IoU, Dice, precisión, recall y matriz de confusión. La evidencia principal usa cinco fotografías reales de La Candelaria, Bogotá; el dataset sintético se conserva como prueba de regresión.

## Estructura

- `src/segmentation_lab/algorithms.py`: Otsu y K-means implementados con NumPy.
- `src/segmentation_lab/metrics.py`: matriz `[[TN, FP], [FN, TP]]` y métricas binarias.
- `scripts/generate_dataset.py`: genera tres imágenes sintéticas distintas y sus máscaras ground truth.
- `data/real/images/`: cinco fotografías reales reutilizables de Wikimedia Commons.
- `data/real/sources.json`: autor, licencia, pagina de archivo y URL de descarga de cada fotografia.
- `data/real/annotations.json`: polígonos manuales, independientes de Otsu/K-means.
- `data/real/README.md`: definición de la clase `urban_built_surface`, atribución y límites de las anotaciones.
- `scripts/build_real_masks.py`: rasterización manual determinista y escritura PNG canónica.
- `scripts/export_labelme_masks.py`: conversión determinista de anotaciones LabelMe revisadas; no aprueba ground truth.
- `docs/qg0_assisted_ground_truth.md`: decisión de viabilidad y estado QG0/QG1.
- `notebooks/evaluacion_segmentacion.ipynb`: notebook reproducible con originales, ground truth, resultados y mapas de errores.
- `tests/test_segmentation.py`: pruebas unitarias.

## Ejecutar en VS Code Windows / Python 3.12

Desde la raíz del repositorio:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
python scripts/build_real_masks.py
python scripts/generate_dataset.py
python -m pytest -q
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/evaluacion_segmentacion.ipynb
```

En VS Code seleccione `.venv\Scripts\python.exe` como intérprete y abra el notebook. Las fotografías se versionan localmente y las máscaras se regeneran desde polígonos documentados, sin red ni algoritmos de segmentación. La prueba de regeneración compara los bytes PNG para detectar diferencias entre entornos.

## Criterio de evaluación

Las máscaras se interpretan como foreground cuando su valor es mayor que cero. Para cada imagen se comparan las predicciones contra ground truth. IoU y Dice evalúan solapamiento; precisión penaliza falsos positivos; recall penaliza falsos negativos; la matriz conserva los conteos absolutos. Son referencias manuales conservadoras a nivel de polígono, no anotaciones pixel-perfect: sombras, vegetación, vehículos y oclusiones introducen ambigüedad. Los resultados no se consideran validados hasta ejecutar las pruebas y el notebook completo en el entorno local.

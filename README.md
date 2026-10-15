# Laboratorio UNIR: Evaluación de la segmentación

Implementación reproducible de segmentación binaria mediante **Otsu** y **K-means**, con evaluación propia mediante IoU, Dice, precisión, recall y matriz de confusión.

## Estructura

- `src/segmentation_lab/algorithms.py`: Otsu y K-means implementados con NumPy.
- `src/segmentation_lab/metrics.py`: matriz `[[TN, FP], [FN, TP]]` y métricas binarias.
- `scripts/generate_dataset.py`: genera tres imágenes sintéticas distintas y sus máscaras ground truth.
- `notebooks/evaluacion_segmentacion.ipynb`: notebook reproducible con visualizaciones y tabla de resultados.
- `tests/test_segmentation.py`: pruebas unitarias.

## Ejecutar en VS Code Windows / Python 3.12

Desde la raíz del repositorio:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
python scripts/generate_dataset.py
python -m pytest -q
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/evaluacion_segmentacion.ipynb
```

En VS Code seleccione `.venv\Scripts\python.exe` como intérprete y abra el notebook. El dataset se regenera sin descargar datos, así que la evidencia es repetible.

## Criterio de evaluación

Las máscaras se interpretan como foreground cuando su valor es mayor que cero. Para cada imagen se comparan las predicciones contra ground truth. IoU y Dice evalúan solapamiento; precisión penaliza falsos positivos; recall penaliza falsos negativos; la matriz conserva los conteos absolutos. Los resultados no se consideran validados hasta ejecutar las pruebas y el notebook completo en el entorno local.

# Laboratorio UNIR: Evaluación de la segmentación

Comparación de cuatro segmentadores frente a *ground truth* con un evaluador propio (matriz de confusión, IoU, Dice, precisión, recall y exactitud):

| Segmentador | Familia | Papel |
|---|---|---|
| Otsu | Umbralización global | Línea base |
| K-means (k=2) | Agrupamiento de intensidades | Línea base |
| Region Growing | Crecimiento de regiones por semillas (Tema 10.4) | Método principal |
| Watershed con marcadores | Inundación morfológica (Tema 10.7) | Método principal |

- **CrackForest Dataset (CFD)**: cinco imágenes de pavimento con anotación oficial de grietas. Es la evaluación cuantitativa principal.
- **La Candelaria (Bogotá)**: cinco fotografías urbanas con máscaras manuales propias de `urban_built_surface`. Es una aplicación secundaria con métricas orientativas.

## Estructura

- `src/segmentation_lab/`: implementación propia de todo el pipeline.
  - `metrics.py`: evaluador (matriz de confusión y métricas).
  - `algorithms.py`: Otsu y K-means.
  - `region_growing.py`: crecimiento de regiones por semillas.
  - `watershed.py`: Watershed con marcadores (inundación por cola de prioridad).
  - `preprocessing.py`: gris, gaussiano, Sobel y resta de fondo.
  - `pipeline.py`: parámetros fijos por escenario.
  - `visualization.py`: figuras de pasos intermedios y de errores.
- `notebooks/evaluacion_segmentacion.ipynb`: notebook ejecutado que muestra los pasos principales de cada método y la evaluación completa.
- `deliverables/informe_academico.pdf`: memoria (máx. 6 páginas). `results.json` contiene los resultados canónicos y `figures/` las figuras.
- `scripts/run_integrated_evaluation.py`: evaluación completa (JSON y figuras).
- `scripts/build_integrated_notebook.py`, `scripts/build_academic_report.py`: generan el notebook y la memoria.
- `scripts/prepare_cfd_benchmark.py`: descarga auditada y selección determinista de CFD.
- `scripts/validate_deliverables.py`: comprueba hashes CFD y coherencia JSON–notebook–PDF.
- `tests/`: pruebas unitarias de métricas y segmentadores.

## Reproducir

Requiere Python 3.12. La validación final necesita `pdfinfo` y `pdftotext`: en macOS se instalan con `brew install poppler` y en Windows vienen con MiKTeX.

macOS / Linux:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt

mkdir -p tmp
git clone https://github.com/cuilimeng/CrackForest-dataset.git tmp/cfd_official
git -C tmp/cfd_official checkout ed57c2d96754e6c7d105805cd29aeeb3a799f267
python scripts/prepare_cfd_benchmark.py --prepare      # o --audit si ya está preparado

python scripts/run_integrated_evaluation.py
python scripts/build_integrated_notebook.py
export PYTHONPATH="$PWD/src:$PWD"
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/evaluacion_segmentacion.ipynb --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=600
python scripts/build_academic_report.py
python -m pytest -q
python scripts/validate_deliverables.py
```

Windows (PowerShell): los mismos comandos, con estas diferencias:
- crear el entorno con `py -3.12 -m venv .venv` y activarlo con `.\.venv\Scripts\Activate.ps1`;
- crear la carpeta con `New-Item -ItemType Directory -Force tmp`;
- fijar el path con `$env:PYTHONPATH = "$PWD\src;$PWD"`.

## Datos y licencias

Las fotografías de La Candelaria se atribuyen en `data/real/sources.json`. CFD declara uso exclusivo para investigación no comercial. Por eso este repositorio no redistribuye sus imágenes ni anotaciones: el manifiesto `data/benchmarks/cfd5/manifest.json` registra commit, regla de selección y hashes, y `prepare_cfd_benchmark.py` reconstruye la muestra.

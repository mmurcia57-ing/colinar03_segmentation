# Laboratorio UNIR: Evaluación de la segmentación

Comparación reproducible de las implementaciones propias de Otsu y K-means mediante IoU, Dice, precisión, recall, exactitud y matrices de confusión.

La Candelaria, Bogotá, es el caso de estudio principal con cinco fotografías reales. Sus máscaras de `urban_built_surface` son referencias manuales preliminares: fueron creadas independientemente de los segmentadores, pero no superaron una validación visual completa y no se presentan como *ground truth* validado.

CrackForest Dataset (CFD) es un benchmark complementario con anotaciones oficiales. Se usan cinco pares seleccionados mediante una regla determinista y el mismo preprocesamiento fijo para ambos métodos.

## Estructura principal

- `src/segmentation_lab/`: algoritmos y métricas existentes.
- `notebooks/evaluacion_segmentacion.ipynb`: único notebook académico, ejecutado con los dos escenarios.
- `scripts/run_integrated_evaluation.py`: evaluación conjunta y figuras canónicas.
- `scripts/prepare_cfd_benchmark.py`: auditoría y selección reproducible de CFD.
- `scripts/build_integrated_notebook.py`: generación del notebook sin duplicar algoritmos.
- `scripts/build_academic_report.py`: informe académico de seis páginas.
- `scripts/validate_deliverables.py`: hashes CFD y coherencia JSON–notebook–PDF.
- `scripts/export_labelme_masks.py`: conversión determinista de anotaciones LabelMe revisadas; no aprueba ground truth.
- `docs/qg0_assisted_ground_truth.md`: decisión de viabilidad y estado QG0/QG1.
- `data/benchmarks/cfd5/manifest.json`: procedencia, commit, licencia, etiquetas, hashes y controles.
- `deliverables/`: resultados JSON, figuras, fuente Markdown e informe PDF.

## Reproducir en Windows con Python 3.12

Desde la raíz del repositorio:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

New-Item -ItemType Directory -Force tmp | Out-Null
git clone https://github.com/cuilimeng/CrackForest-dataset.git tmp/cfd_official
git -C tmp/cfd_official checkout ed57c2d96754e6c7d105805cd29aeeb3a799f267
python scripts/prepare_cfd_benchmark.py --prepare

python scripts/run_integrated_evaluation.py
python scripts/build_integrated_notebook.py
$env:PYTHONPATH = "$PWD\src;$PWD"
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/evaluacion_segmentacion.ipynb --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=300
python scripts/build_academic_report.py
python -m pytest -q
python scripts/validate_deliverables.py
```

Si el benchmark ya está preparado, use `python scripts/prepare_cfd_benchmark.py --audit` en lugar de `--prepare`. La validación final del PDF requiere que `pdfinfo` y `pdftotext` estén disponibles; MiKTeX los proporciona en el entorno usado para esta entrega.

## Datos, licencia y publicación

Las fotografías de La Candelaria se atribuyen en `data/real/sources.json`. CFD declara uso exclusivo para investigación no comercial en su repositorio oficial. Por prudencia, este repositorio no redistribuye las imágenes, archivos `.mat`, `.seg` ni máscaras binarias derivadas de CFD; el manifiesto registra los hashes y los scripts permiten reconstruir la muestra desde el commit oficial.

Los entornos virtuales, pesos de modelos, datos CFD crudos y experimentos asistidos pesados permanecen excluidos del control de versiones. El informe y las figuras compactas se publican como evidencia del laboratorio académico.

## Estado metodológico

La integridad, alineación y polaridad de las cinco parejas CFD fueron verificadas. QG1 no está aprobado porque las referencias de La Candelaria siguen siendo preliminares. Las métricas urbanas deben interpretarse como provisionales; las conclusiones cuantitativas frente a anotaciones externas corresponden al benchmark CFD.

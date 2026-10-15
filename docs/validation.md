# Registro de validación

Fecha de preparación: 2026-10-15 (America/Bogota).

## Evidencia registrada

- Repositorio objetivo verificado: `https://github.com/mmurcia57-ing/colinar03_segmentation`.
- `main` conserva el commit inicial `fbbc913`; el trabajo está en `dev/evaluacion-segmentacion`.
- Dataset presente y versionado: `data/images/scene_1.png`, `scene_2.png`, `scene_3.png` y sus tres archivos `data/masks/*_mask.png`.
- Dataset real presente en `data/real/images/` con cinco fotografías y fuentes/licencias en `data/real/sources.json`.
- Ground truth independiente definido en `data/real/annotations.json` y rasterizado por `scripts/build_real_masks.py`.
- Notebook válido como JSON y con seis celdas: `notebooks/evaluacion_segmentacion.ipynb`.
- `git diff --check` debe ejecutarse en VS Code después de clonar la rama.

## Evidencia ejecutada

- `python scripts/build_real_masks.py`: creó 5 máscaras PNG a partir de `annotations.json`.
- `python -m pytest -q`: `4 passed`.
- `jupyter nbconvert --execute --to notebook --inplace notebooks/evaluacion_segmentacion.ipynb --ExecutePreprocessor.kernel_name=python3`: completó 4 celdas de código y dejó 12 salidas persistidas.
- El notebook imprimió resultados para las 5 imágenes, con IoU/Dice/precisión/recall y matrices `[[TN, FP], [FN, TP]]` para ambos métodos.

Promedios calculados desde las salidas ejecutadas: Otsu IoU `0.342545`, Dice `0.500933`, precisión `0.753879`, recall `0.395598`; K-means IoU `0.341752`, Dice `0.500237`, precisión `0.754377`, recall `0.395024`. Estos valores describen esta anotación manual y no constituyen una afirmación de superioridad general.

## Estado de ejecución

No se marca `pytest` ni la ejecución del notebook como aprobados en esta máquina: el host solo ofrece Python 3.13/3.14, no Python 3.12, y el lanzador quedó sin salida fiable durante la instalación de dependencias. Ejecutar en VS Code con Python 3.12:

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts/generate_dataset.py
python -m pytest -q
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/evaluacion_segmentacion.ipynb
```

Registrar en la entrega la salida completa de esos tres comandos; solo entonces declarar la validación aprobada.

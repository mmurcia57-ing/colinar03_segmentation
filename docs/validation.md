# Registro de validación

Fecha de preparación: 2026-10-15 (America/Bogota).

## Evidencia registrada

- Repositorio objetivo verificado: `https://github.com/mmurcia57-ing/colinar03_segmentation`.
- `main` conserva el commit inicial `fbbc913`; el trabajo está en `dev/evaluacion-segmentacion`.
- Dataset presente y versionado: `data/images/scene_1.png`, `scene_2.png`, `scene_3.png` y sus tres archivos `data/masks/*_mask.png`.
- Dataset real presente en `data/real/images/` con cinco fotografías y fuentes/licencias en `data/real/sources.json`.
- Ground truth independiente definido en `data/real/annotations.json` y rasterizado por `scripts/build_real_masks.py`.
- Notebook válido como JSON y con seis celdas: `notebooks/evaluacion_segmentacion.ipynb`.
- Las pruebas comprueban que las máscaras no sean vacías ni de cobertura total, que las cinco fuentes tengan autor/licencia/página verificable y que la regeneración produzca los mismos bytes PNG.

## Evidencia ejecutada

- `python scripts/build_real_masks.py`: creó 5 máscaras PNG a partir de `annotations.json`.
- `python -m pytest -q` con carpeta temporal dentro del workspace: `7 passed`.
- `python -m jupyter nbconvert --execute --to notebook --inplace notebooks/evaluacion_segmentacion.ipynb --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=180`: completó 4 celdas de código; el notebook quedó con 12 salidas persistidas.
- El notebook imprimió resultados para las 5 imágenes, con IoU/Dice/precisión/recall y matrices `[[TN, FP], [FN, TP]]` para ambos métodos.

Promedios calculados desde las salidas ejecutadas: Otsu IoU `0.342545`, Dice `0.500933`, precisión `0.753879`, recall `0.395598`, exactitud `0.401838`; K-means IoU `0.341752`, Dice `0.500237`, precisión `0.754377`, recall `0.395024`, exactitud `0.400755`. Estos valores describen esta anotación manual y no constituyen una afirmación de superioridad general.

## Estado de ejecución

La validación de este cambio se ejecutó en el host disponible con Python 3.14; no sustituye la comprobación solicitada en Python 3.12.10. En VS Code con Python 3.12:

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts/build_real_masks.py
python scripts/generate_dataset.py
python -m pytest -q
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/evaluacion_segmentacion.ipynb --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=180
```

Registrar en la entrega la salida completa de esos comandos. Si una regeneración cambia bytes, revisar primero versión del checkout y ejecutar la prueba byte a byte antes de aceptar cambios.

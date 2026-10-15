# Registro de validación

Fecha de preparación: 2026-10-15 (America/Bogota).

## Evidencia registrada

- Repositorio objetivo verificado: `https://github.com/mmurcia57-ing/colinar03_segmentation`.
- `main` conserva el commit inicial `fbbc913`; el trabajo está en `dev/evaluacion-segmentacion`.
- Dataset presente y versionado: `data/images/scene_1.png`, `scene_2.png`, `scene_3.png` y sus tres archivos `data/masks/*_mask.png`.
- Notebook válido como JSON y con cinco celdas: `notebooks/evaluacion_segmentacion.ipynb`.
- `git diff --check` debe ejecutarse en VS Code después de clonar la rama.

## Estado de ejecución

No se marca `pytest` ni la ejecución del notebook como aprobados en esta máquina: el host solo ofrece Python 3.13/3.14, no Python 3.12, y el lanzador quedó sin salida fiable durante la instalación de dependencias. Ejecutar en VS Code con Python 3.12:

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts/generate_dataset.py
python -m pytest -q
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/evaluacion_segmentacion.ipynb
```

Registrar en la entrega la salida completa de esos tres comandos; solo entonces declarar la validación aprobada.

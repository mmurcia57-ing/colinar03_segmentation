# Flujo local de propuestas asistidas

## Estado

MobileSAM `vit_t` se ejecuta en el entorno aislado `.venv-ai` con PyTorch CPU. La Intel UHD integrada no ofrece CUDA. Las propuestas se guardan en `artifacts/ai_proposals/` y no sustituyen `data/real/masks/`.

## Diagnóstico local ejecutado

Fecha de ejecución: 2026-10-15. El diagnóstico reproducible se ejecuta con:

```powershell
.\.venv-ai\Scripts\python.exe scripts/diagnose_local.py
```

Resultado resumido: Windows 11, Python 3.12.10, Intel integrada sin CUDA, aproximadamente 19.7 GB de RAM, 237.4 GB libres, PyTorch `2.14.1+cpu`, OpenCV `5.0.0`, scikit-image `0.26.0`, Matplotlib `3.11.2`, LabelMe `7.8.1` y checkpoint MobileSAM de 38.84 MB.

SAM 2 y SAM original no se instalaron: no era necesario asumir dependencias pesadas para un equipo sin CUDA. MobileSAM fue seleccionado porque el checkpoint y el runtime CPU ya están disponibles y permiten conservar anotaciones editables mediante LabelMe. La inferencia CPU es viable, pero lenta.

## Generación

```powershell
.\.venv-ai\Scripts\Activate.ps1
python scripts/generate_mobile_sam_proposals.py
python scripts/make_review_sheets.py
```

El script usa la máscara existente solo como semilla de interacción para conservar candidatos que intersectan la región de interés. Esto reduce anotación manual, pero no convierte la máscara ni la salida automática en ground truth.

## Revisión y aprobación

1. Abra cada PNG de `artifacts/review_sheets/`.
2. Revise original, máscara, superposición y contornos.
3. Abra el JSON correspondiente en LabelMe: `labelme data/real/images/01_calle_mural.jpg --output artifacts/ai_proposals/01_calle_mural_mobile_sam_proposal.json`.
4. Añada/elimine polígonos con la clase exacta `urban_built_surface`, excluyendo cielo, vegetación, personas y vehículos.
5. Exporte una máscara binaria y guárdela como copia revisada, nunca sobre el archivo original.
6. Registre la decisión en un archivo de revisión con `accepted`, `accepted_with_corrections` o `rejected`, nombre del revisor y fecha.

QG1 permanece bloqueado hasta que las cinco fotografías tengan una decisión visual. Solo después de copiar explícitamente las máscaras aprobadas a una ubicación de evaluación se debe ejecutar Otsu/K-means y calcular las métricas.

## Corrección interactiva de la propuesta rechazada

La generación original unió todos los segmentos que intersectaban la máscara semilla; por ello podía incorporar cielo, vegetación y personas, y también perder regiones que no intersectaban la semilla. La herramienta `scripts/correct_mobile_sam.py` ya no selecciona esos cuatro componentes: usa `SamPredictor` de MobileSAM con prompts positivos/negativos independientes para cada región de trabajo.

Demostración para `01_calle_mural` en Windows PowerShell, desde la raíz del repositorio:

```powershell
.\.venv-ai\Scripts\Activate.ps1
python scripts\correct_mobile_sam.py --image data\real\images\01_calle_mural.jpg
```

En la ventana, haz clic sobre la fotografía original:

- clic izquierdo: punto positivo sobre la fachada o pavimento;
- clic derecho: punto negativo sobre cielo, vegetación, personas u otro elemento ajeno;
- cada clic ejecuta `SamPredictor` y muestra el tiempo de inferencia;
- `c`: confirmar la propuesta actual como una región independiente;
- comienza la siguiente región con nuevos puntos, sin mezclar prompts;
- `d`: eliminar la última región confirmada;
- `u`: deshacer;
- `r`: reiniciar;
- `s`: guardar;
- `q`: cerrar.

Al guardar se crean, sin sobrescribir archivos existentes, `artifacts/ai_corrections/01_calle_mural_corrected.png`, `01_calle_mural_corrected.json` y `01_calle_mural_decisions.json`. El JSON de decisiones conserva regiones incluidas/excluidas, fuente, clase y estado pendiente de revisión. La máscara corregida tampoco es ground truth hasta la aprobación visual explícita.

Prueba técnica sin abrir una ventana:

```powershell
python scripts\correct_mobile_sam.py --self-test --image data\real\images\01_calle_mural.jpg
```

La prueba ejecuta un prompt positivo y uno negativo con MobileSAM CPU y registra `score`, píxeles e `inference_seconds`. La máscara preliminar rechazada aparece solo como contexto visual y no interviene en la selección.

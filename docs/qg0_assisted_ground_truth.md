# QG0/QG1: decisión sobre ground truth asistido

Fecha de auditoría: 2026-10-15.

## QG0: viabilidad técnica

La inspección del entorno disponible encontró Python 3.14.2, sin `torch`, `torchvision`, `sam2`, `segment_anything`, `cv2` ni `labelme`. `nvidia-smi` tampoco está disponible. Por tanto, no se ha demostrado que SAM 2, SAM original o MobileSAM puedan ejecutarse aquí, y no se descargaron pesos ni dependencias pesadas. Tampoco se puede declarar validación específica en Python 3.12.10 desde este host.

Decisión: no presentar una máscara generada automáticamente como ground truth. Se conserva la clase `urban_built_surface` y se ofrece un flujo de revisión humana asistida compatible con LabelMe. El flujo es reproducible y el exportador no selecciona regiones por Otsu ni K-means.

## QG1: estado por fotografía

| Imagen | Estado | Motivo |
|---|---|---|
| `01_calle_mural.jpg` | REVIEW REQUIRED | La máscara poligonal preliminar incluye áreas de mural/vegetación y no sigue todos los límites visibles. |
| `02_casa_cuervo.jpg` | REVIEW REQUIRED | La máscara separa el jardín, pero sus polígonos atraviesan arquitectura, senderos y vegetación sin contornos suficientes. |
| `03_sombras_tarde.jpg` | REVIEW REQUIRED | Las regiones de tejados, árboles, parque y vehículos requieren selección por objeto; la geometría actual es demasiado aproximada. |
| `04_vendedor_peaton.jpg` | REVIEW REQUIRED | Hay personas, puestos, vehículos y fachadas entrelazados; los polígonos actuales no son una referencia pixel-level defendible. |
| `05_calle_candelaria.jpg` | REVIEW REQUIRED | Las fachadas y el pavimento están parcialmente solapados con peatones y zonas de perspectiva; se necesita ajuste visual. |

QG1 queda bloqueado. La prueba de que una máscara no sea vacía o completamente blanca no equivale a validación semántica.

## Flujo para completar QG1

1. Instalar LabelMe en Python 3.12.
2. Abrir `data/real/images/` y crear un polígono por región construida visible.
3. Etiquetar exclusivamente con `urban_built_surface`; no etiquetar cielo, vegetación, personas, vehículos ni sombras sin superficie construida.
4. Guardar un JSON editable por fotografía en `data/real/labelme/`.
5. Revisar cada JSON sobre la imagen original y conservar una copia de la lámina de revisión.
6. Exportar con `python scripts/export_labelme_masks.py`.
7. Promover las máscaras a `data/real/masks/` solo después de la revisión humana y actualizar el estado de cada imagen a PASS.

El script `export_labelme_masks.py` solo convierte polígonos ya etiquetados y escribe PNG con el mismo rasterizador determinista del proyecto; no produce propuestas semánticas por sí mismo.

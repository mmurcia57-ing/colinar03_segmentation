# Anotaciones editables pendientes de revisión

Esta carpeta debe contener un JSON LabelMe por fotografía, creado y revisado visualmente sobre el archivo original. Los JSON no se rellenan automáticamente porque las propuestas actuales no han superado QG1.

Cada región objetivo debe tener la etiqueta exacta `urban_built_surface`. El exportador es:

```powershell
python scripts/export_labelme_masks.py
```

El resultado se escribe en `data/real/masks_reviewed/` para evitar sobrescribir las máscaras preliminares. Solo tras la revisión humana y el registro de una lámina original/máscara/superposición se deben promover las máscaras a la evaluación principal.

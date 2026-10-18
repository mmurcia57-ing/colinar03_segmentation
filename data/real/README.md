# Evidencia real: La Candelaria

Las cinco fotografias de `images/` son archivos descargados de Wikimedia Commons. La atribucion, licencia, pagina de origen y URL de descarga se conservan en [`sources.json`](sources.json). Antes de redistribuir el dataset hay que conservar esa atribucion y revisar las condiciones de cada licencia.

La unica clase evaluada es `urban_built_surface`: fachadas, cubiertas, balcones, puertas, ventanas y pavimento. Se excluyen cielo, vegetacion, personas y vehiculos. Las mascaras de `masks/` se construyen exclusivamente desde los poligonos manuales de [`annotations.json`](annotations.json); no se obtienen de Otsu, K-means ni de otro resultado automatico.

La rasterizacion usa una regla de centro de pixel y un escritor PNG grayscale canonico implementados en `scripts/build_real_masks.py`. El objetivo es que una regeneracion con Python 3.12 en Windows produzca exactamente los mismos bytes, no solo la misma imagen visual. La prueba `test_real_masks_regenerate_byte_for_byte` comprueba esa propiedad.

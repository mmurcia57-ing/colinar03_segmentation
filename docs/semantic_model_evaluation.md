# Evaluación de propuesta semántica ADE20K

## Resultado

SegFormer B0 se ejecutó sobre las cinco fotografías usando únicamente las imágenes JPEG originales. No se utilizaron máscaras anteriores, MobileSAM, Otsu ni K-means. La alternativa es técnicamente viable en CPU, pero las propuestas no son suficientemente precisas para sustituir un ground truth visualmente validado.

## Modelo y entorno

- Identificador: `nvidia/segformer-b0-finetuned-ade-512-512`.
- Revisión: `main`.
- Arquitectura: SegFormer B0 fine-tuned on ADE20K, entrada 512x512.
- Pesos: `model.safetensors`, aproximadamente 15 MB; snapshot aproximado 45.4 MB.
- Transformers: `5.20.0`.
- PyTorch: `2.14.1+cpu`.
- Dispositivo: CPU, sin CUDA.
- Licencia: NVIDIA Source Code License for SegFormer, uso no comercial para investigación/evaluación.
- Fuente de licencia: https://github.com/NVlabs/SegFormer/blob/master/LICENSE.

## Mapeo reproducible

Se incluyeron exclusivamente estas clases ADE20K:

| ID | Clase ADE20K | Mapeo |
|---:|---|---|
| 0 | wall | `urban_built_surface` |
| 1 | building | `urban_built_surface` |
| 6 | road | `urban_built_surface` |
| 11 | sidewalk | `urban_built_surface` |
| 25 | house | `urban_built_surface` |
| 48 | skyscraper | `urban_built_surface` |
| 52 | path | `urban_built_surface` |
| 53 | stairs | `urban_built_surface` |
| 61 | bridge | `urban_built_surface` |

No inclusión se hizo por color, proximidad o por usar las máscaras previas. Se excluyeron explícitamente, entre otras, `sky`, `tree`, `grass`, `plant`, `person`, `car`, `bus`, `truck`, `van`, `minibike`, `bicycle` y `fountain`. ADE20K no contiene una clase explícita para fachada o tejado; `floor` se dejó fuera porque también puede representar interiores.

## Tiempos CPU medidos

| Imagen | Inferencia |
|---|---:|
| 01_calle_mural | 1.255 s |
| 02_casa_cuervo | 1.340 s |
| 03_sombras_tarde | 1.018 s |
| 04_vendedor_peaton | 1.310 s |
| 05_calle_candelaria | 1.118 s |

La inferencia se hizo a 512x512 y la clase ganadora se amplió a la resolución original. Esto evita consumir varios GB al ampliar simultáneamente los 150 logits.

## Revisión visual y decisión

- `01_calle_mural`: incorpora vegetación y zonas ajenas junto con muros y vía.
- `02_casa_cuervo`: reconoce parte de la edificación, pero mezcla superficies del patio, fuente y vegetación.
- `03_sombras_tarde`: detecta edificios y vías de forma fragmentaria, con confusión en árboles y techos.
- `04_vendedor_peaton`: incluye zonas de mercado, personas, vehículos y objetos junto con el pavimento.
- `05_calle_candelaria`: incluye personas y regiones no construidas por la baja resolución semántica.

Decisión: **propuestas rechazadas como ground truth**. QG1 no se aprueba automáticamente.

## Alternativa pública recomendada

Si el objetivo es cerrar el laboratorio con un ground truth verificable sin anotación manual extensa, debe separarse el experimento local de La Candelaria de una referencia pública anotada. La alternativa académica recomendada es **Cityscapes**, cuyo dataset proporciona anotaciones semánticas finas para escenas urbanas y clases como `building`, `road`, `sidewalk`, `vegetation`, `sky`, `person`, `car`, `bus`, `truck`, `bicycle` y `terrain`. No representa fotografías de La Candelaria, por lo que serviría como conjunto de referencia/metodología, no como ground truth de las cinco fotografías locales. La decisión debe documentar esa diferencia.

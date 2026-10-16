# Evaluación de métodos de segmentación en imágenes urbanas de La Candelaria, Bogotá, con validación complementaria mediante CrackForest Dataset

**Estado metodológico:** La Candelaria es el caso de estudio principal. Sus máscaras manuales son referencias preliminares y no constituyen ground truth validado. CFD es el benchmark complementario con anotaciones oficiales. QG1 no está aprobado.

## Objetivo
Comparar Otsu y K-means en cinco escenas urbanas reales de La Candelaria y contrastar su comportamiento, con métricas verificables, sobre cinco imágenes de CrackForest Dataset (CFD) con anotaciones oficiales.

## Metodología
Otsu usa un histograma de 256 niveles y clasifica como foreground las intensidades mayores al umbral. K-means es una implementación 1-D propia con k=2, centroides iniciales mínimo/máximo, semilla 7, máximo 100 iteraciones y tolerancia 1e-4; el grupo de centroide mayor es foreground. Las métricas son IoU, Dice, precisión, recall y exactitud; la matriz se ordena como `[[TN, FP], [FN, TP]]`.

En La Candelaria, ambos métodos reciben las imágenes RGB y la implementación existente usa la media aritmética de R, G y B, sin ajuste por fotografía. Las máscaras de `urban_built_surface` se elaboraron independientemente de Otsu y K-means, pero tienen errores visuales conocidos.

En CFD, ambos métodos reciben exactamente `255 - round(mean(R,G,B))`, sin ajuste por imagen. La referencia es `groundTruth/<id>.mat`, campo `groundTruth.Segmentation`: etiqueta 1 = pavimento intacto y 2 = grieta. `seg/<id>.seg` se utiliza solo como control de integridad.

## Resultados provisionales — La Candelaria
| Imagen | Método | IoU | Dice | Precisión | Recall | Matriz |
|---|---|---:|---:|---:|---:|---|
| 01_calle_mural | Otsu | 0.4211 | 0.5927 | 0.7562 | 0.4873 | `[[98404, 279162], [910768, 865666]]` |
| 01_calle_mural | K-means | 0.4197 | 0.5913 | 0.7562 | 0.4854 | `[[99571, 277995], [914089, 862345]]` |
| 02_casa_cuervo | Otsu | 0.2564 | 0.4081 | 0.6665 | 0.2941 | `[[3307446, 2412829], [11575312, 4822813]]` |
| 02_casa_cuervo | K-means | 0.2564 | 0.4081 | 0.6665 | 0.2941 | `[[3307446, 2412829], [11575312, 4822813]]` |
| 03_sombras_tarde | Otsu | 0.4308 | 0.6022 | 0.7458 | 0.5049 | `[[677061, 169549], [487878, 497512]]` |
| 03_sombras_tarde | K-means | 0.4308 | 0.6022 | 0.7458 | 0.5049 | `[[677061, 169549], [487878, 497512]]` |
| 04_vendedor_peaton | Otsu | 0.4135 | 0.5851 | 0.5862 | 0.5840 | `[[4996950, 3766147], [3800091, 5335440]]` |
| 04_vendedor_peaton | K-means | 0.4132 | 0.5848 | 0.5864 | 0.5833 | `[[5003989, 3759108], [3806853, 5328678]]` |
| 05_calle_candelaria | Otsu | 0.2284 | 0.3718 | 0.5784 | 0.2740 | `[[24140, 42886], [155926, 58838]]` |
| 05_calle_candelaria | K-means | 0.2275 | 0.3707 | 0.5778 | 0.2729 | `[[24204, 42822], [156157, 58607]]` |

## Resultados CFD — benchmark complementario
| Imagen | Método | IoU | Dice | Precisión | Recall | Matriz |
|---|---|---:|---:|---:|---:|---|
| 001 | Otsu | 0.0266 | 0.0519 | 0.0267 | 0.8629 | `[[94256, 57513], [251, 1580]]` |
| 001 | K-means | 0.0248 | 0.0484 | 0.0249 | 0.8689 | `[[89486, 62283], [240, 1591]]` |
| 024 | Otsu | 0.0301 | 0.0584 | 0.0301 | 0.9269 | `[[45198, 104885], [257, 3260]]` |
| 024 | K-means | 0.0299 | 0.0581 | 0.0300 | 0.9301 | `[[44368, 105715], [246, 3271]]` |
| 050 | Otsu | 0.0169 | 0.0332 | 0.0169 | 0.8493 | `[[72572, 79416], [243, 1369]]` |
| 050 | K-means | 0.0161 | 0.0317 | 0.0161 | 0.8610 | `[[67423, 84565], [224, 1388]]` |
| 074 | Otsu | 0.0276 | 0.0537 | 0.0277 | 0.9007 | `[[77634, 73640], [231, 2095]]` |
| 074 | K-means | 0.0258 | 0.0502 | 0.0258 | 0.9106 | `[[71362, 79912], [208, 2118]]` |
| 116 | Otsu | 0.0178 | 0.0350 | 0.0179 | 0.9178 | `[[75954, 76137], [124, 1385]]` |
| 116 | K-means | 0.0187 | 0.0367 | 0.0187 | 0.9105 | `[[80123, 71968], [135, 1374]]` |

## Trazabilidad de CFD
Repositorio oficial: https://github.com/cuilimeng/CrackForest-dataset.git
Commit: `ed57c2d96754e6c7d105805cd29aeeb3a799f267`
Licencia indicada por el repositorio: Non-commercial research purposes only (official repository README).
Selección reproducible: pares completos con dimensiones coincidentes, etiquetas exactamente {1,2}, equivalencia `seg == groundTruth - 1`, exclusión documentada de 042 y cinco posiciones equidistantes sobre los IDs elegibles ordenados. IDs: 001, 024, 050, 074 y 116.

## Discusión y conclusiones
En ambos escenarios Otsu y K-means producen resultados cercanos porque separan globalmente dos grupos sobre la misma representación de intensidad. Esto no demuestra equivalencia general. En La Candelaria, la intensidad no representa de forma suficiente la semántica de superficies construidas y las referencias preliminares impiden interpretar las métricas como exactitud real.

En CFD, el recall alto coincide con la detección de gran parte de las grietas, pero la precisión, IoU y Dice bajos evidencian numerosos falsos positivos causados por textura, sombras y variación del pavimento. La segmentación global por intensidad es insuficiente para esta estructura fina y minoritaria.

QG1 permanece pendiente por la calidad de las referencias de La Candelaria. La evaluación CFD sí dispone de anotación oficial y sus cinco pares superaron controles técnicos de integridad, alineación y polaridad.

## Referencias
- Otsu, N. (1979). A Threshold Selection Method from Gray-Level Histograms. IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62–66. DOI: 10.1109/TSMC.1979.4310076.
- MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations. Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, 1, 281–297.
- Shi, Y., Cui, L., Qi, Z., Meng, F., & Chen, Z. (2016). Automatic road crack detection using random structured forests. IEEE Transactions on Intelligent Transportation Systems, 17(12), 3434–3445.
- CrackForest Dataset: https://github.com/cuilimeng/CrackForest-dataset.git, commit `ed57c2d96754e6c7d105805cd29aeeb3a799f267`; licencia de uso no comercial para investigación.

### Fuentes de las fotografías de La Candelaria
- 01_calle_mural.jpg: La Candelaria, Bogota, Colombia (5818105341).jpg; Pedro Szekely; CC BY-SA 2.0; https://commons.wikimedia.org/wiki/File:La_Candelaria,_Bogota,_Colombia_(5818105341).jpg
- 02_casa_cuervo.jpg: Casa Cuervo Urisarri.jpg; Instituto Caro y Cuervo; EXIF author Juan Andres Rodriguez Suarez; CC BY-SA 4.0; https://commons.wikimedia.org/wiki/File:Casa_Cuervo_Urisarri.jpg
- 03_sombras_tarde.jpg: Tarde en la Candelaria - panoramio.jpg; Amel Segré; CC BY-SA 3.0; https://commons.wikimedia.org/wiki/File:Tarde_en_la_Candelaria_-_panoramio.jpg
- 04_vendedor_peaton.jpg: 2017 Bogotá carrera 10 con calle 12 vendedor de jugo de naranja.jpg; Felipe Restrepo Acosta; CC BY-SA 4.0; https://commons.wikimedia.org/wiki/File:2017_Bogot%C3%A1_carrera_10_con_calle_12_vendedor_de_jugo_de_naranja.jpg
- 05_calle_candelaria.jpg: La Candelaria.JPG; Tequendamia; Public domain; https://commons.wikimedia.org/wiki/File:La_Candelaria.JPG

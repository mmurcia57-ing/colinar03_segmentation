"""Genera el notebook académico; los algoritmos viven en src/segmentation_lab y no se duplican."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "evaluacion_segmentacion.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


CELLS = [
    md(
        "# Evaluación de segmentadores frente a *ground truth*\n\n"
        "**Asignatura:** Percepción Computacional (COEINTAR) - Enero 2026 - PER 15708 — UNIR. "
        "Trabajo: Evaluación de la segmentación.\n\n"
        "**Integrantes:** Mateo Torres Arredondo, Edna Mayerly Alvarez y Manuel Alejandro Murcia. "
        "**Profesor:** Luis Fernando Barajas Ardila. **Fecha:** 18/10/2026.\n\n"
        "**Objetivo.** Construir cuatro segmentadores y evaluar su rendimiento frente a imágenes de *ground truth* "
        "con un evaluador implementado desde cero.\n\n"
        "| Segmentador | Familia | Papel |\n|---|---|---|\n"
        "| Otsu | Umbralización global | Línea base |\n"
        "| K-means (k=2) | Agrupamiento de intensidades | Línea base |\n"
        "| Region Growing | Crecimiento de regiones por semillas (Tema 10.4) | Método principal |\n"
        "| Watershed con marcadores | Inundación morfológica (Tema 10.7) | Método principal |\n\n"
        "**Datos.** (1) *CrackForest Dataset* (CFD): 5 imágenes de pavimento con anotación oficial de grietas → "
        "evaluación cuantitativa principal. (2) La Candelaria (Bogotá): 5 fotografías urbanas con máscaras manuales "
        "propias → aplicación secundaria con métricas orientativas.\n\n"
        "Todo el código de segmentación, preprocesamiento y métricas es propio (`src/segmentation_lab/`). Única "
        "excepción: la transformada de distancia euclídea `scipy.ndimage.distance_transform_edt`, usada para colocar "
        "marcadores de fondo en Watershed."
    ),
    code(
        "from pathlib import Path\n"
        "import inspect, json, sys\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from IPython.display import HTML, Image as DisplayImage, Markdown, display\n\n"
        "ROOT = Path.cwd().resolve()\n"
        "if ROOT.name == 'notebooks':\n"
        "    ROOT = ROOT.parent\n"
        "sys.path.insert(0, str(ROOT))\n"
        "sys.path.insert(0, str(ROOT / 'src'))\n\n"
        "from segmentation_lab import binary_metrics, confusion_matrix, metrics, region_growing, watershed\n"
        "from segmentation_lab import visualization as viz\n"
        "from segmentation_lab.pipeline import CFD_PARAMS, METHODS, preprocess_cfd, run_methods\n"
        "from scripts.run_integrated_evaluation import evaluate_all, load_cfd\n\n"
        "def show_source(obj):\n"
        "    display(Markdown('```python\\n' + inspect.getsource(obj) + '\\n```'))\n\n"
        "def show_table(rows, fields):\n"
        "    head = ''.join(f'<th>{name}</th>' for name in fields)\n"
        "    body = ''.join(\n"
        "        '<tr>' + ''.join(f'<td>{row[f]:.4f}</td>' if isinstance(row[f], float) else f'<td>{row[f]}</td>' for f in fields) + '</tr>'\n"
        "        for row in rows)\n"
        "    display(HTML(f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'))\n\n"
        "print('Python:', sys.version.split()[0])"
    ),
    md(
        "## 1. Evaluador propio (matriz de confusión y métricas)\n\n"
        "Para una máscara predicha $S$ y un *ground truth* $GT$, ambos binarios:\n\n"
        "$TP=|S\\cap GT|$, $FP=|S\\setminus GT|$, $FN=|GT\\setminus S|$, $TN$ = resto.\n\n"
        "$$IoU=\\frac{TP}{TP+FP+FN}\\qquad Dice=\\frac{2TP}{2TP+FP+FN}\\qquad "
        "Precisión=\\frac{TP}{TP+FP}\\qquad Recall=\\frac{TP}{TP+FN}$$\n\n"
        "La implementación solo usa operaciones lógicas de NumPy (sin bibliotecas de métricas):"
    ),
    code("show_source(metrics)"),
    code(
        "# Comprobación con un caso calculable a mano: TP=1, FP=1, FN=1, TN=1\n"
        "truth = np.array([[1, 1, 0, 0]], dtype=bool)\n"
        "prediction = np.array([[1, 0, 1, 0]], dtype=bool)\n"
        "print('Matriz [[TN, FP], [FN, TP]] =', confusion_matrix(truth, prediction).tolist())\n"
        "print({k: round(v, 4) for k, v in binary_metrics(truth, prediction).items()})  # IoU=1/3, Dice=0.5"
    ),
    md(
        "## 2. Datos: CrackForest Dataset (CFD)\n\n"
        "Cinco imágenes (IDs 001, 024, 050, 074, 116) elegidas con una regla determinista que no mira resultados "
        "(`data/benchmarks/cfd5/manifest.json`). El *ground truth* es la anotación oficial `groundTruth.Segmentation` "
        "(etiqueta 2 = grieta). Las grietas ocupan solo ~1–2 % de los píxeles, lo que hace el problema exigente."
    ),
    code(
        "pairs = load_cfd()\n"
        "fig, axes = plt.subplots(2, 5, figsize=(17, 5.2), dpi=110)\n"
        "for col, pair in enumerate(pairs):\n"
        "    axes[0, col].imshow(pair['image']); axes[0, col].set_title(f\"CFD {pair['id']}\")\n"
        "    axes[1, col].imshow(pair['truth'], cmap='gray')\n"
        "    axes[1, col].set_title(f\"GT: {pair['truth'].mean():.2%} grieta\")\n"
        "for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## 3. Preprocesamiento común\n\n"
        "Los cuatro métodos reciben **la misma entrada** y parámetros fijos para todas las imágenes:\n\n"
        "1. Gris = media de R, G y B.\n"
        "2. Suavizado gaussiano propio ($\\sigma=1.5$) para atenuar la textura del asfalto.\n"
        "3. Inversión $255-I$: la grieta (oscura) pasa a ser el objeto claro.\n"
        "4. Resta de fondo $I' = 4\\,(I - G_{15} * I) + 128$: corrige la iluminación no uniforme y realza estructuras finas.\n\n"
        "Se muestran los pasos sobre la imagen 001."
    ),
    code(
        "example = next(p for p in pairs if p['id'] == '001')\n"
        "steps = preprocess_cfd(example['image'])\n"
        "x = steps['input']\n"
        "viz.preprocessing_figure(steps, [('gray', '1. Gris'), ('smoothed', '2. Gaussiano σ=1.5'),\n"
        "                                 ('inverted', '3. Invertida'), ('input', '4. Sin fondo (entrada)')])\n"
        "plt.show()\n"
        "outputs = run_methods(x, CFD_PARAMS)\n"
        "print('Parámetros CFD:', CFD_PARAMS)"
    ),
    md(
        "## 4. Líneas base: Otsu y K-means\n\n"
        "**Otsu** elige el umbral $t$ que maximiza la varianza entre clases "
        "$\\sigma_B^2(t)=\\frac{(\\mu_T\\,\\omega(t)-\\mu(t))^2}{\\omega(t)(1-\\omega(t))}$. **K-means** (k=2) alterna "
        "asignación al centroide más cercano y actualización de medias; su frontera es el punto medio de los centroides. "
        "Ambos deciden con un único umbral global, sin información espacial."
    ),
    code("viz.thresholds_figure(x, outputs); plt.show()"),
    md(
        "## 5. Segmentador 1: crecimiento de regiones por semillas (Tema 10.4)\n\n"
        "- **Semillas:** píxeles con intensidad ≥ percentil 99.8 de la entrada (los más oscuros del original).\n"
        "- **Función de similitud / pertenencia:** un vecino $p$ (vecindad 8) se agrega si "
        "$|I(p)-\\mu_R|\\le T$, con $\\mu_R$ la media actual de la región y $T=30$.\n"
        "- **Recorrido:** búsqueda en anchura; el orden de entrada de cada píxel permite visualizar la evolución.\n\n"
        "Implementación propia:"
    ),
    code("show_source(region_growing)"),
    code("viz.region_growing_figure(x, outputs); plt.show()"),
    md(
        "## 6. Segmentador 2: Watershed con marcadores (Tema 10.7)\n\n"
        "1. **Relieve:** magnitud del gradiente de Sobel de la entrada.\n"
        "2. **Candidatos:** píxeles por encima del umbral de Otsu; **mapa de distancias** euclídeas a ellos.\n"
        "3. **Marcadores:** objeto = percentil ≥ 99.5; fondo = píxeles a más de 3 px de cualquier candidato. "
        "Los marcadores evitan la sobresegmentación propia del Watershed sin marcadores.\n"
        "4. **Inundación:** cola de prioridad (algoritmo de Meyer); cada píxel recibe la etiqueta de la cuenca que "
        "lo alcanza primero al subir el nivel del agua. Las fronteras entre cuencas son las **líneas de cuenca**.\n\n"
        "Implementación propia:"
    ),
    code("show_source(watershed)"),
    code("viz.watershed_figure(x, outputs); plt.show()"),
    md("## 7. Evaluación de la imagen 001: superposición de falsos positivos y falsos negativos"),
    code(
        "masks = {m: outputs[m]['mask'] for m in METHODS}\n"
        "viz.errors_figure(example['truth'], masks, 'CFD 001'); plt.show()\n"
        "show_table([{'método': m, **binary_metrics(example['truth'], masks[m])} for m in METHODS],\n"
        "           ['método', 'iou', 'dice', 'precision', 'recall', 'accuracy'])"
    ),
    md(
        "## 8. Evaluación completa en CFD (5 imágenes × 4 métodos)\n\n"
        "Se recalcula toda la evaluación y se comprueba que coincide con `deliverables/results.json`, la fuente de "
        "las tablas del informe."
    ),
    code(
        "computed = json.loads(json.dumps(evaluate_all(write_outputs=False, create_figures=False)))\n"
        "stored = json.loads((ROOT / 'deliverables' / 'results.json').read_text(encoding='utf-8'))\n"
        "assert computed == stored, 'El notebook y deliverables/results.json no coinciden.'\n"
        "print('NOTEBOOK_JSON_MATCH=PASS')\n"
        "cfd = computed['cfd']\n"
        "show_table(cfd['rows'], ['image', 'method', 'iou', 'dice', 'precision', 'recall', 'accuracy'])"
    ),
    code(
        "print('Promedio por imagen:')\n"
        "show_table([{'método': m, **v} for m, v in cfd['mean_per_image'].items()],\n"
        "           ['método', 'iou', 'dice', 'precision', 'recall', 'accuracy'])\n"
        "print('Matrices agregadas [[TN, FP], [FN, TP]]:')\n"
        "for m, item in cfd['aggregate_pixel_weighted'].items():\n"
        "    print(f'  {m:15s}', item['confusion_matrix'])"
    ),
    code("display(DisplayImage(filename=str(ROOT / cfd['overview_figure']), width=1050))"),
    md(
        "### Sensibilidad a los parámetros\n\n"
        "Region Growing depende de la tolerancia $T$: si es pequeña, la región apenas crece desde las semillas; si "
        "es grande, se desborda hacia el asfalto. Watershed depende de dónde se colocan los marcadores de fondo."
    ),
    code(
        "show_table(cfd['sensitivity']['region_growing_tolerance'], ['tolerance', 'mean_iou', 'mean_dice', 'mean_precision', 'mean_recall'])\n"
        "show_table(cfd['sensitivity']['watershed_background_distance'], ['background_distance', 'mean_iou', 'mean_dice', 'mean_precision', 'mean_recall'])"
    ),
    md(
        "## 9. Aplicación secundaria: La Candelaria (Bogotá)\n\n"
        "Cinco fotografías urbanas reales (Wikimedia Commons, licencias en `data/real/sources.json`). La clase objetivo "
        "`urban_built_surface` (fachadas, cubiertas y pavimento) se anotó manualmente con polígonos, sin usar los "
        "segmentadores. Son anotaciones propias aproximadas, no expertas, por lo que **las métricas son orientativas**. "
        "Las imágenes se procesan a lado máximo 1024 px y las máscaras se reescalan a la resolución original para evaluar."
    ),
    code(
        "cand = computed['la_candelaria']\n"
        "show_table(cand['rows'], ['image', 'method', 'iou', 'dice', 'precision', 'recall'])\n"
        "show_table([{'método': m, **v} for m, v in cand['mean_per_image'].items()], ['método', 'iou', 'dice', 'precision', 'recall'])\n"
        "display(DisplayImage(filename=str(ROOT / cand['overview_figure']), width=1050))"
    ),
    md(
        "## 10. Conclusiones\n\n"
        "- **CFD:** Otsu y K-means detectan casi toda la grieta (recall alto) pero marcan como grieta gran parte del "
        "asfalto (precisión ≈ 0.04), porque un único umbral global no distingue la grieta de la textura con la misma "
        "intensidad. Region Growing y Watershed añaden **conectividad espacial**: parten de semillas/marcadores "
        "fiables y solo crecen por píxeles conectados y similares, multiplicando por más de 7 el Dice de las líneas base.\n"
        "- **Region Growing** es muy sensible a la tolerancia (desbordamiento a partir de $T\\approx40$) y a la "
        "colocación de semillas: tramos de grieta sin semilla quedan como falsos negativos (azul).\n"
        "- **Watershed** cierra mejor los tramos discontinuos gracias al gradiente, pero produce manchas de falsos "
        "positivos donde un marcador de objeto cae sobre una mancha oscura del asfalto.\n"
        "- **La Candelaria:** la clase semántica \"superficie construida\" no se corresponde con una intensidad "
        "(cielo claro, fachadas oscuras en sombra), por lo que los cuatro métodos rinden de forma parecida y limitada.\n"
        "- **Limitaciones:** 5 imágenes por escenario; parámetros fijados por escenario sobre las mismas imágenes "
        "evaluadas; las referencias de La Candelaria son aproximadas."
    ),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.12"}
    nb["cells"] = CELLS
    NOTEBOOK.write_text(nbf.writes(nb), encoding="utf-8")
    print(NOTEBOOK)


if __name__ == "__main__":
    main()

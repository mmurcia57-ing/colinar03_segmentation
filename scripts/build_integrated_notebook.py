"""Build the single executable academic notebook without duplicating algorithms."""
from __future__ import annotations

import shutil
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "evaluacion_segmentacion.ipynb"
BACKUP = ROOT / "artifacts" / "notebook_history" / "evaluacion_segmentacion_lacandelaria_pre_cfd.ipynb"


def main() -> None:
    if NOTEBOOK.exists() and not BACKUP.exists():
        BACKUP.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(NOTEBOOK, BACKUP)

    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.12"}
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Evaluación de métodos de segmentación en imágenes urbanas de La Candelaria, Bogotá, "
            "con validación complementaria mediante CrackForest Dataset\n\n"
            "**Caso principal:** cinco fotografías reales de La Candelaria.  "
            "**Benchmark complementario:** cinco pares imagen/anotación oficial de CrackForest Dataset (CFD).\n\n"
            "Se comparan las implementaciones existentes de Otsu y K-means. Las referencias manuales de La Candelaria "
            "son preliminares y sus métricas son provisionales; no se presentan como *ground truth* validado. CFD aporta "
            "una evaluación cuantitativa externa con anotaciones oficiales."
        ),
        nbf.v4.new_markdown_cell(
            "## Metodología\n\n"
            "- **Otsu:** histograma de 256 niveles y *foreground* para intensidades mayores que el umbral.\n"
            "- **K-means:** implementación unidimensional propia, `k=2`, inicialización mínimo/máximo, semilla 7, "
            "máximo 100 iteraciones y tolerancia `1e-4`; el grupo claro es *foreground*.\n"
            "- **La Candelaria:** conversión interna a media aritmética RGB, sin ajuste por fotografía.\n"
            "- **CFD:** preprocesamiento fijo autorizado `255 - round(mean(R,G,B))` para ambos métodos.\n"
            "- **Métricas:** IoU, Dice, precisión, recall, exactitud y matrices `[[TN, FP], [FN, TP]]`."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json, sys\n"
            "from IPython.display import HTML, Image as DisplayImage, display\n\n"
            "ROOT = Path.cwd().resolve()\n"
            "if ROOT.name == 'notebooks':\n"
            "    ROOT = ROOT.parent\n"
            "sys.path.insert(0, str(ROOT))\n"
            "sys.path.insert(0, str(ROOT / 'src'))\n"
            "from scripts.run_integrated_evaluation import evaluate_all\n\n"
            "computed = evaluate_all(write_outputs=False, create_figures=False)\n"
            "stored = json.loads((ROOT / 'deliverables' / 'results.json').read_text(encoding='utf-8'))\n"
            "assert computed == stored, 'El notebook y deliverables/results.json no coinciden.'\n"
            "print('NOTEBOOK_JSON_MATCH=PASS')\n"
            "print('Python:', sys.version.split()[0])\n\n"
            "def show_metrics(rows):\n"
            "    fields = ['image','method','iou','dice','precision','recall','accuracy']\n"
            "    head = ''.join(f'<th>{name}</th>' for name in fields)\n"
            "    body = []\n"
            "    for row in rows:\n"
            "        cells = []\n"
            "        for field in fields:\n"
            "            value = row[field]\n"
            "            cells.append(f'<td>{value:.4f}</td>' if isinstance(value, float) else f'<td>{value}</td>')\n"
            "        body.append('<tr>' + ''.join(cells) + '</tr>')\n"
            "    display(HTML('<table><thead><tr>' + head + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table>'))"
        ),
        nbf.v4.new_markdown_cell(
            "## Caso de estudio principal: La Candelaria\n\n"
            "Las cinco fotografías documentan escenas urbanas reales. La clase preliminar `urban_built_surface` incluye "
            "fachadas, cubiertas, balcones, puertas, ventanas y pavimento, y excluye cielo, vegetación, personas y "
            "vehículos. Las máscaras se elaboraron independientemente de Otsu y K-means, pero no superaron una "
            "validación visual completa; por ello todos los indicadores de esta sección son **provisionales**."
        ),
        nbf.v4.new_code_cell(
            "case = computed['case_study']\n"
            "show_metrics(case['rows'])\n"
            "print('Matrices por imagen, orden [[TN, FP], [FN, TP]]:')\n"
            "display(case['confusion_matrices'])\n"
            "print('Agregado por píxel:')\n"
            "display(case['aggregate_pixel_weighted'])"
        ),
        nbf.v4.new_code_cell(
            "display(DisplayImage(filename=str(ROOT / case['overview_figure']), width=1050))"
        ),
        nbf.v4.new_markdown_cell(
            "## Benchmark complementario: CrackForest Dataset\n\n"
            "Se usan los IDs `001`, `024`, `050`, `074` y `116`, seleccionados mediante una regla determinista sin "
            "consultar predicciones ni métricas. La referencia procede de `groundTruth/<id>.mat`, campo "
            "`groundTruth.Segmentation`: etiqueta 2 = grieta/*foreground* y etiqueta 1 = pavimento intacto/*background*. "
            "Los archivos `.seg` se utilizan únicamente para comprobar integridad. Las cinco parejas superaron la "
            "inspección técnica de dimensiones, alineación y polaridad."
        ),
        nbf.v4.new_code_cell(
            "cfd = computed['complementary_benchmark']\n"
            "show_metrics(cfd['rows'])\n"
            "print('Matrices por imagen, orden [[TN, FP], [FN, TP]]:')\n"
            "display(cfd['confusion_matrices'])\n"
            "print('Agregado por píxel:')\n"
            "display(cfd['aggregate_pixel_weighted'])"
        ),
        nbf.v4.new_code_cell(
            "display(DisplayImage(filename=str(ROOT / cfd['overview_figure']), width=1050))"
        ),
        nbf.v4.new_markdown_cell(
            "## Discusión conjunta\n\n"
            "En La Candelaria, la similitud visual y numérica de Otsu y K-means se explica porque ambos reducen cada "
            "escena a una separación global de dos grupos sobre la misma intensidad media. Ese supuesto no representa "
            "una clase semántica urbana y las métricas dependen de referencias preliminares; solo describen el experimento.\n\n"
            "En CFD, ambos métodos recuperan muchas grietas (recall alto), pero producen numerosos falsos positivos por "
            "textura, sombras y variaciones del pavimento. La baja precisión, IoU y Dice demuestra la limitación de la "
            "segmentación global por intensidad para estructuras finas y minoritarias.\n\n"
            "La conclusión no es que los métodos sean equivalentes en general, sino que bajo este preprocesamiento fijo y "
            "estas muestras generan particiones muy próximas. QG1 permanece sin aprobar porque las referencias de La "
            "Candelaria no están visualmente validadas."
        ),
        nbf.v4.new_markdown_cell(
            "## Reproducibilidad y trazabilidad\n\n"
            "El manifiesto `data/benchmarks/cfd5/manifest.json` registra repositorio, commit, licencia, hashes, etiquetas, "
            "regla de selección y verificaciones. Los datos crudos CFD no se publican en este repositorio; se regeneran "
            "desde el origen oficial para respetar su licencia de uso no comercial en investigación. Los resultados "
            "numéricos canónicos están en `deliverables/results.json`."
        ),
    ]
    NOTEBOOK.write_text(nbf.writes(nb), encoding="utf-8")
    print(NOTEBOOK)


if __name__ == "__main__":
    main()

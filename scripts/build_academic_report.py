"""Build the integrated six-page academic report from computed JSON results."""
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "deliverables"
TITLE = (
    "Evaluación de métodos de segmentación en imágenes urbanas de La Candelaria, Bogotá, "
    "con validación complementaria mediante CrackForest Dataset"
)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def p(text: str, style) -> Paragraph:
    return Paragraph(escape(text), style)


def page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(A4[0] - 1.2 * cm, 0.65 * cm, f"Página {doc.page} de 6")
    canvas.restoreState()


def _by_image(rows: list[dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault(row["image"], {})[row["method"]] = row
    return grouped


def _source_lines() -> list[str]:
    sources = json.loads((ROOT / "data" / "real" / "sources.json").read_text(encoding="utf-8"))["images"]
    return [
        f"{item['local_file']}: {item['title']}; {item['author']}; {item['license']}; {item['source_page']}"
        for item in sources
    ]


def write_markdown(path: Path, manifest: dict, results: dict) -> None:
    case = results["case_study"]
    cfd = results["complementary_benchmark"]
    lines = [
        f"# {TITLE}",
        "",
        "**Estado metodológico:** La Candelaria es el caso de estudio principal. Sus máscaras manuales son referencias preliminares y no constituyen ground truth validado. CFD es el benchmark complementario con anotaciones oficiales. QG1 no está aprobado.",
        "",
        "## Objetivo",
        "Comparar Otsu y K-means en cinco escenas urbanas reales de La Candelaria y contrastar su comportamiento, con métricas verificables, sobre cinco imágenes de CrackForest Dataset (CFD) con anotaciones oficiales.",
        "",
        "## Metodología",
        "Otsu usa un histograma de 256 niveles y clasifica como foreground las intensidades mayores al umbral. K-means es una implementación 1-D propia con k=2, centroides iniciales mínimo/máximo, semilla 7, máximo 100 iteraciones y tolerancia 1e-4; el grupo de centroide mayor es foreground. Las métricas son IoU, Dice, precisión, recall y exactitud; la matriz se ordena como `[[TN, FP], [FN, TP]]`.",
        "",
        "En La Candelaria, ambos métodos reciben las imágenes RGB y la implementación existente usa la media aritmética de R, G y B, sin ajuste por fotografía. Las máscaras de `urban_built_surface` se elaboraron independientemente de Otsu y K-means, pero tienen errores visuales conocidos.",
        "",
        "En CFD, ambos métodos reciben exactamente `255 - round(mean(R,G,B))`, sin ajuste por imagen. La referencia es `groundTruth/<id>.mat`, campo `groundTruth.Segmentation`: etiqueta 1 = pavimento intacto y 2 = grieta. `seg/<id>.seg` se utiliza solo como control de integridad.",
        "",
        "## Resultados provisionales — La Candelaria",
        "| Imagen | Método | IoU | Dice | Precisión | Recall | Matriz |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in case["rows"]:
        matrix = case["confusion_matrices"][row["image"]][row["method"]]
        lines.append(
            f"| {row['image']} | {row['method']} | {fmt(row['iou'])} | {fmt(row['dice'])} | "
            f"{fmt(row['precision'])} | {fmt(row['recall'])} | `{matrix}` |"
        )
    lines += ["", "## Resultados CFD — benchmark complementario", "| Imagen | Método | IoU | Dice | Precisión | Recall | Matriz |", "|---|---|---:|---:|---:|---:|---|"]
    for row in cfd["rows"]:
        matrix = cfd["confusion_matrices"][row["image"]][row["method"]]
        lines.append(
            f"| {row['image']} | {row['method']} | {fmt(row['iou'])} | {fmt(row['dice'])} | "
            f"{fmt(row['precision'])} | {fmt(row['recall'])} | `{matrix}` |"
        )
    lines += [
        "",
        "## Trazabilidad de CFD",
        f"Repositorio oficial: {manifest['official_repository']}",
        f"Commit: `{manifest['official_commit']}`",
        f"Licencia indicada por el repositorio: {manifest['license']}",
        "Selección reproducible: pares completos con dimensiones coincidentes, etiquetas exactamente {1,2}, equivalencia `seg == groundTruth - 1`, exclusión documentada de 042 y cinco posiciones equidistantes sobre los IDs elegibles ordenados. IDs: 001, 024, 050, 074 y 116.",
        "",
        "## Discusión y conclusiones",
        "En ambos escenarios Otsu y K-means producen resultados cercanos porque separan globalmente dos grupos sobre la misma representación de intensidad. Esto no demuestra equivalencia general. En La Candelaria, la intensidad no representa de forma suficiente la semántica de superficies construidas y las referencias preliminares impiden interpretar las métricas como exactitud real.",
        "",
        "En CFD, el recall alto coincide con la detección de gran parte de las grietas, pero la precisión, IoU y Dice bajos evidencian numerosos falsos positivos causados por textura, sombras y variación del pavimento. La segmentación global por intensidad es insuficiente para esta estructura fina y minoritaria.",
        "",
        "QG1 permanece pendiente por la calidad de las referencias de La Candelaria. La evaluación CFD sí dispone de anotación oficial y sus cinco pares superaron controles técnicos de integridad, alineación y polaridad.",
        "",
        "## Referencias",
        "- Otsu, N. (1979). A Threshold Selection Method from Gray-Level Histograms. IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62–66. DOI: 10.1109/TSMC.1979.4310076.",
        "- MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations. Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, 1, 281–297.",
        "- Shi, Y., Cui, L., Qi, Z., Meng, F., & Chen, Z. (2016). Automatic road crack detection using random structured forests. IEEE Transactions on Intelligent Transportation Systems, 17(12), 3434–3445.",
        f"- CrackForest Dataset: {manifest['official_repository']}, commit `{manifest['official_commit']}`; licencia de uso no comercial para investigación.",
        "",
        "### Fuentes de las fotografías de La Candelaria",
    ]
    lines.extend(f"- {line}" for line in _source_lines())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _table_style(font_size: float = 6.6) -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24445c")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#8a969c")),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#edf3f7")]),
            ("TOPPADDING", (0, 0), (-1, -1), 2.6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
        ]
    )


def build_pdf(path: Path, manifest: dict, results: dict) -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="MainTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22, spaceAfter=13))
    styles.add(ParagraphStyle(name="SectionX", parent=styles["Heading2"], fontSize=12.3, leading=14, spaceBefore=3, spaceAfter=5, textColor=colors.HexColor("#173b51")))
    styles.add(ParagraphStyle(name="SmallX", parent=styles["BodyText"], fontSize=7.5, leading=8.8))
    styles.add(ParagraphStyle(name="TinyX", parent=styles["BodyText"], fontSize=6.3, leading=7.3))
    body = styles["BodyText"]
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=1.15 * cm, leftMargin=1.15 * cm, topMargin=1.0 * cm, bottomMargin=1.1 * cm)
    story: list = []
    case = results["case_study"]
    cfd = results["complementary_benchmark"]

    story += [
        p(TITLE, styles["MainTitle"]),
        p("Laboratorio de Percepción Computacional — UNIR", styles["Heading2"]),
        Spacer(1, 8),
        p("Planteamiento", styles["SectionX"]),
        p("La Candelaria, Bogotá, constituye el caso de estudio principal: cinco fotografías reales permiten observar cómo dos segmentadores clásicos responden a escenas urbanas con fachadas, cubiertas, pavimento, vegetación, cielo, personas y vehículos. Se comparan las implementaciones existentes de Otsu y K-means sin modificarlas.", body),
        Spacer(1, 8),
        p("Objetivo", styles["SectionX"]),
        p("Comparar visual y cuantitativamente Otsu y K-means en La Candelaria, y complementar el análisis con cinco imágenes de CrackForest Dataset (CFD), cuya anotación oficial permite una evaluación externa y verificable.", body),
        Spacer(1, 8),
        p("Transparencia metodológica", styles["SectionX"]),
        p("Las máscaras manuales de La Candelaria se elaboraron independientemente de Otsu y K-means, pero contienen errores visuales importantes y no superaron una validación completa. Sus métricas son provisionales y no equivalen a exactitud semántica real. CFD no reemplaza el caso urbano: funciona como benchmark complementario con ground truth oficial.", body),
        Spacer(1, 8),
        p("Estado", styles["SectionX"]),
        p("La integración, la ejecución reproducible y la verificación técnica de CFD se documentan en este informe. QG1 no se declara aprobado porque la referencia semántica de La Candelaria sigue pendiente de validación.", body),
        PageBreak(),
    ]

    story += [
        p("Datos, implementación y métricas", styles["SectionX"]),
        p("Caso principal — La Candelaria", styles["SectionX"]),
        p("Se usan cinco fotografías reales obtenidas de Wikimedia Commons y la clase urban_built_surface: fachadas, cubiertas, balcones, puertas, ventanas y pavimento. Se excluyen cielo, vegetación, personas y vehículos. Las referencias manuales son preliminares. La conversión de RGB es la media aritmética de sus tres canales, aplicada internamente por el código existente, sin parámetros por fotografía.", body),
        Spacer(1, 6),
        p("Benchmark complementario — CFD", styles["SectionX"]),
        p(f"Fuente oficial: {manifest['official_repository']}; commit {manifest['official_commit']}. Licencia declarada: {manifest['license']}", body),
        p("La referencia procede de groundTruth/<id>.mat, campo groundTruth.Segmentation: 1 = pavimento intacto/background y 2 = grieta/foreground. El archivo seg/<id>.seg reconstruye groundTruth - 1 y se usa solo como control de integridad. Los IDs 001, 024, 050, 074 y 116 se eligieron por una regla determinista, sin observar predicciones ni métricas.", body),
        Spacer(1, 6),
        p("Parámetros comunes y específicos", styles["SectionX"]),
        p("Otsu: histograma de 256 niveles; se maximiza la varianza entre clases y foreground corresponde a intensidad mayor que el umbral. K-means: k=2, datos 1-D, centroides iniciales mínimo/máximo, semilla 7, máximo 100 iteraciones, tolerancia 1e-4 y foreground en el centroide mayor. En CFD se aplica a ambos, de forma fija, entrada = 255 - round(mean(R,G,B)); no hay ajuste por imagen. En La Candelaria no se invierte la intensidad.", body),
        Spacer(1, 6),
        p("Evaluación", styles["SectionX"]),
        p("Se calculan IoU, Dice, precisión, recall y exactitud. Las matrices usan el orden [[TN, FP], [FN, TP]]. Los mapas de error distinguen FP en rojo, FN en azul y TP en verde. Las predicciones de Otsu y K-means nunca construyen las referencias.", body),
        PageBreak(),
    ]

    case_grouped = _by_image(case["rows"])
    case_data = [["Imagen", "O IoU", "O Dice", "O Prec.", "O Rec.", "K IoU", "K Dice", "K Prec.", "K Rec."]]
    for image_id, methods in case_grouped.items():
        o, k = methods["Otsu"], methods["K-means"]
        case_data.append([image_id[:14], fmt(o["iou"]), fmt(o["dice"]), fmt(o["precision"]), fmt(o["recall"]), fmt(k["iou"]), fmt(k["dice"]), fmt(k["precision"]), fmt(k["recall"])])
    case_table = Table(case_data, colWidths=[3.15 * cm] + [1.63 * cm] * 8, repeatRows=1)
    case_table.setStyle(_table_style(5.9))
    aggregate_case = case["aggregate_pixel_weighted"]
    case_matrix_data = [["Método", "Matriz agregada [[TN,FP],[FN,TP]]", "IoU", "Dice", "Precisión", "Recall"]]
    for method in ("Otsu", "K-means"):
        item = aggregate_case[method]
        case_matrix_data.append([method, str(item["confusion_matrix"]), fmt(item["metrics"]["iou"]), fmt(item["metrics"]["dice"]), fmt(item["metrics"]["precision"]), fmt(item["metrics"]["recall"])])
    case_matrix_table = Table(case_matrix_data, colWidths=[1.8 * cm, 6.2 * cm, 1.7 * cm, 1.7 * cm, 1.9 * cm, 1.7 * cm])
    case_matrix_table.setStyle(_table_style(6.0))
    story += [
        p("Resultados — La Candelaria (caso principal)", styles["SectionX"]),
        p("O = Otsu; K = K-means. Estos valores comparan contra referencias preliminares y solo describen el experimento.", styles["SmallX"]),
        case_table,
        Spacer(1, 4),
        case_matrix_table,
        Spacer(1, 5),
        RLImage(str(ROOT / case["overview_figure"]), width=17.6 * cm, height=12.5 * cm),
        p("La similitud visual entre ambos métodos proviene de una partición global en dos grupos de la misma intensidad; no equivale a reconocimiento semántico de superficies construidas.", styles["SmallX"]),
        PageBreak(),
    ]

    metric_data = [["Imagen", "Método", "IoU", "Dice", "Precisión", "Recall"]]
    for row in cfd["rows"]:
        metric_data.append([row["image"], row["method"], fmt(row["iou"]), fmt(row["dice"]), fmt(row["precision"]), fmt(row["recall"])])
    metric_table = Table(metric_data, colWidths=[2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.5 * cm, 2.2 * cm], repeatRows=1)
    metric_table.setStyle(_table_style(6.6))
    matrix_data = [["Imagen", "Otsu [[TN,FP],[FN,TP]]", "K-means [[TN,FP],[FN,TP]]"]]
    for identifier, methods in cfd["confusion_matrices"].items():
        matrix_data.append([identifier, str(methods["Otsu"]), str(methods["K-means"])])
    matrix_table = Table(matrix_data, colWidths=[2.1 * cm, 7.0 * cm, 7.0 * cm], repeatRows=1)
    matrix_table.setStyle(_table_style(6.0))
    aggregate_cfd = cfd["aggregate_pixel_weighted"]
    aggregate_data = [["Método", "IoU", "Dice", "Precisión", "Recall", "Matriz agregada"]]
    for method in ("Otsu", "K-means"):
        item = aggregate_cfd[method]
        aggregate_data.append([method, fmt(item["metrics"]["iou"]), fmt(item["metrics"]["dice"]), fmt(item["metrics"]["precision"]), fmt(item["metrics"]["recall"]), str(item["confusion_matrix"])])
    aggregate_table = Table(aggregate_data, colWidths=[2.0 * cm, 1.6 * cm, 1.6 * cm, 2.0 * cm, 1.7 * cm, 7.0 * cm])
    aggregate_table.setStyle(_table_style(6.2))
    story += [
        p("Resultados — CFD (benchmark complementario)", styles["SectionX"]),
        p("Las cinco referencias son anotaciones oficiales independientes; el mismo preprocesamiento y los mismos parámetros se aplican a todas las imágenes.", styles["SmallX"]),
        Spacer(1, 4), metric_table, Spacer(1, 7),
        p("Matrices de confusión por imagen", styles["SectionX"]), matrix_table, Spacer(1, 7),
        p("Agregado ponderado por píxel", styles["SectionX"]), aggregate_table, Spacer(1, 8),
        p("El recall agregado es alto, pero la precisión es muy baja. La gran cantidad de FP muestra que textura, sombra y cambios de pavimento se confunden con grietas.", body),
        PageBreak(),
    ]

    story += [
        p("Evidencia visual y Quality Gate de CFD", styles["SectionX"]),
        RLImage(str(ROOT / cfd["overview_figure"]), width=17.6 * cm, height=12.5 * cm),
        Spacer(1, 5),
        p("Lectura: la referencia oficial marca la grieta en blanco; en los mapas de error FP = rojo, FN = azul y TP = verde. El predominio del rojo explica la baja precisión de ambos métodos.", body),
        Spacer(1, 6),
        p("Verificaciones realizadas", styles["SectionX"]),
        p("Los cinco pares tienen dimensiones coincidentes; sus hashes están registrados; las etiquetas son exactamente {1,2}; cada archivo .seg coincide con groundTruth - 1; la imagen y la máscara fueron revisadas visualmente para confirmar alineación y polaridad de grieta como foreground. No se detectó un par que exigiera sustitución.", body),
        Spacer(1, 6),
        p("Alcance de la validación", styles["SectionX"]),
        p("La inspección técnica valida los cinco pares CFD seleccionados para este benchmark complementario. No valida las máscaras preliminares de La Candelaria y, por tanto, no modifica el estado pendiente de QG1.", body),
        PageBreak(),
    ]

    co = aggregate_cfd["Otsu"]["metrics"]
    ck = aggregate_cfd["K-means"]["metrics"]
    story += [
        p("Discusión, conclusiones y limitaciones", styles["SectionX"]),
        p("Hallazgos conjuntos", styles["SectionX"]),
        p(f"En CFD, Otsu obtuvo IoU {fmt(co['iou'])}, Dice {fmt(co['dice'])}, precisión {fmt(co['precision'])} y recall {fmt(co['recall'])}; K-means obtuvo IoU {fmt(ck['iou'])}, Dice {fmt(ck['dice'])}, precisión {fmt(ck['precision'])} y recall {fmt(ck['recall'])}. Los resultados son próximos porque ambos separan globalmente dos grupos sobre la misma intensidad invertida. Esta observación se limita a las muestras y parámetros ensayados.", body),
        Spacer(1, 5),
        p("En las escenas de La Candelaria, la iluminación, sombras, color y diversidad semántica hacen que una partición binaria de intensidad no represente directamente urban_built_surface. Las máscaras preliminares permiten conservar y analizar el experimento, pero sus errores impiden usar sus métricas como estimación de exactitud semántica real.", body),
        Spacer(1, 5),
        p("Conclusión técnica", styles["SectionX"]),
        p("Otsu y K-means son reproducibles y útiles como líneas base, pero resultan insuficientes para los dos tipos de estructura evaluados: una clase urbana semántica heterogénea y grietas finas sobre pavimento texturizado. CFD satisface la comparación cuantitativa frente a anotaciones externas; La Candelaria sigue siendo el núcleo visual y metodológico del laboratorio, con la limitación explícita de sus referencias.", body),
        Spacer(1, 5),
        p("Referencias", styles["SectionX"]),
        p("Otsu, N. (1979). A Threshold Selection Method from Gray-Level Histograms. IEEE TSMC, 9(1), 62–66. DOI 10.1109/TSMC.1979.4310076. MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations. Fifth Berkeley Symposium, 1, 281–297.", styles["SmallX"]),
        p("Shi, Y., Cui, L., Qi, Z., Meng, F., & Chen, Z. (2016). Automatic road crack detection using random structured forests. IEEE T-ITS, 17(12), 3434–3445. CrackForest Dataset: repositorio y commit indicados en el manifiesto; licencia de uso no comercial para investigación.", styles["SmallX"]),
        p("Fotografías de La Candelaria: Pedro Szekely (CC BY-SA 2.0); Instituto Caro y Cuervo/Juan Andrés Rodríguez Suárez (CC BY-SA 4.0); Amel Segré (CC BY-SA 3.0); Felipe Restrepo Acosta (CC BY-SA 4.0); Tequendamia (dominio público). URLs completas: data/real/sources.json.", styles["TinyX"]),
        Spacer(1, 5),
        p("Estado final", styles["SectionX"]),
        p("La ejecución, las pruebas y la coherencia entre JSON, notebook y PDF se verifican técnicamente. QG1 no se declara aprobado y el trabajo no se presenta como validación académica definitiva de la clase urbana mientras las máscaras de La Candelaria sigan siendo preliminares.", body),
    ]
    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    results = json.loads((OUTPUT / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data" / "benchmarks" / "cfd5" / "manifest.json").read_text(encoding="utf-8"))
    if results["complementary_benchmark"]["benchmark_manifest"] != "data/benchmarks/cfd5/manifest.json":
        raise ValueError("Integrated results do not reference the expected CFD manifest.")
    write_markdown(OUTPUT / "informe_academico.md", manifest, results)
    build_pdf(OUTPUT / "informe_academico.pdf", manifest, results)
    print(OUTPUT / "informe_academico.pdf")


if __name__ == "__main__":
    main()

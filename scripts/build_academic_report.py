"""Genera la memoria académica (máx. 6 páginas, PDF y Markdown) a partir de deliverables/results.json."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image as RLImage
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "deliverables"
METHODS = ("Otsu", "K-means", "Region Growing", "Watershed")
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DejaVu", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique", boldItalic="DejaVu-Bold")


COURSE = "Percepción Computacional (COEINTAR) - Enero 2026 - PER 15708"
ASSIGNMENT = "Trabajo: Evaluación de la segmentación"
AUTHORS = (("Torres Arredondo", "Mateo"), ("Alvarez", "Edna Mayerly"), ("Murcia", "Manuel Alejandro"))
PROFESSOR = "Luis Fernando Barajas Ardila"
DATE = "18/10/2026"


def fmt(value: float) -> str:
    return f"{value:.4f}"


# --- Texto de la memoria (marcado de ReportLab: <b>, <i>, <sub>, <super>) -----------------

INTRO = [
    "La segmentación divide una imagen en regiones homogéneas que corresponden a los objetos de interés. Este "
    "trabajo construye cuatro segmentadores y mide su rendimiento frente a imágenes de <i>ground truth</i> con un "
    "evaluador implementado desde cero.",
    "<b>Problema principal.</b> Detectar grietas en pavimento con el <i>CrackForest Dataset</i> (CFD; Shi et al., "
    "2016), que aporta anotaciones humanas oficiales. Es un problema exigente: la grieta es una estructura fina, "
    "oscura y minoritaria (1–2 % de los píxeles) sobre un asfalto muy texturizado. <b>Aplicación secundaria.</b> "
    "Segmentar la superficie construida en cinco fotografías urbanas de La Candelaria (Bogotá) con máscaras "
    "manuales propias.",
    "<b>Objetivos:</b> (1) implementar dos métodos del Tema 10 —crecimiento de regiones por semillas y Watershed "
    "con marcadores— y dos líneas base —Otsu y K-means—; (2) implementar un evaluador propio (matriz de confusión, "
    "IoU, Dice, precisión y recall); (3) comparar los cuatro métodos sobre varias imágenes y analizar sus fallos.",
]

PREPROCESSING = (
    "<b>2.1 Preprocesamiento común.</b> Los cuatro métodos reciben la misma entrada, con parámetros fijos para todas "
    "las imágenes del escenario. En CFD: gris como media de R, G y B; suavizado gaussiano separable propio "
    "(σ = 1.5) para atenuar la textura; inversión 255 − I para que la grieta sea el objeto claro; y resta de "
    "fondo I′ = 4·(I − G<sub>15</sub>∗I) + 128, que elimina la iluminación no uniforme y realza estructuras finas. En "
    "La Candelaria: reducción a lado máximo 1024 px (coste computacional), gris y suavizado σ = 1.5; las máscaras "
    "se devuelven a resolución original por vecino más cercano antes de evaluar."
)

BASELINES = (
    "<b>2.2 Líneas base.</b> <i>Otsu</i> (1979) elige el umbral t que maximiza la varianza entre clases "
    "σ<sub>B</sub><super>2</super>(t) = [μ<sub>T</sub>·ω(t) − μ(t)]<super>2</super> / [ω(t)·(1 − ω(t))], con ω(t) y "
    "μ(t) la probabilidad y la media acumuladas del histograma. <i>K-means</i> (MacQueen, 1967) con k = 2 sobre "
    "intensidades alterna la asignación al centroide más cercano y la actualización de medias (inicialización "
    "mín./máx., tolerancia 10<super>−4</super>); el grupo claro es el objeto. Ambos deciden con un único umbral "
    "global y no usan información espacial."
)

REGION_GROWING = (
    "<b>2.3 Crecimiento de regiones por semillas (Tema 10.4).</b> Técnica <i>bottom-up</i> de agregación local "
    "(Adams y Bischof, 1994). Semillas: píxeles con intensidad ≥ percentil P de la entrada (CFD: P = 99.8, los "
    "más oscuros del original). La región R crece por búsqueda en anchura con vecindad 8; la función de "
    "pertenencia acepta un vecino p si |I(p) − μ<sub>R</sub>| ≤ T, donde μ<sub>R</sub> es la media actual de la "
    "región, que se actualiza con cada inserción (CFD: T = 30; La Candelaria: P = 70, T = 25). El orden de entrada "
    "de cada píxel se guarda para visualizar la evolución del crecimiento."
)

WATERSHED = (
    "<b>2.4 Watershed con marcadores (Tema 10.7).</b> La imagen se interpreta como un relieve topográfico: la "
    "magnitud del gradiente de Sobel √(G<sub>x</sub><super>2</super> + G<sub>y</sub><super>2</super>). Para evitar "
    "la sobresegmentación se inunda solo desde marcadores: objeto = intensidad ≥ percentil 99.5; fondo = píxeles "
    "a más de d = 3 px de cualquier candidato (I > umbral de Otsu), según un mapa de distancias euclídeas (La "
    "Candelaria: percentil 80, d = 10). La inundación sigue el algoritmo de cola de prioridad de Meyer (1994): se "
    "extrae siempre el píxel de menor nivel, que hereda la etiqueta de la cuenca que lo alcanzó; las fronteras "
    "entre cuencas forman las líneas de división (diques)."
)

AUTHORSHIP = (
    "<b>2.5 Autoría del código.</b> Todos los algoritmos —gaussiano, Sobel, Otsu, K-means, crecimiento de regiones, "
    "inundación Watershed y métricas— están implementados en NumPy/Python en <i>src/segmentation_lab/</i>. Única "
    "parte de software existente: <i>scipy.ndimage.distance_transform_edt</i> (Virtanen et al., 2020), usada para "
    "el mapa de distancias de los marcadores de fondo."
)

METRICS = [
    "<b>3.1 Definiciones.</b> Para una máscara predicha S y el <i>ground truth</i> GT, ambos binarios: "
    "TP = |S ∩ GT|, FP = |S ∖ GT|, FN = |GT ∖ S| y TN = resto. Se calculan IoU = TP/(TP + FP + FN), "
    "Dice = 2TP/(2TP + FP + FN), precisión = TP/(TP + FP), recall = TP/(TP + FN) y exactitud. La exactitud se "
    "reporta pero no se usa para comparar: en CFD un detector que no marque nada obtendría ≈ 0.98 por el "
    "desbalance de clases. IoU y Dice ignoran TN y miden solapamiento real con el objeto.",
    "<b>3.2 Implementación propia.</b> <i>confusion_matrix</i> convierte ambas máscaras a booleanos y cuenta "
    "TN, FP, FN y TP con operaciones lógicas (¬GT ∧ ¬S, ¬GT ∧ S, GT ∧ ¬S, GT ∧ S), devolviendo "
    "[[TN, FP], [FN, TP]]; <i>binary_metrics</i> deriva las cinco métricas con protección ante denominadores nulos. "
    "No se usa ninguna biblioteca de métricas. Pruebas unitarias verifican casos calculables a mano (p. ej. "
    "TP = FP = FN = TN = 1 → IoU = 1/3, Dice = 0.5). Se reportan métricas por imagen, el promedio por imagen y la "
    "matriz agregada ponderada por píxel. El notebook recalcula todo y comprueba que coincide con "
    "<i>deliverables/results.json</i>, fuente de las tablas de esta memoria.",
]

DATA_CFD = (
    "<b>4.1 CFD.</b> Cinco imágenes de 480×320 (IDs 001, 024, 050, 074 y 116) seleccionadas por una regla "
    "determinista sin mirar resultados (pares completos y consistentes; posiciones equidistantes sobre los IDs "
    "ordenados; se excluye 042 por desalineación conocida). <i>Ground truth</i>: <i>groundTruth.Segmentation</i>, "
    "etiqueta 2 = grieta."
)


def conclusions(cfd: dict, cand: dict) -> list[str]:
    m = cfd["mean_per_image"]
    rg_sens = {row["tolerance"]: row for row in cfd["sensitivity"]["region_growing_tolerance"]}
    ratio = m["Region Growing"]["dice"] / m["Otsu"]["dice"]
    return [
        f"<b>Umbral global frente a crecimiento de regiones.</b> En CFD, Otsu y K-means alcanzan recall ≈ "
        f"{m['Otsu']['recall']:.2f} pero precisión ≈ {m['Otsu']['precision']:.2f}: el asfalto tiene muchos píxeles "
        "con la misma intensidad que la grieta y un umbral global no puede separarlos (rojo en la figura 2). Region "
        "Growing y Watershed añaden conectividad espacial: parten de semillas o marcadores fiables y solo crecen por "
        f"píxeles conectados y similares. Así multiplican el Dice por ≈ {ratio:.0f} "
        f"(RG {fmt(m['Region Growing']['dice'])}, WS {fmt(m['Watershed']['dice'])}).",
        "<b>Region Growing</b> es el más preciso, pero sensible a sus parámetros (tabla 3): con T = 20 crece poco "
        f"(recall {rg_sens[20.0]['mean_recall']:.2f}), y con T = 40 se desborda hacia el asfalto (precisión "
        f"{rg_sens[40.0]['mean_precision']:.2f}). Los tramos de grieta sin semilla o interrumpidos por un píxel más "
        "claro quedan sin detectar (azul en 024 y 074). <b>Watershed</b> tiene mayor recall porque el gradiente "
        "cierra tramos discontinuos, pero cada marcador de objeto que cae sobre una mancha oscura genera una cuenca "
        "falsa (manchas rojas en 116). Acercar los marcadores de fondo (d = 3) limita ese crecimiento.",
        "<b>La Candelaria.</b> La superficie construida no corresponde a un nivel de intensidad: el cielo es claro "
        "y las fachadas en sombra son oscuras. Por eso los cuatro métodos rinden de forma parecida y limitada "
        f"(Dice medio {min(v['dice'] for v in cand['mean_per_image'].values()):.2f}–"
        f"{max(v['dice'] for v in cand['mean_per_image'].values()):.2f}). Segmentar una clase semántica requiere "
        "color, textura o aprendizaje, no solo intensidad.",
        "<b>Limitaciones.</b> Cinco imágenes por escenario. Los parámetros se fijaron por escenario tras un barrido "
        "exploratorio sobre las mismas imágenes evaluadas, por lo que las cifras de RG y WS son optimistas; la "
        "tabla 3 muestra cuánto varían. Las referencias de La Candelaria son polígonos manuales aproximados, no "
        "anotación experta. <b>Conclusión:</b> para estructuras finas sobre fondo texturizado, los métodos de "
        "crecimiento de regiones del Tema 10 superan claramente a la umbralización global, a costa de depender de "
        "la colocación de semillas o marcadores.",
    ]


REFERENCES = [
    "Adams, R., &amp; Bischof, L. (1994). Seeded region growing. <i>IEEE TPAMI</i>, 16(6), 641–647.",
    "Beucher, S., &amp; Meyer, F. (1993). The morphological approach to segmentation: the watershed transformation. "
    "En <i>Mathematical Morphology in Image Processing</i> (pp. 433–481). Marcel Dekker.",
    "Gonzalez, R. C., &amp; Woods, R. E. (2018). <i>Digital Image Processing</i> (4.ª ed.). Pearson.",
    "MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. "
    "<i>Proc. 5th Berkeley Symposium</i>, 1, 281–297.",
    "Meyer, F. (1994). Topographic distance and watershed lines. <i>Signal Processing</i>, 38(1), 113–125.",
    "Otsu, N. (1979). A threshold selection method from gray-level histograms. <i>IEEE TSMC</i>, 9(1), 62–66.",
    "Shi, Y., Cui, L., Qi, Z., Meng, F., &amp; Chen, Z. (2016). Automatic road crack detection using random "
    "structured forests. <i>IEEE T-ITS</i>, 17(12), 3434–3445. Datos: github.com/cuilimeng/CrackForest-dataset "
    "(uso no comercial para investigación).",
    "UNIR. <i>Percepción Computacional</i>, Tema 10: Crecimiento de regiones.",
    "Virtanen, P., et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. "
    "<i>Nature Methods</i>, 17, 261–272.",
    "Fotografías de La Candelaria (Wikimedia Commons): P. Szekely (CC BY-SA 2.0); Instituto Caro y Cuervo / "
    "J. A. Rodríguez Suárez (CC BY-SA 4.0); A. Segré (CC BY-SA 3.0); F. Restrepo Acosta (CC BY-SA 4.0); "
    "Tequendamia (dominio público). Enlaces en <i>data/real/sources.json</i>.",
]


# --- Markdown ---------------------------------------------------------------------------

def to_markdown(text: str) -> str:
    text = re.sub(r"</?b>", "**", text)
    text = re.sub(r"</?i>", "*", text)
    text = re.sub(r"<sub>(.*?)</sub>", r"_\1", text)
    text = re.sub(r"<super>(.*?)</super>", r"^\1", text)
    return text.replace("&amp;", "&")


def metric_rows(rows: list[dict]) -> list[list[str]]:
    return [[r["image"], r["method"], fmt(r["iou"]), fmt(r["dice"]), fmt(r["precision"]), fmt(r["recall"])] for r in rows]


def write_markdown(path: Path, results: dict) -> None:
    cfd, cand = results["cfd"], results["la_candelaria"]
    lines = [
        f"# {results['title']}",
        "",
        f"**Universidad Internacional de La Rioja (UNIR)** — {COURSE}",
        "",
        f"**{ASSIGNMENT}**",
        "",
        "**Integrantes:** " + "; ".join(f"{name} {surname}" for surname, name in AUTHORS),
        "",
        f"**Profesor:** {PROFESSOR}  ",
        f"**Fecha:** {DATE}",
        "",
        "## 1. Introducción y objetivos",
        "",
    ]
    lines += [to_markdown(t) + "\n" for t in INTRO]
    lines += ["## 2. Métodos de segmentación", ""]
    lines += [to_markdown(t) + "\n" for t in (PREPROCESSING, BASELINES, REGION_GROWING, WATERSHED, AUTHORSHIP)]
    lines += ["## 3. Métricas de evaluación e implementación", ""]
    lines += [to_markdown(t) + "\n" for t in METRICS]
    lines += ["## 4. Resultados experimentales", "", to_markdown(DATA_CFD), ""]
    lines += ["| Imagen | Método | IoU | Dice | Precisión | Recall |", "|---|---|---:|---:|---:|---:|"]
    lines += ["| " + " | ".join(row) + " |" for row in metric_rows(cfd["rows"])]
    lines += ["", "| Método (CFD) | IoU medio | Dice medio | Precisión | Recall | Matriz agregada [[TN,FP],[FN,TP]] |", "|---|---:|---:|---:|---:|---|"]
    for method in METHODS:
        mean = cfd["mean_per_image"][method]
        matrix = cfd["aggregate_pixel_weighted"][method]["confusion_matrix"]
        lines.append(f"| {method} | {fmt(mean['iou'])} | {fmt(mean['dice'])} | {fmt(mean['precision'])} | {fmt(mean['recall'])} | `{matrix}` |")
    lines += ["", "![CFD](figures/cfd_overview.png)", "", "![Pasos](figures/cfd_pasos.png)", ""]
    lines += ["### La Candelaria (métricas orientativas)", "", "| Imagen | Método | IoU | Dice | Precisión | Recall |", "|---|---|---:|---:|---:|---:|"]
    lines += ["| " + " | ".join(row) + " |" for row in metric_rows(cand["rows"])]
    lines += ["", "![La Candelaria](figures/la_candelaria_overview.png)", "", "## 5. Conclusiones", ""]
    lines += [to_markdown(t) + "\n" for t in conclusions(cfd, cand)]
    lines += ["## Referencias", ""] + [f"- {to_markdown(r)}" for r in REFERENCES]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- PDF --------------------------------------------------------------------------------

def _table_style(font_size: float = 7.2) -> TableStyle:
    return TableStyle(
        [
            ("FONTNAME", (0, 0), (-1, -1), "DejaVu"),
            ("FONTNAME", (0, 0), (-1, 0), "DejaVu-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0098cd")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#8a969c")),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eaf5fa")]),
            ("TOPPADDING", (0, 0), (-1, -1), 1.6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
        ]
    )


def _figure(path: Path, width_cm: float) -> RLImage:
    with PILImage.open(path) as image:
        ratio = image.size[1] / image.size[0]
    return RLImage(str(path), width=width_cm * cm, height=width_cm * ratio * cm)


def _header() -> Table:
    blue = colors.HexColor("#0098cd")
    authors = "\n".join(f"{surname}, {name}" for surname, name in AUTHORS)
    data = [
        ["Asignatura", "Datos de los alumnos", "Fecha"],
        ["Percepción\nComputacional", f"Apellidos, Nombre:\n{authors}", DATE],
    ]
    table = Table(data, colWidths=[4.2 * cm, 9.0 * cm, 4.0 * cm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "DejaVu"),
                ("FONTNAME", (0, 1), (0, 1), "DejaVu-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 1), (-1, 1), 10.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9eff7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), blue),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("ALIGN", (1, 1), (1, 1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LINEABOVE", (0, 0), (-1, 0), 0.8, blue),
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, blue),
                ("LINEBELOW", (0, 1), (-1, 1), 0.8, blue),
                ("LINEBEFORE", (1, 0), (2, 1), 0.5, blue),
            ]
        )
    )
    return table


def _cover(title_text: str) -> list:
    blue = colors.HexColor("#0098cd")
    base = getSampleStyleSheet()
    center = ParagraphStyle("CoverCenter", parent=base["BodyText"], fontName="DejaVu", fontSize=11, leading=15, alignment=TA_CENTER)
    brand = ParagraphStyle("CoverBrand", parent=center, fontName="DejaVu-Bold", fontSize=13, leading=17, textColor=blue)
    course = ParagraphStyle("CoverCourse", parent=center, fontSize=12, leading=16, textColor=colors.HexColor("#0b4f6c"))
    assignment = ParagraphStyle("CoverAssignment", parent=center, fontName="DejaVu-Bold", fontSize=14, leading=18)
    title = ParagraphStyle("CoverTitle", parent=center, fontSize=19, leading=25, textColor=blue)
    label = ParagraphStyle("CoverLabel", parent=center, fontName="DejaVu-Bold", fontSize=10.5, leading=14, textColor=colors.HexColor("#0b4f6c"))

    def rule() -> Table:
        line = Table([[""]], colWidths=[12 * cm], rowHeights=[0.1 * cm])
        line.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 1.2, blue)]))
        return line

    story = [
        Spacer(1, 2.2 * cm),
        Paragraph("Universidad Internacional de La Rioja (UNIR)", brand),
        Spacer(1, 0.3 * cm),
        Paragraph(COURSE, course),
        Spacer(1, 2.6 * cm),
        Paragraph(ASSIGNMENT, assignment),
        Spacer(1, 0.5 * cm),
        rule(),
        Spacer(1, 0.5 * cm),
        Paragraph(title_text, title),
        Spacer(1, 0.5 * cm),
        rule(),
        Spacer(1, 3.0 * cm),
        Paragraph("Integrantes", label),
        Spacer(1, 0.15 * cm),
    ]
    story += [Paragraph(f"{name} {surname}", center) for surname, name in AUTHORS]
    story += [
        Spacer(1, 1.0 * cm),
        Paragraph("Profesor", label),
        Spacer(1, 0.15 * cm),
        Paragraph(PROFESSOR, center),
        Spacer(1, 2.6 * cm),
        Paragraph(DATE, center),
        PageBreak(),
    ]
    return story


def page_number(canvas, doc) -> None:
    """Numera las páginas de contenido; la portada no lleva número."""
    canvas.saveState()
    canvas.setFont("DejaVu", 7.5)
    canvas.drawRightString(A4[0] - 1.6 * cm, 0.8 * cm, f"Página {doc.page - 1}")
    canvas.restoreState()


def no_page_number(canvas, doc) -> None:
    pass


def build_pdf(path: Path, results: dict) -> None:
    cfd, cand = results["cfd"], results["la_candelaria"]
    base = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=base["BodyText"], fontName="DejaVu", fontSize=8.6, leading=11.2, alignment=TA_JUSTIFY, spaceAfter=3.5)
    small = ParagraphStyle("Small", parent=body, fontSize=7.3, leading=9.0, spaceAfter=2)
    caption = ParagraphStyle("Caption", parent=small, fontName="DejaVu-Oblique", alignment=TA_CENTER, spaceAfter=5)
    title = ParagraphStyle("TitleX", parent=base["Title"], fontName="DejaVu", fontSize=15, leading=19, textColor=colors.HexColor("#0098cd"), spaceBefore=8, spaceAfter=8)
    h1 = ParagraphStyle("H1", parent=base["Heading2"], fontName="DejaVu-Bold", fontSize=11, leading=13.5, spaceBefore=5, spaceAfter=3, textColor=colors.HexColor("#0b4f6c"))
    doc = SimpleDocTemplate(
        str(path), pagesize=A4, leftMargin=1.9 * cm, rightMargin=1.9 * cm, topMargin=1.4 * cm, bottomMargin=1.5 * cm,
        title=results["title"], author="; ".join(f"{name} {surname}" for surname, name in AUTHORS),
    )

    story: list = _cover(results["title"]) + [_header(), Paragraph(results["title"], title)]
    story.append(Paragraph("1. Introducción y objetivos", h1))
    story += [Paragraph(text, body) for text in INTRO]
    story.append(Paragraph("2. Métodos de segmentación", h1))
    story += [Paragraph(text, body) for text in (PREPROCESSING, BASELINES, REGION_GROWING, WATERSHED, AUTHORSHIP)]
    story.append(Paragraph("3. Métricas de evaluación e implementación del evaluador", h1))
    story += [Paragraph(text, body) for text in METRICS]

    story.append(Paragraph("4. Resultados experimentales y comparativa", h1))
    story.append(Paragraph(DATA_CFD, body))
    grouped: dict[str, dict[str, dict]] = {}
    for row in cfd["rows"]:
        grouped.setdefault(row["image"], {})[row["method"]] = row
    header = ["Imagen"] + [f"{abbr} {metric}" for abbr in ("Otsu", "K-m", "RG", "WS") for metric in ("IoU", "Dice")]
    per_image = [header]
    for image_id, methods in grouped.items():
        per_image.append([image_id] + [fmt(methods[m][f]) for m in METHODS for f in ("iou", "dice")])
    table1 = Table(per_image, colWidths=[1.5 * cm] + [1.95 * cm] * 8)
    table1.setStyle(_table_style(7.0))
    story += [Paragraph("Tabla 1. CFD: IoU y Dice por imagen (RG = Region Growing; WS = Watershed).", caption), table1, Spacer(1, 4)]

    pr_header = ["Imagen"] + [f"{abbr} {metric}" for abbr in ("Otsu", "K-m", "RG", "WS") for metric in ("Prec.", "Rec.")]
    per_image_pr = [pr_header]
    for image_id, methods in grouped.items():
        per_image_pr.append([image_id] + [fmt(methods[m][f]) for m in METHODS for f in ("precision", "recall")])
    table1b = Table(per_image_pr, colWidths=[1.5 * cm] + [1.95 * cm] * 8)
    table1b.setStyle(_table_style(7.0))
    story += [Paragraph("Tabla 1b. CFD: precisión y recall por imagen.", caption), table1b, Spacer(1, 4)]

    summary = [["Método", "IoU", "Dice", "Precisión", "Recall", "Matriz agregada [[TN, FP], [FN, TP]]"]]
    for method in METHODS:
        mean = cfd["mean_per_image"][method]
        summary.append([method, fmt(mean["iou"]), fmt(mean["dice"]), fmt(mean["precision"]), fmt(mean["recall"]), str(cfd["aggregate_pixel_weighted"][method]["confusion_matrix"])])
    table2 = Table(summary, colWidths=[2.6 * cm, 1.6 * cm, 1.6 * cm, 1.8 * cm, 1.6 * cm, 8.0 * cm])
    table2.setStyle(_table_style(7.0))
    story += [KeepTogether([Paragraph("Tabla 2. CFD: métricas promediadas por imagen y matriz de confusión agregada por píxel.", caption), table2]), Spacer(1, 5)]

    story += [
        KeepTogether([
            _figure(ROOT / cfd["steps_figure"], 17.2),
            Paragraph("Figura 1. Pasos intermedios sobre CFD 001: entrada común; semillas y orden de crecimiento de Region Growing; gradiente, mapa de distancias, marcadores y líneas de cuenca de Watershed.", caption),
        ]),
        KeepTogether([
            _figure(ROOT / cfd["overview_figure"], 17.2),
            Paragraph("Figura 2. CFD: imagen, <i>ground truth</i> y mapa de errores de cada método (rojo = FP, azul = FN, verde = TP).", caption),
        ]),
    ]

    sens = [["RG: T", "IoU", "Dice", "Prec.", "Recall", "", "WS: d (px)", "IoU", "Dice", "Prec.", "Recall"]]
    rg_rows = cfd["sensitivity"]["region_growing_tolerance"]
    ws_rows = cfd["sensitivity"]["watershed_background_distance"]
    for index in range(max(len(rg_rows), len(ws_rows))):
        line = []
        if index < len(rg_rows):
            r = rg_rows[index]
            line += [f"{r['tolerance']:g}", fmt(r["mean_iou"]), fmt(r["mean_dice"]), fmt(r["mean_precision"]), fmt(r["mean_recall"])]
        else:
            line += [""] * 5
        line.append("")
        if index < len(ws_rows):
            w = ws_rows[index]
            line += [f"{w['background_distance']:g}", fmt(w["mean_iou"]), fmt(w["mean_dice"]), fmt(w["mean_precision"]), fmt(w["mean_recall"])]
        else:
            line += [""] * 5
        sens.append(line)
    table3 = Table(sens, colWidths=[2.4 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm, 0.4 * cm, 2.4 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm])
    style3 = _table_style(7.0)
    style3.add("BACKGROUND", (5, 0), (5, -1), colors.white)
    style3.add("LINEBEFORE", (5, 0), (5, -1), 0, colors.white)
    table3.setStyle(style3)
    story += [KeepTogether([Paragraph("Tabla 3. CFD: sensibilidad (promedio por imagen) a la tolerancia T de Region Growing y a la distancia d de los marcadores de fondo de Watershed.", caption), table3]), Spacer(1, 5)]

    cand_grouped: dict[str, dict[str, dict]] = {}
    for row in cand["rows"]:
        cand_grouped.setdefault(row["image"], {})[row["method"]] = row
    cand_table = [header]
    for image_id, methods in cand_grouped.items():
        cand_table.append([image_id.split("_", 1)[0]] + [fmt(methods[m][f]) for m in METHODS for f in ("iou", "dice")])
    cand_table.append(["Media"] + [fmt(cand["mean_per_image"][m][f]) for m in METHODS for f in ("iou", "dice")])
    table4 = Table(cand_table, colWidths=[1.5 * cm] + [1.95 * cm] * 8)
    style4 = _table_style(7.0)
    style4.add("FONTNAME", (0, -1), (-1, -1), "DejaVu-Bold")
    table4.setStyle(style4)
    story += [
        Paragraph(
            "<b>4.2 La Candelaria (métricas orientativas).</b> Cinco fotografías urbanas; la referencia "
            "<i>urban_built_surface</i> (fachadas, cubiertas y pavimento) son polígonos manuales propios trazados sin "
            "usar los segmentadores. Al ser contornos aproximados, las métricas solo "
            "sirven para comparar métodos entre sí.",
            body,
        ),
        KeepTogether([Paragraph("Tabla 4. La Candelaria: IoU y Dice por imagen (01–05) y media.", caption), table4]),
        Spacer(1, 4),
        KeepTogether([
            _figure(ROOT / cand["overview_figure"], 13.5),
            Paragraph("Figura 3. La Candelaria: referencia manual y mapas de error de los cuatro métodos.", caption),
        ]),
    ]

    story.append(Paragraph("5. Conclusiones", h1))
    story += [Paragraph(text, body) for text in conclusions(cfd, cand)]
    story.append(Paragraph("Referencias", h1))
    story += [Paragraph(text, small) for text in REFERENCES]
    doc.build(story, onFirstPage=no_page_number, onLaterPages=page_number)


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    results = json.loads((OUTPUT / "results.json").read_text(encoding="utf-8"))
    write_markdown(OUTPUT / "informe_academico.md", results)
    build_pdf(OUTPUT / "informe_academico.pdf", results)
    print(OUTPUT / "informe_academico.pdf")


if __name__ == "__main__":
    main()

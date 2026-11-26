"""Professional PDF report generation for AI Data Analyst."""
from datetime import datetime, timezone
from io import BytesIO
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


def _table(data, font_size=7, header_color="#17324D"):
    table = Table(data, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#D8DEE8")),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _paragraphs(text, style):
    if not text:
        return []
    return [
        Paragraph(escape(part), style)
        for part in str(text).splitlines() if part.strip()
    ]


def build_pdf_report(
    filename, df, profile, numeric, findings=None, *,
    report_title="AI Data Analyst | Analysis Report",
    prepared_by="", executive_summary="", methodology="",
    sections=None, cleaning_report=None, anomalies=None,
    model_result=None, chart_images=None,
):
    """Build a PDF report from selected analysis outputs and user-authored context."""
    selected = set(sections if sections is not None else [
        "Dataset overview", "Key findings", "Numeric summary",
    ])
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, rightMargin=17 * mm, leftMargin=17 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportNote", parent=styles["BodyText"], fontSize=8,
        leading=10, textColor=colors.HexColor("#596579"),
    ))
    story = [
        Paragraph(escape(report_title or "Analysis Report"), styles["Title"]),
        Paragraph("Source dataset: " + escape(str(filename)), styles["BodyText"]),
        Paragraph(
            "Generated: " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            styles["ReportNote"],
        ),
    ]
    if prepared_by.strip():
        story.append(Paragraph("Prepared by: " + escape(prepared_by.strip()), styles["ReportNote"]))
    story.append(Spacer(1, 6 * mm))

    if executive_summary.strip():
        story.append(Paragraph("Executive summary", styles["Heading2"]))
        story.extend(_paragraphs(executive_summary, styles["BodyText"]))
        story.append(Spacer(1, 3 * mm))

    if "Dataset overview" in selected:
        rows = [
            ["Metric", "Value"],
            ["Rows", f"{profile['rows']:,}"],
            ["Columns", f"{profile['columns']:,}"],
            ["Missing cells", f"{profile['missing_cells']:,} ({profile['missing_pct']:.2f}%)"],
            ["Duplicate rows", f"{profile['duplicate_rows']:,}"],
            ["Heuristic quality score", f"{profile['quality_score']:.1f}/100"],
        ]
        story += [
            Paragraph("Dataset overview", styles["Heading2"]),
            _table(rows, font_size=8), Spacer(1, 4 * mm),
        ]

    if methodology.strip():
        story.append(Paragraph("Methodology and limitations", styles["Heading2"]))
        story.extend(_paragraphs(methodology, styles["BodyText"]))
        story.append(Spacer(1, 3 * mm))

    if "Key findings" in selected and findings:
        story.append(Paragraph("Key findings", styles["Heading2"]))
        story.extend(
            Paragraph("• " + escape(str(item)), styles["BodyText"])
            for item in findings[:15]
        )

    if "Numeric summary" in selected and numeric is not None and not numeric.empty:
        story.append(Paragraph("Numeric summary", styles["Heading2"]))
        data = [list(map(str, numeric.columns))]
        data.extend(numeric.head(25).round(3).astype(str).values.tolist())
        story.append(_table(data, font_size=6))

    if "Correlation matrix" in selected and len(df.select_dtypes(include="number").columns) >= 2:
        corr = df.select_dtypes(include="number").corr()
        corr_data = [["Variable"] + [str(c)[:18] for c in corr.columns[:8]]]
        for index, row in corr.iloc[:8, :8].iterrows():
            corr_data.append([str(index)[:22]] + [f"{v:.2f}" for v in row.iloc[:8]])
        story += [
            Paragraph("Correlation matrix (Pearson)", styles["Heading2"]),
            _table(corr_data, font_size=6), Spacer(1, 3 * mm),
        ]

    if "Cleaning audit" in selected and cleaning_report:
        audit = cleaning_report.get("per_column")
        if audit is not None and not audit.empty:
            story.append(Paragraph("Cleaning audit by column", styles["Heading2"]))
            data = [list(map(str, audit.columns))]
            data.extend(audit.head(25).fillna("").astype(str).values.tolist())
            story.append(_table(data, font_size=6))

    if "Anomaly results" in selected and anomalies is not None and not anomalies.empty:
        story.append(Paragraph("Potential anomaly records", styles["Heading2"]))
        data = [list(map(str, anomalies.columns))]
        data.extend(anomalies.head(20).fillna("").astype(str).values.tolist())
        story.append(_table(data, font_size=6))

    if "Predictive model results" in selected and model_result:
        metrics = model_result.get("metrics", {})
        if metrics:
            story.append(Paragraph("Predictive model baseline", styles["Heading2"]))
            story.append(_table(
                [["Metric", "Value"]] +
                [[str(k), f"{v:.4f}" if isinstance(v, (int, float)) else str(v)]
                 for k, v in metrics.items()],
                font_size=7,
            ))
            story.append(Paragraph(
                "Model metrics are baseline estimates. Review the target definition, "
                "feature leakage, sampling, validation design and assumptions before use.",
                styles["ReportNote"],
            ))

    if "Dataset sample" in selected and df is not None and not df.empty:
        sample = df.head(12).iloc[:, :8].copy().fillna("")
        data = [list(map(str, sample.columns))]
        data.extend(sample.astype(str).values.tolist())
        story.append(Paragraph("Dataset sample (first 12 rows; up to 8 columns)", styles["Heading2"]))
        story.append(_table(data, font_size=5))

    for chart in chart_images or []:
        title, image_bytes = chart
        story.append(Paragraph(escape(str(title)), styles["Heading2"]))
        story.append(Image(BytesIO(image_bytes), width=165 * mm, height=88 * mm))
        story.append(Spacer(1, 3 * mm))

    story += [
        Spacer(1, 5 * mm),
        Paragraph(
            "Interpretation note: the quality score is a transparent heuristic "
            "(70% completeness, 30% row uniqueness), not a certification. "
            "Correlation indicates association, not causation. Anomaly flags are "
            "screening signals, not proof of errors. Validate data definitions, "
            "units, sampling and analytical assumptions before decisions.",
            styles["ReportNote"],
        ),
    ]
    doc.build(story)
    return buf.getvalue()


def build_text_report_pdf(text, report_title="AI Smart Assistant | Technical Report", prepared_by=""):
    """Render a Markdown-like assistant response as a clean, downloadable PDF."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    from xml.sax.saxutils import escape
    from io import BytesIO

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=19 * mm, leftMargin=19 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TechnicalBody", parent=styles["BodyText"], fontSize=9,
        leading=13, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="TechnicalH1", parent=styles["Heading1"], textColor=colors.HexColor("#17324D"),
        spaceBefore=12, spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="TechnicalH2", parent=styles["Heading2"], textColor=colors.HexColor("#245B7A"),
        spaceBefore=9, spaceAfter=5,
    ))
    story = [Paragraph(escape(report_title), styles["Title"])]
    if prepared_by:
        story.append(Paragraph("Prepared by: " + escape(prepared_by), styles["Italic"]))
    story.append(Spacer(1, 7 * mm))
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line:
            story.append(Spacer(1, 2 * mm))
            continue
        if line.startswith("### "):
            story.append(Paragraph(escape(line[4:]), styles["Heading3"]))
        elif line.startswith("## "):
            story.append(Paragraph(escape(line[3:]), styles["TechnicalH2"]))
        elif line.startswith("# "):
            story.append(Paragraph(escape(line[2:]), styles["TechnicalH1"]))
        elif line.startswith(("- ", "* ")):
            story.append(Paragraph("&#8226; " + escape(line[2:]), styles["TechnicalBody"]))
        else:
            story.append(Paragraph(escape(line), styles["TechnicalBody"]))
    doc.build(story)
    return buffer.getvalue()

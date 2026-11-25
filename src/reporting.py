"""PDF summary report generation."""
from io import BytesIO
from datetime import datetime, timezone
from xml.sax.saxutils import escape
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

def build_pdf_report(filename,df,profile,numeric,findings=None):
    buf=BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=A4,rightMargin=17*mm,leftMargin=17*mm,topMargin=17*mm,bottomMargin=17*mm)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="NoteSmall",parent=styles["BodyText"],fontSize=8,leading=10,textColor=colors.HexColor("#596579")))
    story=[Paragraph("AI Data Analyst | Dataset Report",styles["Title"]),Paragraph("File: "+escape(str(filename)),styles["BodyText"]),
      Paragraph("Generated: "+datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),styles["NoteSmall"]),Spacer(1,7*mm)]
    rows=[["Metric","Value"],["Rows",f"{profile['rows']:,}"],["Columns",f"{profile['columns']:,}"],
      ["Missing cells",f"{profile['missing_cells']:,} ({profile['missing_pct']:.2f}%)"],
      ["Duplicate rows",f"{profile['duplicate_rows']:,}"],["Heuristic quality score",f"{profile['quality_score']:.1f}/100"]]
    t=Table(rows,colWidths=[65*mm,90*mm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17324D")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
      ("GRID",(0,0),(-1,-1),.4,colors.HexColor("#D8DEE8")),("PADDING",(0,0),(-1,-1),7)]))
    story += [Paragraph("Dataset overview",styles["Heading2"]),t,Spacer(1,5*mm)]
    if findings:
        story.append(Paragraph("Key observations",styles["Heading2"]))
        story.extend(Paragraph("• "+escape(str(x)),styles["BodyText"]) for x in findings[:12])
    if not numeric.empty:
        story.append(Paragraph("Numeric summary",styles["Heading2"]))
        data=[list(numeric.columns)]+numeric.head(20).round(3).astype(str).values.tolist()
        tab=Table(data,repeatRows=1)
        tab.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17324D")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
          ("FONTSIZE",(0,0),(-1,-1),7),("GRID",(0,0),(-1,-1),.25,colors.HexColor("#D8DEE8")),("PADDING",(0,0),(-1,-1),4)]))
        story.append(tab)
    story.append(Spacer(1,6*mm))
    story.append(Paragraph("The quality score is a transparent heuristic (70% completeness, 30% row uniqueness), not a certification. Correlation is not causation. Validate modeling results before operational use.",styles["NoteSmall"]))
    doc.build(story)
    return buf.getvalue()

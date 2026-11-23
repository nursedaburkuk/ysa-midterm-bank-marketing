"""make_report.py'deki rapor içeriğini (story) JSON bloklarına çevirir; Word sürümü bu dosyadan üretilir."""
import json
import sys

from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, Spacer, Table

from src import make_report as M


def walk(items, out):
    for it in items:
        if isinstance(it, KeepTogether):
            walk(it._content, out)
        elif isinstance(it, Paragraph):
            out.append({"type": it.style.name, "text": it.text})
        elif isinstance(it, Image):
            out.append({"type": "image", "path": it.filename, "w": it.drawWidth, "h": it.drawHeight})
        elif isinstance(it, Table):
            cells = it._cellvalues
            if cells and isinstance(cells[0][0], Image):
                out.append({"type": "images", "items": [{"path": c.filename, "w": c.drawWidth, "h": c.drawHeight} for c in cells[0]]})
            else:
                out.append({"type": "table", "widths": list(it._colWidths),
                            "rows": [[c.text if isinstance(c, Paragraph) else str(c) for c in r] for r in cells]})
        elif isinstance(it, PageBreak):
            out.append({"type": "pagebreak"})
        elif isinstance(it, Spacer):
            out.append({"type": "spacer", "h": it.height})


out = []
walk(M.story, out)
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
print(len(out), "blok")

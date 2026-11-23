// report_blocks.json -> report.docx
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType,
  AlignmentType, ShadingType, BorderStyle, Footer, PageNumber, HeadingLevel, PageBreak,
} = require("docx");

const [, , blocksPath, outPath] = process.argv;
const blocks = JSON.parse(fs.readFileSync(blocksPath, "utf8"));

const PAGE_W = 11906, MARGIN_LR = 1247, MARGIN_TB = 1134;
const CONTENT_W = PAGE_W - 2 * MARGIN_LR;
const BODY_FONT = "Times New Roman", HEAD_FONT = "Arial";

function decode(t) {
  return t.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&").replace(/&quot;/g, '"');
}
// <b>...</b> işaretlerini kalın TextRun'lara çevirir
function runs(text, opts = {}) {
  const parts = text.split(/(<b>|<\/b>)/);
  let bold = false; const out = [];
  for (const part of parts) {
    if (part === "<b>") { bold = true; continue; }
    if (part === "</b>") { bold = false; continue; }
    if (!part) continue;
    out.push(new TextRun({ text: decode(part), bold: bold || opts.bold, italics: opts.italics, size: opts.size, font: opts.font }));
  }
  return out;
}
const border = { style: BorderStyle.SINGLE, size: 4, color: "9AA5B1" };
const borders = { top: border, bottom: border, left: border, right: border };

function makeTable(b) {
  let widths = b.widths.map(w => Math.round(w * 20));
  const total = widths.reduce((a, c) => a + c, 0);
  if (total > CONTENT_W) widths = widths.map(w => Math.floor(w * CONTENT_W / total));
  const tw = widths.reduce((a, c) => a + c, 0);
  return new Table({
    width: { size: tw, type: WidthType.DXA },
    columnWidths: widths,
    rows: b.rows.map((r, i) => new TableRow({
      tableHeader: i === 0,
      cantSplit: true,
      children: r.map((c, j) => new TableCell({
        borders,
        width: { size: widths[j], type: WidthType.DXA },
        shading: i === 0 ? { fill: "E8EDF3", type: ShadingType.CLEAR, color: "auto" } : undefined,
        margins: { top: 40, bottom: 40, left: 60, right: 60 },
        children: [new Paragraph({ keepNext: i < b.rows.length - 1, children: runs(c, { size: 16, font: HEAD_FONT, bold: i === 0 }) })],
      })),
    })),
  });
}
function img(path, wPt, hPt) {
  return new ImageRun({ type: "png", data: fs.readFileSync(path),
    transformation: { width: Math.round(wPt * 96 / 72), height: Math.round(hPt * 96 / 72) } });
}

const children = [];
let onCover = true;
for (const b of blocks) {
  switch (b.type) {
    case "title":
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
        children: runs(b.text, { bold: true, size: 38, font: HEAD_FONT }) })); break;
    case "sub":
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
        children: runs(b.text, { size: 24, font: BODY_FONT }) })); break;
    case "spacer":
      if (onCover && b.h > 20) children.push(new Paragraph({ spacing: { before: Math.round(b.h * 20) }, children: [] }));
      break;
    case "pagebreak":
      onCover = false;
      children.push(new Paragraph({ children: [new PageBreak()] })); break;
    case "h1":
      children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: runs(b.text) })); break;
    case "h2":
      children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: runs(b.text) })); break;
    case "body":
      children.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 300 },
        children: runs(b.text) })); break;
    case "ref":
      children.push(new Paragraph({ spacing: { after: 80 }, indent: { left: 280, hanging: 280 },
        children: runs(b.text, { size: 20 }) })); break;
    case "cap":
      children.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 60, after: 180 },
        keepNext: /^Tablo/.test(b.text), children: runs(b.text, { italics: true, size: 19 }) })); break;
    case "table":
      children.push(makeTable(b)); break;
    case "image":
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, children: [img(b.path, b.w, b.h)] })); break;
    case "images":
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true,
        children: b.items.flatMap((it, k) => k ? [new TextRun("   "), img(it.path, it.w, it.h)] : [img(it.path, it.w, it.h)]) }));
      break;
    default: throw new Error("bilinmeyen blok: " + b.type);
  }
}

const doc = new Document({
  creator: "Nurseda Akgül",
  title: "MLP ile Sınıflandırma — Bank Marketing",
  styles: {
    default: { document: { run: { font: BODY_FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: HEAD_FONT, size: 28, bold: true }, paragraph: { spacing: { before: 240, after: 120 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: HEAD_FONT, size: 23, bold: true }, paragraph: { spacing: { before: 160, after: 80 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  sections: [{
    properties: {
      titlePage: true,
      page: { pageNumbers: { start: 0 }, size: { width: PAGE_W, height: 16838 }, margin: { top: MARGIN_TB, bottom: MARGIN_TB, left: MARGIN_LR, right: MARGIN_LR } },
    },
    footers: {
      first: new Footer({ children: [new Paragraph({ children: [] })] }),
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], size: 17, font: HEAD_FONT })] })] }),
    },
    children,
  }],
});
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(outPath, buf); console.log("yazıldı:", outPath); });

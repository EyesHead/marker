"""Generate small, deterministic source PDFs; tests never need ReportLab.

All content is synthetic. Coordinates below use PDF points, with y measured
from the top of a 360 x 400 page. Run with the bundled ReportLab runtime.
"""

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject, RectangleObject
from reportlab import rl_config
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas


SIZE = (360, 400)


def canvas_at(path, title):
    result = Canvas(str(path) if isinstance(path, Path) else path,
                    pagesize=SIZE, invariant=1, pageCompression=1)
    result.setTitle(title)
    result.setAuthor("Marker offline tests")
    result.setFont("Helvetica", 10)
    return result


def text(canvas, x, top, value, font="Helvetica"):
    canvas.setFont(font, 10)
    canvas.drawString(x, SIZE[1] - top - 10, value)


def line(canvas, x0, y0, x1, y1):
    canvas.line(x0, SIZE[1] - y0, x1, SIZE[1] - y1)


def image(canvas, pixels, bbox):
    x0, y0, x1, y1 = bbox
    canvas.drawImage(ImageReader(pixels), x0, SIZE[1] - y1,
                     width=x1-x0, height=y1-y0, mask="auto")


def glyph(color="black"):
    result = Image.new("RGB", (96, 96), "white")
    draw = ImageDraw.Draw(result)
    draw.line([(20, 20), (70, 44), (20, 68)], fill=color, width=7)
    draw.line([(20, 80), (70, 80)], fill=color, width=7)
    return result


def ruled(canvas, top, bottom, caption, values, *, open_top=False, open_bottom=False):
    if caption:
        text(canvas, 30, top-20, caption)
    for x in (30, 130, 230, 330):
        line(canvas, x, top, x, bottom)
    step = (bottom-top)/len(values)
    for row in range(len(values)+1):
        if row == 0 and open_top or row == len(values) and open_bottom:
            continue
        line(canvas, 30, top + row*step, 330, top + row*step)
    for row, cells in enumerate(values):
        for col, value in enumerate(cells):
            text(canvas, 35 + col*100, top + row*step + 8, value)


def continuation(path):
    canvas = canvas_at(path, "Synthetic three-page interrupted header")
    text(canvas, 30, 256, "Table 7: stress classes")
    for x in (30, 130, 230, 330):
        line(canvas, x, 280, x, 380)
    line(canvas, 30, 280, 330, 280)
    line(canvas, 130, 320, 330, 320)
    text(canvas, 40, 290, "Stress")
    text(canvas, 140, 290, "Class A")
    text(canvas, 240, 290, "Class B")
    text(canvas, 140, 330, "Header begins")
    text(canvas, 240, 330, "Header begins")
    image(canvas, glyph(), (45, 350, 61, 366))
    canvas.showPage()
    text(canvas, 30, 7, "Table 7 (continued)")
    for x in (30, 130, 230, 330):
        line(canvas, x, 30, x, 380)
    for y in (150, 200, 260):
        line(canvas, 130, y, 330, y)
    text(canvas, 40, 40, "state continues")
    image(canvas, glyph("blue"), (45, 95, 61, 111))
    for col in (140, 240):
        text(canvas, col, 40, "Header middle")
        text(canvas, col, 160, "first value")
        text(canvas, col, 210, "second value")
        text(canvas, col, 275, "row begins")
    canvas.showPage()
    text(canvas, 30, 7, "Table 7 (continued)")
    for x in (30, 130, 230, 330):
        line(canvas, x, 30, x, 170)
    for y in (70, 110, 170):
        line(canvas, 30, y, 330, y)
    text(canvas, 40, 40, "ends")
    image(canvas, glyph("red"), (95, 42, 111, 58))
    for col in (140, 240):
        text(canvas, col, 40, "row ends")
        text(canvas, col, 80, "third value")
        text(canvas, col, 125, "fourth value")
    ruled(canvas, 250, 330, "Table 8: unrelated", [("Other", "X", "Y"), ("1", "2", "3")])
    canvas.showPage()
    canvas.save()


def native_figure(path):
    canvas = canvas_at(path, "Synthetic native raster figure")
    pixels = Image.new("RGB", (2048, 1024), "white")
    draw = ImageDraw.Draw(pixels)
    draw.rectangle((0, 0, 1023, 1023), fill=(10, 40, 90))
    draw.rectangle((1024, 0, 2047, 1023), fill=(180, 20, 35))
    draw.line((0, 512, 2047, 512), fill="yellow", width=1)
    image(canvas, pixels, (40, 50, 200, 130))
    text(canvas, 40, 150, "Figure 1: native pixels")
    line(canvas, 40, 220, 112, 220)
    line(canvas, 40, 256, 112, 256)
    canvas.showPage()
    canvas.save()


def split_formula(path):
    canvas = canvas_at(path, "Synthetic raster and text formula glyphs")
    text(canvas, 30, 40, "Limit:")
    text(canvas, 74, 40, "x")
    image(canvas, glyph(), (85, 40, 101, 56))
    text(canvas, 106, 40, "0")
    canvas.setFont("Helvetica", 10)
    canvas.drawString(30, SIZE[1]-72, "x")
    canvas.setFont("Helvetica", 6)
    canvas.drawString(36, SIZE[1]-68, "2")
    ruled(canvas, 100, 180, "Table 3: mixed content", [("Formula", "Value", "Note"), ("x", "2", "diagram")])
    image(canvas, glyph(), (54, 146, 70, 162))
    text(canvas, 76, 146, "0")
    diagram = Image.new("RGB", (240, 160), "white")
    draw = ImageDraw.Draw(diagram)
    draw.rectangle((10, 10, 230, 150), outline="blue", width=5)
    draw.line((20, 140, 120, 25, 220, 140), fill="red", width=5)
    image(canvas, diagram, (275, 145, 311, 169))
    canvas.showPage()
    canvas.save()


def scan(path):
    canvas = canvas_at(path, "Synthetic page scan without a text layer")
    pixels = Image.new("RGB", (1080, 1200), "white")
    draw = ImageDraw.Draw(pixels)
    draw.text((90, 90), "SCANNED PAGE ONLY", fill="black")
    for y in (180, 260, 340, 420):
        draw.line((90, y, 990, y), fill="black", width=3)
    for x in (90, 390, 690, 990):
        draw.line((x, 180, x, 420), fill="black", width=3)
    draw.text((100, 200), "alpha   12   24", fill="black")
    image(canvas, pixels, (0, 0, 360, 400))
    canvas.showPage()
    canvas.save()


def damaged_text(path):
    # Visible glyphs stay readable; only Courier's Unicode map is damaged.
    stream = BytesIO()
    canvas = canvas_at(stream, "Synthetic damaged text with a good neighbouring region")
    text(canvas, 30, 40, "Trustworthy selectable paragraph.")
    text(canvas, 30, 160, "Readable on paper; broken text mapping.", "Courier")
    canvas.showPage()
    canvas.save()
    reader = PdfReader(stream)
    mapping = ("/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
               "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
               "/CMapName /Damaged def\n/CMapType 2 def\n1 begincodespacerange\n<00> <FF>\n"
               "endcodespacerange\n95 beginbfchar\n")
    mapping += "\n".join(f"<{code:02X}> <FFFD>" for code in range(32, 127))
    mapping += "\nendbfchar\nendcmap\nCMapName currentdict /CMap defineresource pop\nend end"
    writer = PdfWriter(clone_from=reader)
    for font in writer.pages[0]["/Resources"]["/Font"].values():
        if font.get_object()["/BaseFont"] == "/Courier":
            cmap = DecodedStreamObject()
            cmap.set_data(mapping.encode("ascii"))
            font.get_object()[NameObject("/ToUnicode")] = writer._add_object(cmap)
    # Avoid a random writer ID; preserve the invariant ReportLab source ID.
    writer._ID = reader.trailer["/ID"]
    writer.write(path)


def unrelated(path):
    canvas = canvas_at(path, "Synthetic unrelated neighbouring tables")
    ruled(canvas, 280, 380, "Table 10: pressures", [("Name", "Min", "Max"), ("A", "1", "2")])
    canvas.showPage()
    ruled(canvas, 30, 130, "Table 11: temperatures", [("Name", "Min", "Max"), ("B", "3", "4")])
    ruled(canvas, 240, 340, "Table 12: distances", [("Name", "Min", "Max"), ("C", "5", "6")])
    canvas.showPage()
    canvas.save()


def borderless(path):
    canvas = canvas_at(path, "Synthetic multiple borderless tables")
    for top, label in ((40, "Table 21: speeds"), (230, "Table 22: loads")):
        text(canvas, 30, top-20, label)
        for row, cells in enumerate((("Item", "Low", "High"), ("A", "10", "20"),
                                     ("B", "30", "40"), ("C", "50", "60"))):
            for x, value in zip((30, 150, 270), cells):
                text(canvas, x, top + row*22, value)
    canvas.showPage()
    canvas.save()


def ambiguous(path):
    canvas = canvas_at(path, "Synthetic visually ambiguous closed continuation")
    ruled(canvas, 280, 380, "", [("Type", "Min", "Max"), ("A", "1", "2")])
    canvas.showPage()
    ruled(canvas, 30, 130, "", [("Type", "Min", "Max"), ("B", "3", "4")])
    canvas.showPage()
    canvas.save()


def rotated_crop(path):
    stream = BytesIO()
    native_figure(stream)
    reader = PdfReader(stream)
    writer = PdfWriter(clone_from=reader)
    writer.pages[0].cropbox = RectangleObject((20, 100, 320, 380))
    writer.pages[0].rotate(90)
    for rotation in (180, 270):
        page = writer.add_page(reader.pages[0])
        page.cropbox = RectangleObject((20, 100, 320, 380))
        page.rotate(rotation)
    writer._ID = reader.trailer["/ID"]
    writer.write(path)


def refined_header(path):
    canvas = canvas_at(path, "Synthetic header with finer columns on the next page")
    text(canvas, 30, 256, "Table 30: grouped columns")
    for x in (30, 130, 330):
        line(canvas, x, 280, x, 380)
    line(canvas, 30, 280, 330, 280)
    line(canvas, 130, 320, 330, 320)
    text(canvas, 40, 290, "Item")
    text(canvas, 140, 290, "Grouped values")
    text(canvas, 140, 330, "Header begins")
    canvas.showPage()
    text(canvas, 30, 7, "Table 30 (continued)")
    for x in (30, 130, 330):
        line(canvas, x, 30, x, 170)
    line(canvas, 230, 70, 230, 170)
    for y in (70, 110, 170):
        line(canvas, 30, y, 330, y)
    text(canvas, 40, 40, "Name ends")
    text(canvas, 140, 40, "Header ends")
    for y in (80, 125):
        for x, value in ((40, "A"), (140, "1"), (240, "2")):
            text(canvas, x, y, value)
    canvas.showPage()
    canvas.save()


def conflicting_captions(path):
    canvas = canvas_at(path, "Synthetic reused table numbers with different captions")
    ruled(canvas, 280, 380, "Table 1: pressures", [("Name", "Min", "Max"), ("A", "1", "2")])
    canvas.showPage()
    ruled(canvas, 30, 130, "Table 1: temperatures", [("Name", "Min", "Max"), ("B", "3", "4")])
    canvas.showPage()
    canvas.save()


def image_layers(path):
    canvas = canvas_at(path, "Synthetic overlapping native rasters and source overlays")
    background = Image.new("RGB", (2048, 1024), (10, 40, 90))
    overlay = Image.new("RGBA", (512, 256), (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle((0, 0, 255, 255), fill=(255, 0, 0, 255))
    image(canvas, background, (40, 50, 200, 130))
    image(canvas, overlay, (40, 50, 200, 130))
    text(canvas, 45, 90, "OVERLAY")
    canvas.showPage()
    image(canvas, background, (40, 50, 200, 130))
    line(canvas, 30, 90, 210, 90)
    canvas.showPage()
    canvas.save()


def open_neighbours(path):
    canvas = canvas_at(path, "Synthetic independent adjacent open grids")
    for columns in ((20, 60, 100), (150, 190, 230)):
        for x in columns:
            line(canvas, x, 80, x, 180)
        for y in (120, 160):
            line(canvas, columns[0], y, columns[-1], y)
        for x in columns[:-1]:
            text(canvas, x+4, 90, "A")
            text(canvas, x+4, 130, "1")
    text(canvas, 110, 130, "gap")
    canvas.showPage()
    canvas.save()


def form_instances(path):
    canvas = canvas_at(path, "Synthetic nested and reused native-image forms")
    pixels = Image.new("RGB", (800, 400), (10, 40, 90))
    ImageDraw.Draw(pixels).rectangle((400, 0, 799, 399), fill=(180, 20, 35))
    canvas.beginForm("inner", 0, 0, 80, 40)
    canvas.drawImage(ImageReader(pixels), 0, 0, width=80, height=40)
    canvas.endForm()
    canvas.beginForm("outer", 0, 0, 100, 80)
    canvas.saveState()
    canvas.translate(20, 10)
    canvas.scale(0.5, 0.5)
    canvas.doForm("inner")
    canvas.restoreState()
    canvas.endForm()
    for x, y, sx, sy, rotation in ((120, 200, 1, 1, 0), (220, 80, 2, 1, 0), (80, 260, 1, 1, 90)):
        canvas.saveState()
        canvas.translate(x, y)
        canvas.scale(sx, sy)
        canvas.rotate(rotation)
        canvas.doForm("outer")
        canvas.restoreState()
    text(canvas, 145, 175, "A")
    canvas.showPage()
    canvas.saveState()
    canvas.translate(180, 200)
    canvas.rotate(30)
    canvas.drawImage(ImageReader(pixels), 0, 0, width=80, height=40)
    canvas.restoreState()
    canvas.showPage()
    canvas.save()
    reader = PdfReader(BytesIO(Path(path).read_bytes()))
    writer = PdfWriter(clone_from=reader)
    writer.add_page(reader.pages[0]).rotate(90)
    writer._ID = reader.trailer["/ID"]
    writer.write(path)


def crossing_overlay(path):
    canvas = canvas_at(path, "Synthetic text crossing a native raster boundary")
    pixels = Image.new("RGB", (800, 400), (10, 40, 90))
    image(canvas, pixels, (40, 50, 200, 130))
    text(canvas, 190, 90, "OVERLAY")
    canvas.showPage()
    canvas.save()


def generate(directory):
    rl_config.useA85 = False
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    generators = {"three_page_continuation": continuation, "native_figure": native_figure,
                  "split_formula": split_formula, "scan": scan, "damaged_text": damaged_text,
                  "unrelated_tables": unrelated, "borderless_tables": borderless,
                  "ambiguous_tables": ambiguous, "rotated_crop": rotated_crop,
                  "refined_header": refined_header, "conflicting_captions": conflicting_captions,
                  "image_layers": image_layers, "open_neighbours": open_neighbours,
                  "form_instances": form_instances, "crossing_overlay": crossing_overlay}
    for name, generator in generators.items():
        generator(directory / f"{name}.pdf")


if __name__ == "__main__":
    generate(Path(__file__).parent)

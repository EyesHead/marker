"""Deterministic seven-column physical cells with three matching body baselines."""

from pathlib import Path
from io import BytesIO
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, DictionaryObject, NameObject, NumberObject
from PIL import Image, ImageDraw
from reportlab.lib.utils import ImageReader
from reportlab import rl_config
from reportlab.pdfgen.canvas import Canvas


def generate(path):
    rl_config.useA85 = False
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=(560, 250), invariant=1, pageCompression=1)
    canvas.setTitle("Synthetic coefficient line correspondence")
    canvas.setAuthor("Marker offline tests")
    cuts = [20 + 74 * column for column in range(8)]
    for x in cuts:
        canvas.line(x, 60, x, 190)
    for y in (60, 160, 190):
        canvas.line(cuts[0], y, cuts[-1], y)
    canvas.setFont("Helvetica", 10)
    for column, x in enumerate(cuts[:-1]):
        if column == 6:
            pixels = Image.new("RGB", (240, 60), "white")
            ImageDraw.Draw(pixels).text((8, 8), "Value 6", fill="black", font_size=36)
            canvas.drawImage(ImageReader(pixels), x + 8, 170, width=55, height=14)
        else:
            canvas.drawString(
                x + 8, 174, "Coefficient" if column == 0 else f"Value {column}"
            )
    for index, y in enumerate((135, 110, 85)):
        canvas.setFont("Symbol", 12)
        canvas.drawString(cuts[0] + 8, y, "α" if index < 2 else "β")
        if index < 2:
            canvas.setFont("Helvetica", 7)
            canvas.drawString(cuts[0] + 17, y - 3, str(index + 1))
        canvas.setFont("Helvetica", 12)
        for column, x in enumerate(cuts[1:-1], start=1):
            canvas.drawString(x + 8, y, f"{index+1},{column}")
    canvas.showPage()
    canvas.save()
    # Explicit Symbol advance widths make the source usable by extractors that
    # otherwise assign zero width to this standard PDF font's a/b glyphs.
    reader = PdfReader(stream)
    writer = PdfWriter(clone_from=reader)
    for font in writer.pages[0]["/Resources"]["/Font"].values():
        item = font.get_object()
        if item["/BaseFont"] == "/Symbol":
            item[NameObject("/Encoding")] = DictionaryObject(
                {
                    NameObject("/Differences"): ArrayObject(
                        [NumberObject(97), NameObject("/alpha"), NameObject("/beta")]
                    )
                }
            )
            item[NameObject("/FirstChar")] = NumberObject(97)
            item[NameObject("/LastChar")] = NumberObject(98)
            item[NameObject("/Widths")] = ArrayObject(
                [NumberObject(631), NumberObject(549)]
            )
    writer._ID = reader.trailer["/ID"]
    writer.write(path)


if __name__ == "__main__":
    generate(Path(__file__).with_suffix(".pdf"))

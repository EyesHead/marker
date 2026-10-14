"""Generate a deterministic, synthetic PDF with 125 rows and two cells per row.

Run with the bundled Python runtime that provides reportlab.  The regression
extracts the text and physical cell boxes from the committed PDF.
"""

from pathlib import Path

from reportlab import rl_config
from reportlab.pdfgen.canvas import Canvas


def generate(destination: Path):
    # Use binary PDF streams so Git treats the fixture as a binary resource.
    rl_config.useA85 = False
    width, height = 600, 2350
    left, right, top, row_height = 50, 550, 50, 18
    canvas = Canvas(str(destination), pagesize=(width, height), invariant=1)
    canvas.setTitle("Synthetic 125-row table regression")
    canvas.setAuthor("Marker offline tests")
    canvas.setFont("Helvetica", 10)
    for x in (left, 300, right):
        canvas.line(x, height - top, x, height - top - 125 * row_height)
    for row in range(126):
        y = height - top - row * row_height
        canvas.line(left, y, right, y)
    for row in range(125):
        baseline = height - top - row * row_height - 12
        canvas.drawString(left + 5, baseline, str(row))
        canvas.drawString(305, baseline, f"value-{row:03d}")
    canvas.showPage()
    canvas.save()


if __name__ == "__main__":
    generate(Path(__file__).with_suffix(".pdf"))

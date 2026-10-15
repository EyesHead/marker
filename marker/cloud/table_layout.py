"""Place complete validated table paragraphs on trusted native source baselines.

This is a final rendering step. Canonical HTML remains in the stage cache;
source geometry supplies every CSS number below. No PDF or model I/O occurs.
"""

from collections import Counter
from math import isfinite
import re
import unicodedata

from bs4 import BeautifulSoup, NavigableString, Tag

from marker.cloud.content import bounds, expression_groups, source_lines
from marker.cloud.regions import assess_text_quality
from marker.processors.table_geometry import grid_positions


class TableLayoutError(ValueError):
    """A safe repair reason when full paragraphs cannot preserve source rows."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def _fail(identity=None):
    cell = f" {identity}" if identity is not None else ""
    raise TableLayoutError(
        f"Trusted source table cell{cell} must preserve its native baseline order "
        "with one complete p or br slot per source line"
    )


def _visible_text(node):
    if isinstance(node, NavigableString):
        return str(node)
    if node.name == "annotation":
        return ""
    if node.name == "mfrac":
        parts = [child for child in node.contents if isinstance(child, Tag)]
        if len(parts) == 2:
            return _visible_text(parts[0]) + "/" + _visible_text(parts[1])
    text = "".join(_visible_text(child) for child in node.contents)
    return " " + text + " " if node.name == "math" else text


def _tokens(text):
    text = unicodedata.normalize("NFC", text).replace("−", "-").replace("\u00ad", "")
    return re.findall(r"[+-]?\d+(?:[.,]\d+)*|[^\W\d_]+", text, re.UNICODE)


def _protected_text(line):
    mathematical = {id(atom) for group in expression_groups(line) for atom in group}
    words = [
        atom
        for atom in line
        if atom.get("kind", "text") == "text" and id(atom) not in mathematical
    ]
    if not words or not assess_text_quality(words, bounds(words)).trusted:
        return Counter()
    return Counter(_tokens(" ".join(atom.get("text", "") for atom in words)))


def _one_vertical_math_object(blocks, lines):
    if len(blocks) != 1 or len(lines) <= 1:
        return False
    block = blocks[0]
    formulas = block.find_all("math")
    if len(formulas) != 1 or block.find("img"):
        return False
    if any(
        isinstance(child, NavigableString)
        and child.strip()
        and not any(parent.name == "math" for parent in child.parents)
        for child in block.descendants
    ):
        return False
    if not formulas[0].find(
        [
            "mfrac",
            "mtable",
            "mroot",
            "msub",
            "msup",
            "msubsup",
            "munder",
            "mover",
            "munderover",
            "mmultiscripts",
        ]
    ):
        return False
    atoms = [atom for line in lines for atom in line]
    if len({atom.get("source_page", 0) for atom in atoms}) != 1:
        return False
    return all(expression_groups(line) and not _protected_text(line) for line in lines)


def _check_positions(html, ordered):
    # Use the same bounded HTML occupancy calculation as canonical validation.
    # The import is lazy because whole_tables also calls this pure renderer.
    from marker.cloud.client import ResponseValidationError
    from marker.cloud.whole_tables import _TableParser, _validate_structure

    parser = _TableParser()
    try:
        parser.feed(html)
        parser.close()
    except (ResponseValidationError, ValueError, TypeError, RecursionError):
        _fail()
    tables = [child for child in parser.root.children if not isinstance(child, str)]
    if (
        len(parser.stack) != 1
        or len(tables) != 1
        or tables[0].tag != "table"
        or any(
            isinstance(child, str) and child.strip() for child in parser.root.children
        )
    ):
        _fail()
    try:
        _, positions = _validate_structure(tables[0])
    except (ResponseValidationError, ValueError, TypeError, RecursionError):
        _fail()
    expected = {f"c{index}": position[:4] for index, position in enumerate(ordered)}
    for cell, row, column, rowspan, colspan in positions:
        identity = cell.attrs.get("data-cell-id")
        if identity not in expected:
            _fail()
        if (row, column, rowspan, colspan) != expected[identity]:
            _fail(identity)


def _blocks(node, soup, identity):
    """Retain whole p contents; an existing br already supplies a line slot."""
    paragraphs = node.find_all("p", recursive=False)
    if paragraphs:
        if any(
            (isinstance(child, Tag) and child.name != "p")
            or (isinstance(child, NavigableString) and child.strip())
            for child in node.contents
        ):
            _fail(identity)
    else:
        paragraph = soup.new_tag("p")
        for child in list(node.contents):
            paragraph.append(child.extract())
        paragraphs = [paragraph]
    blocks = []
    for paragraph in paragraphs:
        if any(br.parent is not paragraph for br in paragraph.find_all("br")):
            _fail(identity)
        current = soup.new_tag("p")
        for child in list(paragraph.contents):
            if isinstance(child, Tag) and child.name == "br":
                if _visible_text(current).strip() or current.find(["math", "img"]):
                    blocks.append(current)
                current = soup.new_tag("p")
            else:
                current.append(child.extract())
        if _visible_text(current).strip() or current.find(["math", "img"]):
            blocks.append(current)
    return blocks


def _font(line):
    words = [atom for atom in line if atom.get("kind", "text") == "text"]
    sizes = [
        float(atom.get("size", atom["bbox"][3] - atom["bbox"][1])) for atom in words
    ]
    default = max(
        6, min(12, bounds(line, layout=True)[3] - bounds(line, layout=True)[1])
    )
    size = max(sizes, default=default)
    if not isfinite(size) or size <= 0 or size > 144:
        _fail()
    names = " ".join(atom.get("fontname", "").lower() for atom in words)
    family = (
        "monospace"
        if "courier" in names or "mono" in names
        else ("serif" if "times" in names or "roman" in names else "Arial,sans-serif")
    )
    return size, family


def apply_source_table_layout(html, grid):
    """Render validated canonical HTML on a trusted grid's physical canvas.

    ``grid=None`` or an inferred grid keeps normal HTML flow. A trusted source
    requires an exact, unambiguous mapping of whole p/br slots to source lines;
    otherwise TableLayoutError provides a bounded repair reason. Call this
    before accepting canonical HTML and again only when composing final output,
    never cache the returned presentation as model input.
    """
    if not grid or not grid.get("trusted", True):
        return html
    geometry = grid_positions(grid)
    if geometry is None:
        _fail()
    xs, ys, positions = geometry
    if not all(isfinite(value) for value in xs + ys):
        _fail()
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    if len(tables) != 1:
        _fail()
    table = tables[0]
    # Only code-generated native figure dimensions exist in canonical HTML.
    # Never turn arbitrary model CSS into a trusted final layout.
    for node in table.find_all(True) + [table]:
        if "style" in node.attrs and not (
            node.name == "img"
            and re.fullmatch(
                r"width:[0-9]+(?:\.[0-9]+)?pt;height:[0-9]+(?:\.[0-9]+)?pt",
                node["style"],
            )
        ):
            raise TableLayoutError(
                "Source table layout accepts canonical HTML without model CSS"
            )
    ordered = sorted(positions, key=lambda position: position[:2])
    cells = table.find_all(["td", "th"])
    identities = [cell.get("data-cell-id") for cell in cells]
    expected_ids = {f"c{index}" for index in range(len(ordered))}
    if len(identities) != len(expected_ids) or set(identities) != expected_ids:
        _fail()
    _check_positions(html, ordered)
    rendered = []
    for index, (_, column, _, colspan, original) in enumerate(ordered):
        identity = f"c{index}"
        node = next(cell for cell in cells if cell["data-cell-id"] == identity)
        lines = source_lines(original.get("atoms", []))
        blocks = _blocks(node, soup, identity)
        if _one_vertical_math_object(blocks, lines):
            lines = [[atom for line in lines for atom in line]]
        if len(blocks) != len(lines):
            _fail(identity)
        cell_box = original.get("layout_bbox", original["bbox"])
        width = xs[column + colspan] - xs[column]
        height = cell_box[3] - cell_box[1]
        if not all(isfinite(value) for value in cell_box) or width <= 0 or height <= 0:
            _fail(identity)
        placements = []
        for block, line in zip(blocks, lines):
            if not block.find("img") and _protected_text(line) - Counter(
                _tokens(_visible_text(block))
            ):
                _fail(identity)
            box = bounds(line, layout=True)
            left, top = box[0] - cell_box[0], box[1] - cell_box[1]
            if (
                not all(isfinite(value) for value in box)
                or left < -2
                or top < -2
                or top >= height
            ):
                _fail(identity)
            size, family = _font(line)
            placements.append((block, max(0, left), max(0, top), size, family, line))
        rendered.append((node, width, height, placements))
    rows = table.find_all("tr")
    if len(rows) != len(ys) - 1:
        _fail()
    width = xs[-1] - xs[0]
    table["style"] = (
        f"table-layout:fixed;border-collapse:collapse;width:{width:.3f}pt;"
        f"min-width:{width:.3f}pt;max-width:none"
    )
    for column_group in table.find_all("colgroup", recursive=False):
        column_group.decompose()
    group = soup.new_tag("colgroup")
    for left, right in zip(xs, xs[1:]):
        column = soup.new_tag("col")
        column["width"] = f"{(right - left) / width * 100:.3f}%"
        group.append(column)
    caption = table.find("caption", recursive=False)
    if caption is None:
        table.insert(0, group)
    else:
        caption.insert_after(group)
    for row, top, bottom in zip(rows, ys, ys[1:]):
        row["style"] = f"height:{bottom - top:.3f}pt"
    for node, width, height, placements in rendered:
        node.clear()
        node["style"] = "padding:0;vertical-align:top"
        canvas = soup.new_tag("div")
        canvas["data-source-layout"] = "cell"
        canvas["style"] = (
            f"position:relative;width:{width:.3f}pt;height:{height:.3f}pt;"
            "overflow:visible"
        )
        for block, left, top, size, family, line in placements:
            block["style"] = (
                f"position:absolute;left:{left:.3f}pt;top:{top:.3f}pt;"
                f"margin:0;font-size:{size:.3f}pt;font-family:{family};"
                "line-height:1;white-space:nowrap"
            )
            block["data-source-page"] = str(line[0].get("source_page", grid["page"]))
            block["data-source-y"] = f"{bounds(line)[1]:.3f}"
            canvas.append(block)
        node.append(canvas)
    return str(table)

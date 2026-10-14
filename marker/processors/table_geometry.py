"""Build HTML from PDF ruling lines and position source content inside cells.

PDF coordinates use points, x increasing rightwards and y downwards. Geometry
is kept internally for stitching; no table JSON is exported. Raster formulas
are preserved as lossless resources rather than silently losing their glyphs.
"""
from copy import deepcopy
from html import escape

from pdfplumber.table import edges_to_intersections, intersections_to_cells, merge_edges

TOLERANCE = 1.5


def _near(a, b):
    return abs(a - b) <= TOLERANCE


def _intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _bbox(obj):
    return tuple(float(obj[k]) for k in ('x0', 'top', 'x1', 'bottom'))


def _layout_bbox(item):
    """Physical bboxes stay fixed; only this separate box may be shifted."""
    return item.get('layout_bbox', item['bbox'])


def _line(x0, y0, x1, y1, orientation):
    return {'x0': x0, 'top': y0, 'x1': x1, 'bottom': y1,
            'width': x1-x0, 'height': y1-y0, 'orientation': orientation}


def _components(edges):
    """Connected ruling-line components, including open page-break borders."""
    parents = list(range(len(edges)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for i, a in enumerate(edges):
        for j in range(i):
            b = edges[j]
            if a['orientation'] == b['orientation']:
                continue
            h, v = (a, b) if a['orientation'] == 'h' else (b, a)
            if (h['x0']-TOLERANCE <= v['x0'] <= h['x1']+TOLERANCE
                    and v['top']-TOLERANCE <= h['top'] <= v['bottom']+TOLERANCE):
                parents[root(i)] = root(j)
    groups = {}
    for i, edge in enumerate(edges):
        groups.setdefault(root(i), []).append(edge)
    return list(groups.values())


def _horizontal_border(edges, y, x0, x1):
    intervals = sorted((max(x0, e['x0']), min(x1, e['x1'])) for e in edges
                       if e['orientation'] == 'h' and _near(e['top'], y) and e['x1'] > x0 and e['x0'] < x1)
    end = x0
    for start, stop in intervals:
        if start > end + TOLERANCE:
            return False
        end = max(end, stop)
    return end >= x1-TOLERANCE


def _cuts(cells, first, last):
    values = sorted({float(_layout_bbox(c)[i]) for c in cells for i in (first, last)})
    result = []
    for value in values:
        if not result or not _near(value, result[-1]):
            result.append(value)
    return result


def grid_positions(grid):
    cells = grid['cells']
    xs, ys = _cuts(cells, 0, 2), _cuts(cells, 1, 3)
    occupied = set()
    positions = []
    for cell in cells:
        x0, y0, x1, y1 = _layout_bbox(cell)
        c0, c1 = (min(range(len(xs)), key=lambda i: abs(xs[i]-x)) for x in (x0, x1))
        r0, r1 = (min(range(len(ys)), key=lambda i: abs(ys[i]-y)) for y in (y0, y1))
        slots = {(r, c) for r in range(r0, r1) for c in range(c0, c1)}
        if not slots or occupied.intersection(slots):
            return None
        occupied.update(slots)
        positions.append((r0, c0, r1-r0, c1-c0, cell))
    if len(occupied) != (len(xs)-1)*(len(ys)-1):
        return None
    return xs, ys, positions


def _open_outer_edges(group, edges, content):
    """Recover a disconnected side of a cell interrupted across a page break.

    An isolated vertical edge must span the same height as the component,
    stay within one existing column width, and enclose actual source content.
    Closed independent components are never combined this way.
    """
    verticals = [edge for edge in group if edge['orientation'] == 'v']
    horizontals = [edge for edge in group if edge['orientation'] == 'h']
    if len(verticals) < 2 or len(horizontals) < 2:
        return group
    xs = sorted({edge['x0'] for edge in verticals})
    if len(xs) < 2:
        return group
    y0, y1 = min(edge['top'] for edge in group), max(edge['bottom'] for edge in group)
    if (_horizontal_border(group, y0, xs[0], xs[-1]) and
            _horizontal_border(group, y1, xs[0], xs[-1])):
        return group
    pitch = max(b-a for a, b in zip(xs, xs[1:]))
    result = list(group)
    for edge in edges:
        if edge in group or edge['orientation'] != 'v':
            continue
        x = edge['x0']
        distance = min(abs(x-xs[0]), abs(x-xs[-1]))
        if xs[0] <= x <= xs[-1] or distance > pitch*1.25:
            continue
        if not (_near(edge['top'], y0) and _near(edge['bottom'], y1)):
            continue
        if any(other['orientation'] == 'h' and
               other['x0']-TOLERANCE <= x <= other['x1']+TOLERANCE and
               edge['top']-TOLERANCE <= other['top'] <= edge['bottom']+TOLERANCE for other in edges):
            continue  # A side connected to another grid is never an orphan.
        strip = (min(x, xs[0]) if x < xs[0] else xs[-1], y0,
                 xs[0] if x < xs[0] else max(x, xs[-1]), y1)
        if any(_intersection(_bbox(item), strip) > 0 for item in content):
            result.append(edge)
    return result


def extract_grids(pdf_page):
    """Find complete grids, closing only missing outer page-break borders.

    Missing internal lines stay missing: their cells span the corresponding
    row/column cuts. This preserves unruled numeric bands without guessing new
    visible lines from text spacing.
    """
    edges = [e for e in pdf_page.edges if (e['width'] if e['orientation'] == 'h' else e['height']) >= 5]
    if not edges:
        return []
    edges = merge_edges(edges, TOLERANCE, TOLERANCE, 3, 3)
    words = pdf_page.extract_words(extra_attrs=['fontname', 'size'], x_tolerance=1.5, y_tolerance=2)
    result = []
    for component in _components(edges):
        group = _open_outer_edges(component, edges, [*words, *pdf_page.images])
        h = [e for e in group if e['orientation'] == 'h']
        v = [e for e in group if e['orientation'] == 'v']
        if len(h) < 2 or len(v) < 3:
            continue
        bounds = [min(e['x0'] for e in group), min(e['top'] for e in group),
                  max(e['x1'] for e in group), max(e['bottom'] for e in group)]
        # Thin PDF rectangles have two sides. Their vertical endpoints can be
        # half a stroke away from the snapped horizontal ruling; use the same
        # canonical boundary instead of creating a spurious sub-point row.
        for index in (1, 3):
            nearby = [e['top'] for e in h if _near(e['top'], bounds[index])]
            if nearby:
                bounds[index] = min(nearby, key=lambda y: abs(y-bounds[index]))
        x0, y0, x1, y1 = bounds
        if x1-x0 < 40 or y1-y0 < 10:
            continue
        working = list(group)
        if not _horizontal_border(group, y0, x0, x1):
            working.append(_line(x0, y0, x1, y0, 'h'))
        if not _horizontal_border(group, y1, x0, x1):
            working.append(_line(x0, y1, x1, y1, 'h'))
        working = merge_edges(working, TOLERANCE, TOLERANCE, 3, 3)
        boxes = intersections_to_cells(edges_to_intersections(working, TOLERANCE, TOLERANCE))
        if len(boxes) < 2:
            continue
        page = pdf_page.page_number-1
        cells = [{'bbox': tuple(box), 'source_bbox': tuple(box), 'source_page': page,
                  'source_boxes': ((page, tuple(box)),), 'atoms': [],
                  'open_top': _near(box[1], y0) and not _horizontal_border(group, box[1], box[0], box[2]),
                  'open_bottom': _near(box[3], y1) and not _horizontal_border(group, box[3], box[0], box[2])}
                 for box in boxes]
        grid = {'bbox': tuple(bounds), 'source_bbox': tuple(bounds), 'cells': cells,
                'page': page, 'last_page': page, 'trusted': True, 'method': 'lines'}
        if not grid_positions(grid):
            continue
        atoms = [{'kind': 'text', 'bbox': _bbox(w), 'text': w['text'], 'size': float(w['size']),
                  'bold': 'bold' in w['fontname'].lower(), 'italic': any(s in w['fontname'].lower() for s in ('italic', 'oblique'))}
                 for w in words if _intersection(_bbox(w), bounds) > 0]
        atoms.extend({'kind': 'image', 'bbox': _bbox(image), 'image_index': i}
                     for i, image in enumerate(pdf_page.images) if _intersection(_bbox(image), bounds) > 0)
        # Each source atom belongs to exactly one physical cell, by maximum
        # overlap (small glyph overhangs must not duplicate into adjacent cells).
        for atom in atoms:
            cell = max(cells, key=lambda c: _intersection(c['bbox'], atom['bbox']))
            overlap = _intersection(cell['bbox'], atom['bbox'])
            area = (atom['bbox'][2]-atom['bbox'][0])*(atom['bbox'][3]-atom['bbox'][1])
            if overlap < area*0.8:
                break
            atom['source_page'] = grid['page']
            atom['source_bbox'] = atom['bbox']
            cell['atoms'].append(atom)
        else:
            if any(c['atoms'] for c in cells):
                result.append(grid)
    return sorted(result, key=lambda g: (g['bbox'][1], g['bbox'][0]))


def attach_images(grid, block, page):
    image = None
    for cell in grid['cells']:
        for atom in cell['atoms']:
            if atom['kind'] != 'image':
                continue
            if image is None:
                image = page.get_image(highres=True)
            box = atom['bbox']
            sx, sy = image.width/page.polygon.width, image.height/page.polygon.height
            crop = image.crop((round(box[0]*sx), round(box[1]*sy), round(box[2]*sx), round(box[3]*sy)))
            name = f'{block.id.to_path()}_source_{atom["image_index"]}.png'
            atom['src'] = name
            block.embedded_images[name] = crop


def render_grid(grid):
    geometry = grid_positions(grid)
    if geometry is None:
        raise ValueError('Invalid or overlapping physical table grid')
    xs, ys, positions = geometry
    width = xs[-1]-xs[0]
    html = [f'<table data-pdf-geometry="true" style="border-collapse:collapse;table-layout:fixed;width:{width:.3f}pt;max-width:none"><colgroup>']
    html.extend(f'<col style="width:{(b-a)/width*100:.4f}%">' for a, b in zip(xs, xs[1:]))
    html.append('</colgroup><tbody>')
    for row in range(len(ys)-1):
        html.append(f'<tr style="height:{ys[row+1]-ys[row]:.3f}pt">')
        for r, col, rowspan, colspan, cell in sorted(positions, key=lambda p: (p[0], p[1])):
            if r != row:
                continue
            x0, y0, x1, y1 = _layout_bbox(cell)
            attrs = (f' rowspan="{rowspan}"' if rowspan > 1 else '') + (f' colspan="{colspan}"' if colspan > 1 else '')
            html.append(f'<td{attrs} style="padding:0;border:1px solid #888;vertical-align:top"><div style="position:relative;height:{y1-y0:.3f}pt;overflow:visible">')
            # Preserve absolute baselines inside large, unruled cells. The alpha
            # glyph and all six values therefore remain on the same source line.
            for atom in sorted(cell['atoms'], key=lambda a: (_layout_bbox(a)[1], _layout_bbox(a)[0])):
                ax0, ay0, ax1, ay1 = _layout_bbox(atom)
                placement = f'position:absolute;left:{(ax0-x0)/(x1-x0)*100:.4f}%;top:{ay0-y0:.3f}pt;'
                if atom['kind'] == 'image':
                    if not atom.get('src'):
                        continue
                    name = escape(atom.get('src', ''), quote=True)
                    latex = escape(atom.get('latex', ''), quote=True)
                    html.append(f'<img src="{name}" alt="{latex}" data-source-page="{atom["source_page"]}" style="{placement}width:{ax1-ax0:.3f}pt;height:{ay1-ay0:.3f}pt;max-width:none">')
                else:
                    style = placement + f'font-size:{atom["size"]:.3f}pt;line-height:1;white-space:pre;'
                    if atom['bold']:
                        style += 'font-weight:bold;'
                    if atom['italic']:
                        style += 'font-style:italic;'
                    text = ''.join(ch for ch in atom['text'] if ch >= ' ' or ch == '\n')
                    html.append(f'<span style="{style}">{escape(text)}</span> ')
            html.append('</div></td>')
        html.append('</tr>')
    html.append('</tbody></table>')
    return ''.join(html)


def can_join_grids(first, second):
    if first.get('last_page', first['page']) + 1 != second['page']:
        return False
    if not all(_near(first['bbox'][i], second['bbox'][i]) for i in (0, 2)):
        return False
    outgoing = [c for c in first['cells'] if c['open_bottom']]
    incoming = [c for c in second['cells'] if c['open_top']]
    return bool(outgoing and incoming) and all(any(_near(a['bbox'][0], b['bbox'][0]) and _near(a['bbox'][2], b['bbox'][2])
                                                 for b in incoming) for a in outgoing)


def join_grids(first, second):
    if not can_join_grids(first, second):
        return None
    return stack_grids(first, second, merge_open=True)


def stack_grids(first, second, *, merge_open=False):
    """Build a virtual layout, retaining all original boxes and source parts.

    A joined cell has one physical ``bbox`` for its first fragment and a
    ``source_boxes`` tuple for every fragment. Its full display extent lives
    in ``layout_bbox``. Atoms always crop through source_page/source_bbox.
    """
    if first.get('last_page', first['page']) + 1 != second['page']:
        return None
    if not all(_near(first['bbox'][i], second['bbox'][i]) for i in (0, 2)):
        return None
    merged = deepcopy(first)
    following = deepcopy(second)
    first_box, second_box = _layout_bbox(first), _layout_bbox(second)
    shift = first_box[3]-second_box[1]
    for grid in (merged, following):
        for cell in grid['cells']:
            cell.setdefault('source_page', grid['page'])
            cell.setdefault('source_bbox', tuple(cell['bbox']))
            cell.setdefault('source_boxes', ((cell['source_page'], cell['source_bbox']),))
            for atom in cell['atoms']:
                atom.setdefault('source_page', grid['page'])
                atom.setdefault('source_bbox', tuple(atom['bbox']))
    for cell in following['cells']:
        for item in [cell, *cell['atoms']]:
            box = _layout_bbox(item)
            item['layout_bbox'] = (box[0], box[1]+shift, box[2], box[3]+shift)
    for cell in merged['cells']:
        if not merge_open or not cell.get('open_bottom'):
            continue
        partner = next((c for c in following['cells'] if c.get('open_top') and
                        _near(cell['bbox'][0], c['bbox'][0]) and _near(cell['bbox'][2], c['bbox'][2])), None)
        if partner is None:
            return None
        box = _layout_bbox(cell)
        cell['layout_bbox'] = (box[0], box[1], box[2], partner['layout_bbox'][3])
        cell['atoms'].extend(partner['atoms'])
        cell['source_boxes'] += partner['source_boxes']
        cell['open_bottom'] = partner['open_bottom']
        following['cells'].remove(partner)
    merged['cells'].extend(following['cells'])
    merged['layout_bbox'] = (first_box[0], first_box[1], first_box[2], second_box[3]+shift)
    merged['last_page'] = second.get('last_page', second['page'])
    merged['source_parts'] = first.get('source_parts', ((first['page'], tuple(first['bbox'])),)) + second.get(
        'source_parts', ((second['page'], tuple(second['bbox'])),))
    merged['trusted'] = first.get('trusted', True) and second.get('trusted', True)
    return merged if grid_positions(merged) else None

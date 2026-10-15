"""Optional source-labelled close-ups, without reducing existing request evidence."""
from PIL import Image, ImageDraw

from marker.cloud.client import schema_prompt
from marker.cloud.context import RecognitionFailure, fit_request


def needs_math_closeups(cells):
    return any(atom['kind'] == 'image'
               and atom['bbox'][2]-atom['bbox'][0] <= 160
               and atom['bbox'][3]-atom['bbox'][1] <= 50
               for cell in cells for line in cell['lines'] for atom in line['atoms'])


def with_math_closeups(source, cells, images, client, prompt, schema):
    """Use spare image/context capacity for readable native raster expressions.

    Original source resources remain untouched. If adding evidence would shrink
    any existing request image, keep the original request instead.
    """
    if len(images) >= client.cloud_max_images or any(image.info.get('source_closeups') for image in images):
        return images
    candidates = {}
    for cell in cells:
        for line in cell['lines']:
            for atom in line['atoms']:
                if atom['kind'] == 'image':
                    box = atom['bbox']
                    if box[2]-box[0] <= 160 and box[3]-box[1] <= 50:
                        candidates.setdefault(atom['id'], atom)
    if not candidates:
        return images
    text = schema_prompt(prompt, schema)
    baseline = fit_request(text, images, client.cloud_context_tokens, client.cloud_max_output_tokens)
    tiles = []
    for identity, atom in list(candidates.items())[:8]:
        # This is a disposable request copy; retained resources keep native pixels.
        try:
            picture = source.resource(atom['page'], tuple(atom['bbox'])).convert('RGBA')
        except RecognitionFailure:
            return images
        factor = min(3, 300/picture.width, 128/picture.height)
        size = (max(1, round(picture.width*factor)), max(1, round(picture.height*factor)))
        zoom = picture.resize(size, Image.Resampling.LANCZOS)
        tile = Image.new('RGB', (320, size[1]+24), 'white')
        ImageDraw.Draw(tile).text((4, 4), identity, fill='black')
        tile.paste(zoom, (4, 22), zoom)
        tiles.append(tile)
    columns = min(2, len(tiles))
    row_height = max(tile.height for tile in tiles)
    sheet = Image.new('RGB', (320*columns, row_height*((len(tiles)+columns-1)//columns)), 'white')
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index%columns)*320, (index//columns)*row_height))
    try:
        fitted = fit_request(text, [*images, sheet], client.cloud_context_tokens, client.cloud_max_output_tokens)
    except RecognitionFailure:
        fitted = []
    if fitted and all(before.size == after.size for before, after in zip(baseline, fitted)):
        sheet.info['source_closeups'] = True
        return [*images, sheet]
    # Two repeated context images each cost a fixed image overhead. Packing
    # their already fitted pixels and native close-ups into one labelled canvas
    # can reclaim that overhead WITHOUT resampling or dropping context pixels.
    for factor in (2, 1):
        rows = []
        for identity, atom in list(candidates.items())[:8]:
            try:
                native = source.resource(atom['page'], tuple(atom['bbox'])).convert('RGBA')
            except RecognitionFailure:
                return images
            if max(native.size) > 600:
                return images
            picture = native.resize((native.width*factor, native.height*factor), Image.Resampling.LANCZOS) if factor > 1 else native
            row = Image.new('RGBA', (max(140, picture.width+4), picture.height+18), 'white')
            ImageDraw.Draw(row).text((2, 2), identity, fill='black')
            row.paste(picture, (2, 18))
            rows.append(row)
        width = max(image.width for image in [*baseline, *rows])
        shelves, shelf, used = [], [], 0
        for row in rows:
            if shelf and used + row.width + 4 > width:
                shelves.append(shelf)
                shelf, used = [], 0
            shelf.append(row)
            used += row.width + 4
        if shelf:
            shelves.append(shelf)
        height = sum(image.height+4 for image in baseline) + sum(max(row.height for row in shelf)+4 for shelf in shelves)
        if max(width, height) > 1536:
            continue
        packed = Image.new('RGBA', (width, height), 'white')
        top = 0
        for image in baseline:
            packed.paste(image.convert('RGBA'), (0, top))
            top += image.height+4
        for shelf in shelves:
            left = 0
            for row in shelf:
                packed.paste(row, (left, top))
                left += row.width+4
            top += max(row.height for row in shelf)+4
        try:
            fitted = fit_request(text, [packed], client.cloud_context_tokens, client.cloud_max_output_tokens)
        except RecognitionFailure:
            continue
        if fitted[0].size == packed.size:
            packed.info['source_closeups'] = True
            return [packed]
    return images

"""Visible PDF coordinates and independently owned source pixels.

Every public bbox is in points, measured from the top left of the displayed
CropBox after page rotation. PDFium crops, words, table edges and native image
metadata all use that same basis. Request sizing belongs to CloudClient.
"""

from contextlib import ExitStack
from dataclasses import dataclass
from math import ceil, floor, hypot, isfinite
from pathlib import Path

import pdfplumber
from pdfplumber.utils import extract_words
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw
from PIL import Image


def intersection(first, second):
    return max(0, min(first[2], second[2])-max(first[0], second[0])) * max(
        0, min(first[3], second[3])-max(first[1], second[1])
    )


def object_bbox(item):
    return tuple(float(item[key]) for key in ("x0", "top", "x1", "bottom"))


def _rotate_bbox(box, width, height, rotation):
    x0, y0, x1, y1 = box
    if rotation == 90:
        return height-y1, x0, height-y0, x1
    if rotation == 180:
        return width-x1, height-y1, width-x0, height-y0
    if rotation == 270:
        return y0, width-x1, y1, width-x0
    return tuple(box)


def _ancestor_matrix(obj):
    """Compose this placed instance's containing forms, from inner to outer."""
    matrix = pdfium.PdfMatrix()
    parent = obj.container
    while parent is not None:
        matrix = matrix.multiply(parent.get_matrix())
        parent = parent.container
    return matrix


def _native_transpose(matrix):
    """Lossless orientation for axis-aligned placements, excluding resampling."""
    a, b, c, d, _, _ = matrix.get()
    tolerance = max(1, abs(a), abs(b), abs(c), abs(d))*1e-7
    if abs(b) <= tolerance and abs(c) <= tolerance and a*d != 0:
        orientation = (Image.Transpose.ROTATE_180 if a < 0 and d < 0 else
                       Image.Transpose.FLIP_LEFT_RIGHT if a < 0 else
                       Image.Transpose.FLIP_TOP_BOTTOM if d < 0 else None)
        return True, orientation
    if abs(a) <= tolerance and abs(d) <= tolerance and b*c != 0:
        orientation = (Image.Transpose.ROTATE_90 if b > 0 and c < 0 else
                       Image.Transpose.ROTATE_270 if b < 0 and c > 0 else
                       Image.Transpose.TRANSVERSE if b > 0 else Image.Transpose.TRANSPOSE)
        return True, orientation
    return False, None


def _native_extraction_orientation(obj):
    """PDFium renders the child matrix; ancestors need a lossless transpose.

    Individually tilted components can cancel into an axis-aligned total,
    but rendering the child first would already resample/rotate its pixels.
    """
    child_supported, _ = _native_transpose(obj.get_matrix())
    ancestor_supported, orientation = _native_transpose(_ancestor_matrix(obj))
    return child_supported and ancestor_supported, orientation


@dataclass(frozen=True)
class NativeImage:
    """One displayed image instance; width/height describe its source pixels.

    ``bbox`` is the visible clipped box; ``full_bbox`` includes any pixels
    outside the CropBox. ``index`` also identifies image atoms in table grids.
    Extraction applies lossless page rotation, so oriented output dimensions
    can be swapped relative to width/height.
    """

    page: int
    index: int
    bbox: tuple[float, float, float, float]
    width: int
    height: int
    full_bbox: tuple[float, float, float, float]
    object_index: int


class SourcePage:
    """A pdfplumber-compatible view with normalized, copied object records."""

    def __init__(self, page, visible_bbox, size, repair_word):
        self._page = page.crop(visible_bbox, strict=False)
        self._offset = visible_bbox[:2]
        self.page_number = page.page_number
        self.width, self.height = size
        self.bbox = (0.0, 0.0, self.width, self.height)
        self._repair_word = repair_word
        self._rotation = page.rotation

    def _normalize(self, item):
        result = dict(item)
        for key in ("x0", "x1"):
            result[key] = float(item[key])-self._offset[0]
        for key in ("top", "bottom"):
            result[key] = float(item[key])-self._offset[1]
        result["bbox"] = object_bbox(result)
        return result

    @property
    def edges(self):
        return [self._normalize(item) for item in self._page.edges]

    @property
    def images(self):
        return [self._normalize(item) for item in self._page.images]

    @property
    def curves(self):
        return [self._normalize(item) for item in self._page.curves]

    def extract_words(self, **kwargs):
        # Normalize the size on copied chars: rotated pdfminer chars report
        # their advance as size. Keep real size boundaries such as x / ^2.
        chars = [dict(char, size=char['size'] if char['upright'] else char['width'])
                 for char in self._page.chars]
        if self._rotation == 180:
            kwargs.setdefault('line_dir', 'btt')
            kwargs.setdefault('char_dir', 'rtl')
        elif self._rotation in (90, 270):
            kwargs.setdefault('line_dir_rotated', 'rtl' if self._rotation == 90 else 'ltr')
            kwargs.setdefault('char_dir_rotated', 'ttb' if self._rotation == 90 else 'btt')
        words = []
        for word in extract_words(chars, **kwargs):
            normalized = self._normalize(word)
            if normalized['x0'] >= normalized['x1'] or normalized['top'] >= normalized['bottom']:
                normalized = self._repair_word(normalized)
            words.append(normalized)
        return words


class PdfSource:
    """Context-managed text/geometry extraction and clipped PDFium rendering.

    Full resources never pass through the cloud request resize path. A tiny
    vector crop renders only its own rectangle, rather than a 600-DPI page.
    Returned PIL images are copied before closing their PDFium bitmap.
    """

    def __init__(self, filepath: str | Path, *, crop_dpi: int = 600):
        self.filepath = str(filepath)
        self.crop_dpi = crop_dpi
        self._stack = None
        self._pdf = None
        self._plumber = None
        self._pages = {}
        self._views = {}
        self._images = {}
        self._char_boxes = {}

    def __enter__(self):
        if self._stack is not None:
            raise RuntimeError("PdfSource is already open")
        stack = ExitStack()
        try:
            self._pdf = stack.enter_context(pdfium.PdfDocument(self.filepath))
            self._pdf.init_forms()
            self._plumber = stack.enter_context(pdfplumber.open(self.filepath))
        except Exception:
            stack.close()
            self._pdf = self._plumber = None
            raise
        self._stack = stack
        return self

    def __exit__(self, *args):
        for page in self._pages.values():
            page.close()
        self._pages.clear()
        self._views.clear()
        self._images.clear()
        self._char_boxes.clear()
        if self._stack is not None:
            self._stack.close()
        self._stack = self._pdf = self._plumber = None

    def _check_open(self):
        if self._stack is None:
            raise RuntimeError("PdfSource must be used inside its context manager")

    def __len__(self):
        self._check_open()
        return len(self._pdf)

    @property
    def page_count(self):
        return len(self)

    def _page(self, page):
        self._check_open()
        if not isinstance(page, int) or not 0 <= page < len(self._pdf):
            raise IndexError("PDF page is outside the source document")
        if page not in self._pages:
            self._pages[page] = self._pdf[page]
        return self._pages[page]

    def page_size(self, page):
        return tuple(self._page(page).get_size())

    def pdf_page(self, page) -> SourcePage:
        """Normalized vector edges/images/words for physical table detection."""
        pdf_page = self._page(page)
        if page not in self._views:
            media = pdf_page.get_mediabox()
            crop = pdf_page.get_bbox()  # PDFium's intersection of Media/CropBox.
            rotation = pdf_page.get_rotation()
            relative = (crop[0]-media[0], media[3]-crop[3],
                        crop[2]-media[0], media[3]-crop[1])
            visible = _rotate_bbox(relative, media[2]-media[0], media[3]-media[1], rotation)
            plumber_page = self._plumber.pages[page]
            # pdfplumber's CropBox normalization for rotated pages differs from
            # PDFium's visible viewport; derive the crop from raw PDF boxes.
            ox, oy = plumber_page.mediabox[:2]
            visible = (visible[0]+ox, visible[1]+oy, visible[2]+ox, visible[3]+oy)
            self._views[page] = SourcePage(plumber_page, visible, self.page_size(page),
                                           lambda word: self._repair_word(page, word))
        return self._views[page]

    def _pdfium_box(self, page, box):
        pdf_page = self._page(page)
        left, bottom, right, top = pdf_page.get_bbox()
        x0, y0, x1, y1 = box
        return _rotate_bbox((x0-left, top-y1, x1-left, top-y0),
                            right-left, top-bottom, pdf_page.get_rotation())

    def _repair_word(self, page, word):
        """Recover actual ink boxes when a damaged Unicode map breaks metrics."""
        if page not in self._char_boxes:
            textpage = self._page(page).get_textpage()
            try:
                characters = []
                for index in range(textpage.count_chars()):
                    code = pdfium_raw.FPDFText_GetUnicode(textpage, index)
                    characters.append((chr(code), self._pdfium_box(page, textpage.get_charbox(index))))
                self._char_boxes[page] = characters
            finally:
                textpage.close()
        characters = self._char_boxes[page]
        text = ''.join(character for character, _ in characters)
        value = word['text']
        start, candidates = 0, []
        while value:
            offset = text.find(value, start)
            if offset < 0:
                break
            boxes = [box for _, box in characters[offset:offset+len(value)]]
            box = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                   max(b[2] for b in boxes), max(b[3] for b in boxes))
            candidates.append(box)
            start = offset+1
        if candidates:
            box = min(candidates, key=lambda candidate: abs(candidate[0]-word['x0']) + abs(candidate[1]-word['top']))
            result = dict(word, bbox=box, x0=box[0], top=box[1], x1=box[2], bottom=box[3])
            result['width'], result['height'] = box[2]-box[0], box[3]-box[1]
            return result
        return word

    def page_words(self, page, bbox=None):
        words = self.pdf_page(page).extract_words(
            extra_attrs=["fontname", "size"], x_tolerance=1.5, y_tolerance=2
        )
        for word in words:
            word.update(kind="text", source_page=page, source_bbox=word["bbox"])
        return [word for word in words if bbox is None or intersection(word["bbox"], bbox) > 0 or
                (bbox[0] <= word['x0'] <= bbox[2] and bbox[1] <= word['top'] <= bbox[3] and
                 (word['width'] <= 0 or word['height'] <= 0))]

    def _image_bounds(self, page, obj):
        # PDFium child bounds include the image's own matrix, but omit every
        # containing Form's matrix. The container chain is instance-specific,
        # even when the PDF reuses the same Form stream at several placements.
        matrix = _ancestor_matrix(obj)
        points = [matrix.on_point(*point) for point in obj.get_quad_points()]
        box = (min(point[0] for point in points), min(point[1] for point in points),
               max(point[0] for point in points), max(point[1] for point in points))
        return self._pdfium_box(page, box)

    def page_images(self, page, bbox=None) -> list[NativeImage]:
        pdf_page = self._page(page)
        if page not in self._images:
            objects = list(pdf_page.get_objects(filter=[pdfium_raw.FPDF_PAGEOBJ_IMAGE]))
            candidates = [(index, obj, self._image_bounds(page, obj)) for index, obj in enumerate(objects)]
            records = []
            used = set()
            for index, image in enumerate(self.pdf_page(page).images):
                box = object_bbox(image)
                matching = [(i, obj, bounds) for i, obj, bounds in candidates
                            if i not in used and intersection(box, bounds) > 0]
                if not matching:
                    from marker.cloud.context import RecognitionFailure
                    raise RecognitionFailure("A native raster instance could not be located for source extraction")
                srcsize = tuple(image.get('srcsize', (0, 0)))
                object_index, obj, bounds = min(matching, key=lambda entry: (
                    tuple(entry[1].get_px_size()) != srcsize,
                    sum(abs(a-b) for a, b in zip(box, entry[2]))))
                used.add(object_index)
                width, height = obj.get_px_size()
                records.append(NativeImage(page, index, box, int(width), int(height), bounds, object_index))
            self._images[page] = records
        return [image for image in self._images[page] if bbox is None or intersection(image.bbox, bbox) > 0]

    def extract_image(self, image: NativeImage) -> Image.Image:
        """Decode pixels/masks; apply lossless ancestor and page orientation.

        Unsupported image or ancestor components raise RecognitionFailure.
        Use resource() to retain their correctly placed composite rendering.
        """
        page = self._page(image.page)
        objects = list(page.get_objects(filter=[pdfium_raw.FPDF_PAGEOBJ_IMAGE]))
        obj = objects[image.object_index]
        supported, orientation = _native_extraction_orientation(obj)
        if not supported:
            from marker.cloud.context import RecognitionFailure
            raise RecognitionFailure("Native raster placement requires composite source rendering")
        bitmap = obj.get_bitmap(render=True, scale_to_original=True)
        try:
            pixels = bitmap.to_pil().copy()
        finally:
            bitmap.close()
        if orientation is not None:
            pixels = pixels.transpose(orientation)
        rotation = page.get_rotation()
        transpose = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180,
                     270: Image.Transpose.ROTATE_90}
        return pixels.transpose(transpose[rotation]) if rotation else pixels

    def _valid_crop(self, page, bbox, dpi):
        width, height = self.page_size(page)
        if len(bbox) != 4 or not all(isfinite(value) for value in bbox):
            raise ValueError("Source crop requires four finite coordinates")
        x0, y0, x1, y1 = bbox
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("Source crop must be inside the visible page")
        if not isfinite(dpi) or dpi <= 0:
            raise ValueError("Source crop DPI must be positive")
        return width, height

    def crop(self, page, bbox, dpi=None) -> Image.Image:
        """Render this source rectangle only; bbox never uses virtual coordinates."""
        dpi = self.crop_dpi if dpi is None else dpi
        width, height = self._valid_crop(page, bbox, dpi)
        x0, y0, x1, y1 = bbox
        scale = dpi/72
        left, top = floor(x0*scale), floor(y0*scale)
        # PDFium matrices use float32 while pdfplumber coordinates use Python
        # floats. Ignore their tiny boundary drift, capped at 0.001 pixel,
        # without changing source coordinates or the retained raster DPI.
        spans = ((x1-x0)*scale, (y1-y0)*scale)
        pixel_width, pixel_height = (max(1, ceil(span-min(1e-3, max(1e-9, span*1e-7))))
                                     for span in spans)
        cuts = (left, ceil(height*scale)-top-pixel_height,
                ceil(width*scale)-left-pixel_width, top)
        # PDFium rounds each cut separately. Choose integer pixel cuts first
        # to avoid losing a pixel to floating point ceil(2066.666... * scale).
        crop = tuple(max(0, (cut-1e-7)/scale) for cut in cuts)
        bitmap = self._page(page).render(scale=scale, crop=crop, draw_annots=False)
        try:
            return bitmap.to_pil().convert("RGB").copy()
        finally:
            bitmap.close()

    def resource(self, page, bbox, dpi=None) -> Image.Image:
        """Retain native detail, including in composites with text/vector content.

        ``dpi`` may raise the retained-resource sampling. Use crop() for a
        deliberately smaller request/diagnostic image; resource() never lowers
        the configured vector DPI or an intersecting raster's native density.
        """
        dpi = self.crop_dpi if dpi is None else dpi
        self._valid_crop(page, bbox, dpi)
        exact = [image for image in self.page_images(page) if all(
            abs(a-b) <= 0.1 for a, b in zip(image.bbox, bbox)
        )]
        if len(exact) == 1 and self.isolated_image(exact[0]):
            image = exact[0]
            pixels = self.extract_image(image)
            full = image.full_bbox
            if full != image.bbox:
                sx, sy = pixels.width/(full[2]-full[0]), pixels.height/(full[3]-full[1])
                pixels = pixels.crop(tuple(round(value) for value in (
                    (bbox[0]-full[0])*sx, (bbox[1]-full[1])*sy,
                    (bbox[2]-full[0])*sx, (bbox[3]-full[1])*sy)))
            return pixels
        return self.crop(page, bbox, dpi=max(self.crop_dpi, dpi, self._raster_dpi(page, bbox)))

    def _raster_dpi(self, page, bbox):
        """Maximum pixel/point density through image and ancestor transforms.

        The largest singular value of the inverse pixel placement accounts
        for anisotropic scaling/shear as well as rotation. Page rotation is
        orthogonal, so it does not change this density. Raising sample counts
        on a stretched axis preserves detail; it does not invent new detail.
        """
        objects = list(self._page(page).get_objects(filter=[pdfium_raw.FPDF_PAGEOBJ_IMAGE]))
        density = 0.0
        for image in self.page_images(page, bbox):
            obj = objects[image.object_index]
            matrix = obj.get_matrix().multiply(_ancestor_matrix(obj))
            a, b, c, d, _, _ = matrix.get()
            determinant = a*d-b*c
            if determinant == 0:
                from marker.cloud.context import RecognitionFailure
                raise RecognitionFailure("Native raster placement has no usable area")
            q = (image.width*d/determinant, -image.width*c/determinant,
                 -image.height*b/determinant, image.height*a/determinant)
            # Stable closed form for the largest singular value of a 2x2
            # matrix. Unlike subtracting nearly equal squared traces, this
            # does not add a spurious pixel for uniform native sampling.
            native = (hypot(q[0]+q[3], q[1]-q[2]) +
                      hypot(q[0]-q[3], q[1]+q[2]))*36
            if not isfinite(native):
                from marker.cloud.context import RecognitionFailure
                raise RecognitionFailure("Native raster placement density is not finite")
            density = max(density, native)
        return density

    def isolated_image(self, image: NativeImage) -> bool:
        """Whether native extraction alone reproduces this displayed rectangle."""
        box, page = image.bbox, image.page
        objects = list(self._page(page).get_objects(filter=[pdfium_raw.FPDF_PAGEOBJ_IMAGE]))
        obj = objects[image.object_index]
        supported, _ = _native_extraction_orientation(obj)
        if not supported:
            return False
        if self.page_words(page, box) or any(other.index != image.index and
                                             intersection(other.bbox, box) > 0 for other in self.page_images(page)):
            return False
        for edge in self.pdf_page(page).edges:
            if edge['orientation'] == 'h':
                if box[1] < edge['top'] < box[3] and max(box[0], edge['x0']) < min(box[2], edge['x1']):
                    return False
            elif box[0] < edge['x0'] < box[2] and max(box[1], edge['top']) < min(box[3], edge['bottom']):
                return False
        return not any(intersection(object_bbox(curve), box) > 0 for curve in self.pdf_page(page).curves)

    def text_quality(self, page, bbox=None):
        from marker.cloud.regions import assess_text_quality
        width, height = self.page_size(page)
        bbox = (0, 0, width, height) if bbox is None else tuple(bbox)
        return assess_text_quality(self.page_words(page, bbox), bbox, self.page_images(page, bbox))

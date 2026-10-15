"""Keep completed image resources as lossless, lazily decoded PNG snapshots."""

from copy import deepcopy
from io import BytesIO
import os
import sys

from PIL import Image, PngImagePlugin


_PNG_MODES = {"1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"}


class _RetainedPng(PngImagePlugin.PngImageFile):
    def __init__(self, stream, mode):
        self._retained_stream = stream
        stream.seek(0)
        super().__init__(stream)
        # PNG represents both modes with 16-bit samples. Pillow's decoder
        # supports either destination, so retain the caller's integer mode.
        if mode in {"I", "I;16"} and self.mode in {"I", "I;16"}:
            self._mode = mode
        self._retained_mode = self.mode
        self._retained_size = self.size
        self._retained_info = deepcopy(self.info)
        self._retained_palette = self._palette_state()

    def _palette_state(self):
        if self.palette is None:
            return None
        return self.palette.mode, self.palette.rawmode, bytes(self.palette.palette)

    def _untouched(self):
        return bool(
            self.tile
            and not self._retained_stream.closed
            and self.mode == self._retained_mode
            and self.size == self._retained_size
            and self.info == self._retained_info
            and self._palette_state() == self._retained_palette
        )

    def save(self, fp, format=None, **params):
        filename = fp if isinstance(fp, (str, bytes, os.PathLike)) else getattr(fp, "name", "")
        png = format and format.upper() == "PNG"
        if not format and isinstance(filename, (str, bytes, os.PathLike)):
            png = os.path.splitext(os.fspath(filename))[1].lower() in (".png", b".png")
        if not png or params or not self._untouched() or fp is self._retained_stream:
            return super().save(fp, format=format, **params)
        # No decoder, second PNG encoding, temporary file, or resource file
        # descriptor is needed for the converter's ordinary final PNG save.
        with self._retained_stream.getbuffer() as encoded:
            if isinstance(fp, (str, bytes, os.PathLike)):
                with open(fp, "w+b") as destination:
                    destination.write(encoded)
            else:
                if fp is sys.stdout:
                    fp = getattr(fp, "buffer", fp)
                fp.write(encoded)


def retain_png(image: Image.Image) -> Image.Image:
    """Own a PNG snapshot without holding its source's decoded image core.

    Modes supported by the PNG output path keep their samples and alpha;
    other modes follow that path's existing RGB conversion. Integer samples
    follow Pillow's normal PNG representation. Reading or editing the returned
    image uses Pillow normally; untouched PNG saves copy its encoded bytes.
    An already retained, untouched resource is reused during output assembly.
    """
    if isinstance(image, _RetainedPng) and image._untouched():
        return image
    if image.mode not in _PNG_MODES:
        image = image.convert("RGB")
    stream = BytesIO()
    image.save(stream, "PNG")
    return _RetainedPng(stream, image.mode)

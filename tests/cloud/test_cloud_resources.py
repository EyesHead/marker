"""Lossless retained resources own PNG bytes without retaining decoded cores."""

import builtins
import gc
from io import BytesIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

from PIL import Image, PngImagePlugin

from marker.cloud.resources import retain_png


FIXTURES = Path(__file__).parent / "fixtures"


def sample(mode):
    image = Image.new(mode, (19, 11))
    if mode == "P":
        image.putpalette([value for index in range(256) for value in (index, 255-index, index//2)])
        image.info["transparency"] = bytes(range(256))
    for y in range(image.height):
        for x in range(image.width):
            value = (x*17 + y*23) % 256
            color = {
                "1": value > 127,
                "L": value,
                "P": value,
                "LA": (value, 255-value),
                "RGB": (value, 255-value, value//2),
                "RGBA": (value, 255-value, value//2, (x*29 + y*11) % 256),
                "I": value*257,
                "I;16": value*257,
            }[mode]
            image.putpixel((x, y), color)
    return image


class RetainedPngTests(unittest.TestCase):
    def test_supported_modes_dimensions_and_pixels_are_preserved(self):
        for mode in ("1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"):
            with self.subTest(mode=mode):
                original = sample(mode)
                expected = original.tobytes()
                expected_rgba = original.convert("RGBA").tobytes()
                retained = retain_png(original)
                self.assertIsInstance(retained, Image.Image)
                self.assertEqual((retained.mode, retained.size), (original.mode, original.size))
                self.assertTrue(retained.tile)
                self.assertEqual(retained.tobytes(), expected)
                self.assertEqual(retained.convert("RGBA").tobytes(), expected_rgba)
                self.assertEqual(original.tobytes(), expected)

    def test_palette_with_alpha_keeps_every_visible_color(self):
        image = Image.new("P", (3, 1))
        image.putpalette(bytes([255, 0, 0, 0, 0, 255, 0, 127, 0, 255, 255, 255]), rawmode="RGBA")
        image.putdata([0, 1, 2])
        expected = image.convert("RGBA").tobytes()
        retained = retain_png(image)
        self.assertEqual(retained.mode, "P")
        self.assertEqual(retained.convert("RGBA").tobytes(), expected)

    def test_unmodified_png_save_copies_bytes_without_loading(self):
        image = sample("RGBA")
        expected = BytesIO()
        image.save(expected, "PNG")
        retained = retain_png(image)
        for format in ("PNG", "png"):
            with self.subTest(format=format), patch.object(retained, "load", side_effect=AssertionError("decoded PNG during save")):
                output = BytesIO()
                retained.save(output, format=format)
                self.assertEqual(output.getvalue(), expected.getvalue())
                self.assertTrue(retained.tile)

    def test_unicode_path_extension_and_bytes_path_use_lazy_save(self):
        image = sample("RGB")
        retained = retain_png(image)
        with tempfile.TemporaryDirectory(prefix="retained png ") as temporary:
            for name in ("ресурс.PNG", "resource.png"):
                path = Path(temporary) / name
                target = path if name.startswith("ресурс") else os.fsencode(path)
                with patch.object(retained, "load", side_effect=AssertionError("decoded PNG during save")):
                    retained.save(target)
                with Image.open(path) as saved:
                    self.assertEqual(saved.tobytes(), image.tobytes())

    def test_retention_and_in_memory_save_never_open_an_os_file(self):
        image = sample("RGB")
        image.save(BytesIO(), "PNG")
        with patch.object(builtins, "open", side_effect=AssertionError("unexpected resource file")):
            retained = [retain_png(image) for _ in range(40)]
            for resource in retained:
                resource.save(BytesIO(), "PNG")
        self.assertTrue(all(resource.tile for resource in retained))

    def test_original_can_close_before_retained_image_is_used(self):
        image = sample("RGBA")
        expected = image.tobytes()
        retained = retain_png(image)
        image.close()
        self.assertEqual(retained.getpixel((2, 3)), (103, 152, 51, 91))
        self.assertEqual(retained.copy().tobytes(), expected)

    def test_pixel_edits_are_saved_instead_of_original_png(self):
        retained = retain_png(sample("RGBA"))
        retained.putpixel((2, 3), (7, 9, 11, 13))
        output = BytesIO()
        retained.save(output, "PNG")
        with Image.open(BytesIO(output.getvalue())) as saved:
            self.assertEqual(saved.getpixel((2, 3)), (7, 9, 11, 13))

    def test_palette_edits_are_saved(self):
        retained = retain_png(sample("P"))
        palette = [value for _ in range(256) for value in (10, 20, 30)]
        retained.putpalette(palette)
        output = BytesIO()
        retained.save(output, "PNG")
        with Image.open(BytesIO(output.getvalue())) as saved:
            self.assertEqual(saved.convert("RGBA").getpixel((0, 0))[:3], (10, 20, 30))

    def test_explicit_png_options_use_normal_pillow_encoding(self):
        retained = retain_png(sample("RGB"))
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("Resource note", "specified metadata")
        output = BytesIO()
        retained.save(output, "PNG", pnginfo=pnginfo, compress_level=0)
        with Image.open(BytesIO(output.getvalue())) as saved:
            self.assertEqual(saved.info["Resource note"], "specified metadata")
            self.assertEqual(saved.tobytes(), sample("RGB").tobytes())

    def test_info_changes_use_normal_pillow_encoding(self):
        retained = retain_png(sample("RGB"))
        retained.info["transparency"] = (0, 255, 0)
        output = BytesIO()
        retained.save(output, "PNG")
        with Image.open(BytesIO(output.getvalue())) as saved:
            self.assertEqual(saved.info["transparency"], (0, 255, 0))

    def test_non_png_output_uses_normal_pillow_save(self):
        original = sample("RGB")
        retained = retain_png(original)
        output = BytesIO()
        retained.save(output, "BMP")
        with Image.open(BytesIO(output.getvalue())) as saved:
            self.assertEqual(saved.format, "BMP")
            self.assertEqual(saved.tobytes(), original.tobytes())

    def test_resources_remain_usable_after_source_closes(self):
        from marker.cloud.source import PdfSource

        with PdfSource(FIXTURES / "content_lines.pdf") as source:
            original = source.crop(0, (40, 70, 110, 90))
            expected = original.tobytes()
            retained = retain_png(original)
        self.assertEqual(retained.tobytes(), expected)

    def test_large_white_resource_drops_decoded_source_and_compresses(self):
        original = Image.new("RGBA", (2400, 1600), (255, 255, 255, 255))
        raw_size = original.width*original.height*4
        reference = weakref.ref(original)
        retained = retain_png(original)
        del original
        gc.collect()
        self.assertIsNone(reference())
        self.assertIsNone(retained.im)
        with patch.object(retained, "load", side_effect=AssertionError("decoded PNG during save")):
            output = BytesIO()
            retained.save(output, "PNG")
        self.assertLess(len(output.getvalue()), raw_size//100)

    def test_releasing_retained_resources_leaves_no_global_cache(self):
        retained = retain_png(sample("RGB"))
        reference = weakref.ref(retained)
        del retained
        gc.collect()
        self.assertIsNone(reference())

    def test_untouched_retained_resource_is_reused_without_another_encoding(self):
        first = retain_png(sample("RGBA"))
        with patch.object(first, "save", side_effect=AssertionError("encoded twice")):
            self.assertIs(retain_png(first), first)
        self.assertIsNone(first.im)

    def test_loaded_or_edited_retained_resource_gets_a_fresh_compressed_snapshot(self):
        for edit in (False, True):
            with self.subTest(edit=edit):
                first = retain_png(sample("RGBA"))
                first.load()
                if edit:
                    first.putpixel((2, 3), (7, 9, 11, 13))
                expected = first.tobytes()
                second = retain_png(first)
                self.assertIsNot(second, first)
                self.assertIsNone(second.im)
                self.assertTrue(second.tile)
                self.assertEqual(second.tobytes(), expected)

    def test_unsupported_mode_matches_existing_rgb_png_output(self):
        image = Image.new("CMYK", (12, 9), (25, 50, 75, 10))
        expected = image.convert("RGB")
        retained = retain_png(image)
        self.assertEqual(retained.mode, "RGB")
        self.assertEqual(retained.tobytes(), expected.tobytes())
        self.assertEqual(image.mode, "CMYK")

    def test_integer_png_values_match_existing_png_serialization(self):
        image = Image.new("I", (3, 1))
        image.putdata([-1, 256, 70000])
        baseline = BytesIO()
        image.save(baseline, "PNG")
        with Image.open(BytesIO(baseline.getvalue())) as expected:
            values = list(expected.getdata())
        retained = retain_png(image)
        self.assertEqual(list(retained.getdata()), values)
        self.assertEqual(list(image.getdata()), [-1, 256, 70000])


if __name__ == "__main__":
    unittest.main()

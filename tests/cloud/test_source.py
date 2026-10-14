"""Offline extraction and quality gates, exercised against actual PDF bytes."""

import unittest
from pathlib import Path

from marker.cloud.tables import detect_tables

FIXTURES = Path(__file__).parent / "fixtures"


class PdfSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from marker.cloud.source import PdfSource
        from marker.cloud.regions import analyze_page, assess_text_quality
        cls.PdfSource = PdfSource
        cls.analyze_page = staticmethod(analyze_page)
        cls.assess_text_quality = staticmethod(assess_text_quality)

    def test_native_raster_resource_keeps_original_pixels(self):
        with self.PdfSource(FIXTURES / "native_figure.pdf") as source:
            self.assertEqual(source.page_size(0), (360, 400))
            images = source.page_images(0)
            self.assertEqual(len(images), 1)
            self.assertEqual(images[0].bbox, (40, 50, 200, 130))
            self.assertEqual((images[0].width, images[0].height), (2048, 1024))
            pixels = source.extract_image(images[0])
            self.assertEqual(pixels.size, (2048, 1024))
            self.assertEqual(pixels.getpixel((100, 100))[:3], (10, 40, 90))
            self.assertEqual(pixels.getpixel((1500, 100))[:3], (180, 20, 35))
            resource = source.resource(0, images[0].bbox)
            self.assertEqual(resource.size, (2048, 1024))

    def test_vector_crop_uses_dpi_and_original_top_down_coordinates(self):
        with self.PdfSource(FIXTURES / "native_figure.pdf") as source:
            pixels = source.crop(0, (40, 220, 112, 256), dpi=600)
            self.assertEqual(pixels.size, (600, 300))
            self.assertLess(pixels.convert("L").getextrema()[0], 100)
            with self.assertRaises(ValueError):
                source.crop(0, (0, 0, 0, 12), dpi=600)
            with self.assertRaises(ValueError):
                source.crop(0, (-1, 0, 10, 12), dpi=600)

    def test_rotated_nonzero_cropbox_shares_word_image_and_crop_coordinates(self):
        with self.PdfSource(FIXTURES / "rotated_crop.pdf") as source:
            self.assertEqual(source.page_size(0), (280, 300))
            image = source.page_images(0)[0]
            self.assertEqual(image.bbox, (170, 20, 250, 180))
            crop = source.crop(0, image.bbox, dpi=72)
            self.assertEqual(crop.size, (80, 160))
            native = source.resource(0, image.bbox)
            self.assertEqual(native.size, (1024, 2048))
            self.assertEqual(native.getpixel((100, 100))[:3], (10, 40, 90))
            self.assertEqual(native.getpixel((100, 1500))[:3], (180, 20, 35))
            before = source.page_words(0)
            figure = next(word for word in before if word["text"] == "Figure")
            self.assertAlmostEqual(figure["bbox"][1], 20)
            text_crop = source.crop(0, figure["bbox"], dpi=144)
            self.assertLess(text_crop.convert("L").getextrema()[0], 100)
            self.assertEqual(source.page_words(0), before)
        self.assertEqual(native.getpixel((100, 100))[:3], (10, 40, 90))
        self.assertLess(text_crop.convert("L").getextrema()[0], 100)

    def test_other_page_rotations_preserve_complete_words_and_source_crops(self):
        with self.PdfSource(FIXTURES / "rotated_crop.pdf") as source:
            for page, dimensions in ((1, (300, 280)), (2, (280, 300))):
                with self.subTest(page=page):
                    self.assertEqual(source.page_size(page), dimensions)
                    words = source.page_words(page)
                    self.assertIn("Figure", [word["text"] for word in words])
                    for word in words:
                        self.assertLess(source.crop(page, word["bbox"], dpi=144).convert("L").getextrema()[0], 100)

    def test_font_size_boundaries_keep_superscript_separate_from_base_text(self):
        with self.PdfSource(FIXTURES / "split_formula.pdf") as source:
            words = source.page_words(0, (20, 60, 55, 77))
            self.assertEqual([word["text"] for word in words], ["x", "2"])
            self.assertEqual([word["size"] for word in words], [10, 6])

    def test_source_closes_and_returned_images_remain_usable(self):
        source = self.PdfSource(FIXTURES / "native_figure.pdf")
        with source:
            crop = source.crop(0, (40, 50, 200, 130), dpi=72)
            words = source.page_words(0)
            self.assertIn("Figure", [word["text"] for word in words])
        self.assertEqual(crop.size, (160, 80))
        self.assertLess(crop.convert("L").getextrema()[0], 100)
        with self.assertRaises(RuntimeError):
            source.page_words(0)

    def test_overlapping_native_instances_and_text_overlay_remain_available(self):
        with self.PdfSource(FIXTURES / "image_layers.pdf") as source:
            images = source.page_images(0)
            self.assertEqual(len(images), 2)
            self.assertEqual([(image.width, image.height) for image in images], [(2048, 1024), (512, 256)])
            first, second = [source.extract_image(image) for image in images]
            self.assertEqual(first.getpixel((100, 100))[:3], (10, 40, 90))
            self.assertEqual(second.getpixel((100, 100)), (255, 0, 0, 255))
            self.assertEqual(second.getpixel((400, 100))[3], 0)
            regions = self.analyze_page(source, 0)
            self.assertEqual(len(regions), 1)
            self.assertEqual(regions[0].kind, "image")
            self.assertIsNone(regions[0].image, "A composite must use its source crop")
            self.assertEqual(len([atom for atom in regions[0].atoms if atom["kind"] == "image"]), 2)
            self.assertIn("OVERLAY", [atom.get("text") for atom in regions[0].atoms])
            composite = source.resource(0, images[0].bbox, dpi=72)
            self.assertEqual(composite.size, (160, 80))
            self.assertEqual(composite.getpixel((100, 10))[:3], (10, 40, 90))

    def test_resource_crop_keeps_a_vector_overlay_that_crosses_raster_bounds(self):
        with self.PdfSource(FIXTURES / "image_layers.pdf") as source:
            image = source.page_images(1)[0]
            resource = source.resource(1, image.bbox, dpi=72)
            self.assertEqual(resource.size, (160, 80))
            self.assertLess(resource.getpixel((100, 40))[0], 10)

    def test_scan_has_untrusted_full_page_region_and_native_image(self):
        with self.PdfSource(FIXTURES / "scan.pdf") as source:
            self.assertEqual(source.page_words(0), [])
            self.assertEqual(detect_tables(source), [])
            quality = source.text_quality(0)
            self.assertFalse(quality.trusted)
            self.assertIn("missing_text", quality.reasons)
            regions = self.analyze_page(source, 0)
            self.assertEqual(len(regions), 1)
            self.assertEqual(regions[0].kind, "image")
            self.assertFalse(regions[0].quality.trusted)
            self.assertEqual(source.extract_image(regions[0].image).size, (1080, 1200))

    def test_damaged_text_is_rejected_without_poisoning_neighbour_region(self):
        with self.PdfSource(FIXTURES / "damaged_text.pdf") as source:
            good = source.text_quality(0, (20, 30, 340, 80))
            damaged = source.text_quality(0, (20, 150, 340, 200))
            self.assertTrue(good.trusted)
            self.assertFalse(damaged.trusted)
            self.assertIn("invalid_characters", damaged.reasons)
            regions = self.analyze_page(source, 0)
            self.assertEqual(len(regions), 2)
            self.assertEqual([region.quality.trusted for region in regions], [True, False])

    def test_quality_rejects_unmapped_glyphs_and_outside_text(self):
        good = [{"text": "Valid 1,70", "bbox": (0, 0, 80, 10)}]
        self.assertTrue(self.assess_text_quality(good, (0, 0, 100, 20)).trusted)
        for text in ("(cid:128)(cid:129)", "\ufffd\ufffd\ufffd", "a\x00b", "\ue123\ue124"):
            with self.subTest(text=text):
                words = [{"text": text, "bbox": (0, 0, 80, 10)}]
                self.assertFalse(self.assess_text_quality(words, (0, 0, 100, 20)).trusted)
        outside = [{"text": "outside", "bbox": (200, 0, 230, 10)}]
        self.assertFalse(self.assess_text_quality(outside, (0, 0, 100, 20)).trusted)

    def test_split_formula_keeps_raster_and_text_atoms_in_source_regions(self):
        with self.PdfSource(FIXTURES / "split_formula.pdf") as source:
            regions = self.analyze_page(source, 0)
            inline = [region for region in regions if region.bbox[1] < 60]
            self.assertTrue(inline)
            atoms = [atom for region in inline for atom in region.atoms]
            self.assertIn("x", [atom.get("text") for atom in atoms])
            self.assertIn("0", [atom.get("text") for atom in atoms])
            images = [atom for atom in atoms if atom["kind"] == "image"]
            self.assertEqual(len(images), 1)
            self.assertEqual(images[0]["source_bbox"], (85, 40, 101, 56))
            self.assertEqual(source.extract_image(source.page_images(0)[0]).size, (96, 96))


if __name__ == "__main__":
    unittest.main()

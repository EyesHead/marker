"""Regressions derived from the incomplete 72-page whole-object conversion."""

import unittest

from PIL import Image

from marker.cloud.client import CloudClient, ResponseValidationError
from marker.cloud.regions import Region, assess_text_quality
from marker.converters.ollama_document import (
    StagedOllamaPdfConverter,
    _article_validator,
)
from test_client import OfflineCloud


class Source:
    def resource(self, page, box):
        return Image.new("RGB", (128, 48), "white")

    def crop(self, page, box, **kwargs):
        return self.resource(page, box)


def region(kind="text", text=".", *, raster=False):
    box = (0, 0, 80, 12)
    atoms = [{"kind": "text", "text": text, "bbox": box,
              "size": 10, "fontname": "ArialMT"}]
    if raster:
        atoms.append({"kind": "image", "bbox": (70, 0, 80, 12)})
    return Region(0, box, kind, tuple(atoms), assess_text_quality(atoms[:1], box),
                  inline_group=(0, box))


class StagedResponseRegressionTests(unittest.TestCase):
    def test_known_whole_equation_in_an_ordinary_font_is_still_math(self):
        owner = region("math", "Run")
        owner = Region(owner.page, owner.bbox, owner.kind, owner.atoms, owner.quality)
        with OfflineCloud([r"<math>R_{un}</math>"]) as offline:
            client = CloudClient({"cloud_max_attempts": 1}, client=offline.client)
            converter = StagedOllamaPdfConverter(artifact_dict={"cloud_client": client})
            result = converter._recognize_object(Source(), owner)
            self.assertTrue(result.complete, result.reasons)
            self.assertIn('display="block"', result.html)
            self.assertEqual(result.markdown, r"$$R_{un}$$")

    def test_article_preserves_numbers_decimals_signs_and_single_letter_words(self):
        for original, changed in (("Предел 450", "Предел 45"),
                                  ("Предел 0,42", "Предел 0,4"),
                                  ("Предел -450", "Предел 450"),
                                  ("и элемент", "элемент")):
            with self.subTest(original=original), self.assertRaises(ResponseValidationError):
                _article_validator(f"<p>{changed}</p>", {}, expected_text=[original])
            _article_validator(f"<p>{original}</p>", {}, expected_text=[original])

    def test_unknown_object_cannot_drop_trusted_prose(self):
        for raw in ("<p>Стальная</p>", "<article><p>Стальная</p></article>", "<math>x</math>"):
            owner = region("unknown", "Стальная конструкция")
            with self.subTest(raw=raw), OfflineCloud([raw]) as offline:
                client = CloudClient({"cloud_max_attempts": 1}, client=offline.client)
                converter = StagedOllamaPdfConverter(artifact_dict={"cloud_client": client})
                result = converter._recognize_object(Source(), owner)
                self.assertFalse(result.complete)

    def test_unknown_true_figure_can_retain_original_labels_as_pixels(self):
        owner = region("unknown", "Стальная конструкция")
        with OfflineCloud(["<figure/>"]) as offline:
            client = CloudClient({"cloud_max_attempts": 1}, client=offline.client)
            converter = StagedOllamaPdfConverter(artifact_dict={"cloud_client": client})
            result = converter._recognize_object(Source(), owner)
            self.assertTrue(result.complete, result.reasons)
            self.assertEqual(len(result.resources), 1)

    def test_native_punctuation_is_preserved_without_model_interpretation(self):
        with OfflineCloud([]) as offline:
            client = CloudClient(client=offline.client)
            converter = StagedOllamaPdfConverter(artifact_dict={"cloud_client": client})
            for text in (".", ",", ";", ":", "…", "(", ")"):
                owner = region(text=text)
                job = {"kind": "text", "region": owner}
                result = converter._recognize_job(Source(), job)
                self.assertTrue(result.complete, text)
                self.assertEqual(result.markdown, text)
                self.assertEqual(result.kind, "text")
            self.assertEqual(client.request_count, 0)

    def test_complete_paragraph_and_math_fragment_does_not_need_article_wrapper(self):
        html = _article_validator(
            "<p>должна быть <math>h_{ef}/30+40</math> мм</p>", {},
            expected_text=["должна быть мм"], require_math=True,
        )
        self.assertIn('application/x-tex', html)
        self.assertIn("h_{ef}/30+40", html)

    def test_two_complete_blocks_can_be_wrapped_without_changing_content(self):
        html = _article_validator("<p>Начало</p><p>Конец</p>", {},
                                  expected_text=["Начало Конец"])
        self.assertEqual(html, "<p>Начало</p><p>Конец</p>")

    def test_fragment_normalization_still_rejects_truncation_and_executable_markup(self):
        for raw in ("<p>Начало", "<article><p>Начало</article>",
                    "<p>Начало</p> explanation", "<script>alert(1)</script>",
                    '<p onclick="x()">Начало</p>'):
            with self.subTest(raw=raw), self.assertRaises(ResponseValidationError):
                _article_validator(raw, {})

    def test_mixed_raster_formula_and_native_prose_is_not_forced_to_bare_math(self):
        owner = region("image", "должна быть не менее мм", raster=True)
        with OfflineCloud(["<p>должна быть не менее <math>h_{ef}/30+40</math> мм</p>"]) as offline:
            client = CloudClient(client=offline.client)
            converter = StagedOllamaPdfConverter(artifact_dict={"cloud_client": client})
            result = converter._recognize_job(Source(), {"kind": "image", "region": owner})
            self.assertTrue(result.complete, result.reasons)
            self.assertIn("должна быть не менее", result.markdown)
            self.assertIn("$h_{ef}/30+40$", result.markdown)
            self.assertEqual(result.resources, {})
            self.assertEqual(client.request_count, 1)

    def test_mixed_response_cannot_drop_native_prose_or_known_math(self):
        for raw in ("<p>должна быть не менее мм</p>", "<p><math>x</math></p>"):
            owner = region("image", "должна быть не менее мм", raster=True)
            with self.subTest(raw=raw), OfflineCloud([raw]) as offline:
                client = CloudClient({"cloud_max_attempts": 1}, client=offline.client)
                converter = StagedOllamaPdfConverter(artifact_dict={"cloud_client": client})
                result = converter._recognize_job(Source(), {"kind": "image", "region": owner})
                self.assertFalse(result.complete)


if __name__ == "__main__":
    unittest.main()

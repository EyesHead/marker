"""The model sees original word geometry, not only flattened cell strings."""

from copy import deepcopy
import json
import unittest

from marker.cloud.content import prompt_source_evidence
from marker.cloud.whole_tables import recognize_whole_table
from test_whole_tables import FakeClient, FakeSource, logical


def words():
    return [
        {"kind": "text", "text": "R", "bbox": (10, 20, 22, 38),
         "source_bbox": (10, 20, 22, 38), "source_page": 0, "size": 14},
        {"kind": "text", "text": "bun", "bbox": (22, 29, 38, 39),
         "source_bbox": (22, 29, 38, 39), "source_page": 0, "size": 8},
    ]


class PromptGeometryTests(unittest.TestCase):
    def test_prompt_coordinates_and_scripts_preserve_original_objects(self):
        atoms = words()
        original = deepcopy(atoms)
        evidence = prompt_source_evidence(atoms, (0, 0, 100, 100), 0)
        self.assertEqual(evidence["atoms"][0]["bbox"], [100, 200, 220, 380])
        self.assertEqual(evidence["atoms"][1]["bbox"], [220, 290, 380, 390])
        hint = evidence["script_hints"][0]
        self.assertEqual((hint["base"], hint["position"], hint["text"]), ("R", "subscript", "bun"))
        self.assertEqual(hint["base_atom"], evidence["atoms"][0]["id"])
        self.assertEqual(hint["atom_ids"], [evidence["atoms"][1]["id"]])
        self.assertEqual(atoms, original)

    def test_table_request_carries_geometry_and_baselines_for_each_physical_cell(self):
        table = logical(words())
        client = FakeClient(r"<table><tr><td><math>R_{bun}</math></td></tr></table>")
        result = recognize_whole_table(FakeSource(), table, client)
        self.assertTrue(result.complete, result.reasons)
        self.assertEqual(len(client.calls), 1)
        payload = json.loads(client.calls[0][0].split("SOURCE:\n", 1)[1])
        cell = payload["parts"][0]["cells"][0]
        self.assertEqual([a["text"] for a in cell["atoms"]], ["R", "bun"])
        self.assertEqual(cell["script_hints"][0]["text"], "bun")
        self.assertEqual(cell["atoms"][1]["bbox"], [36.667, 64.444, 63.333, 86.667])


if __name__ == "__main__":
    unittest.main()

import unittest
from marker.cloud.tables import TablePart


class AppendixTableLabelTests(unittest.TestCase):
    def test_unicode_appendix_and_numeric_table_labels(self):
        for caption, expected in (
            ("Таблица И.1", "и.1"),
            ("Таблица К.1", "к.1"),
            ("Table AA.12.3", "aa.12.3"),
            ("Таблица 16", "16"),
            ("Продолжение таблицы И.5", "и.5"),
        ):
            with self.subTest(caption=caption):
                self.assertEqual(
                    TablePart(0, (0, 0, 100, 100), caption=caption).table_id, expected
                )

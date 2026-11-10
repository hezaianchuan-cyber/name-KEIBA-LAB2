import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nar_exacta_research as exacta


class ExactaResearchTests(unittest.TestCase):
    def setUp(self):
        self.group = [{"horse_no": str(i)} for i in range(1, 9)]
        self.first = {str(i): 1.0 / i for i in range(1, 9)}
        self.second = {str(i): 1.0 / (9 - i) for i in range(1, 9)}

    def test_six_ordered_tickets_and_no_self_pair(self):
        tickets = exacta.build_exacta_tickets(self.group, self.first, self.second)
        self.assertEqual(len(tickets), 6)
        self.assertTrue(all(len(t) == 2 and t[0] != t[1] for t in tickets))
        self.assertEqual(len({tuple(t) for t in tickets}), 6)

    def test_requires_complete_scores(self):
        with self.assertRaises(ValueError):
            exacta.build_exacta_tickets(self.group, {"1": 1.0}, self.second)


if __name__ == "__main__":
    unittest.main()

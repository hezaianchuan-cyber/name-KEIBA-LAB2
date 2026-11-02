import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nar_backtest as nb


class PayoutParsingTests(unittest.TestCase):
    def payout_row(self, winner="1", win_pay="250"):
        row = [""] * 53
        row[4], row[5] = winner, win_pay
        row[7], row[8] = "1", "120"
        row[10], row[11] = "2", "110"
        row[13], row[14] = "3", "130"
        row[24], row[25], row[26] = "1", "2", "400"
        row[28], row[29], row[30] = "1", "2", "600"
        row[32], row[33], row[34] = "1", "2", "200"
        row[36], row[37], row[38] = "1", "3", "250"
        row[40], row[41], row[42] = "2", "3", "300"
        row[44], row[45], row[46], row[47] = "1", "2", "3", "800"
        row[49], row[50], row[51], row[52] = "1", "2", "3", "1500"
        return row

    def test_dead_heat_rows_are_merged_and_all_bet_types_match(self):
        payouts = {}
        key = ("大井", "20261103", "10")
        nb.merge_payout_row(payouts, key, self.payout_row("1", "250"))
        nb.merge_payout_row(payouts, key, self.payout_row("2", "300"))
        payout = payouts[key]

        self.assertEqual(payout["source_rows"], 2)
        self.assertEqual(nb.payout_for("win", ["1"], payout), 250)
        self.assertEqual(nb.payout_for("win", ["2"], payout), 300)
        self.assertEqual(nb.payout_for("place", ["2"], payout), 110)
        self.assertEqual(nb.payout_for("quinella", ["2", "1"], payout), 400)
        self.assertEqual(nb.payout_for("exacta", ["1", "2"], payout), 600)
        self.assertEqual(nb.payout_for("exacta", ["2", "1"], payout), 0)
        self.assertEqual(nb.payout_for("wide", ["2", "1"], payout), 200)
        self.assertEqual(nb.payout_for("trio", ["3", "2", "1"], payout), 800)
        self.assertEqual(nb.payout_for("trifecta", ["1", "2", "3"], payout), 1500)
        self.assertEqual(nb.payout_for("trifecta", ["3", "2", "1"], payout), 0)

    def test_duplicate_payout_combinations_are_not_duplicated(self):
        payouts = {}
        key = ("大井", "20261103", "10")
        row = self.payout_row()
        nb.merge_payout_row(payouts, key, row)
        nb.merge_payout_row(payouts, key, row)
        self.assertEqual(len(payouts[key]["win"]), 1)
        self.assertEqual(payouts[key]["source_rows"], 2)


if __name__ == "__main__":
    unittest.main()

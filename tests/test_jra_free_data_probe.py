import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from jra_free_data_probe import parse_body_weight, parse_horse_name, parse_result


class JRAFreeDataProbeTests(unittest.TestCase):
    def test_extracts_runner_and_trifecta_payout(self):
        cells = [
            "1", "枠4青", "7", "ゴッドブルービー", "牡6", "58.0",
            "角田 大和", "1:43.3", "", "7 6 6 4", "36.7",
            "486(+4)", "角田 晃一", "3",
        ]
        row = "".join(f"<td>{cell}</td>" for cell in cells)
        html = f"""
        <html><body>
          <h1>レース結果</h1>
          <p>2026年11月15日（日曜） 3回福島4日 発走時刻：15時11分</p>
          <p>天候雨 ダート稍重</p>
          <h2>11レース 福島民友カップ</h2>
          <p>コース：1,700メートル（ダート・右）</p>
          <table><tr>{row}</tr></table>
          <section>3連複 4-7-15 73,440円 208番人気</section>
        </body></html>
        """
        report = parse_result(html, "https://www.jra.go.jp/sample")
        self.assertEqual(report["runner_count"], 1)
        self.assertEqual(report["runners"][0]["horse_name"], "ゴッドブルービー")
        self.assertEqual(report["runners"][0]["win_popularity"], 3)
        self.assertEqual(report["runners"][0]["frame_number"], 4)
        self.assertEqual(report["runners"][0]["body_weight_kg"], 486)
        self.assertEqual(report["runners"][0]["body_weight_change_kg"], 4)
        self.assertEqual(report["race"]["venue"], "福島")
        self.assertEqual(report["race"]["race_number"], 11)
        self.assertEqual(report["race"]["distance_m"], 1700)
        self.assertEqual(report["trifecta_payout"]["combination"], "4-7-15")
        self.assertEqual(report["trifecta_payout"]["payout_yen_per_100"], 73440)

    def test_parses_trifecta_with_single_digit_horse_numbers(self):
        cells = [
            "1", "枠1白", "2", "テストホース", "牡4", "57.0",
            "騎手名", "1:35.0", "", "1 1", "35.0",
            "480(+0)", "調教師名", "1",
        ]
        row = "".join(f"<td>{cell}</td>" for cell in cells)
        html = f"""
        <p>2026年11月15日（日曜） 3回京都4日</p>
        <p>1レース テスト</p>
        <p>コース：1,600メートル（芝・右）</p>
        <table><tr>{row}</tr></table>
        <p>3連複 2-3-9 220円 10番人気</p>
        """
        report = parse_result(html, "https://www.jra.go.jp/sample")
        self.assertEqual(report["trifecta_payout"]["combination"], "2-3-9")
        self.assertEqual(report["trifecta_payout"]["payout_yen_per_100"], 220)
        self.assertEqual(report["trifecta_payout"]["popularity"], 10)

    def test_normalizes_foreign_bred_marker_and_equipment(self):
        raw, name, equipment, is_foreign = parse_horse_name(
            "マルガイ シンビリーブ ブリンカー着用"
        )
        self.assertEqual(raw, "マルガイ シンビリーブ ブリンカー着用")
        self.assertEqual(name, "シンビリーブ")
        self.assertEqual(equipment, ["ブリンカー着用"])
        self.assertTrue(is_foreign)

    def test_parses_body_weight_and_change(self):
        self.assertEqual(parse_body_weight("486(+4)"), (486, 4))
        self.assertEqual(parse_body_weight("unknown"), (None, None))


if __name__ == "__main__":
    unittest.main()

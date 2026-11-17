import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from jra_free_data_probe import parse_result


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
        self.assertEqual(report["runners"][0]["win_popularity"], "3")
        self.assertEqual(report["race"]["venue"], "福島")
        self.assertEqual(report["race"]["race_number"], 11)
        self.assertEqual(report["trifecta_payout"]["combination"], "4-7-15")
        self.assertEqual(report["trifecta_payout"]["payout_yen_per_100"], 73440)


if __name__ == "__main__":
    unittest.main()

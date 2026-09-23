"""The regime label of a bar must not depend on the bars after it (bug B3).

Viterbi decoding and the segment smoothing both looked ahead: the label of bar
t changed with the bars that followed, so training labels (whole series) and
live labels (a 600-bar window ending at t) differed on ~7% of the bars.
"""
import unittest

import numpy as np
import pandas as pd

from feature_engineering.crypto_regime_detector import CryptoRegimeDetector


def _market(rows: int = 1500, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.choice([-0.002, 0.0, 0.002], size=rows // 100 + 1), 100)[:rows]
    close = 30000 * np.exp(np.cumsum(drift + rng.normal(0, 0.004, rows)))
    spread = np.abs(rng.normal(0, 0.003, rows)) * close
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.001, rows)),
        "high": close + spread,
        "low": close - spread,
        "close": close,
        "volume": rng.lognormal(3, 0.5, rows),
    }, index=pd.date_range("2025-01-01", periods=rows, freq="15min"))


class RegimeCausalityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = _market()
        cls.detector = CryptoRegimeDetector()
        cls.detector.fit_predict(cls.df.iloc[:1000])
        cls.full = cls.detector.predict(cls.df)

    def test_label_of_a_bar_ignores_later_bars(self):
        for t in (650, 900, 1200, 1499):
            prefix = self.detector.predict(self.df.iloc[:t + 1])
            self.assertEqual(prefix["regime"].iloc[-1], self.full["regime"].iloc[t], "bar %d" % t)
            self.assertAlmostEqual(prefix["confidence"].iloc[-1], self.full["confidence"].iloc[t], places=9)

    def test_live_window_matches_whole_series(self):
        for t in (900, 1200, 1499):
            window = self.detector.predict(self.df.iloc[t - 600:t + 1])
            self.assertEqual(window["regime"].iloc[-1], self.full["regime"].iloc[t], "bar %d" % t)


if __name__ == "__main__":
    unittest.main()

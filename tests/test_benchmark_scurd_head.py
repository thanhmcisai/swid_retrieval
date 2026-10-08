import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from swid_retrieval.benchmark_scurd_head import benchmark_case


class ScurdHeadBenchmarkTest(unittest.TestCase):
    def test_precomputed_episodes_match_online_cpu_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"ROOT_PATH": tmp, "RESULTS_DIR": tmp,
                                      "OUT_DIR": str(Path(tmp) / "logs")}, clear=False):
                engine = importlib.import_module(
                    "swid_retrieval._engines.variance_retrieval_evidence_colab")
        rng = np.random.RandomState(9)
        weak = rng.randn(160, 32).astype(np.float32)
        strong = rng.randn(160, 32).astype(np.float32)
        labels = np.repeat(np.arange(16), 10)
        online = benchmark_case(engine, weak, strong, labels, "cpu", False, 2, 1)
        precomputed = benchmark_case(engine, weak, strong, labels, "cpu", True, 2, 1)
        self.assertEqual(online["mean_loss"], precomputed["mean_loss"])
        self.assertEqual(online["steps"], 2)
        self.assertGreater(online["ms_per_episode"], 0)
        self.assertGreater(precomputed["precompute_s"], 0)


if __name__ == "__main__":
    unittest.main()

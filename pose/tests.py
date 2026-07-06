import numpy as np
from django.test import SimpleTestCase

from .ai_utils import fuse_features


class FuseFeaturesTests(SimpleTestCase):
    def test_fuse_features_returns_expected_input_size(self):
        front_features = np.zeros(156, dtype=np.float32)
        side_features = np.ones(156, dtype=np.float32)

        fused_features, _ = fuse_features(
            front_features,
            side_features,
            {"category": "male", "confidence": 0.8, "ratios": {}},
            {"category": "female", "confidence": 0.9, "ratios": {}},
        )

        self.assertEqual(fused_features.shape[0], 292)

"""Cross-framework numerical parity for positional encoding.

Positional encoding has no learned parameters, so — unlike attention weights
(Day 6) — it's possible to compare PyTorch's and TensorFlow's output
directly without loading matching weights into both. This runs only if both
frameworks happen to be importable in the current environment; otherwise it
is skipped rather than failing whichever single-framework venv is active.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
tf = pytest.importorskip("tensorflow")

from tf_impl.model.embeddings import PositionalEncoding as TFPositionalEncoding
from torch_impl.model.embeddings import PositionalEncoding as TorchPositionalEncoding


def test_positional_encoding_matches_across_frameworks():
    d_model, max_len = 16, 32
    torch_pe = TorchPositionalEncoding(d_model, max_len=max_len, dropout=0.0).eval()
    tf_pe = TFPositionalEncoding(d_model, max_len=max_len, dropout=0.0)

    x_np = np.zeros((1, max_len, d_model), dtype=np.float32)
    with torch.no_grad():
        torch_out = torch_pe(torch.from_numpy(x_np)).numpy()
    tf_out = tf_pe(tf.constant(x_np), training=False).numpy()

    np.testing.assert_allclose(torch_out, tf_out, atol=1e-5)

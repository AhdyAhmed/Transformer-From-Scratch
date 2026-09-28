"""Validation behaviour of TransformerConfig, checked for both frameworks."""

import pytest

from tf_impl.config import TransformerConfig as TFConfig
from torch_impl.config import TransformerConfig as TorchConfig

CONFIGS = [TorchConfig, TFConfig]


@pytest.mark.parametrize("Config", CONFIGS)
def test_defaults_are_valid(Config):
    cfg = Config()
    assert cfg.d_k == cfg.d_model // cfg.num_heads


@pytest.mark.parametrize("Config", CONFIGS)
def test_d_model_must_divide_by_heads(Config):
    with pytest.raises(ValueError, match="divisible"):
        Config(d_model=250, num_heads=8)


@pytest.mark.parametrize("Config", CONFIGS)
@pytest.mark.parametrize("bad", [-0.1, 1.0, 2.0])
def test_dropout_range(Config, bad):
    with pytest.raises(ValueError, match="dropout"):
        Config(dropout=bad)


@pytest.mark.parametrize("Config", CONFIGS)
def test_label_smoothing_range(Config):
    with pytest.raises(ValueError, match="label_smoothing"):
        Config(label_smoothing=1.5)


@pytest.mark.parametrize("Config", CONFIGS)
def test_shared_embeddings_need_shared_vocab(Config):
    with pytest.raises(ValueError, match="share_embeddings"):
        Config(share_embeddings=True, src_vocab_size=100, tgt_vocab_size=200)


@pytest.mark.parametrize("Config", CONFIGS)
def test_paper_base_values(Config):
    cfg = Config.paper_base()
    assert (cfg.d_model, cfg.num_heads, cfg.num_layers, cfg.d_ff) == (512, 8, 6, 2048)
    assert cfg.d_k == 64


@pytest.mark.parametrize("Config", CONFIGS)
def test_save_load_roundtrip(Config, tmp_path):
    cfg = Config(d_model=128, num_heads=4, num_layers=2)
    path = tmp_path / "cfg.json"
    cfg.save(path)
    assert Config.load(path) == cfg

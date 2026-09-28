"""Framework-agnostic tests: the two configs must never drift apart.

Both config modules are stdlib-only, so this file runs in either the torch
or the tensorflow virtualenv (or a bare Python install).
"""

import json
from dataclasses import fields

import pytest

from tf_impl.config import TransformerConfig as TFConfig
from torch_impl.config import TransformerConfig as TorchConfig
import tf_impl.config as tf_cfg
import torch_impl.config as torch_cfg


def test_same_field_names_and_defaults():
    torch_fields = {f.name: f.default for f in fields(TorchConfig)}
    tf_fields = {f.name: f.default for f in fields(TFConfig)}
    assert torch_fields == tf_fields


def test_same_special_token_ids():
    for name in ("PAD_IDX", "BOS_IDX", "EOS_IDX", "UNK_IDX"):
        assert getattr(torch_cfg, name) == getattr(tf_cfg, name)


def test_same_paper_base():
    assert TorchConfig.paper_base().to_dict() == TFConfig.paper_base().to_dict()


def test_json_roundtrip_is_cross_compatible(tmp_path):
    """A config saved by one framework loads in the other."""
    path = tmp_path / "cfg.json"
    TorchConfig(d_model=128, num_heads=4).save(path)
    loaded = TFConfig.load(path)
    assert loaded.d_model == 128 and loaded.num_heads == 4
    assert json.loads(path.read_text()) == loaded.to_dict()

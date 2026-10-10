"""Day 17 tests for the PyTorch training loop and checkpointing."""
import torch

from data.bpe_tokenizer import BPETokenizer
from data.parallel_data import ParallelDataset
from torch_impl.config import TransformerConfig
from torch_impl.data.dataset import TorchBatchLoader
from torch_impl.model.transformer import Transformer
from torch_impl.training.losses import LabelSmoothingLoss
from torch_impl.training.lr_schedule import NoamSchedule
from torch_impl.training.train import load_checkpoint, run_epoch, save_checkpoint


def _fixture(tmp_path):
    tokenizer = BPETokenizer()
    tokenizer.train(["a man runs", "a woman walks", "ein mann rennt", "eine frau geht"], 48)
    tok_path = tmp_path / "tokenizer.json"
    tokenizer.save(tok_path)
    pairs = [("a man runs", "ein mann rennt"), ("a woman walks", "eine frau geht")]
    dataset = ParallelDataset(pairs, tokenizer, max_len=12)
    loader = TorchBatchLoader(dataset, batch_size=2, shuffle=False)
    config = TransformerConfig(d_model=16, num_heads=4, num_layers=1, d_ff=32,
                              max_seq_len=12, dropout=0.0,
                              src_vocab_size=tokenizer.vocab_size,
                              tgt_vocab_size=tokenizer.vocab_size,
                              warmup_steps=4, batch_size=2)
    return tokenizer, tok_path, loader, config


def test_run_epoch_updates_weights_and_returns_finite_metrics(tmp_path):
    _, _, loader, config = _fixture(tmp_path)
    model = Transformer.from_config(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    before = model.output_projection.weight.detach().clone()
    loss_fn = LabelSmoothingLoss(config.label_smoothing, config.pad_idx)
    loss, step, tokens_per_sec, elapsed = run_epoch(
        model, loader, loss_fn, torch.device("cpu"), optimizer,
        NoamSchedule(config.d_model, config.warmup_steps), 0, config.grad_clip)
    assert step == 1
    assert torch.isfinite(torch.tensor(loss))
    assert tokens_per_sec > 0 and elapsed > 0
    assert not torch.equal(before, model.output_projection.weight.detach())


def test_checkpoint_roundtrip_restores_model_optimizer_and_progress(tmp_path):
    _, tok_path, loader, config = _fixture(tmp_path)
    model = Transformer.from_config(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Initialize optimizer state with one real update.
    run_epoch(model, loader, LabelSmoothingLoss(config.label_smoothing, config.pad_idx),
              torch.device("cpu"), optimizer, NoamSchedule(config.d_model, 4), 0)
    path = tmp_path / "latest.pt"
    save_checkpoint(path, model, optimizer, config, 2, 7, 1.25, tok_path)
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    restored = Transformer.from_config(config)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.001)
    payload = load_checkpoint(path, restored, restored_optimizer, torch.device("cpu"))
    assert payload["epoch"] == 2 and payload["global_step"] == 7
    assert payload["best_val_loss"] == 1.25
    for key, value in restored.state_dict().items():
        assert torch.equal(value, expected[key])
    assert restored_optimizer.state_dict()["state"]

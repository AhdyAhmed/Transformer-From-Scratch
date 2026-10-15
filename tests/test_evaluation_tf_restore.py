import numpy as np
import tensorflow as tf

from evaluation.evaluate import _load_tf_model
from tf_impl.config import TransformerConfig
from tf_impl.model.transformer import Transformer
from tf_impl.training.lr_schedule import NoamSchedule


def test_full_tensorflow_checkpoint_graph_restores_model_and_optimizer(tmp_path):
    config = TransformerConfig(
        d_model=8,
        num_heads=2,
        num_layers=1,
        d_ff=16,
        dropout=0.0,
        max_seq_len=8,
        src_vocab_size=16,
        tgt_vocab_size=16,
        warmup_steps=4,
        batch_size=2,
    )
    config_path = tmp_path / "config.json"
    config.save(config_path)
    model = Transformer.from_config(config)
    dummy = tf.constant([[config.bos_idx, config.eos_idx]], dtype=tf.int32)
    masks = model.create_masks(dummy, dummy)
    model(dummy, dummy, *masks, training=False)
    for index, variable in enumerate(model.trainable_variables):
        variable.assign(tf.ones_like(variable) * (index + 1) / 100.0)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=NoamSchedule(config.d_model, config.warmup_steps),
        beta_1=config.adam_beta1,
        beta_2=config.adam_beta2,
        epsilon=config.adam_eps,
    )
    optimizer.build(model.trainable_variables)
    optimizer.iterations.assign(17)
    epoch = tf.Variable(3, dtype=tf.int64, trainable=False, name="epoch")
    best = tf.Variable(1.25, dtype=tf.float32, trainable=False, name="best_val_loss")
    checkpoint = tf.train.Checkpoint(
        model=model, optimizer=optimizer, epoch=epoch, best_val_loss=best
    )
    manager = tf.train.CheckpointManager(checkpoint, str(tmp_path / "best"), max_to_keep=1)
    manager.save(checkpoint_number=3)
    expected = [variable.numpy().copy() for variable in model.trainable_variables]

    restored, _, metadata, restore_context = _load_tf_model(
        config_path, tmp_path / "best"
    )

    assert metadata == {"epoch": 3, "step": 17, "best_validation_loss": 1.25}
    assert all(
        np.array_equal(left, right.numpy())
        for left, right in zip(expected, restored.trainable_variables)
    )
    assert restore_context is not None

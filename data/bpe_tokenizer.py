"""A from-scratch byte-pair-encoding (BPE) tokenizer — Day 14.

Shared by both frameworks (``torch_impl/data/dataset.py`` and
``tf_impl/data/dataset.py``, Day 15) rather than duplicated per framework.
Unlike model code — where each framework needs its own from-scratch
implementation to demonstrate understanding of *that framework* — a
tokenizer has no torch/TensorFlow dependency at all, and training it twice
independently would only risk the two frameworks silently training on
different vocabularies: the exact kind of drift ``tests/test_config_parity.py``
(Day 2) exists to prevent for hyperparameters. One tokenizer, one trained
vocab file, both frameworks load the same thing.

Implements the standard BPE algorithm (Sennrich et al., 2016) from scratch —
no ``tokenizers``/``sentencepiece`` dependency — consistent with this repo's
"from scratch" scope extending to more than just the model architecture.

Algorithm summary:
    1. Pre-tokenize text into words (regex: runs of word characters, or a
       single non-word/non-space character).
    2. Represent each word as a list of characters, with an end-of-word
       marker appended to the last character — so "er" at a word boundary
       and "er" mid-word can be learned as distinct subwords.
    3. Repeatedly find the most frequent adjacent symbol pair across the
       whole corpus and merge it into a new symbol, recording the merge
       rule, until ``vocab_size`` is reached.
    4. Encoding applies the learned merge rules, in the order they were
       learned, to new text.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<pad>", "<bos>", "<eos>", "<unk>"
# Order matters: this must match torch_impl/config.py's PAD_IDX=0, BOS_IDX=1,
# EOS_IDX=2, UNK_IDX=3 exactly, since both the model and the tokenizer need
# to agree on what id 0 means. Checked directly in tests/test_bpe_tokenizer.py.
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

END_OF_WORD = "</w>"
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _pretokenize(text: str) -> list[str]:
    """Split text into word/punctuation tokens, lowercased for a smaller,
    more learnable vocabulary at the toy corpus sizes this repo trains on.
    (A case-preserving variant would be a straightforward extension.)
    """
    return _WORD_RE.findall(text.lower())


def _word_to_symbols(word: str) -> list[str]:
    """'hello' -> ['h', 'e', 'l', 'l', 'o</w>']"""
    if not word:
        return []
    chars = list(word)
    chars[-1] = chars[-1] + END_OF_WORD
    return chars


class BPETokenizer:
    """Byte-pair-encoding tokenizer: ``train()`` once on a corpus, then
    ``encode()``/``decode()`` text against the learned vocabulary.
    """

    def __init__(self) -> None:
        self.merges: list[tuple[str, str]] = []  # learned merge rules, in learned order
        self.vocab: dict[str, int] = {}  # symbol -> id
        self.inverse_vocab: dict[int, str] = {}  # id -> symbol

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, corpus: list[str], vocab_size: int) -> None:
        """Learn merge rules and build the vocabulary from a list of texts.

        Deterministic: ties in pair frequency are broken by lexicographic
        order on the pair itself, so training twice on the same corpus with
        the same ``vocab_size`` always produces an identical vocabulary and
        merge list — verified directly in ``tests/test_bpe_tokenizer.py``.
        This determinism is what lets both frameworks train independently
        (Day 15) and still end up with byte-identical vocabularies.
        """
        if vocab_size < len(SPECIAL_TOKENS):
            raise ValueError(
                f"vocab_size ({vocab_size}) must be at least {len(SPECIAL_TOKENS)} "
                "to fit the special tokens alone"
            )

        word_freqs: Counter[tuple[str, ...]] = Counter()
        for text in corpus:
            for word in _pretokenize(text):
                word_freqs[tuple(_word_to_symbols(word))] += 1

        # Base vocabulary: every symbol (character, possibly end-of-word-marked)
        # that appears anywhere in the corpus.
        base_symbols = sorted({sym for word in word_freqs for sym in word})

        self.merges = []
        words: dict[tuple[str, ...], int] = dict(word_freqs)

        while len(SPECIAL_TOKENS) + len(base_symbols) + len(self.merges) < vocab_size:
            pair_counts = self._count_pairs(words)
            if not pair_counts:
                break  # corpus exhausted: no more adjacent pairs left to merge
            best_pair = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            words = self._merge_pair(words, best_pair)
            self.merges.append(best_pair)

        merged_symbols = ["".join(pair) for pair in self.merges]
        all_symbols = SPECIAL_TOKENS + base_symbols + merged_symbols
        self.vocab = {sym: idx for idx, sym in enumerate(all_symbols)}
        self.inverse_vocab = {idx: sym for sym, idx in self.vocab.items()}

    @staticmethod
    def _count_pairs(words: dict[tuple[str, ...], int]) -> Counter[tuple[str, str]]:
        pair_counts: Counter[tuple[str, str]] = Counter()
        for symbols, freq in words.items():
            for a, b in zip(symbols, symbols[1:]):
                pair_counts[(a, b)] += freq
        return pair_counts

    @staticmethod
    def _merge_pair(
        words: dict[tuple[str, ...], int], pair: tuple[str, str]
    ) -> dict[tuple[str, ...], int]:
        merged = "".join(pair)
        new_words: dict[tuple[str, ...], int] = {}
        for symbols, freq in words.items():
            new_symbols: list[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            key = tuple(new_symbols)
            new_words[key] = new_words.get(key, 0) + freq
        return new_words

    # ------------------------------------------------------------------ #
    # Encoding / decoding
    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _apply_merges(self, symbols: list[str]) -> list[str]:
        for pair in self.merges:
            if len(symbols) < 2:
                break
            merged = "".join(pair)
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """text -> list of token ids. Symbols the trained vocab never saw
        (an unseen character, for instance) fall back to ``<unk>``."""
        ids: list[int] = []
        unk_id = self.vocab[UNK_TOKEN]
        for word in _pretokenize(text):
            symbols = self._apply_merges(_word_to_symbols(word))
            ids.extend(self.vocab.get(sym, unk_id) for sym in symbols)
        if add_special_tokens:
            ids = [self.vocab[BOS_TOKEN], *ids, self.vocab[EOS_TOKEN]]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """list of token ids -> text. Lossy by construction: lowercased,
        and whitespace is normalized to single spaces between words, since
        that's all the pre-tokenizer preserved about the original text in
        the first place."""
        symbols = [self.inverse_vocab.get(i, UNK_TOKEN) for i in ids]
        if skip_special_tokens:
            symbols = [s for s in symbols if s not in SPECIAL_TOKENS]
        text = "".join(symbols)
        return text.replace(END_OF_WORD, " ").strip()

    def encode_batch(self, texts: list[str], add_special_tokens: bool = False) -> list[list[int]]:
        return [self.encode(t, add_special_tokens) for t in texts]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        data = {"vocab": self.vocab, "merges": self.merges}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text())
        tok = cls()
        tok.vocab = {k: int(v) for k, v in data["vocab"].items()}
        tok.inverse_vocab = {v: k for k, v in tok.vocab.items()}
        tok.merges = [tuple(pair) for pair in data["merges"]]
        return tok

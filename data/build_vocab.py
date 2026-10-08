"""Train the shared BPE tokenizer and save it to disk — Day 14.

Usage:
    python -m data.build_vocab --vocab-size 2000 --out data/tokenizer.json
    python -m data.build_vocab --files corpus_en.txt corpus_de.txt --vocab-size 8000 --out data/tokenizer.json

With no ``--files`` given, trains on a small built-in toy English/German
corpus, so the tokenizer (and anything built on top of it) can be exercised
without needing the real Multi30k dataset downloaded yet (Day 15).

Both frameworks' data loaders will point at the same ``--out`` file, so this
script is run exactly once per corpus/vocab-size choice, not once per
framework — see ``data/bpe_tokenizer.py`` for why.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data.bpe_tokenizer import BPETokenizer

_TOY_CORPUS = [
    "a man in a red shirt is climbing a rock",
    "two young children are playing in the park",
    "a dog runs across the green field",
    "the woman is reading a book on the bench",
    "a group of people are standing near the water",
    "ein mann in einem roten hemd klettert auf einen felsen",
    "zwei kleine kinder spielen im park",
    "ein hund rennt ueber das gruene feld",
    "die frau liest ein buch auf der bank",
    "eine gruppe von menschen steht in der naehe des wassers",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--files", nargs="*", default=None, help="plain-text files, one sentence per line"
    )
    parser.add_argument("--vocab-size", type=int, default=2000)
    parser.add_argument("--out", default="data/tokenizer.json")
    args = parser.parse_args()

    if args.files:
        corpus: list[str] = []
        for path in args.files:
            corpus.extend(Path(path).read_text(encoding="utf-8").splitlines())
    else:
        print("No --files given; training on the small built-in toy corpus.")
        corpus = _TOY_CORPUS

    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=args.vocab_size)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_path)
    print(f"Trained a {tokenizer.vocab_size}-token vocabulary from {len(corpus)} lines -> {out_path}")


if __name__ == "__main__":
    main()

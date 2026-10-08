"""Download Multi30k (English <-> German image captions) — Day 15.

Usage:
    python -m data.download_multi30k                 # -> data/multi30k/{train,val,test}.{en,de}
    python -m data.download_multi30k --out my_dir

Then build the shared vocabulary from *both* sides (a joint BPE vocab, so
``share_embeddings=True`` is possible), once, for both frameworks:

    python -m data.build_vocab \\
        --files data/multi30k/train.en data/multi30k/train.de \\
        --vocab-size 8000 --out data/multi30k/tokenizer.json

NOTE: the default ``--base-url`` points at the public multi30k/dataset
GitHub repository's raw task-1 files. That URL could not be reached from the
environment this script was written in (no network access), so the
download itself is unverified — if it 404s, download the six ``*.gz``
files from https://github.com/multi30k/dataset manually, gunzip them, and
save them as ``train.en``, ``train.de``, ``val.en``, ``val.de``,
``test.en``, ``test.de`` in the output directory. What *was* verified is
everything after the network call: ``fetch_all`` is tested against local
``file://`` URLs in ``tests/test_parallel_data.py``.

Multi30k has its own license terms (CC BY-NC-SA for the descriptions);
that is why the data is downloaded on demand and git-ignored rather than
committed to this repo.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"

# remote file stem -> local file stem
SPLITS = {
    "train": "train",
    "val": "val",
    "test_2016_flickr": "test",
}
LANGS = ("en", "de")


def _download_and_gunzip(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, gzip.GzipFile(fileobj=response) as gz:
        with open(dest, "wb") as out:
            shutil.copyfileobj(gz, out)


def fetch_all(base_url: str = DEFAULT_BASE_URL, out_dir: str | Path = "data/multi30k") -> list[Path]:
    """Download every split/language pair, returning the paths written."""
    out_dir = Path(out_dir)
    if not base_url.endswith("/"):
        base_url += "/"
    written: list[Path] = []
    for remote_stem, local_stem in SPLITS.items():
        for lang in LANGS:
            url = f"{base_url}{remote_stem}.{lang}.gz"
            dest = out_dir / f"{local_stem}.{lang}"
            print(f"  {url} -> {dest}")
            _download_and_gunzip(url, dest)
            written.append(dest)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--out", default="data/multi30k")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()

    print(f"Downloading Multi30k to {args.out} ...")
    written = fetch_all(args.base_url, args.out)
    print(f"Done: {len(written)} files.")


if __name__ == "__main__":
    main()

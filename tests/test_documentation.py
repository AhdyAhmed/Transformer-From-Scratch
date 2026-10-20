"""Regression checks for the portfolio documentation added on Day 27."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def test_all_local_markdown_links_resolve():
    broken: list[str] = []
    for document in ROOT.rglob("*.md"):
        for target in MARKDOWN_LINK.findall(document.read_text(encoding="utf-8")):
            target = target.strip().strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not broken, "broken local documentation links:\n" + "\n".join(broken)


def test_architecture_svg_is_valid_and_accessible():
    svg_path = ROOT / "assets" / "transformer-architecture.svg"
    root = ET.parse(svg_path).getroot()
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert root.attrib["role"] == "img"
    assert root.attrib["aria-labelledby"] == "title description"
    assert root.find("svg:title", namespace).text
    assert root.find("svg:desc", namespace).text
    visible_text = " ".join(
        node.text or "" for node in root.findall(".//svg:text", namespace)
    )
    for component in ("Encoder layer", "Decoder layer", "Memory", "Translation"):
        assert component in visible_text


def test_detailed_transformer_architecture_image_is_documented():
    image_path = ROOT / "assets" / "full-transformer-architecture.webp"
    image = image_path.read_bytes()
    assert image[:4] == b"RIFF"
    assert image[8:12] == b"WEBP"
    for document in (ROOT / "README.md", ROOT / "design.md"):
        assert "assets/full-transformer-architecture.webp" in document.read_text(
            encoding="utf-8"
        )


def test_root_results_table_matches_generated_metrics():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    day25 = json.loads(
        (ROOT / "results" / "day25" / "metrics.json").read_text(encoding="utf-8")
    )
    expected_values = (
        f"{day25['pytorch']['parameters']:,}",
        f"{day25['tensorflow']['parameters']:,}",
        f"{day25['pytorch']['training']['best_validation_loss']:.4f}",
        f"{day25['tensorflow']['training']['best_validation_loss']:.4f}",
        f"{day25['pytorch']['evaluation']['bleu']['score']:.4f}",
        f"{day25['tensorflow']['evaluation']['bleu']['score']:.4f}",
    )
    for value in expected_values:
        assert value in readme
    assert "not a controlled framework benchmark" in readme

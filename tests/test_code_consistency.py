"""Static guarantees for the mirrored PyTorch and TensorFlow source trees."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORKS = ("torch_impl", "tf_impl")


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _public_symbols(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and not node.name.startswith("_")
    }


def _public_methods(node: ast.ClassDef, *, tensorflow: bool) -> set[str]:
    ignored = {"get_config", "layers"} if tensorflow else set()
    methods = {
        child.name
        for child in node.body
        if isinstance(child, ast.FunctionDef)
        and not child.name.startswith("_")
        and child.name not in ignored
    }
    if not tensorflow and "forward" in methods:
        methods.remove("forward")
        methods.add("call")
    return methods


def test_model_modules_and_public_contracts_are_parallel() -> None:
    """Keep equivalent model files, symbols, and class operations aligned."""
    assert (ROOT / "torch_impl" / "config.py").read_bytes() == (
        ROOT / "tf_impl" / "config.py"
    ).read_bytes()
    torch_dir = ROOT / "torch_impl" / "model"
    tf_dir = ROOT / "tf_impl" / "model"
    torch_files = {path.name for path in torch_dir.glob("*.py")}
    tf_files = {path.name for path in tf_dir.glob("*.py")}
    assert torch_files == tf_files

    for filename in sorted(torch_files - {"__init__.py"}):
        torch_tree = _tree(torch_dir / filename)
        tf_tree = _tree(tf_dir / filename)
        assert _public_symbols(torch_tree) == _public_symbols(tf_tree), filename

        tf_classes = {
            node.name: node for node in tf_tree.body if isinstance(node, ast.ClassDef)
        }
        for torch_class in (
            node for node in torch_tree.body if isinstance(node, ast.ClassDef)
        ):
            assert _public_methods(torch_class, tensorflow=False) == _public_methods(
                tf_classes[torch_class.name], tensorflow=True
            ), f"{filename}:{torch_class.name}"


def test_public_implementation_apis_are_documented_and_typed() -> None:
    """Require docstrings and annotations on non-test public APIs."""
    failures: list[str] = []
    for framework in FRAMEWORKS:
        for path in (ROOT / framework).rglob("*.py"):
            if "tests" in path.parts:
                continue
            for node in ast.walk(_tree(path)):
                if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    continue
                if node.name.startswith("_"):
                    continue
                location = f"{path.relative_to(ROOT)}:{node.lineno}:{node.name}"
                if ast.get_docstring(node) is None:
                    failures.append(f"{location} missing docstring")
                if isinstance(node, ast.FunctionDef):
                    if node.returns is None:
                        failures.append(f"{location} missing return annotation")
                    args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
                    if args and args[0].arg in {"self", "cls"}:
                        args = args[1:]
                    for argument in args:
                        if argument.annotation is None:
                            failures.append(
                                f"{location} missing annotation for {argument.arg}"
                            )
    assert not failures, "\n".join(failures)

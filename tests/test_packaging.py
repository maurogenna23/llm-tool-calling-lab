"""Two dependency lists exist because Hugging Face Spaces ignores pyproject.toml.

Two lists drift. This is the only thing keeping them honest.
"""

from __future__ import annotations

import re
import tomllib

from assistant.config import ROOT


def _name(spec: str) -> str:
    return re.split(r"[><=\[!~]", spec)[0].strip().lower()


def _pyproject_dependencies() -> set[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return {_name(dep) for dep in data["project"]["dependencies"]}


def _requirements() -> set[str]:
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return {_name(line) for line in lines if line.strip() and not line.lstrip().startswith("#")}


def test_requirements_matches_pyproject() -> None:
    assert _requirements() == _pyproject_dependencies()


def test_the_entry_point_spaces_looks_for_exists() -> None:
    assert (ROOT / "app.py").is_file()


def test_env_example_documents_every_key_the_registry_needs() -> None:
    from assistant.config import MODELS

    documented = (ROOT / ".env.example").read_text(encoding="utf-8")
    for model in MODELS:
        if model.requires_env:
            assert model.requires_env in documented, f"{model.key} needs {model.requires_env}"


def test_secrets_are_not_committed() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in ignored
    assert "*.db" in ignored


def test_no_stray_database_files_in_the_repo() -> None:
    """A relative ARNIE_DB_PATH used to create one wherever you launched from."""
    strays = [path.name for path in ROOT.glob("*.db")]
    tracked = [name for name in strays if name != "arnie.db"]
    assert not tracked, f"unexpected database files at the project root: {tracked}"


def test_readme_is_not_a_stub() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert len(readme) > 2000
    for section in ("## Running it", "## Tests", "## Cost"):
        assert section in readme

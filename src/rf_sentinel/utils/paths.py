"""Path resolution and directory management utilities."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist.

    Parameters
    ----------
    path : str | Path
        Directory path to ensure exists.

    Returns
    -------
    Path
        The resolved directory path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_project_root() -> Path:
    """Get the project root directory (where pyproject.toml lives).

    Falls back to current working directory if pyproject.toml is not found
    by walking up the directory tree.
    """
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return current


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    """Resolve a path, making it absolute if relative.

    Parameters
    ----------
    path : str | Path
        Path to resolve.
    base : Path, optional
        Base directory for relative paths. Defaults to CWD.

    Returns
    -------
    Path
        Absolute resolved path.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    base = base or Path.cwd()
    return (base / p).resolve()


def get_checkpoint_dir(cfg: dict) -> Path:
    """Get the checkpoint directory from config."""
    output_dir = cfg.get("project", {}).get("output_dir", "artifacts")
    return ensure_dir(Path(output_dir) / "checkpoints")


def get_report_dir(cfg: dict) -> Path:
    """Get the report directory from config."""
    report_dir = cfg.get("project", {}).get("report_dir", "reports")
    return ensure_dir(Path(report_dir))


def get_figures_dir(cfg: dict) -> Path:
    """Get the figures directory from config."""
    report_dir = cfg.get("project", {}).get("report_dir", "reports")
    return ensure_dir(Path(report_dir) / "figures")

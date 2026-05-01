"""Logging utilities using Rich console."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()

_CONFIGURED = False


def get_logger(name: str = "rf_sentinel", level: int = logging.INFO) -> logging.Logger:
    """Get a Rich-configured logger.

    Parameters
    ----------
    name : str
        Logger name.
    level : int
        Logging level.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    global _CONFIGURED
    logger = logging.getLogger(name)

    if not _CONFIGURED:
        logger.setLevel(level)
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        handler.setLevel(level)
        fmt = logging.Formatter("%(message)s", datefmt="[%X]")
        handler.setFormatter(fmt)
        logger.addHandler(handler)

        # Prevent duplicate logs
        logger.propagate = False
        _CONFIGURED = True

    return logger


def print_header(title: str, subtitle: str = "") -> None:
    """Print a styled header to the console."""
    console.print()
    console.rule(f"[bold cyan]{title}[/bold cyan]")
    if subtitle:
        console.print(f"  [dim]{subtitle}[/dim]")
    console.print()


def print_success(msg: str) -> None:
    """Print a success message."""
    console.print(f"[bold green]OK[/bold green] {msg}")


def print_warning(msg: str) -> None:
    """Print a warning message."""
    console.print(f"[bold yellow]![/bold yellow] {msg}")


def print_error(msg: str) -> None:
    """Print an error message."""
    console.print(f"[bold red]X[/bold red] {msg}")


def print_info(msg: str) -> None:
    """Print an info message."""
    console.print(f"[bold blue]i[/bold blue] {msg}")

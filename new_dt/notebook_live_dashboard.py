from __future__ import annotations

from typing import Any

from rich.console import Console

from . import all_models_dashboard_cli as base


def _notebook_display_available() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ in {
            "ZMQInteractiveShell",
            "Shell",
        }
    except Exception:
        return False


def _render_html(renderable: Any) -> str:
    console = Console(
        record=True,
        force_terminal=False,
        color_system="truecolor",
        width=160,
    )
    console.print(renderable)
    return console.export_html(inline_styles=True, clear=True)


class NotebookSafeDashboard(base.Dashboard):
    """Keep one replaceable dashboard output in Jupyter/Colab."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._notebook_mode = _notebook_display_available()
        self._display_handle = None

    def start(self) -> None:
        if self.args.no_dashboard:
            return
        if not self._notebook_mode:
            super().start()
            return

        from IPython.display import HTML, display

        self._display_handle = display(
            HTML(_render_html(self.render())),
            display_id=True,
        )

    def update(self) -> None:
        if not self._notebook_mode:
            super().update()
            return
        if self.args.no_dashboard:
            return

        from IPython.display import HTML

        html = HTML(_render_html(self.render()))
        if self._display_handle is None:
            from IPython.display import display

            self._display_handle = display(html, display_id=True)
        else:
            self._display_handle.update(html)

    def snapshot(self, *, model, step: int) -> None:
        if not self._notebook_mode:
            super().snapshot(model=model, step=step)
            return

        from IPython.display import HTML, display
        from rich.panel import Panel

        title = f"Static snapshot · {base.MODEL_LABELS[model]} · step {step:,}"
        panel = Panel(self._table(), title=title, border_style="yellow")
        display(HTML(_render_html(panel)))

    def stop(self) -> None:
        if not self._notebook_mode:
            super().stop()
            return
        self.update()
        self._display_handle = None


def install_notebook_safe_dashboard() -> None:
    """Install before importing a runner that captures base.Dashboard."""

    base.Dashboard = NotebookSafeDashboard  # type: ignore[assignment]

from new_dt import all_models_dashboard_cli as base
from new_dt.notebook_live_dashboard import (
    NotebookSafeDashboard,
    install_notebook_safe_dashboard,
)


def test_notebook_dashboard_installs_on_base_runner(monkeypatch):
    original = base.Dashboard
    try:
        install_notebook_safe_dashboard()
        assert base.Dashboard is NotebookSafeDashboard
    finally:
        monkeypatch.setattr(base, "Dashboard", original)


def test_notebook_dashboard_is_base_dashboard_subclass():
    assert issubclass(NotebookSafeDashboard, base.Dashboard)

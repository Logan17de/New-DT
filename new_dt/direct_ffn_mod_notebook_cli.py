from __future__ import annotations

from .notebook_live_dashboard import install_notebook_safe_dashboard


def main(argv: list[str] | None = None) -> int:
    install_notebook_safe_dashboard()

    # Import only after installing the notebook-safe Dashboard, because the
    # dedicated runner captures the base Dashboard class at import time.
    from . import direct_ffn_mod_variants_training_cli as runner

    return runner.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

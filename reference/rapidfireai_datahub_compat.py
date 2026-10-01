"""Compatibility helpers for RapidFireAI on Datahub/JupyterHub.

RapidFireAI's notebook UI assumes the browser can directly reach the evals
dispatcher on ``http://127.0.0.1:8851``. That works for local notebooks, but
fails on hosted JupyterHub deployments such as UCSD Datahub because the browser
is running on the user's machine, not inside the notebook container.

This module patches RapidFireAI so the interactive controller talks to the
dispatcher through JupyterHub's authenticated port proxy instead.
"""

from __future__ import annotations

import os
from typing import Any


def is_jupyterhub_environment() -> bool:
    """Return True when running inside JupyterHub/Datahub."""
    return any(
        os.getenv(name)
        for name in (
            "JUPYTERHUB_USER",
            "JUPYTERHUB_SERVICE_PREFIX",
            "JUPYTERHUB_API_TOKEN",
        )
    )


def get_jupyterhub_proxy_base(port: int = 8851) -> str:
    """Return the same-origin proxy base path for a container port."""
    return f"/hub/user-redirect/proxy/{port}"


def datahub_dispatcher_diagnostics() -> dict[str, Any]:
    """Summarize the dispatcher URL RapidFireAI should use on this host."""
    from rapidfireai.utils.constants import DispatcherConfig

    return {
        "is_jupyterhub_environment": is_jupyterhub_environment(),
        "dispatcher_host": DispatcherConfig.HOST,
        "dispatcher_port": DispatcherConfig.PORT,
        "default_dispatcher_url": f"http://{DispatcherConfig.HOST}:{DispatcherConfig.PORT}",
        "recommended_proxy_url": get_jupyterhub_proxy_base(DispatcherConfig.PORT),
        "jupyterhub_user": os.getenv("JUPYTERHUB_USER"),
        "jupyterhub_service_prefix": os.getenv("JUPYTERHUB_SERVICE_PREFIX"),
    }


def patch_rapidfireai_for_datahub() -> dict[str, Any]:
    """Patch RapidFireAI's notebook UI to use JupyterHub proxy URLs.

    Call this once before constructing ``Experiment(..., mode="evals")``.
    """
    from rapidfireai.evals.utils import constants as eval_constants
    from rapidfireai.evals.utils.notebook_ui import NotebookUI
    from rapidfireai.utils.constants import DispatcherConfig

    if not is_jupyterhub_environment():
        return {
            "patched": False,
            "reason": "Not running inside JupyterHub/Datahub",
            **datahub_dispatcher_diagnostics(),
        }

    proxy_base = get_jupyterhub_proxy_base(DispatcherConfig.PORT)

    def _patched_get_dispatcher_url() -> str:
        return proxy_base

    original_init = NotebookUI.__init__
    original_generate_html = NotebookUI._generate_html

    def _patched_init(
        self,
        dispatcher_url: str = "http://127.0.0.1:8851",
        refresh_rate_seconds: float = 3.0,
        auth_token: str | None = None,
    ) -> None:
        if dispatcher_url.rstrip("/") == f"http://127.0.0.1:{DispatcherConfig.PORT}":
            dispatcher_url = proxy_base
        original_init(
            self,
            dispatcher_url=dispatcher_url,
            refresh_rate_seconds=refresh_rate_seconds,
            auth_token=auth_token,
        )

    def _patched_generate_html(self) -> str:
        html = original_generate_html(self)
        html = html.replace("credentials: 'omit'", "credentials: 'include'")
        html = html.replace("credentials: 'same-origin'", "credentials: 'include'")
        return html

    eval_constants.get_dispatcher_url = _patched_get_dispatcher_url
    NotebookUI.__init__ = _patched_init
    NotebookUI._generate_html = _patched_generate_html

    return {
        "patched": True,
        "dispatcher_proxy_url": proxy_base,
        **datahub_dispatcher_diagnostics(),
    }

"""Dispatch: model name -> (rf | numpy | torch) implementation."""
from __future__ import annotations

from ..utils import get_logger, set_seed
from .task import Task

log = get_logger(__name__)
_BACKEND_LOGGED = False


def resolve_backend(cfg: dict) -> str:
    pref = cfg["train"].get("backend", "auto")
    if pref == "auto":
        try:
            import torch  # noqa: F401
            return "torch"
        except ImportError:
            return "numpy"
    return pref


def run_model(name: str, task: Task, cfg: dict, seed: int) -> dict:
    """name in {'rf','mlp','sage'}; returns dict with prob_val, prob_test, n_params, hidden, epochs..."""
    global _BACKEND_LOGGED
    set_seed(seed)
    if name == "rf":
        from ..models.rf import fit_predict_rf
        return fit_predict_rf(task, cfg, seed)
    backend = resolve_backend(cfg)
    if not _BACKEND_LOGGED:
        log.info("Neural back-end: %s", backend)
        _BACKEND_LOGGED = True
    if backend == "torch":
        from ..models.nn_torch import fit_predict_torch
        return fit_predict_torch(name, task, cfg, seed)
    from ..models.nn_numpy import fit_predict_numpy
    return fit_predict_numpy(name, task, cfg, seed)

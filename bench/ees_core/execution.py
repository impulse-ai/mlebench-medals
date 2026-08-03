"""Execution wrappers for EES operator budgets and typed retries.

Operators run in a subprocess so a wall-clock budget can hard-kill them. The
subprocess start method matters for soundness:

- ``fork`` is fast but unsound on macOS once native libraries (torch, OpenMP,
  Accelerate, lightgbm) are loaded in the parent: forked children SIGSEGV or
  deadlock. The default there is ``spawn``.
- ``fork`` remains the default on Linux for startup performance.
- ``EES_OPERATOR_START_METHOD`` (``fork`` | ``spawn`` | ``forkserver``)
  overrides the platform default.

``spawn``/``forkserver`` require the operator invocation to be picklable, so
callers pass an :class:`OperatorCall` wrapping a module-level function and
picklable args. Bare callables remain accepted for backward compatibility, but
only work under ``fork`` unless they are themselves picklable. An
``OperatorCall`` whose payload is not picklable (test seams inject in-memory
fakes/backbones) falls back to ``fork`` with a ``RuntimeWarning``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import multiprocessing as mp
import os
import pickle
import queue
import sys
import time
import warnings
from typing import Any, Callable

from bench.ees_core.candidates import CandidateResult

_START_METHODS = ("fork", "spawn", "forkserver")


@dataclass(frozen=True)
class OperatorBudget:
    seconds: float
    phase: str = "operator_wall_clock"


@dataclass(frozen=True)
class OperatorCall:
    """Picklable operator invocation spec.

    ``func`` must be a module-level function and ``args``/``kwargs`` must be
    picklable (Paths, strings, numbers, bools, dataclasses of those) for the
    call to run under the ``spawn``/``forkserver`` start methods. ``env``
    entries are applied to ``os.environ`` for the duration of the call and
    restored afterwards (restoration matters for the in-process path; in a
    subprocess the mutation dies with the child).
    """

    func: Callable[..., CandidateResult]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)

    def __call__(self) -> CandidateResult:
        old_values = {key: os.environ.get(key) for key in self.env}
        try:
            os.environ.update(self.env)
            return self.func(*self.args, **self.kwargs)
        finally:
            for key, old_value in old_values.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value


def _configured_start_method() -> str:
    override = os.environ.get("EES_OPERATOR_START_METHOD", "").strip().lower()
    if override:
        if override not in _START_METHODS:
            raise ValueError(
                f"EES_OPERATOR_START_METHOD must be one of {_START_METHODS}, got {override!r}"
            )
        return override
    return "spawn" if sys.platform == "darwin" else "fork"


def _resolve_start_method(fn: "OperatorCall | Callable[[], CandidateResult]") -> str:
    method = _configured_start_method()
    if method == "fork":
        return method
    try:
        pickle.dumps(fn)
        return method
    except Exception as exc:
        if isinstance(fn, OperatorCall):
            func_name = getattr(fn.func, "__name__", repr(fn.func))
            warnings.warn(
                f"OperatorCall({func_name}) is not picklable ({exc}); falling back to the "
                "'fork' start method for this operator. This is expected for test seams "
                "that inject in-memory objects; production operator calls must stay "
                "picklable so they run under 'spawn' on macOS.",
                RuntimeWarning,
                stacklevel=3,
            )
            return "fork"
        raise TypeError(
            f"operator callable {fn!r} is not picklable and cannot run under the "
            f"'{method}' start method; pass an OperatorCall wrapping a module-level "
            "function with picklable args (or set EES_OPERATOR_START_METHOD=fork)."
        ) from exc


def _with_execution_metrics(
    candidate: CandidateResult,
    *,
    attempts: int,
    elapsed_seconds: float,
    budget: OperatorBudget,
    retry_reasons: list[str],
) -> CandidateResult:
    metrics = dict(candidate.metrics)
    timed_out = elapsed_seconds >= budget.seconds
    metrics["execution"] = {
        "attempts": attempts,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "budget_seconds": budget.seconds,
        "timed_out": timed_out,
        "timeout_phase": budget.phase if timed_out else None,
        "retry_reasons": retry_reasons,
    }
    return CandidateResult(
        candidate_id=candidate.candidate_id,
        recipe_id=candidate.recipe_id,
        success=candidate.success,
        submission_path=candidate.submission_path,
        validation=candidate.validation,
        score=candidate.score,
        medal=candidate.medal,
        final_artifact_source=candidate.final_artifact_source,
        artifact_paths=candidate.artifact_paths,
        metrics=metrics,
        no_score_reason=candidate.no_score_reason,
        score_direction=candidate.score_direction,
    )


def _exception_candidate(operator_id: str, recipe_id: str, exc: Exception) -> CandidateResult:
    return CandidateResult(
        candidate_id=operator_id,
        recipe_id=recipe_id,
        success=False,
        submission_path=None,
        validation={},
        score=None,
        medal="none",
        final_artifact_source=f"ees_core:{operator_id}",
        no_score_reason=f"{type(exc).__name__}: {exc}",
    )


def _timeout_candidate(operator_id: str, recipe_id: str) -> CandidateResult:
    return CandidateResult(
        candidate_id=operator_id,
        recipe_id=recipe_id,
        success=False,
        submission_path=None,
        validation={},
        score=None,
        medal="none",
        final_artifact_source=f"ees_core:{operator_id}",
        no_score_reason="operator_timeout",
    )


def _operator_worker(fn: "OperatorCall | Callable[[], CandidateResult]", result_queue) -> None:
    try:
        result_queue.put(("candidate", fn()))
    except Exception as exc:
        result_queue.put(("exception", type(exc).__name__, str(exc)))


def _run_once_with_timeout(
    *,
    operator_id: str,
    recipe_id: str,
    fn: "OperatorCall | Callable[[], CandidateResult]",
    timeout_seconds: float,
) -> CandidateResult:
    if timeout_seconds <= 0:
        try:
            return fn()
        except Exception as exc:
            return _exception_candidate(operator_id, recipe_id, exc)
    try:
        context = mp.get_context(_resolve_start_method(fn))
    except ValueError:
        context = mp.get_context()
    result_queue = context.Queue(maxsize=1)
    process = context.Process(target=_operator_worker, args=(fn, result_queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(1)
        if process.is_alive():
            process.kill()
            process.join(1)
        return _timeout_candidate(operator_id, recipe_id)
    try:
        payload = result_queue.get_nowait()
    except queue.Empty:
        if process.exitcode == 0:
            return _exception_candidate(operator_id, recipe_id, RuntimeError("operator returned no result"))
        return _exception_candidate(operator_id, recipe_id, RuntimeError(f"operator process exited with {process.exitcode}"))
    if payload[0] == "candidate":
        return payload[1]
    return _exception_candidate(operator_id, recipe_id, RuntimeError(f"{payload[1]}: {payload[2]}"))


def run_operator_with_retries(
    *,
    operator_id: str,
    recipe_id: str,
    fn: "OperatorCall | Callable[[], CandidateResult]",
    budget: OperatorBudget,
    max_attempts: int = 1,
) -> CandidateResult:
    started = time.time()
    retry_reasons: list[str] = []
    last: CandidateResult | None = None
    attempts = max(1, max_attempts)
    for attempt in range(1, attempts + 1):
        remaining = budget.seconds - (time.time() - started)
        candidate = _run_once_with_timeout(
            operator_id=operator_id,
            recipe_id=recipe_id,
            fn=fn,
            timeout_seconds=remaining if budget.seconds > 0 else 0,
        )
        last = candidate
        if candidate.success and candidate.submission_path is not None:
            return _with_execution_metrics(
                candidate,
                attempts=attempt,
                elapsed_seconds=time.time() - started,
                budget=budget,
                retry_reasons=retry_reasons,
            )
        reason = candidate.no_score_reason or "failed"
        retry_reasons.append(reason)
        if reason == "operator_timeout":
            break
    assert last is not None
    return _with_execution_metrics(
        last,
        attempts=attempts,
        elapsed_seconds=time.time() - started,
        budget=budget,
        retry_reasons=retry_reasons,
    )

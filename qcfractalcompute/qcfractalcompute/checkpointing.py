"""
Checkpoint-aware task handling for executors that opt in with a ``checkpoint`` block.

A checkpointed record keeps its restart state in ``<root>/<record_id>/``, which psi4 receives through the
AtomicInput's ``keywords.function_kwargs.checkpoint_dir``. The record id is the key because it is the only
identifier that survives a reset, a reclaim by another manager, and a move to another node; psi4's own
job-identity hash guards against reusing the directory for a different specification.

psi4 commits only at stage boundaries and writes ``saptdft_status.json`` next to its manifest. The manager reads
that file to report progress and to decide what a failure means:

* retryable (walltime, a lost worker, SIGKILL/SIGTERM, an OOM kill): resubmitted to the same executor without
  returning the task, so the server's reset budget is not spent on a computation that is still progressing;
* not retryable (SCF non-convergence, checkpoint identity mismatch or corruption, or no new completed stage
  across ``max_stalled_attempts`` consecutive attempts): returned as an error naming the stage;
* anything else: returned unchanged, so the server's normal policy applies.

Manager-side bookkeeping lives in ``<root>/.qcfractalcompute/<record_id>.json`` so a different manager that
reclaims the record continues the stall count.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from qcportal.compression import CompressionEnum, compress, decompress
from qcportal.record_models import RecordTask

STATUS_FILENAME = "saptdft_status.json"
LEDGER_DIRNAME = ".qcfractalcompute"
LEDGER_SCHEMA_VERSION = 1

FATAL_ERROR_TYPE = "checkpoint_fatal"
STALLED_ERROR_TYPE = "checkpoint_stalled"

# psi4 exception classes that will fail the same way on every attempt.
_FATAL_PSI4_ERRORS = {
    "SCFConvergenceError",
    "ConvergenceError",
    "TDSCFConvergenceError",
    "ValidationError",
    "UpgradeHelper",
    "MissingMethodError",
    "ManagedMethodError",
}

# Fragments of psi4's final exception line that identify the same failures when psi4 did not get to write its
# status file, or when QCEngine re-labelled the error (ValidationError -> input_error, SCFConvergenceError ->
# unknown_error). Only the final line is searched: the error message also carries psi4's whole stdout.
_FATAL_MESSAGE_MARKERS = (
    "Could not converge SCF iterations",
    "checkpoint identity mismatch",
)

# Task-level error types that mean the worker or its allocation went away.
_LOST_ERROR_TYPES = {"ManagerLost", "WorkerLost", "KilledWorker", "execution_error", "random_error"}

_LOST_MESSAGE_MARKERS = (
    "Killed",
    "SIGKILL",
    "SIGTERM",
    "signal 9",
    "signal 15",
    "Out Of Memory",
    "out of memory",
    "oom-kill",
    "oom_kill",
    "DUE TO TIME LIMIT",
    "CANCELLED",
    "NODE_FAIL",
)

logger = logging.getLogger(__name__)


class CheckpointConfig(BaseModel):
    """Checkpoint/restart settings for one executor."""

    model_config = ConfigDict(extra="forbid")

    root: str
    """Shared-filesystem directory every compute node and the manager can see. Each record checkpoints into
    ``<root>/<record_id>/``."""

    programs: list[str] = ["psi4"]
    """Programs whose tasks are checkpointed."""

    methods: list[str] = ["sapt(dft)*", "dft-d*(sapt)"]
    """Case-insensitive glob patterns for checkpointed methods. Other tasks on the executor run unchanged."""

    max_stalled_attempts: int = Field(3, gt=0)
    """Consecutive retryable failures without a newly completed stage before the record is failed."""

    claim_exact: bool = True
    """Claim at most one task per worker slot instead of three, so claimed tasks do not wait for days behind
    pending allocations while another manager on the same tag sits idle."""

    cleanup: Literal["delete", "keep"] = "delete"
    """What to do with the checkpoint directory after the server accepts a successful result."""

    def root_path(self) -> Path:
        return Path(os.path.expanduser(os.path.expandvars(self.root)))

    def record_dir(self, record_id: int) -> Path:
        return self.root_path() / str(record_id)

    def ledger_path(self, record_id: int) -> Path:
        return self.root_path() / LEDGER_DIRNAME / f"{record_id}.json"


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    with tmp_path.open("w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def read_status(record_dir: Path) -> dict[str, Any] | None:
    """psi4's operator status for a checkpoint directory, or None when it is absent or unreadable."""

    try:
        status = json.loads((record_dir / STATUS_FILENAME).read_text())
    except (OSError, ValueError):
        return None
    return status if isinstance(status, dict) else None


def completed_stages(status: dict[str, Any] | None) -> list[str]:
    if not status:
        return []
    stages = status.get("completed_stages")
    return list(stages) if isinstance(stages, list) else []


def read_ledger(config: CheckpointConfig, record_id: int) -> dict[str, Any]:
    try:
        ledger = json.loads(config.ledger_path(record_id).read_text())
    except (OSError, ValueError):
        ledger = None
    if not isinstance(ledger, dict) or ledger.get("schema_version") != LEDGER_SCHEMA_VERSION:
        ledger = {"schema_version": LEDGER_SCHEMA_VERSION, "record_id": record_id, "stalled": 0, "attempts": []}
    return ledger


def write_ledger(config: CheckpointConfig, record_id: int, ledger: dict[str, Any]) -> None:
    _atomic_write_json(config.ledger_path(record_id), ledger)


def task_method_program(task: RecordTask) -> tuple[str | None, str | None]:
    if task.function != "qcengine.compute" or task.function_kwargs_compressed is None:
        return None, None
    kwargs = task.function_kwargs
    input_data = kwargs.get("input_data") or {}
    model = input_data.get("model") or {}
    return model.get("method"), kwargs.get("program")


def wants_checkpoint(config: CheckpointConfig, task: RecordTask) -> bool:
    method, program = task_method_program(task)
    if method is None or program is None:
        return False
    if program.lower() not in {p.lower() for p in config.programs}:
        return False
    method = method.lower()
    return any(fnmatch.fnmatchcase(method, pattern.lower()) for pattern in config.methods)


def inject_checkpoint_dir(task: RecordTask, record_dir: Path) -> RecordTask:
    """A copy of ``task`` whose AtomicInput carries ``keywords.function_kwargs.checkpoint_dir``."""

    kwargs = decompress(task.function_kwargs_compressed, CompressionEnum.zstd)
    input_data = dict(kwargs["input_data"])
    keywords = dict(input_data.get("keywords") or {})
    function_kwargs = dict(keywords.get("function_kwargs") or {})
    function_kwargs["checkpoint_dir"] = str(record_dir)
    keywords["function_kwargs"] = function_kwargs
    input_data["keywords"] = keywords
    kwargs = {**kwargs, "input_data": input_data}
    compressed, _, _ = compress(kwargs, CompressionEnum.zstd)
    return task.model_copy(update={"function_kwargs_compressed": compressed})


@dataclass
class Verdict:
    action: Literal["retry", "fatal", "stalled", "passthrough"]
    reason: str
    stage: str | None


def _current_attempt(status: dict[str, Any] | None, submitted_at: float) -> dict[str, Any] | None:
    """psi4's record of the attempt this task started, if psi4 got far enough to write one."""

    if not status:
        return None
    attempts = status.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return None
    attempt = attempts[-1]
    if not isinstance(attempt, dict):
        return None
    # Allow for clock skew between the manager host and the compute node.
    if float(attempt.get("started_at", 0.0)) < submitted_at - 300.0:
        return None
    return attempt


def classify_failure(
    error: dict[str, Any],
    status: dict[str, Any] | None,
    *,
    submitted_at: float,
    completed_at_submit: int,
    stalled_before: int,
    max_stalled_attempts: int,
) -> Verdict:
    error_type = str(error.get("error_type", ""))
    message = str(error.get("error_message", ""))
    stage = status.get("next_stage") if status else None

    attempt = _current_attempt(status, submitted_at)
    psi4_error = attempt.get("error") if attempt and attempt.get("outcome") == "failed" else None
    if isinstance(psi4_error, dict):
        stage = psi4_error.get("stage") or stage
        if psi4_error.get("type") in _FATAL_PSI4_ERRORS:
            return Verdict("fatal", f"{psi4_error['type']}: {psi4_error.get('message', '')}", stage)
        # psi4 finished the attempt with an ordinary exception; leave it to the server's policy.
        return Verdict("passthrough", f"{psi4_error.get('type')}: {psi4_error.get('message', '')}", stage)

    final_line = _last_line(message)
    final_exception = final_line.split(":", 1)[0].rsplit(".", 1)[-1]
    if final_exception in _FATAL_PSI4_ERRORS or any(marker in final_line for marker in _FATAL_MESSAGE_MARKERS):
        return Verdict("fatal", f"{error_type}: {final_line}", stage)

    lost = error_type in _LOST_ERROR_TYPES or any(marker in message for marker in _LOST_MESSAGE_MARKERS)
    if not lost:
        return Verdict("passthrough", f"{error_type}: {_last_line(message)}", stage)

    progressed = len(completed_stages(status)) > completed_at_submit
    stalled = 0 if progressed else stalled_before + 1
    if stalled >= max_stalled_attempts:
        return Verdict(
            "stalled",
            f"no stage completed in {stalled} consecutive attempts (last failure {error_type}: {_last_line(message)})",
            stage,
        )
    return Verdict("retry", f"{error_type}: {_last_line(message)}", stage)


def _last_line(message: str) -> str:
    lines = [line.strip() for line in message.strip().splitlines() if line.strip()]
    return lines[-1][:500] if lines else ""


def fatal_result(original: dict[str, Any], verdict: Verdict, record_id: int, record_dir: Path) -> dict[str, Any]:
    """A FailedOperation that names the stage, keeping the original error in the message and extras."""

    error = dict(original.get("error") or {})
    error_type = STALLED_ERROR_TYPE if verdict.action == "stalled" else FATAL_ERROR_TYPE
    stage = verdict.stage or "unknown stage"
    extras = dict(error.get("extras") or {})
    extras.update(
        {
            "checkpoint_dir": str(record_dir),
            "checkpoint_stage": verdict.stage,
            "original_error_type": error.get("error_type"),
        }
    )
    message = f"[checkpoint] record {record_id} failed at {stage}: {verdict.reason}"
    if error.get("error_message"):
        message += "\n\n" + str(error["error_message"])
    return {
        **original,
        "success": False,
        "error": {"error_type": error_type, "error_message": message, "extras": extras},
    }


def describe_progress(status: dict[str, Any] | None) -> tuple[str, str, int]:
    """(done summary, running stage, attempt number) for the progress table."""

    if not status:
        return "-", "starting", 0
    done = completed_stages(status)
    selected = status.get("selected_stages") or []
    last = done[-1] if done else "-"
    summary = f"{len(done)}/{len(selected)} ({last})" if selected else last
    attempts = status.get("attempts") or []
    return summary, str(status.get("next_stage") or "-"), len(attempts)


def remove_record_dir(config: CheckpointConfig, record_id: int) -> None:
    """Delete a record's checkpoint directory and its ledger."""

    record_dir = config.record_dir(record_id)
    if record_dir.is_dir():
        # Rename first so a half-deleted directory is never mistaken for a checkpoint.
        doomed = record_dir.parent / f".deleting-{record_id}-{uuid.uuid4().hex}"
        os.replace(record_dir, doomed)
        shutil.rmtree(doomed, ignore_errors=True)
    try:
        config.ledger_path(record_id).unlink()
    except FileNotFoundError:
        pass


def list_record_dirs(config: CheckpointConfig) -> dict[int, Path]:
    root = config.root_path()
    if not root.is_dir():
        return {}
    return {int(p.name): p for p in root.iterdir() if p.is_dir() and p.name.isdigit()}


def now() -> float:
    return time.time()

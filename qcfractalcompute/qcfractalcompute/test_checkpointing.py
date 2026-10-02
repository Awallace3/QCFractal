from __future__ import annotations

import json
import logging
import time
from collections import defaultdict

import pytest
import yaml

from qcfractalcompute import checkpointing
from qcfractalcompute.apps.models import AppTaskResult
from qcfractalcompute.compress import compress_result
from qcfractalcompute.compute_manager import ComputeManager
from qcfractalcompute.config import SlurmExecutorConfig
from qcportal.compression import CompressionEnum, compress
from qcportal.metadata_models import TaskReturnMetadata
from qcportal.record_models import RecordTask


def _task(task_id=1, record_id=101, method="sapt(dft)-d4(i)", program="psi4", keywords=None, function="qcengine.compute"):
    kwargs = {
        "input_data": {
            "schema_name": "qcschema_input",
            "id": str(record_id),
            "driver": "energy",
            "model": {"method": method, "basis": "aug-cc-pvdz"},
            "keywords": dict(keywords or {}),
            "molecule": {"symbols": ["He"], "geometry": [0.0, 0.0, 0.0]},
        },
        "program": program,
    }
    compressed, _, _ = compress(kwargs, CompressionEnum.zstd)
    return RecordTask(
        id=task_id,
        record_id=record_id,
        function=function,
        function_kwargs_compressed=compressed,
        compute_tag="*",
        compute_priority=1,
        required_programs=[program],
    )


def _write_status(record_dir, *, completed, selected=None, attempts=None):
    record_dir.mkdir(parents=True, exist_ok=True)
    selected = selected or ["hf_dimer_scf", "hf_monomer_a_scf", "hf_monomer_b_scf", "final"]
    status = {
        "schema_version": 1,
        "state": "running",
        "selected_stages": selected,
        "completed_stages": completed,
        "next_stage": next((s for s in selected if s not in completed), None),
        "attempts": attempts or [],
        "updated_at": time.time(),
    }
    (record_dir / checkpointing.STATUS_FILENAME).write_text(json.dumps(status))
    return status


def _failed(error_type, message):
    return {"success": False, "error": {"error_type": error_type, "error_message": message}}


def _classify(error, status, *, completed_at_submit=0, stalled_before=0, max_stalled=3, submitted_at=None):
    return checkpointing.classify_failure(
        error["error"],
        status,
        submitted_at=time.time() - 10 if submitted_at is None else submitted_at,
        completed_at_submit=completed_at_submit,
        stalled_before=stalled_before,
        max_stalled_attempts=max_stalled,
    )


def test_checkpoint_config_parses_and_defaults_off(tmp_path):
    base = """
        compute_tags: ['*']
        cores_per_worker: 1
        memory_per_worker: 1.0
        max_nodes: 2
        workers_per_node: 1
        walltime: "72:00:00"
    """
    assert SlurmExecutorConfig(**yaml.safe_load(base)).checkpoint is None

    with_ckpt = yaml.safe_load(base)
    with_ckpt["checkpoint"] = {"root": str(tmp_path / "ckpt"), "max_stalled_attempts": 2}
    executor = SlurmExecutorConfig(**with_ckpt)
    assert executor.checkpoint.record_dir(42) == tmp_path / "ckpt" / "42"
    assert executor.checkpoint.max_stalled_attempts == 2
    assert executor.checkpoint.claim_exact is True

    with_ckpt["checkpoint"]["bogus"] = 1
    with pytest.raises(ValueError):
        SlurmExecutorConfig(**with_ckpt)


@pytest.mark.parametrize(
    "method, program, function, expected",
    [
        ("sapt(dft)-d4(i)", "psi4", "qcengine.compute", True),
        ("SAPT(DFT)", "Psi4", "qcengine.compute", True),
        ("dft-d4(sapt)", "psi4", "qcengine.compute", True),
        ("sapt0", "psi4", "qcengine.compute", False),
        ("sapt(dft)", "nwchem", "qcengine.compute", False),
        ("sapt(dft)", "psi4", "qcengine.compute_procedure", False),
    ],
)
def test_wants_checkpoint(tmp_path, method, program, function, expected):
    config = checkpointing.CheckpointConfig(root=str(tmp_path))
    assert checkpointing.wants_checkpoint(config, _task(method=method, program=program, function=function)) is expected


def test_inject_checkpoint_dir_keeps_existing_keywords(tmp_path):
    task = _task(keywords={"sapt_dft_functional": "pbe0", "function_kwargs": {"bsse_type": "nocp"}})
    injected = checkpointing.inject_checkpoint_dir(task, tmp_path / "101")

    keywords = injected.function_kwargs["input_data"]["keywords"]
    assert keywords["sapt_dft_functional"] == "pbe0"
    assert keywords["function_kwargs"] == {"bsse_type": "nocp", "checkpoint_dir": str(tmp_path / "101")}
    # The claimed task itself is untouched
    assert "checkpoint_dir" not in task.function_kwargs["input_data"]["keywords"]["function_kwargs"]
    assert injected.id == task.id and injected.record_id == task.record_id


def test_classify_uses_psi4_status_for_hard_failures(tmp_path):
    now = time.time()
    attempt = {
        "started_at": now,
        "outcome": "failed",
        "error": {"stage": "hf_monomer_a_scf", "type": "SCFConvergenceError", "message": "Could not converge"},
    }
    status = _write_status(tmp_path, completed=["hf_dimer_scf"], attempts=[attempt])
    # QCEngine relabels the psi4 exception, so the error_type alone would not identify it
    verdict = _classify(_failed("unknown_error", "lots of stdout"), status, submitted_at=now - 5)
    assert (verdict.action, verdict.stage) == ("fatal", "hf_monomer_a_scf")

    attempt["error"]["type"] = "RuntimeError"
    verdict = _classify(_failed("unknown_error", "lots of stdout"), status, submitted_at=now - 5)
    assert verdict.action == "passthrough"


def test_classify_ignores_a_stale_attempt(tmp_path):
    now = time.time()
    attempt = {
        "started_at": now - 86400,
        "outcome": "failed",
        "error": {"stage": "hf_dimer_scf", "type": "SCFConvergenceError", "message": "old"},
    }
    status = _write_status(tmp_path, completed=[], attempts=[attempt])
    verdict = _classify(_failed("ManagerLost", "Compute worker lost"), status, submitted_at=now)
    assert verdict.action == "retry"


@pytest.mark.parametrize(
    "error_type, message",
    [
        ("input_error", "...\nValidationError: SAPT(DFT) checkpoint identity mismatch for /x/101: keywords"),
        (
            "unknown_error",
            "SAPT(DFT) checkpoint restored\n...\npsi4.driver.p4util.exceptions.SCFConvergenceError: Could not converge SCF iterations in 100 iterations.",
        ),
    ],
)
def test_classify_fatal_from_final_exception_line(error_type, message):
    verdict = _classify(_failed(error_type, message), None)
    assert verdict.action == "fatal"


def test_classify_does_not_match_markers_in_stdout():
    message = "SCFConvergenceError mentioned in output\nRuntimeError: disk quota exceeded"
    assert _classify(_failed("unknown_error", message), None).action == "passthrough"


def test_classify_lost_worker_retries_while_progressing(tmp_path):
    status = _write_status(tmp_path, completed=["hf_dimer_scf", "hf_monomer_a_scf"])
    error = _failed("ManagerLost", "Compute worker lost")

    assert _classify(error, status, completed_at_submit=1, stalled_before=2).action == "retry"
    assert _classify(error, status, completed_at_submit=2, stalled_before=1).action == "retry"
    verdict = _classify(error, status, completed_at_submit=2, stalled_before=2)
    assert (verdict.action, verdict.stage) == ("stalled", "hf_monomer_b_scf")


@pytest.mark.parametrize(
    "error_type, message",
    [
        ("execution_error", "psi4 exited"),
        ("unknown_error", "slurmstepd: error: *** JOB 123 ON node001 CANCELLED AT 2026 DUE TO TIME LIMIT ***"),
        ("unknown_error", "Killed"),
    ],
)
def test_classify_kills_are_retryable(error_type, message):
    assert _classify(_failed(error_type, message), None).action == "retry"


def test_fatal_result_names_the_stage(tmp_path):
    verdict = checkpointing.Verdict("stalled", "no stage completed in 3 consecutive attempts", "hf_dimer_scf")
    original = _failed("ManagerLost", "Compute worker lost")
    original["error"]["extras"] = {"slurm_job_id": "77"}
    failed = checkpointing.fatal_result(original, verdict, 101, tmp_path / "101")

    error = failed["error"]
    assert failed["success"] is False
    assert error["error_type"] == checkpointing.STALLED_ERROR_TYPE
    assert error["error_message"].startswith("[checkpoint] record 101 failed at hf_dimer_scf")
    assert "Compute worker lost" in error["error_message"]
    assert error["extras"]["slurm_job_id"] == "77"
    assert error["extras"]["original_error_type"] == "ManagerLost"
    assert error["extras"]["checkpoint_stage"] == "hf_dimer_scf"


def test_remove_record_dir_and_ledger(tmp_path):
    config = checkpointing.CheckpointConfig(root=str(tmp_path))
    _write_status(config.record_dir(7), completed=[])
    checkpointing.write_ledger(config, 7, checkpointing.read_ledger(config, 7))
    assert set(checkpointing.list_record_dirs(config)) == {7}

    checkpointing.remove_record_dir(config, 7)
    assert checkpointing.list_record_dirs(config) == {}
    assert not config.ledger_path(7).exists()
    assert list(tmp_path.glob(".deleting-*")) == []


class _Executor:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint


class _Config:
    def __init__(self, checkpoint):
        self.executors = {"slurm": _Executor(checkpoint)}


class _FakeManager:
    """The checkpoint methods of ComputeManager, without parsl or a server."""

    _prepare_checkpoint_tasks = ComputeManager._prepare_checkpoint_tasks
    _track_checkpoint_task = ComputeManager._track_checkpoint_task
    _triage_checkpoint_results = ComputeManager._triage_checkpoint_results
    _checkpoint_after_return = ComputeManager._checkpoint_after_return
    _log_checkpoint_progress = ComputeManager._log_checkpoint_progress

    def __init__(self, checkpoint):
        self.manager_config = _Config(checkpoint)
        self.logger = logging.getLogger("fake_manager")
        self.name = "cluster-host-uuid"
        self._checkpoint_tasks = {}
        self._checkpoint_cleanup = {}
        self._checkpoint_immediate = defaultdict(dict)
        self._record_id_map = {}
        self.submitted = []

    def _submit_tasks(self, executor_label, tasks):
        self.submitted.extend((executor_label, t) for t in tasks)


def _app_result(result):
    return AppTaskResult(success=result.get("success", False), walltime=1.0, result_compressed=compress_result(result))


def test_manager_resubmits_until_stalled(tmp_path):
    config = checkpointing.CheckpointConfig(root=str(tmp_path), max_stalled_attempts=2)
    manager = _FakeManager(config)
    task = _task(task_id=5, record_id=101)
    other = _task(task_id=6, record_id=102, method="sapt0")

    prepared = manager._prepare_checkpoint_tasks("slurm", [task, other])
    assert prepared[1] is other
    assert prepared[0].function_kwargs["input_data"]["keywords"]["function_kwargs"]["checkpoint_dir"] == str(
        config.record_dir(101)
    )
    assert set(manager._checkpoint_tasks) == {5}

    # First allocation completes a stage, then dies: resubmitted, not returned
    _write_status(config.record_dir(101), completed=["hf_dimer_scf"])
    results = {"slurm": {5: _app_result(_failed("ManagerLost", "lost")), 6: _app_result(_failed("x", "y"))}}
    manager._triage_checkpoint_results(results)
    assert set(results["slurm"]) == {6}
    assert [t.id for _, t in manager.submitted] == [5]
    assert manager._checkpoint_tasks[5]["completed_at_submit"] == 1

    # Second allocation makes no progress: still retried (1 of 2)
    results = {"slurm": {5: _app_result(_failed("ManagerLost", "lost"))}}
    manager._triage_checkpoint_results(results)
    assert results["slurm"] == {}
    assert checkpointing.read_ledger(config, 101)["stalled"] == 1

    # Third allocation makes no progress either: returned as a stalled error naming the stage
    results = {"slurm": {5: _app_result(_failed("ManagerLost", "lost"))}}
    manager._triage_checkpoint_results(results)
    returned = results["slurm"][5].result
    assert returned["error"]["error_type"] == checkpointing.STALLED_ERROR_TYPE
    assert "hf_monomer_a_scf" in returned["error"]["error_message"]
    assert 5 not in manager._checkpoint_tasks
    ledger = checkpointing.read_ledger(config, 101)
    assert ledger["stalled"] == 0
    assert [a["verdict"] for a in ledger["attempts"]] == ["retry", "retry", "stalled"]


def test_manager_cleans_up_only_after_acceptance(tmp_path):
    config = checkpointing.CheckpointConfig(root=str(tmp_path))
    manager = _FakeManager(config)
    accepted, rejected = _task(task_id=1, record_id=201), _task(task_id=2, record_id=202)
    manager._prepare_checkpoint_tasks("slurm", [accepted, rejected])
    for record_id in (201, 202):
        _write_status(config.record_dir(record_id), completed=["final"], selected=["final"])

    results = {"slurm": {1: _app_result({"success": True}), 2: _app_result({"success": True})}}
    manager._triage_checkpoint_results(results)
    assert config.record_dir(201).exists() and config.record_dir(202).exists()

    manager._checkpoint_after_return(
        TaskReturnMetadata(accepted_ids=[1], rejected_info=[(2, "record is not running")])
    )
    assert not config.record_dir(201).exists()
    assert config.record_dir(202).exists()
    assert manager._checkpoint_cleanup == {}


def test_manager_keeps_checkpoint_on_hard_failure(tmp_path):
    config = checkpointing.CheckpointConfig(root=str(tmp_path))
    manager = _FakeManager(config)
    manager._prepare_checkpoint_tasks("slurm", [_task(task_id=3, record_id=301)])
    attempt = {
        "started_at": time.time(),
        "outcome": "failed",
        "error": {"stage": "dft_monomer_a_scf", "type": "SCFConvergenceError", "message": "maxiter"},
    }
    _write_status(config.record_dir(301), completed=["hf_dimer_scf"], attempts=[attempt])

    results = {"slurm": {3: _app_result(_failed("unknown_error", "stdout"))}}
    manager._triage_checkpoint_results(results)
    assert results["slurm"][3].result["error"]["error_type"] == checkpointing.FATAL_ERROR_TYPE
    assert manager.submitted == []
    assert manager._checkpoint_cleanup == {}
    assert config.record_dir(301).exists()


def _fail_hard(manager, config, task_id, record_id, attempts):
    attempt = {
        "started_at": time.time(),
        "outcome": "failed",
        "error": {"stage": "hf_dimer_scf", "type": "SCFConvergenceError", "message": "maxiter"},
    }
    _write_status(config.record_dir(record_id), completed=["grac_monomer_a"], attempts=[attempt] * attempts)
    results = {"slurm": {task_id: _app_result(_failed("unknown_error", "stdout"))}}
    manager._triage_checkpoint_results(results)
    return results


def test_reclaimed_fatal_record_is_not_rerun(tmp_path):
    config = checkpointing.CheckpointConfig(root=str(tmp_path))
    manager = _FakeManager(config)
    manager._prepare_checkpoint_tasks("slurm", [_task(task_id=1, record_id=401)])
    _fail_hard(manager, config, 1, 401, attempts=1)
    manager.submitted.clear()

    # The server auto-resets the record and this manager claims it again under a new task id
    assert manager._prepare_checkpoint_tasks("slurm", [_task(task_id=2, record_id=401)]) == []
    held = manager._checkpoint_immediate["slurm"][2].result
    assert held["error"]["error_type"] == checkpointing.FATAL_ERROR_TYPE
    assert "hf_dimer_scf" in held["error"]["error_message"]
    assert manager._record_id_map[2] == 401
    assert 2 not in manager._checkpoint_tasks


def test_fatal_hold_releases_when_someone_else_ran_it(tmp_path):
    config = checkpointing.CheckpointConfig(root=str(tmp_path))
    manager = _FakeManager(config)
    manager._prepare_checkpoint_tasks("slurm", [_task(task_id=1, record_id=402)])
    _fail_hard(manager, config, 1, 402, attempts=1)

    # A later psi4 attempt (another manager, a manual run) means the stored verdict is stale
    _write_status(config.record_dir(402), completed=["grac_monomer_a"], attempts=[{"started_at": 1.0}] * 2)
    assert [t.id for t in manager._prepare_checkpoint_tasks("slurm", [_task(task_id=2, record_id=402)])] == [2]


@pytest.mark.parametrize("hold", [0.0, 1e-9])
def test_fatal_hold_disabled_or_expired(tmp_path, hold):
    config = checkpointing.CheckpointConfig(root=str(tmp_path), fatal_hold=hold)
    manager = _FakeManager(config)
    manager._prepare_checkpoint_tasks("slurm", [_task(task_id=1, record_id=403)])
    _fail_hard(manager, config, 1, 403, attempts=1)
    time.sleep(0.01)
    assert [t.id for t in manager._prepare_checkpoint_tasks("slurm", [_task(task_id=2, record_id=403)])] == [2]


def test_cli_clear_releases_a_held_record(tmp_path):
    from qcfractalcompute import checkpoint_cli

    root = tmp_path / "ckpt"
    config = checkpointing.CheckpointConfig(root=str(root))
    manager = _FakeManager(config)
    manager._prepare_checkpoint_tasks("slurm", [_task(task_id=1, record_id=404)])
    _fail_hard(manager, config, 1, 404, attempts=1)
    assert "fatal" in checkpointing.read_ledger(config, 404)

    cfg = tmp_path / "manager.yml"
    cfg.write_text(yaml.safe_dump({
        "cluster": "t", "server": {"fractal_uri": "http://localhost:7777/"},
        "executors": {"slurm": {"type": "slurm", "compute_tags": ["*"], "cores_per_worker": 1,
                                "memory_per_worker": 1.0, "max_nodes": 1, "workers_per_node": 1,
                                "walltime": "01:00:00", "checkpoint": {"root": str(root)}}},
    }))
    assert checkpoint_cli.main(["--config", str(cfg), "clear", "404"]) == 0
    assert "fatal" not in checkpointing.read_ledger(config, 404)
    assert [t.id for t in manager._prepare_checkpoint_tasks("slurm", [_task(task_id=2, record_id=404)])] == [2]

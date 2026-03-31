from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Dict

from qcarchivetesting.testing_classes import QCATestingSnowflake
from qcfractalcompute.compress import compress_result
from qcfractalcompute.compute_manager import ComputeManager
from qcfractalcompute.config import (
    FractalComputeConfig,
    FractalServerSettings,
    LocalExecutorConfig,
)
from qcportal.external import fsapt_workflow
from qcportal.record_models import RecordStatusEnum
from qcportal.singlepoint.record_models import SinglepointRecord

import pandas as pd
import pytest
from pprint import pprint as pp
from qcelemental.models import Molecule

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_PACKAGE_ROOT = Path(__file__).resolve().parents[1]

sys.path = [
    p
    for p in sys.path
    if p not in {"", str(Path.cwd().resolve()), str(_LOCAL_PACKAGE_ROOT)}
]
sys.path.insert(0, str(_REPO_ROOT))


psi4 = pytest.importorskip("psi4")


FSAPT_CASES = {
    "methane_dimer": {
        "molecule": """0 1
C 0.00000000 0.00000000 0.00000000
H 1.09000000 0.00000000 0.00000000
H -0.36333333 0.83908239 0.59332085
H -0.36333333 0.09428973 -1.02332709
H -0.36333333 -0.93337212 0.43000624
--
0 1
C 6.44536662 -0.26509169 -0.00000000
H 7.53536662 -0.26509169 -0.00000000
H 6.08203329 0.57399070 0.59332085
H 6.08203329 -0.17080196 -1.02332709
H 6.08203329 -1.19846381 0.43000624
symmetry c1
no_reorient
no_com""",
        "basis": "sto-3g",
        "fragments_a": {"MethylA": [1, 2, 3, 4, 5]},
        "fragments_b": {"MethylB": [6, 7, 8, 9, 10]},
        "expected": {
            ("MethylA", "MethylB"): {
                "F-Electrostatics": -0.0014641083910476027,
                "F-Exchange": 9.057316601367232e-06,
                "F-Induction": -6.921433353433796e-06,
                "F-Dispersion": -0.003525991708258379,
                "F-Total": -0.00498796421680936,
            },
            ("All", "All"): {
                "F-Electrostatics": -0.001464108542199844,
                "F-Exchange": 9.05731660548897e-06,
                "F-Induction": -6.921433355451785e-06,
                "F-Dispersion": -0.003525991709862946,
                "F-Total": -0.004987964368812753,
            },
        },
    },
    "multi_fragment_ethane_peptide": {
        "molecule": """
0 1
C   11.54100       27.68600       13.69600
H   12.45900       27.15000       13.44600
C   10.79000       27.96500       12.40600
H   10.55700       27.01400       11.92400
H   9.879000       28.51400       12.64300
H   11.44300       28.56800       11.76200
H   10.90337       27.06487       14.34224
H   11.78789       28.62476       14.21347
--
0 1
C   10.60200       24.81800       6.466000
O   10.95600       23.84000       7.103000
N   10.17800       25.94300       7.070000
C   10.09100       26.25600       8.476000
C   9.372000       27.59000       8.640000
C   11.44600       26.35600       9.091000
C   9.333000       25.25000       9.282000
H   9.874000       26.68900       6.497000
H   9.908000       28.37100       8.093000
H   8.364000       27.46400       8.233000
H   9.317000       27.84600       9.706000
H   9.807000       24.28200       9.160000
H   9.371000       25.57400       10.32900
H   8.328000       25.26700       8.900000
H   11.28800       26.57600       10.14400
H   11.97000       27.14900       8.585000
H   11.93200       25.39300       8.957000
H   10.61998       24.85900       5.366911
units angstrom

symmetry c1
no_reorient
no_com
""",
        "basis": "sto-3g",
        "fragments_a": {
            "Methyl1_A": [1, 2, 7, 8],
            "Methyl2_A": [3, 4, 5, 6],
        },
        "fragments_b": {
            "Peptide_B": [9, 10, 11, 16, 26],
            "T-Butyl_B": [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        },
        "expected": {
            ("Methyl1_A", "Peptide_B"): {
                "F-Electrostatics": 0.7150992302947188,
                "F-Exchange": 0.00013680800590309906,
                "F-Induction": -0.00673583757136201,
                "F-Dispersion": -0.003992035787563479,
                "F-Total": 0.7045081649423892,
            },
            ("Methyl1_A", "T-Butyl_B"): {
                "F-Electrostatics": -0.2042449764424319,
                "F-Exchange": 0.05310422532154325,
                "F-Induction": -0.0008922003831071668,
                "F-Dispersion": -0.06726003034293386,
                "F-Total": -0.219292981846543,
            },
            ("Methyl2_A", "Peptide_B"): {
                "F-Electrostatics": -0.8155064098920732,
                "F-Exchange": 0.03094443957782037,
                "F-Induction": -0.027767087385440918,
                "F-Dispersion": -0.013540224505783295,
                "F-Total": -0.8258692822083376,
            },
            ("Methyl2_A", "T-Butyl_B"): {
                "F-Electrostatics": -0.9356370086747248,
                "F-Exchange": 3.891362354110715,
                "F-Induction": -0.2554777060517417,
                "F-Dispersion": -0.4109667131624368,
                "F-Total": 2.289280926225068,
            },
            ("All", "All"): {
                "F-Electrostatics": -1.2402891647145111,
                "F-Exchange": 3.9755478270159816,
                "F-Induction": -0.2908728313916518,
                "F-Dispersion": -0.49575900379871746,
                "F-Total": 1.9486268271125766,
            },
        },
    },
}


def _run_fisapt_case(case: Dict[str, object]):
    psi4.core.clean()
    psi4.core.clean_options()

    molecule = psi4.geometry(case["molecule"])
    psi4.set_options(
        {
            "basis": case["basis"],
            "scf_type": "df",
            "guess": "sad",
            "freeze_core": "true",
            "FISAPT_FSAPT_FILEPATH": "none",
        }
    )

    plan = psi4.energy("fisapt0", return_plan=True, molecule=molecule)
    return psi4.schema_wrapper.run_qcschema(plan.plan(), clean=True, postclean=True)


def _submit_completed_results(
    snowflake: "QCATestingSnowflake", record_results: Dict[int, object]
) -> None:
    storage_socket = snowflake.get_storage_socket()
    manager_name, _ = snowflake.activate_manager()
    manager_programs = snowflake.activated_manager_programs()

    tasks = storage_socket.tasks.claim_tasks(
        manager_name.fullname, manager_programs, ["*"]
    )
    assert len(tasks) == len(record_results)
    print("Claimed snowflake tasks:")
    pp([{"task_id": task["id"], "record_id": task["record_id"]} for task in tasks])

    task_map = {task["record_id"]: task for task in tasks}
    assert set(task_map) == set(record_results)

    meta = storage_socket.tasks.update_finished(
        manager_name.fullname,
        {
            task_map[record_id]["id"]: compress_result(result.dict())
            for record_id, result in record_results.items()
        },
    )
    assert meta.n_accepted == len(record_results)
    assert meta.n_rejected == 0
    print("Marked all claimed tasks complete")


def _build_fragment_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": system_name,
            "qcel_molecule": Molecule.from_data(case["molecule"]),
            "fragments_a": case["fragments_a"],
            "fragments_b": case["fragments_b"],
        }
        for system_name, case in FSAPT_CASES.items()
    )


def _lookup_pair(
    results: pd.DataFrame, entry_id: str, frag1: str, frag2: str
) -> pd.Series:
    rows = results[
        (results["id"] == entry_id)
        & (results["Frag1"] == frag1)
        & (results["Frag2"] == frag2)
    ]
    assert len(rows) == 1
    return rows.iloc[0]


def _assert_combined_results(results: pd.DataFrame) -> None:
    assert not results.empty
    assert set(results["id"]) == set(FSAPT_CASES)
    print("Combined F-SAPT analysis dataframe:")
    print(results)

    for system_name, case in FSAPT_CASES.items():
        system_results = results[results["id"] == system_name]
        assert not system_results.empty
        print(f"\nRows for system {system_name}:")
        print(system_results)

        for fragment_pair, expected_energies in case["expected"].items():
            row = _lookup_pair(results, system_name, fragment_pair[0], fragment_pair[1])
            print(f"\nChecking {system_name} fragment pair {fragment_pair}:")
            pp(row.to_dict())
            for key, expected_value in expected_energies.items():
                print(
                    f"  {key}: actual={row[key]:.12f}, expected={expected_value:.12f}"
                )
                assert row[key] == pytest.approx(expected_value, abs=1.0e-5)
                print(f"  Assertion passed for {key}")


def _make_compute_config(
    snowflake: "QCATestingSnowflake", base_folder: Path, cluster: str
) -> FractalComputeConfig:
    scratch_dir = base_folder / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    return FractalComputeConfig(
        base_folder=str(base_folder),
        cluster=cluster,
        update_frequency=1,
        update_frequency_jitter=0.0,
        server=FractalServerSettings(
            fractal_uri=snowflake.get_uri(),
            verify=False,
        ),
        executors={
            "local": LocalExecutorConfig(
                cores_per_worker=2,
                memory_per_worker=4,
                max_workers=1,
                compute_tags=["*"],
                scratch_directory=str(scratch_dir),
            )
        },
    )


def _stop_compute_manager(
    compute: ComputeManager, compute_thread: threading.Thread
) -> None:
    compute.stop()
    compute_thread.join(timeout=10)


def test_fsapt_workflow_dataset_matches_psi4_reference(
    snowflake: "QCATestingSnowflake", monkeypatch: pytest.MonkeyPatch
):
    print("\n=== Combined dataset workflow test ===")
    print("Systems under test:")
    pp(
        {
            name: {
                "basis": case["basis"],
                "fragments_a": case["fragments_a"],
                "fragments_b": case["fragments_b"],
            }
            for name, case in FSAPT_CASES.items()
        }
    )

    snowflake_client = snowflake.client()
    specification_name = "fisapt0"
    dataset_name = "FSAPT workflow combined systems"
    qc_spec = fsapt_workflow.default_recommended_fisapt0_specification(
        basis="sto-3g",
        keywords={"FISAPT_FSAPT_FILEPATH": "none"},
        keep_native_files=False,
    )

    ds = fsapt_workflow.create_dataset(
        snowflake_client,
        dataset_name,
        _build_fragment_df(),
        qc_specification=qc_spec,
        specification_name=specification_name,
        submit=False,
        verbose=0,
    )
    print("Created dataset:")
    pp({"dataset_name": dataset_name, "specification_name": specification_name})

    submit_meta = ds.submit(specification_names=[specification_name])
    print("Submission metadata:")
    pp(submit_meta.dict())
    assert submit_meta.n_inserted == len(FSAPT_CASES)

    records = list(ds.iterate_records(specification_names=specification_name))
    assert len(records) == len(FSAPT_CASES)
    print("Records after submit:")

    record_map = {}
    for entry_name, _, record in records:
        record_map[entry_name] = record
        pp(
            {
                "entry_name": entry_name,
                "record_id": record.id,
                "status": record.status,
            }
        )
        assert record.status == RecordStatusEnum.waiting

    atomic_results = {
        system_name: _run_fisapt_case(case) for system_name, case in FSAPT_CASES.items()
    }
    print("Computed reference atomic results for systems:")
    pp(list(atomic_results))

    _submit_completed_results(
        snowflake,
        {
            record_map[system_name].id: atomic_results[system_name]
            for system_name in FSAPT_CASES
        },
    )

    original_to_qcschema_result = SinglepointRecord.to_qcschema_result

    def _patched_to_qcschema_result(self):
        for system_name, record in record_map.items():
            if self.id == record.id:
                return atomic_results[system_name]
        return original_to_qcschema_result(self)

    monkeypatch.setattr(
        SinglepointRecord, "to_qcschema_result", _patched_to_qcschema_result
    )
    print("Patched record-to-qcschema conversion for deterministic analysis")

    results = fsapt_workflow.analyze_dataset(
        snowflake_client, dataset_name, specification_name
    )
    _assert_combined_results(results)


# def test_fsapt_workflow_dataset_with_explicit_compute_manager(
#     snowflake: "QCATestingSnowflake", tmp_path: Path
# ):
#     print("\n=== Explicit compute-manager workflow test ===")
#
#     snowflake.start_job_runner()
#     snowflake_client = snowflake.client()
#     specification_name = "fisapt0"
#     dataset_name = "FSAPT workflow explicit compute manager"
#     qc_spec = fsapt_workflow.default_recommended_fisapt0_specification(
#         basis="sto-3g",
#         keywords={"FISAPT_FSAPT_FILEPATH": "none"},
#         keep_native_files=False,
#     )
#
#     ds = fsapt_workflow.create_dataset(
#         snowflake_client,
#         dataset_name,
#         _build_fragment_df(),
#         qc_specification=qc_spec,
#         specification_name=specification_name,
#         submit=False,
#         verbose=0,
#     )
#     print("Created dataset for compute-manager execution:")
#     pp({"dataset_name": dataset_name, "specification_name": specification_name})
#
#     compute = ComputeManager(
#         _make_compute_config(snowflake, tmp_path / "explicit", "fsapt_explicit_compute")
#     )
#     compute_thread = threading.Thread(
#         target=compute.start,
#         kwargs={"manual_updates": False},
#         daemon=True,
#     )
#     compute_thread.start()
#     time.sleep(1)
#     print("Started ComputeManager")
#
#     try:
#         submit_meta = ds.submit(specification_names=[specification_name])
#         print("Submission metadata:")
#         pp(submit_meta.dict())
#         assert submit_meta.n_inserted == len(FSAPT_CASES)
#
#         records = list(ds.iterate_records(specification_names=specification_name))
#         assert len(records) == len(FSAPT_CASES)
#         print("Records observed after submission:")
#         pp(
#             [
#                 {
#                     "entry_name": entry_name,
#                     "record_id": record.id,
#                     "status": record.status,
#                 }
#                 for entry_name, _, record in records
#             ]
#         )
#         assert all(
#             record.status in {RecordStatusEnum.waiting, RecordStatusEnum.running}
#             for _, _, record in records
#         )
#
#         record_ids = [record.id for _, _, record in records]
#         assert snowflake.await_results(record_ids, timeout=300.0)
#         print("Snowflake reported these record ids finished:")
#         pp(record_ids)
#
#         ds = snowflake_client.get_dataset("singlepoint", dataset_name)
#         print("Dataset status after compute-manager run:")
#         pp(ds.status())
#         for entry, spec, rec in ds.iterate_records():
#             print(rec)
#             pp(rec.error)
#         assert ds.status()[specification_name] == {
#             RecordStatusEnum.complete: len(FSAPT_CASES)
#         }
#
#         results = fsapt_workflow.analyze_dataset(
#             snowflake_client, dataset_name, specification_name
#         )
#         _assert_combined_results(results)
#     finally:
#         _stop_compute_manager(compute, compute_thread)


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(
            [
                __file__,
                "-k",
                "test_fsapt_workflow_dataset",
                "-s",
            ]
        )
    )

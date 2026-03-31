from __future__ import annotations
from qcportal.singlepoint.record_models import SinglepointRecord
from qcportal.record_models import RecordStatusEnum
from qcportal.external import fsapt_workflow
from qcfractalcompute.compress import compress_result

import sys
from pathlib import Path
from typing import Dict
from qcarchivetesting.testing_classes import QCATestingSnowflake

import pandas as pd
import pytest
from qcelemental.models import Molecule
from pprint import pprint as pp

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_PACKAGE_ROOT = Path(__file__).resolve().parents[1]

sys.path = [
    p
    for p in sys.path
    if p not in {"", str(Path.cwd().resolve()), str(_LOCAL_PACKAGE_ROOT)}
]
sys.path.insert(0, str(_REPO_ROOT))


psi4 = pytest.importorskip("psi4")


FSAPT_CASES = [
    {
        "name": "methane_dimer",
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
        "basis": "jun-cc-pvdz",
        "fragments_a": {"MethylA": [1, 2, 3, 4, 5]},
        "fragments_b": {"MethylB": [6, 7, 8, 9, 10]},
        "expected": {
            ("MethylA", "MethylB"): {
                "F-Electrostatics": -0.0023867836548276955,
                "F-Exchange": 0.00011242419533877543,
                "F-Induction": -2.4039823642064496e-05,
                "F-Dispersion": -0.020636082319331096,
                "F-Total": -0.02293448160273215,
            },
            ("All", "All"): {
                "F-Electrostatics": -0.0023867836548276955,
                "F-Exchange": 0.00011242419533877543,
                "F-Induction": -2.4039823642064496e-05,
                "F-Dispersion": -0.020636082319331096,
                "F-Total": -0.02293448160273215,
            },
        },
    },
    {
        "name": "multi_fragment_ethane_peptide",
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
]


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


def _submit_completed_result(
    snowflake: "QCATestingSnowflake", record_id: int, result
) -> None:
    storage_socket = snowflake.get_storage_socket()
    manager_name, _ = snowflake.activate_manager()
    manager_programs = snowflake.activated_manager_programs()

    tasks = storage_socket.tasks.claim_tasks(
        manager_name.fullname, manager_programs, ["*"]
    )
    assert len(tasks) == 1
    assert tasks[0]["record_id"] == record_id

    meta = storage_socket.tasks.update_finished(
        manager_name.fullname,
        {tasks[0]["id"]: compress_result(result.dict())},
    )
    assert meta.n_accepted == 1
    assert meta.n_rejected == 0


def _lookup_pair(results: pd.DataFrame, frag1: str, frag2: str) -> pd.Series:
    rows = results[(results["Frag1"] == frag1) & (results["Frag2"] == frag2)]
    assert len(rows) == 1
    return rows.iloc[0]


@pytest.mark.parametrize("case", FSAPT_CASES, ids=[x["name"] for x in FSAPT_CASES])
def test_fsapt_workflow_dataset_matches_psi4_reference(
    snowflake: "QCATestingSnowflake", case, monkeypatch: pytest.MonkeyPatch
):
    print(f"\n=== Running FSAPT workflow test for case: {case['name']} ===")
    pp(
        {
            "basis": case["basis"],
            "fragments_a": case["fragments_a"],
            "fragments_b": case["fragments_b"],
            "expected_pairs": list(case["expected"].keys()),
        }
    )

    snowflake_client = snowflake.client()

    fragment_df = pd.DataFrame(
        [
            {
                "id": case["name"],
                "qcel_molecule": Molecule.from_data(case["molecule"]),
                "fragments_a": case["fragments_a"],
                "fragments_b": case["fragments_b"],
            }
        ]
    )

    specification_name = "fisapt0"
    dataset_name = f"FSAPT workflow {case['name']}"
    qc_spec = fsapt_workflow.default_recommended_fisapt0_specification(
        basis=case["basis"],
        keywords={"FISAPT_FSAPT_FILEPATH": "none"},  # Can update options here
        keep_native_files=False,
    )

    ds = fsapt_workflow.create_dataset(
        snowflake_client,
        dataset_name,
        fragment_df,
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
    assert submit_meta.n_inserted == 1
    print("Assertion passed: exactly one record was inserted")

    records = list(ds.iterate_records(specification_names=specification_name))
    print(f"Fetched {len(records)} record(s) from dataset")
    assert len(records) == 1
    print("Assertion passed: dataset contains exactly one record")

    _, _, record = records[0]
    print("Record summary:")
    pp({"record_id": record.id, "status": record.status})
    assert record.status == RecordStatusEnum.waiting
    print("Assertion passed: record is initially waiting")

    atomic_result = _run_fisapt_case(case)
    print("Computed Psi4 atomic result")
    _submit_completed_result(snowflake, record.id, atomic_result)
    print("Submitted completed result back to snowflake")

    original_to_qcschema_result = SinglepointRecord.to_qcschema_result

    def _patched_to_qcschema_result(self):
        if self.id == record.id:
            return atomic_result
        return original_to_qcschema_result(self)

    monkeypatch.setattr(
        SinglepointRecord, "to_qcschema_result", _patched_to_qcschema_result
    )
    print("Monkeypatched SinglepointRecord.to_qcschema_result for target record")

    results = fsapt_workflow.analyze_dataset(
        snowflake_client, dataset_name, specification_name
    )
    results.to_pickle("fsapt_analysis_results.pkl")
    print("Analyzed dataset; result table:")
    print(results)
    assert not results.empty
    print("Assertion passed: analysis results are not empty")

    for fragment_pair, expected_energies in case["expected"].items():
        row = _lookup_pair(results, fragment_pair[0], fragment_pair[1])
        print(f"\nChecking fragment pair {fragment_pair}:")
        pp(row.to_dict())
        for key, expected_value in expected_energies.items():
            actual_value = row[key]
            print(
                f"  Verifying {key}: actual={actual_value:.12f}, "
                f"expected={expected_value:.12f}"
            )
            assert row[key] == pytest.approx(expected_value, abs=1.0e-5)
            print(f"  Assertion passed for {key}")


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(
            [
                __file__,
                "-k",
                "test_fsapt_workflow_dataset_matches_psi4_reference",
                "-s",
            ]
        )
    )

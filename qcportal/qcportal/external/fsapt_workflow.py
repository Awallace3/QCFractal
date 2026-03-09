from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import qcelemental as qcel
import tabulate

from qcportal.dataset_models import load_dataset_view
from qcportal.record_models import RecordStatusEnum
from qcportal.singlepoint import (
    QCSpecification,
    SinglepointDataset,
    SinglepointDatasetEntry,
    SinglepointDriver,
    SinglepointProtocols,
)
from qcportal.singlepoint.record_models import (
    NativeFilesProtocolEnum,
    WavefunctionProtocolEnum,
)

try:
    import pandas as pd
    import psi4
except ImportError:
    raise ImportError("Please install pandas and psi4 to use this module")

if TYPE_CHECKING:
    from qcportal import PortalClient


HARTREE_TO_KCALMOL = qcel.constants.hartree2kcalmol
BOHR_TO_ANGSTROM = qcel.constants.bohr2angstroms

REQUIRED_FRAGMENT_COLUMNS = {"id", "qcel_molecule", "fragments_a", "fragments_b"}

FISAPT0_OPTIONS = {
    "scf_type": "df",
    "guess": "sad",
    "freeze_core": "true",
    "FISAPT_FSAPT_FILEPATH": "fsapt",
}


def default_fisapt0_specification(
    *,
    basis: str = "jun-cc-pvdz",
    keywords: Optional[Dict[str, Any]] = None,
    keep_native_files: bool = True,
) -> QCSpecification:
    spec_keywords = dict(FISAPT0_OPTIONS)
    if keywords:
        spec_keywords.update(keywords)

    native_files = (
        NativeFilesProtocolEnum.all
        if keep_native_files
        else NativeFilesProtocolEnum.none
    )

    return QCSpecification(
        program="psi4",
        driver=SinglepointDriver.energy,
        method="fisapt0",
        basis=basis,
        keywords=spec_keywords,
        protocols=SinglepointProtocols(
            wavefunction=WavefunctionProtocolEnum.none,
            stdout=True,
            native_files=native_files,
        ),
    )


def load_fsapt_fragment_data(
    fragment_data: Union[str, "pd.DataFrame"],
) -> "pd.DataFrame":
    if isinstance(fragment_data, pd.DataFrame):
        df = fragment_data.copy()
    else:
        df = pd.read_pickle(fragment_data)

    missing_columns = REQUIRED_FRAGMENT_COLUMNS - set(df.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise RuntimeError(f"Fragment data is missing required columns: {missing_str}")

    if not isinstance(df, pd.DataFrame):
        raise RuntimeError("Fragment data must load into a pandas DataFrame")

    return df


def _check_ds_complete(ds: SinglepointDataset, specification_name: str):
    if specification_name not in ds.specification_names:
        raise RuntimeError(f"Specification {specification_name} not found in dataset")

    stat = ds.status()
    if specification_name not in stat:
        raise RuntimeError(
            f"Specification {specification_name} found in dataset, but not submitted?"
        )

    if set(stat[specification_name].keys()) != {RecordStatusEnum.complete}:
        raise RuntimeError(
            f"Specification {specification_name} not entirely complete for dataset {ds.name}"
        )

    if stat[specification_name][RecordStatusEnum.complete] != len(ds.entry_names):
        raise RuntimeError("Not all entries submitted/completed")


def create_dataset(
    client: PortalClient,
    dataset_name: str,
    fragment_data: Union[str, "pd.DataFrame"],
    qc_specification: Optional[QCSpecification] = None,
    *,
    specification_name: str = "default",
    specification_description: Optional[str] = None,
    submit: bool = False,
    compute_tag: Optional[str] = None,
    compute_priority: Optional[str] = None,
    find_existing: bool = True,
    verbose: int = 1,
) -> SinglepointDataset:
    df = load_fsapt_fragment_data(fragment_data)

    if qc_specification is None:
        qc_specification = default_fisapt0_specification()

    ds = client.add_dataset("singlepoint", dataset_name)

    entries = []
    for _, row in df.iterrows():
        entry_name = str(row["id"])
        attributes = {
            "fragments_a": dict(row["fragments_a"]),
            "fragments_b": dict(row["fragments_b"]),
            "source_id": row["id"],
        }

        entries.append(
            SinglepointDatasetEntry(
                name=entry_name,
                molecule=row["qcel_molecule"],
                attributes=attributes,
            )
        )

    meta = ds.add_entries(entries)
    if not meta.success:
        raise RuntimeError(
            f"Failed to add entries to dataset {ds.name}. Error:\n {meta.error_string}"
        )

    spec_meta = ds.add_specification(
        specification_name, qc_specification, specification_description
    )
    if not spec_meta.success:
        raise RuntimeError(
            f"Failed to add specification {specification_name} to dataset {ds.name}"
        )

    if submit:
        submit_kwargs = {
            "specification_names": [specification_name],
            "find_existing": find_existing,
        }
        if compute_tag is not None:
            submit_kwargs["compute_tag"] = compute_tag
        if compute_priority is not None:
            submit_kwargs["compute_priority"] = compute_priority

        ds.submit(**submit_kwargs)

    if verbose:
        print(f"Dataset {ds.name} [id {ds.id}] prepared with {len(entries)} entries")
        if submit:
            print(ds.status_table())

    return ds


def error_records(
    ds: SinglepointDataset, specification_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    records = []
    kwargs = {}
    if specification_name is not None:
        kwargs["specification_names"] = specification_name

    for entry_name, spec_name, rec in ds.iterate_records(**kwargs):
        if rec.status != RecordStatusEnum.error:
            continue

        records.append(
            {
                "entry_name": entry_name,
                "specification_name": spec_name,
                "record_id": rec.id,
                "error": rec.error,
                "stdout": rec.stdout,
                "stderr": rec.stderr,
            }
        )

    return records


def _require_two_fragments(mol: Any, entry_name: str):
    if len(mol.fragments) != 2:
        raise RuntimeError(
            f"Entry {entry_name} must contain exactly two molecular fragments"
        )


def _scale_optional_energy(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return value * HARTREE_TO_KCALMOL


def _all_fragment_row(
    entry_name: str, mol: Any, atomic_result, qcvars: Dict[str, Any]
) -> Dict[str, Any]:
    _require_two_fragments(mol, entry_name)

    monomer_a = mol.get_fragment(0)
    monomer_b = mol.get_fragment(1)
    len_monomer_a = len(monomer_a.atomic_numbers)
    len_monomer_b = len(monomer_b.atomic_numbers)

    return {
        "id": entry_name,
        "qcel_molecule": mol,
        "ZA": np.array(monomer_a.atomic_numbers),
        "ZB": np.array(monomer_b.atomic_numbers),
        "RA": np.array(monomer_a.geometry).reshape(-1, 3) * BOHR_TO_ANGSTROM,
        "RB": np.array(monomer_b.geometry).reshape(-1, 3) * BOHR_TO_ANGSTROM,
        "TQA": monomer_a.molecular_charge,
        "TQB": monomer_b.molecular_charge,
        "Frag1": "All",
        "Frag2": "All",
        "Frag1_indices": [list(range(1, len_monomer_a + 1))],
        "Frag2_indices": [
            list(range(len_monomer_a + 1, len_monomer_a + len_monomer_b + 1))
        ],
        "F-Electrostatics": _scale_optional_energy(qcvars.get("sapt elst energy")),
        "F-Exchange": _scale_optional_energy(qcvars.get("sapt exch energy")),
        "F-Induction": _scale_optional_energy(qcvars.get("sapt ind energy")),
        "F-Dispersion": _scale_optional_energy(qcvars.get("sapt disp energy")),
        "F-Total": _scale_optional_energy(qcvars.get("sapt total energy")),
        "analysis_type": "all",
        "record_id": atomic_result.extras.get("qcfractal_record_id"),
    }


def _fragment_results(
    atomic_result,
    len_monomer_a: int,
    fragments_a: Dict[str, Sequence[int]],
    fragments_b: Dict[str, Sequence[int]],
    *,
    analysis_type: str,
    links5050: bool,
) -> List[Dict[str, Any]]:
    fragment_rows = []

    fragments_b_shifted = {
        name: [int(idx) + len_monomer_a for idx in indices]
        for name, indices in fragments_b.items()
    }

    for frag_a_name, frag_a_indices in fragments_a.items():
        for frag_b_name, frag_b_indices in fragments_b_shifted.items():
            data = psi4.fsapt_analysis(
                source=atomic_result,
                fragments_a={frag_a_name: list(frag_a_indices)},
                fragments_b={frag_b_name: list(frag_b_indices)},
                analysis_type=analysis_type,
                links5050=links5050,
                print_output=False,
            )

            df = pd.DataFrame(data)
            df = df[(df["Frag1"] != "All") & (df["Frag2"] != "All")]
            if df.empty:
                continue

            for _, row in df.iterrows():
                fragment_rows.append(
                    {
                        "Frag1": row["Frag1"],
                        "Frag2": row["Frag2"],
                        "Frag1_indices": row["Frag1_indices"],
                        "Frag2_indices": row["Frag2_indices"],
                        "F-Electrostatics": row.get("Elst"),
                        "F-Exchange": row.get("Exch"),
                        "F-Induction": row.get("IndAB"),
                        "F-Dispersion": row.get("EDisp", row.get("Disp")),
                        "F-Total": row.get("Total"),
                    }
                )

    if not fragment_rows:
        return []

    df = pd.DataFrame(fragment_rows)
    df["_sort_key"] = (df["Frag1"] == "Other").astype(int) + (
        df["Frag2"] == "Other"
    ).astype(int)
    df = df.sort_values(["_sort_key", "Frag1", "Frag2"])
    df = df.drop_duplicates(subset=["F-Electrostatics", "F-Total"], keep="first")
    return df.drop(columns=["_sort_key"]).to_dict(orient="records")


def _analyze_datasets(
    specification_name: str,
    *datasets: SinglepointDataset,
    analysis_type: str = "reduced",
    links5050: bool = True,
) -> "pd.DataFrame":
    results = []

    for ds in datasets:
        for entry_name, _, rec in ds.iterate_records(
            specification_names=specification_name,
            status=RecordStatusEnum.complete,
        ):
            entry = ds.get_entry(entry_name)
            if entry is None:
                raise RuntimeError(f"Could not load dataset entry {entry_name}")

            mol = entry.molecule
            fragments_a = dict(entry.attributes["fragments_a"])
            fragments_b = dict(entry.attributes["fragments_b"])

            atomic_result = rec.to_qcschema_result()
            if atomic_result.molecule is None:
                raise RuntimeError(
                    f"Record {rec.id} for entry {entry_name} is missing molecule data"
                )

            atomic_result.extras = dict(atomic_result.extras)
            atomic_result.extras["qcfractal_record_id"] = rec.id
            qcvars = atomic_result.extras.get("extra_properties", {})

            if not qcvars:
                raise RuntimeError(
                    f"Record {rec.id} for entry {entry_name} does not contain Psi4 extra_properties"
                )

            all_row = _all_fragment_row(entry_name, mol, atomic_result, qcvars)
            all_row["dataset_name"] = ds.name
            results.append(all_row)

            for row in _fragment_results(
                atomic_result,
                len(mol.get_fragment(0).atomic_numbers),
                fragments_a,
                fragments_b,
                analysis_type=analysis_type,
                links5050=links5050,
            ):
                row.update(
                    {
                        "id": entry_name,
                        "qcel_molecule": mol,
                        "dataset_name": ds.name,
                        "analysis_type": analysis_type,
                        "record_id": rec.id,
                    }
                )
                results.append(row)

    return pd.DataFrame(results)


def analyze_dataset(
    client: PortalClient,
    dataset_name: str,
    specification_name: str,
    *,
    analysis_type: str = "reduced",
    links5050: bool = True,
) -> "pd.DataFrame":
    ds = client.get_dataset("singlepoint", dataset_name)
    _check_ds_complete(ds, specification_name)
    return _analyze_datasets(
        specification_name, ds, analysis_type=analysis_type, links5050=links5050
    )


def analyze_dataset_views(
    specification_name: str,
    *view_paths: str,
    analysis_type: str = "reduced",
    links5050: bool = True,
) -> "pd.DataFrame":
    datasets = [load_dataset_view(path) for path in view_paths]
    return _analyze_datasets(
        specification_name, *datasets, analysis_type=analysis_type, links5050=links5050
    )


def results_summary_str(
    results: Union["pd.DataFrame", Iterable[Dict[str, Any]]],
) -> str:
    if not isinstance(results, pd.DataFrame):
        results = pd.DataFrame(results)

    if results.empty:
        return "No F-SAPT results found"

    keep_columns = [
        "dataset_name",
        "id",
        "Frag1",
        "Frag2",
        "F-Electrostatics",
        "F-Exchange",
        "F-Induction",
        "F-Dispersion",
        "F-Total",
    ]

    summary_df = results.loc[
        :, [c for c in keep_columns if c in results.columns]
    ].copy()
    summary_df = summary_df.sort_values(
        [c for c in ["dataset_name", "id", "Frag1", "Frag2"] if c in summary_df.columns]
    )

    output = tabulate.tabulate(
        summary_df, headers="keys", tablefmt="simple", showindex=False, floatfmt=".8f"
    )

    if {"Frag1", "Frag2", "F-Total"}.issubset(results.columns):
        all_rows = results[(results["Frag1"] == "All") & (results["Frag2"] == "All")]
        if not all_rows.empty:
            output += "\n\n"
            output += f"Completed dimers: {len(all_rows)}\n"
            output += (
                f"Mean All-All total (kcal/mol): {all_rows['F-Total'].mean():.8f}\n"
            )
            output += (
                f"Min  All-All total (kcal/mol): {all_rows['F-Total'].min():.8f}\n"
            )
            output += (
                f"Max  All-All total (kcal/mol): {all_rows['F-Total'].max():.8f}\n"
            )

    return output


def print_results_summary(results: Union["pd.DataFrame", Iterable[Dict[str, Any]]]):
    print(results_summary_str(results))

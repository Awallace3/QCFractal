import psi4
import pytest


def test_fsapt_indices():
    """
    Test F-SAPT fragment index tracking in multi-fragment analysis.

    This test verifies that fsapt_analysis correctly tracks and reports
    which atom indices belong to each fragment pair in the output. Uses
    a more complex system (ethane + N-methylacetamide) with multiple
    user-defined fragments to test index bookkeeping.

    The test validates:
    1. QCSchema plan generation and execution
    2. Correct fragment index assignment in output dictionary
    3. Multi-fragment definitions with links5050 option
    4. DataFrame construction with fragment indices
    5. Molecules of different sizes

    NOTE: This takes a bit longer to run due to size...
    """
    # example testcase from tests/fsapt-allterms/input.dat
    # psi4.set_memory("1 GB")
    # psi4.set_num_threads(12)

    mol = psi4.geometry(
        """
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
"""
    )
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "df",
            "guess": "sad",
            "freeze_core": "true",
        }
    )
    plan = psi4.energy("fisapt0", return_plan=True, molecule=mol)
    atomic_result = psi4.schema_wrapper.run_qcschema(
        plan.plan(),
        # plan.plan(wfn_qcvars_only=False), # Needed if SAPT data not stored on dimer_wfn.
        # clean=True,
        # postclean=True,
    )
    print(atomic_result)
    print(dir(atomic_result))
    data = psi4.fsapt_analysis(
        source=atomic_result,
        # NOTE: 1-indexed for fragments_a and fragments_b
        fragments_a={
            "Methyl1_A": [1, 2, 7, 8],
            "Methyl2_A": [3, 4, 5, 6],
        },
        fragments_b={
            "Peptide_B": [9, 10, 11, 16, 26],
            "T-Butyl_B": [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        },
        links5050=True,
        print_output=False,
    )
    mol_qcel_dict = mol.to_schema(dtype=2)
    frag1_indices = data["Frag1_indices"]
    frag2_indices = data["Frag2_indices"]
    # Using molecule object for all test to ensure right counts from each
    # fragment are achieved. Note +1 for 1-indexing in fsapt_analysis
    all_A = [i + 1 for i in mol_qcel_dict["fragments"][0]]
    expected_frag1_indices = [
        [1, 2, 7, 8],
        [1, 2, 7, 8],
        [3, 4, 5, 6],
        [3, 4, 5, 6],
        [1, 2, 7, 8],
        [3, 4, 5, 6],
        all_A,
        all_A,
        all_A,
    ]
    all_B = [j + 1 for j in mol_qcel_dict["fragments"][1]]
    expected_frag2_indices = [
        [9, 10, 11, 16, 26],
        [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        [9, 10, 11, 16, 26],
        [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        all_B,
        all_B,
        [9, 10, 11, 16, 26],
        [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        all_B,
    ]
    for i, indices in enumerate(frag1_indices):
        # Assert lists are identical
        e = expected_frag1_indices[i]
        sorted_frag = sorted(indices)
        assert sorted_frag == e, (
            "Frag1 indices do not match for fragment "
            f"{i}: expected {e}, got {sorted_frag}"
        )

    for i, indices in enumerate(frag2_indices):
        e = expected_frag2_indices[i]
        sorted_frag = sorted(indices)
        assert sorted_frag == e, (
            "Frag2 indices do not match for fragment "
            f"{i}: expected {e}, got {sorted_frag}"
        )
    ref_dict = {
        "ClosestContact": [
            12.99840199731447,
            6.708905946098247,
            9.420620786025163,
            3.7293279474020324,
            6.708905946098247,
            3.7293279474020324,
            9.420620786025163,
            3.7293279474020324,
            3.7293279474020324,
        ],
        "Disp": [
            -0.003992035787563479,
            -0.06726003034293386,
            -0.013540224505783295,
            -0.4109667131624368,
            -0.07125206613049734,
            -0.4245069376682201,
            -0.017532260293346775,
            -0.4782267435053707,
            -0.49575900379871746,
        ],
        "EDisp": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "Elst": [
            0.7150992302947188,
            -0.2042449764424319,
            -0.8155064098920732,
            -0.9356370086747248,
            0.5108542538522869,
            -1.751143418566798,
            -0.10040717959735446,
            -1.1398819851171567,
            -1.2402891647145111,
        ],
        "Exch": [
            0.00013680800590309906,
            0.05310422532154325,
            0.03094443957782037,
            3.891362354110715,
            0.05324103332744635,
            3.922306793688535,
            0.031081247583723468,
            3.9444665794322584,
            3.9755478270159816,
        ],
        "Frag1": [
            "Methyl1_A",
            "Methyl1_A",
            "Methyl2_A",
            "Methyl2_A",
            "Methyl1_A",
            "Methyl2_A",
            "All",
            "All",
            "All",
        ],
        "Frag2": [
            "Peptide_B",
            "T-Butyl_B",
            "Peptide_B",
            "T-Butyl_B",
            "All",
            "All",
            "Peptide_B",
            "T-Butyl_B",
            "All",
        ],
        "IndAB": [
            -0.007088778360656281,
            -0.015599345365187771,
            -0.026015085738530286,
            -0.17479907268235031,
            -0.022688123725844053,
            -0.2008141584208806,
            -0.033103864099186565,
            -0.1903984180475381,
            -0.22350228214672466,
        ],
        "IndBA": [
            0.0003529407892942706,
            0.014707144982080604,
            -0.0017520016469106318,
            -0.08067863336939138,
            0.015060085771374875,
            -0.082430635016302,
            -0.0013990608576163613,
            -0.06597148838731078,
            -0.06737054924492714,
        ],
        "Total": [
            0.7045081649423892,
            -0.219292981846543,
            -0.8258692822083376,
            2.289280926225068,
            0.4852151830958462,
            1.4634116440167304,
            -0.12136111726594834,
            2.069987944378525,
            1.9486268271125766,
        ],
    }

    for key in [
        "ClosestContact",
        "Elst",
        "Exch",
        "IndAB",
        "IndBA",
        "Disp",
        "EDisp",
        "Total",
    ]:
        for i, value in enumerate(data[key]):
            f1_f2 = f"{data['Frag1'][i]}-{data['Frag2'][i]}"
            print(
                f1_f2,
                ref_dict[key][i],
                value,
            )
            assert compare_values(
                ref_dict[key][i],
                value,
                5,  # compares in kcal/mol, so looser tolerance
                f"Fragment pair {f1_f2}:{i} for key {key}",
            )
    return

def test_fsapt_AtomicOutput():
    """
    Test F-SAPT analysis using QCSchema AtomicResult output (no pandas).

    This test verifies that fsapt_analysis works with QCSchema AtomicResult
    objects returned from run_qcschema, using dictionary output format instead
    of pandas. This approach is useful for integration with QCArchive and
    other QCSchema-compatible workflows. Note, QCArchive will flatten
    arrays, so fsapt_ab_size handles reshaping.

    The test validates:
    1. QCSchema plan generation and execution via run_qcschema
    2. F-SAPT analysis from atomic_results parameter
    3. Dictionary output format without pandas dependency
    """
    mol = psi4.geometry(
        """0 1
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
no_com"""
    )
    psi4.set_options(
        {
            "basis": "jun-cc-pvdz",
            "scf_type": "df",
            "guess": "sad",
            "freeze_core": "true",
            "FISAPT_FSAPT_FILEPATH": "none",
        }
    )
    plan = psi4.energy("fisapt0", return_plan=True, molecule=mol)
    atomic_result = psi4.schema_wrapper.run_qcschema(
        plan.plan(),
        # plan.plan(wfn_qcvars_only=False), # Needed if SAPT data not stored on dimer_wfn.
        clean=True,
        postclean=True,
    )
    fEnergies = psi4.fsapt_analysis(
        source=atomic_result,
        # NOTE: 1-indexed for fragments_a and fragments_b
        fragments_a={
            "MethylA": [1, 2, 3, 4, 5],
        },
        fragments_b={
            "MethylB": [6, 7, 8, 9, 10],
        },
    )
    fEnergies = {
        "Elst": fEnergies["Elst"][0],
        "Exch": fEnergies["Exch"][0],
        "IndAB": fEnergies["IndAB"][0],
        "IndBA": fEnergies["IndBA"][0],
        "Disp": fEnergies["Disp"][0],
        "EDisp": fEnergies["EDisp"][0],
        "Total": fEnergies["Total"][0],
    }
    print(fEnergies)
    fEref = {
        "Elst": -0.0023867836548276955,
        "Exch": 0.00011242419533877543,
        "IndAB": -1.2055155927787574e-05,
        "IndBA": -1.1984667714276922e-05,
        "Disp": -0.020636082319331096,
        "EDisp": 0.0,
        "Total": -0.02293448160273215,
    }

    # python iterate over dictionary keys
    for k in fEref.keys():
        compare_values(fEref[k], fEnergies[k], 2, k)

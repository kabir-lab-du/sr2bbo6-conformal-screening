"""
Shared chemistry utilities: Bartel tau, Shannon radii, formula normalization.
Imported by 01h_merge_and_filter.py, 01j_screening_set.py, and others.
"""
import numpy as np
from pymatgen.core import Composition

SHANNON_R = {
    "Sr": {2: {12: 1.44}},
    "Ba": {2: {12: 1.61}},
    "Ca": {2: {12: 1.34}},
    "Ti": {4: {6: 0.605}},
    "V":  {3: {6: 0.640}, 4: {6: 0.580}, 5: {6: 0.540}},
    "Cr": {3: {6: 0.615}, 4: {6: 0.550}},
    "Mn": {2: {6: 0.830}, 3: {6: 0.645}, 4: {6: 0.530}},
    "Fe": {2: {6: 0.780}, 3: {6: 0.645}},
    "Co": {2: {6: 0.745}, 3: {6: 0.610}},
    "Ni": {2: {6: 0.690}},
    "Cu": {2: {6: 0.730}},
    "Zn": {2: {6: 0.740}},
    "Nb": {5: {6: 0.640}},
    "Mo": {4: {6: 0.650}, 6: {6: 0.590}},
    "Ru": {4: {6: 0.620}},
    "Rh": {3: {6: 0.665}},
    "Pd": {2: {6: 0.860}},
    "W":  {4: {6: 0.660}, 6: {6: 0.600}},
    "Re": {4: {6: 0.630}, 7: {6: 0.530}},
    "Ir": {4: {6: 0.625}},
    "Ta": {5: {6: 0.640}},
    "Hf": {4: {6: 0.710}},
    "Zr": {4: {6: 0.720}},
    "Sc": {3: {6: 0.745}},
    "Y":  {3: {6: 0.900}},
    "Al": {3: {6: 0.535}},
    "Ga": {3: {6: 0.620}},
    "In": {3: {6: 0.800}},
    "Ge": {4: {6: 0.530}},
    "Sn": {4: {6: 0.690}},
    "Sb": {3: {6: 0.760}, 5: {6: 0.600}},
    "Bi": {3: {6: 1.030}},
    "Te": {4: {6: 0.970}, 6: {6: 0.560}},
}
R_O = 1.40  # O2- CN=6


def get_r(el_str, ox_state, cn):
    el_data = SHANNON_R.get(el_str, {})
    ox_int = int(round(ox_state))
    for ox in [ox_int, ox_int - 1, ox_int + 1]:
        if ox in el_data:
            cn_data = el_data[ox]
            if cn in cn_data:
                return cn_data[cn]
            return list(cn_data.values())[0]
    return None


def compute_bartel_tau(formula):
    """
    Bartel 2019 tolerance factor for double perovskites.
    tau = r_O/r_B - n_A*(n_A - (r_A/r_B)/ln(r_A/r_B))
    """
    try:
        comp = Composition(formula)
        guesses = comp.oxi_state_guesses(max_sites=-1)
        if not guesses:
            return 999.0
        oxi = guesses[0]
        cations = {str(el): ox for el, ox in oxi.items() if str(el) != "O"}
        if not cations:
            return 999.0
        radii_12 = {}
        for el_str, ox in cations.items():
            r = get_r(el_str, ox, 12)
            if r is not None:
                radii_12[el_str] = r
        if not radii_12:
            return 999.0
        A_el = max(radii_12, key=lambda e: radii_12[e])
        rA = radii_12[A_el]
        nA = cations[A_el]
        B_els = [el for el in cations if el != A_el]
        rB_list = []
        for el_str in B_els:
            r = get_r(el_str, cations[el_str], 6)
            if r is not None:
                rB_list.append(r)
        if not rB_list:
            return 999.0
        rB = np.mean(rB_list)
        if rB <= 0 or rA <= 0:
            return 999.0
        ratio = rA / rB
        if ratio <= 0 or abs(ratio - 1.0) < 1e-6:
            return 999.0
        tau = R_O / rB - nA * (nA - ratio / np.log(ratio))
        return float(tau)
    except Exception:
        return 999.0


def norm_formula(f):
    try:
        return str(Composition(f).reduced_formula)
    except Exception:
        return f

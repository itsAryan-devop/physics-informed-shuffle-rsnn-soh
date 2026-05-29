"""
Physics-informed layer for the (Shuffle-)RSNN battery SOH model.

Implements Strategy B (feature engineering) and Strategy C (post-processing)
from physics_layer_blueprint.txt. Both strategies leave the reservoir
untouched; only the readout's input feature vector and the final predicted
sequence are modified.

Constraints encoded here:
    1. ECM / impedance-based SOH prior (NASA)
    2. sqrt(t) + linear cycle SEI-growth prior (all datasets)
    3. Arrhenius-weighted effective age (CALCE)
    4. Nernst / OCV consistency estimate (CALCE low-C OCV files)
    5. Monotonic irreversible degradation (all datasets, post-processing)
    6. Range / boundary clipping (all datasets, post-processing)

Design choices:
    * Everything is NumPy; no torch; fits inside the existing ridge-regression
      readout with zero dependency changes.
    * Functions return numpy arrays of the same length as the input so they
      drop in as new columns in summary_features or as in-place replacements
      for predictions.
    * Constants (E_a, k_cal, k_R, OCV coefficients) default to NMC-ballpark
      values from the NREL / Plett references cited in the blueprint.
    * Each function handles empty / degenerate input gracefully (returns
      zeros or identity) so callers can apply them blindly.
"""

from __future__ import annotations

import numpy as np

# Boltzmann constant in eV / K (for Arrhenius in eV units)
KB_EV = 8.617333262e-5

# Common NMC OCV coefficients (simplified two-term Nernst).
# OCV(z) ~ U_0 + alpha * ln(z / (1 - z)) + beta * z,  z = SOC in [0, 1].
NERNST_U0 = 3.5
NERNST_ALPHA = 0.03
NERNST_BETA = 0.8


# ---------------------------------------------------------------------------
# 1. ECM / impedance-based SOH prior (NASA).
# ---------------------------------------------------------------------------

def soh_ecm_prior(impedance_re, impedance_rct, k_R=0.30, floor=0.55,
                  ceil=1.15):
    """Per-cycle SOH prior from Randles ECM resistances.

    SOH_ecm = 1 - k_R * (R_aged - R_0) / R_0,
    where R_0 is the first non-zero R_aged value in the sequence.

    Parameters
    ----------
    impedance_re, impedance_rct : 1-D np.ndarray
        Ohmic and charge-transfer resistances per cycle.
    k_R : float
        Scaling that maps relative resistance growth to SOH drop.
    floor, ceil : float
        Clamp bounds for the prior (soft guard only).

    Returns
    -------
    np.ndarray of shape (n_cycles,)
    """
    r_re = np.asarray(impedance_re, dtype=float).copy()
    r_rct = np.asarray(impedance_rct, dtype=float).copy()
    r_total = r_re + r_rct

    # Replace missing / zero values with the first positive reading so that
    # sequences with sporadic EIS measurements remain usable.
    valid = r_total > 0
    if not valid.any():
        return np.ones_like(r_total)

    first_valid = r_total[valid][0]
    r_filled = np.where(valid, r_total, first_valid)

    prior = 1.0 - k_R * (r_filled - first_valid) / max(first_valid, 1e-9)
    return np.clip(prior, floor, ceil)


# ---------------------------------------------------------------------------
# 2. sqrt(t) + linear cycle SEI-growth prior.
# ---------------------------------------------------------------------------

def soh_sqrt_time_prior(cycle_index, days_from_first=None,
                        k_cal=5e-3, k_cyc=2e-4, floor=0.55, ceil=1.15):
    """SOH = 1 - k_cal * sqrt(days) - k_cyc * cycles, clipped.

    If calendar days are unavailable, fall back to cycle_index / 365 (an
    approximate "days if 1 cycle per day" surrogate). This keeps the prior
    informative on NASA / LG, where absolute timestamps are unknown.
    """
    cycles = np.asarray(cycle_index, dtype=float)
    if days_from_first is None:
        days = cycles / 1.0  # treat each cycle as ~1 day of aging
    else:
        days = np.asarray(days_from_first, dtype=float)

    prior = 1.0 - k_cal * np.sqrt(np.maximum(days, 0.0)) - k_cyc * cycles
    return np.clip(prior, floor, ceil)


# ---------------------------------------------------------------------------
# 3. Arrhenius-weighted effective age (CALCE).
# ---------------------------------------------------------------------------

def arrhenius_effective_age(throughput_ah, temperature_c,
                            E_a=0.4, T_ref_c=25.0):
    """Cumulative throughput weighted by Arrhenius factor relative to T_ref.

    effective_age[k] = sum_{j <= k} throughput_j * exp( (E_a / kB)
                                                        * (1/T_ref - 1/T_j) )
    with temperatures converted to Kelvin. Units: Ah-equivalent @ T_ref.
    """
    thr = np.asarray(throughput_ah, dtype=float)
    T_k = np.asarray(temperature_c, dtype=float) + 273.15
    T_ref_k = T_ref_c + 273.15
    factor = np.exp((E_a / KB_EV) * (1.0 / T_ref_k - 1.0 / np.maximum(T_k, 1.0)))
    return np.cumsum(thr * factor)


# ---------------------------------------------------------------------------
# 4. Nernst / OCV consistency (CALCE low-C OCV files).
# ---------------------------------------------------------------------------

def _invert_nernst(voltage, U_0=NERNST_U0, alpha=NERNST_ALPHA, beta=NERNST_BETA,
                   n_grid=401):
    """Solve V = U_0 + alpha * ln(z/(1-z)) + beta * z for z in (0,1) via a
    tabulated grid (fast for vector inputs, avoids brittle Newton steps)."""
    z = np.linspace(1e-3, 1.0 - 1e-3, n_grid)
    ocv_grid = U_0 + alpha * np.log(z / (1.0 - z)) + beta * z
    # For each requested voltage pick the z whose OCV is closest.
    # Both arrays are monotone in z, so searchsorted works.
    order = np.argsort(ocv_grid)
    sorted_ocv = ocv_grid[order]
    sorted_z = z[order]
    v = np.atleast_1d(np.asarray(voltage, dtype=float))
    idx = np.clip(np.searchsorted(sorted_ocv, v), 0, len(sorted_ocv) - 1)
    return sorted_z[idx]


def nernst_consistent_soh(voltage_end, throughput_ah, reference_capacity_ah,
                          is_low_current_mask=None,
                          U_0=NERNST_U0, alpha=NERNST_ALPHA, beta=NERNST_BETA,
                          floor=0.55, ceil=1.15):
    """Estimate SOH from V_end at end of a slow (near-OCV) discharge.

    Logic: extracted Ah = throughput_ah, remaining SOC fraction = z inferred
    from V_end. Then absolute current capacity = throughput_ah / (1 - z),
    and SOH = absolute_capacity / reference_capacity_ah.

    Rows not flagged as low-C OCV (where V_end is a cutoff voltage, not a
    rest OCV) fall back to the measured throughput / reference ratio so the
    feature is never missing.
    """
    v = np.asarray(voltage_end, dtype=float)
    thr = np.asarray(throughput_ah, dtype=float)
    c_ref = float(reference_capacity_ah) if reference_capacity_ah else 1.0

    z = _invert_nernst(v, U_0=U_0, alpha=alpha, beta=beta)
    denom = np.clip(1.0 - z, 1e-3, 1.0)
    capacity_est = thr / denom
    soh_est = capacity_est / max(c_ref, 1e-9)

    # Fallback for non-OCV rows — use the simple throughput ratio so the
    # column stays informative.
    fallback = thr / max(c_ref, 1e-9)
    if is_low_current_mask is not None:
        mask = np.asarray(is_low_current_mask, dtype=bool)
        soh_est = np.where(mask, soh_est, fallback)

    return np.clip(soh_est, floor, ceil)


# ---------------------------------------------------------------------------
# 5. Post-processing: monotonic projection with capacity-regeneration
#    tolerance.
# ---------------------------------------------------------------------------

def monotonic_projection(y_hat, cycle_order=None, epsilon_reg=0.005):
    """Project a per-cell prediction sequence onto a (nearly) monotone
    non-increasing curve, allowing small upward excursions up to
    ``epsilon_reg`` (default 0.5 percentage points) to model real capacity
    regeneration.

    We compute the pool-adjacent-violators isotonic baseline, then add back
    whatever "bonus" each point had above that baseline, clipped to
    epsilon_reg. Pure numpy — no scikit dependency required.
    """
    y = np.asarray(y_hat, dtype=float).copy()
    if y.size == 0:
        return y

    if cycle_order is None:
        order = np.arange(y.size)
    else:
        order = np.argsort(np.asarray(cycle_order))

    # Sort into time order for projection, then unsort.
    y_sorted = y[order]

    # Running minimum from the left of (y + epsilon_reg * position) — but
    # the classic isotonic-decreasing solution is simpler: enforce
    # y_sorted[i] <= y_sorted[i-1] by taking cumulative minimum.
    baseline = np.minimum.accumulate(y_sorted)
    bonus = np.clip(y_sorted - baseline, 0.0, epsilon_reg)
    projected_sorted = baseline + bonus

    # Unsort back to original row order.
    projected = np.empty_like(y_sorted)
    projected[order] = projected_sorted
    return projected


# ---------------------------------------------------------------------------
# 6. Post-processing: hard range clip.
# ---------------------------------------------------------------------------

def range_clip(y_hat, lo=0.55, hi=1.15):
    """Clip predicted relative SOH to a physically plausible band."""
    return np.clip(np.asarray(y_hat, dtype=float), lo, hi)


# ---------------------------------------------------------------------------
# 7. Post-processing: Coulomb-counting consistency clamp (opt-in).
# ---------------------------------------------------------------------------

def coulomb_clamp(y_hat, throughput_ah, reference_capacity_ah, tol=0.05):
    """If absolute-Ah reconstruction (y_hat * C_ref) diverges from the
    measured throughput by more than ``tol`` (fractional), pull it back to
    the tolerance band. Applies only where throughput is positive."""
    y = np.asarray(y_hat, dtype=float).copy()
    thr = np.asarray(throughput_ah, dtype=float)
    c_ref = float(reference_capacity_ah) if reference_capacity_ah else 1.0
    absolute_ah = y * c_ref
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_error = np.where(thr > 0, (absolute_ah - thr) / thr, 0.0)

    over_mask = rel_error > tol
    under_mask = rel_error < -tol
    absolute_ah = np.where(over_mask, thr * (1.0 + tol), absolute_ah)
    absolute_ah = np.where(under_mask, thr * (1.0 - tol), absolute_ah)
    return absolute_ah / max(c_ref, 1e-9)

"""
similarity_transform.py
=======================
Implements Khazanovich et al. (2001) structural similarity theory for
concrete pavement systems (single-layer Winkler plate model).

All computations are numpy-compatible and work on scalars or arrays.

Unit conventions (must be consistent throughout):
  Length : mm
  Force  : N
  Stress : MPa  = N/mm²
  E      : MPa
  k      : N/mm³  (Winkler subgrade modulus)
  gamma  : N/mm³  (unit weight)
  h      : mm
  alpha  : 1/°C
  P      : MPa    (applied pressure = load per unit area)

Typical concrete values:
  E     = 27600 – 41400 MPa
  h     = 152 – 355 mm
  k     = 0.027 – 0.166 N/mm³
  gamma = 2.36e-5 N/mm³  (≈ 23600 N/m³ converted to N/mm³)
  alpha = 9e-6 /°C
  mu    = 0.15
"""

import numpy as np


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

def gamma_SI_to_mm(gamma_Npm3: float) -> float:
    """Convert unit weight from N/m³ to N/mm³.

    ISLAB .in2 files store unit weight as N/m³ (e.g. 23600).
    Internal similarity calculations require N/mm³.
    """
    return gamma_Npm3 * 1e-9


def k_file_to_mm(k_file: float) -> float:
    """k_winkler from .in2 is already in N/mm³ (e.g. 0.0327).

    Actually ISLAB stores k in MPa/m = N/mm³ × 1000. Need to check:
    The sample file has k=32.68 in the FOUNDATION section and the parser
    reads it as k_winkler=32.68.  Looking at the output it says
    "Coefficient of subgrade reaction (k-value) = 32.680000"
    ISLAB2000 uses kPa/mm = N/mm³ units for k.
    So k=32.68 kPa/mm = 32.68×10⁻³ MPa/mm = 32.68×10⁻³ N/mm³.
    """
    # k in ISLAB output is in kPa/mm = 10^-3 N/mm^3
    # Convert to N/mm³
    return k_file * 1e-3  # kPa/mm → N/mm³ = MPa/m


# ---------------------------------------------------------------------------
# Radius of relative stiffness — Eq. 9
# ---------------------------------------------------------------------------

def compute_l(E: float, h: float, k: float, mu: float = 0.15) -> float:
    """Radius of relative stiffness for a single-layer Winkler plate.

    l = (E · h³ / (12 · k · (1 - μ²)))^(1/4)

    Parameters
    ----------
    E  : elastic modulus (MPa)
    h  : plate thickness (mm)
    k  : Winkler subgrade modulus (N/mm³ = MPa/m)
    mu : Poisson's ratio (dimensionless)

    Returns
    -------
    l  : radius of relative stiffness (mm)
    """
    D = (E * h ** 3) / (12.0 * (1.0 - mu ** 2))  # flexural rigidity, N·mm
    return (D / k) ** 0.25


# ---------------------------------------------------------------------------
# Korenev's non-dimensional temperature gradient — Eq. 12 (single-layer)
# ---------------------------------------------------------------------------

def compute_phi(
    alpha: float,
    mu: float,
    l: float,
    k: float,
    h: float,
    gamma: float,
    delta_T: float,
) -> float:
    """Korenev's non-dimensional temperature gradient φ.

    For a single-layer slab:
        φ = 2·α·(1+μ)·l²·k / (h²·γ) · ΔT

    Parameters
    ----------
    alpha   : thermal expansion coefficient (1/°C)
    mu      : Poisson's ratio
    l       : radius of relative stiffness (mm)
    k       : Winkler modulus (N/mm³)
    h       : slab thickness (mm)
    gamma   : unit weight (N/mm³)  — must be in N/mm³!
    delta_T : temperature difference top-to-bottom (°C)

    Returns
    -------
    phi : dimensionless temperature gradient (scalar or array)
    """
    numerator   = 2.0 * alpha * (1.0 + mu) * (l ** 2) * k * delta_T
    denominator = (h ** 2) * gamma
    return numerator / (denominator + 1e-30)


# ---------------------------------------------------------------------------
# Pressure / self-weight ratio — Eq. 8
# ---------------------------------------------------------------------------

def compute_PQ_ratio(pressure_q: float, gamma: float, h: float) -> float:
    """Dimensionless load-to-self-weight ratio P/Q.

    P   = applied pressure       (MPa = N/mm²)
    Q   = self weight per area   = gamma × h  (N/mm²)
    P/Q = pressure_q / (gamma × h)  [dimensionless]

    Parameters
    ----------
    pressure_q : applied surface pressure (MPa)
    gamma      : unit weight (N/mm³)
    h          : slab thickness (mm)
    """
    Q = gamma * h  # N/mm³ × mm = N/mm² = MPa
    return pressure_q / (Q + 1e-30)


# ---------------------------------------------------------------------------
# Convenience: compute all three similarity scalars at once
# ---------------------------------------------------------------------------

def compute_similarity_params(
    E: float,
    h: float,
    k_file: float,
    mu: float,
    alpha: float,
    gamma_file: float,
    delta_T: float,
    pressure_q: float,
) -> dict:
    """Compute all similarity parameters from raw .in2 parsed values.

    Parameters
    ----------
    E           : elastic modulus (MPa)
    h           : thickness (mm)
    k_file      : k_winkler as read from .in2 (kPa/mm)
    mu          : Poisson's ratio
    alpha       : thermal expansion coefficient (1/°C)
    gamma_file  : unit weight as read from .in2 (N/m³, e.g. 23600)
    delta_T     : temperature difference (°C)  — ΔT in paper
    pressure_q  : applied pressure (MPa)

    Returns
    -------
    dict with keys: l, phi, PQ
    """
    k     = k_file_to_mm(k_file)         # kPa/mm → N/mm³
    gamma = gamma_SI_to_mm(gamma_file)   # N/m³ → N/mm³

    l   = compute_l(E, h, k, mu)
    phi = compute_phi(alpha, mu, l, k, h, gamma, delta_T)
    PQ  = compute_PQ_ratio(pressure_q, gamma, h)

    return {"l": l, "phi": phi, "PQ": PQ}


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example from the paper / sample .in2 file
    params = compute_similarity_params(
        E           = 3.25e4,       # MPa
        h           = 327.12,       # mm
        k_file      = 32.68,        # kPa/mm  (from .in2 FOUNDATION line)
        mu          = 0.15,
        alpha       = 1.191e-5,     # 1/°C
        gamma_file  = 23600,        # N/m³
        delta_T     = 1.0,          # °C (temperature delta from .in2)
        pressure_q  = 0.845,        # MPa
    )
    print("Similarity parameters for sample case:")
    print(f"  l   = {params['l']:.2f} mm")
    print(f"  phi = {params['phi']:.6f}")
    print(f"  P/Q = {params['PQ']:.4f}")

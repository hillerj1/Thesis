#!/usr/bin/env python3
"""Generate ISW convergence (ground-state relative error vs N)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import finite_difference_quantum as fdq
from scipy.sparse.linalg import ArpackNoConvergence, eigs, eigsh


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ISW convergence figure (2nd vs 4th order).")
    p.add_argument(
        "--out",
        default="images/isw_convergence.png",
        help="Output PNG path (default: images/isw_convergence.png).",
    )
    p.add_argument(
        "--L",
        type=float,
        default=10.0,
        help="ISW domain length L for analytical reference (default: 10.0).",
    )
    p.add_argument(
        "--N-min",
        type=int,
        default=200,
        help="Minimum grid size N (default: 200).",
    )
    p.add_argument(
        "--N-max",
        type=int,
        default=2000,
        help="Maximum grid size N (default: 2000).",
    )
    p.add_argument(
        "--N-step",
        type=int,
        default=100,
        help="Step in N (default: 100).",
    )
    return p.parse_args()


def _analytical_isw_ground_energy(*, L: float) -> float:
    return float(np.pi**2 / (2.0 * (float(L) ** 2)))


def _numerical_isw_ground_energy(*, N: int, order: int, L: float) -> float:
    def V0(x):
        return np.zeros_like(x)

    H = fdq.hamiltonian(V0, 0.0, float(L), int(N), order=int(order))

    if int(order) == 2:
        ev = eigsh(
            H,
            k=1,
            which="SA",
            tol=1e-10,
            maxiter=200000,
            return_eigenvectors=False,
        )[0]
        return float(np.real(ev))

    # 4th-order boundary closures can make H slightly non-symmetric -> use eigs + shift-invert.
    try:
        ev = eigs(
            H,
            k=1,
            sigma=0.0,
            which="LM",
            tol=1e-10,
            maxiter=400000,
            return_eigenvectors=False,
        )[0]
    except ArpackNoConvergence as e:
        if getattr(e, "eigenvalues", None) is not None and len(e.eigenvalues) > 0:
            ev = e.eigenvalues[0]
        else:
            raise

    return float(np.real(ev))


def compute_isw_convergence(*, N_values: np.ndarray, L: float) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (errors_2nd, errors_4th, E_ref) where errors are relative ground-state errors
    vs the analytical ISW ground-state energy.
    """
    E_ref = _analytical_isw_ground_energy(L=L)

    errors_2nd: list[float] = []
    errors_4th: list[float] = []

    for N in map(int, N_values):
        E2 = _numerical_isw_ground_energy(N=N, order=2, L=L)
        E4 = _numerical_isw_ground_energy(N=N, order=4, L=L)

        errors_2nd.append(abs(E2 - E_ref) / abs(E_ref))
        errors_4th.append(abs(E4 - E_ref) / abs(E_ref))

    return np.asarray(errors_2nd, dtype=float), np.asarray(errors_4th, dtype=float), E_ref


def plot_isw_convergence(
    *,
    N_values: np.ndarray,
    errors_2nd: np.ndarray,
    errors_4th: np.ndarray,
    out_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "axes.linewidth": 1.1,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )

    fig = plt.figure(figsize=(7.2, 4.2))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(
        N_values,
        errors_2nd,
        "o-",
        color="#1f77b4",
        linewidth=2.0,
        markersize=5.5,
        label="2nd order",
    )
    ax.plot(
        N_values,
        errors_4th,
        "s--",
        color="#ff7f0e",
        linewidth=2.0,
        markersize=5.5,
        label="4th order",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both")

    ax.set_xlabel(r"Grid size $N$")
    ax.set_ylabel(r"Relative error (ground state)")
    ax.set_title("ISW convergence: ground-state relative error vs $N$")
    ax.legend(frameon=False, loc="lower left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    out_path = Path(args.out)

    if out_path.suffix == "":
        out_path = out_path.with_suffix(".png")
    if out_path.suffix.lower() != ".png":
        raise ValueError("Please provide a .png output path.")

    if args.N_step <= 0:
        raise ValueError("--N-step must be positive.")
    if args.N_min < 10:
        raise ValueError("--N-min is too small; use at least 10.")
    if args.N_max < args.N_min:
        raise ValueError("--N-max must be >= --N-min.")

    N_values = np.arange(args.N_min, args.N_max + 1, args.N_step, dtype=int)

    errors_2nd, errors_4th, _E_ref = compute_isw_convergence(N_values=N_values, L=float(args.L))
    plot_isw_convergence(N_values=N_values, errors_2nd=errors_2nd, errors_4th=errors_4th, out_path=out_path)

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


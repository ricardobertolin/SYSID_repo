"""
Activity 4 - Linear identification on REAL data (Ball-and-Beam)
==============================================================

Activity 4 from https://helonayala.github.io/sysid/linear_activities_BLS.html
applied to the real-world ball-and-beam datasets published at:

    https://github.com/helonayala/bab_datasets

Data is loaded exactly as in the package demo notebook
(https://github.com/helonayala/bab_datasets/blob/main/demo_bab_datasets.ipynb):

    data = nod.load_experiment('multisine_05', preprocess=True, plot=True,
                               end_idx=None, resample_factor=50,
                               zoom_last_n=10000, y_dot_method=velMethod)
    u, y, y_ref, y_dot = data

We use the 'multisine_05' broadband experiment and apply a simple 50/50
train/test split: the first half of the record identifies the model and the
second half validates it. The full BLS pipeline from Activities 1-3 is reused:
batch least squares estimation, one-step-ahead (OSA) and free-run (FR)
prediction, RMSE / R2 metrics, and a frequency-domain comparison.

Requires the bab_datasets package:
    pip install --upgrade git+https://github.com/helonayala/bab_datasets.git

Run with:  python activity4_ball_and_beam_real.py
"""

import numpy as np
import matplotlib.pyplot as plt

# Reuse the functions implemented for Activities 1-3
from linear_activities_BLS import (regression_matrix, free_run, evaluate,
                                    amplitude_spectrum)

try:
    import bab_datasets as nod
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The 'bab_datasets' package is required.\n"
        "Install it with:\n"
        "    pip install --upgrade git+https://github.com/helonayala/bab_datasets.git"
    ) from exc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXPERIMENT = "multisine_05"        # broadband excitation experiment
RESAMPLE_FACTOR = 50               # as in the demo notebook -> Ts = 0.05 s
Y_DOT_METHOD = "savgol"            # velMethod used in the demo

N_A, N_B = 2, 2                    # ARX orders (2nd order is enough here)
BIAS = True                        # affine term to absorb any output offset
TRAIN_FRACTION = 0.5              # 50/50 train/test split


# ---------------------------------------------------------------------------
# Data loading (via the bab_datasets package, exactly like the demo notebook)
# ---------------------------------------------------------------------------
def load_data():
    """Load the multisine_05 experiment and return (u, y, ts)."""
    data = nod.load_experiment(
        EXPERIMENT,
        preprocess=True,
        plot=False,                # set True to reproduce the demo's plots
        end_idx=None,
        resample_factor=RESAMPLE_FACTOR,
        zoom_last_n=10000,
        y_dot_method=Y_DOT_METHOD,
    )
    u = np.asarray(data.u, dtype=float).ravel()
    y = np.asarray(data.y, dtype=float).ravel()
    ts = float(data.sampling_time)
    return u, y, ts


def fit_arx(u, y, n_a, n_b, bias=BIAS):
    """Batch least squares ARX fit; returns (theta, Phi, Y)."""
    Phi, Y = regression_matrix(u, y, n_a, n_b, bias=bias)
    theta, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    return theta, Phi, Y


# ===========================================================================
# Activity 4 - 50/50 train/test identification on real multisine data
# ===========================================================================
def main():
    np.set_printoptions(precision=4, suppress=True)

    u, y, ts = load_data()
    n = u.size
    split = int(round(TRAIN_FRACTION * n))

    u_tra, y_tra = u[:split], y[:split]
    u_test, y_test = u[split:], y[split:]

    print("=" * 72)
    print(f"ACTIVITY 4 - REAL DATA ('{EXPERIMENT}', Ts={ts:.4f}s)")
    print("=" * 72)
    print(f"total samples : {n}")
    print(f"train (50%)   : {u_tra.size} samples")
    print(f"test  (50%)   : {u_test.size} samples")

    # ----- estimate the model on the training half (BLS) -------------------
    theta, Phi_tra, Y_tra = fit_arx(u_tra, y_tra, N_A, N_B)
    poles = np.roots(np.r_[1.0, theta[:N_A]])
    print(f"\nARX(n_a={N_A}, n_b={N_B}, bias={BIAS}) theta = "
          f"{np.array2string(theta, precision=4)}")
    print(f"AR pole magnitudes = {np.round(np.abs(poles), 4)} "
          f"({'stable' if np.all(np.abs(poles) < 1) else 'UNSTABLE'})")

    # ----- predictions: one-step-ahead and free-run, on both halves --------
    yhat_TRA_OSA = Phi_tra @ theta
    yhat_TRA_FR = free_run(theta, u_tra, y_tra, N_A, N_B, bias=BIAS)

    Phi_test, Y_test = regression_matrix(u_test, y_test, N_A, N_B, bias=BIAS)
    yhat_TEST_OSA = Phi_test @ theta
    yhat_TEST_FR = free_run(theta, u_test, y_test, N_A, N_B, bias=BIAS)

    # ----- metrics ---------------------------------------------------------
    print("\n" + "-" * 72)
    print(f"{'case':<12}{'RMSE':>14}{'R2':>12}")
    print("-" * 72)
    rows = [
        ("TRAIN  OSA", *evaluate(Y_tra, yhat_TRA_OSA)),
        ("TRAIN  FR ", *evaluate(Y_tra, yhat_TRA_FR)),
        ("TEST   OSA", *evaluate(Y_test, yhat_TEST_OSA)),
        ("TEST   FR ", *evaluate(Y_test, yhat_TEST_FR)),
    ]
    for name, rmse, r2 in rows:
        print(f"{name:<12}{rmse:>14.4e}{r2:>12.4f}")

    # ----- time-domain plot ------------------------------------------------
    t_tra = np.arange(Y_tra.size) * ts
    t_test = (split + np.arange(Y_test.size)) * ts

    fig, ax = plt.subplots(2, 1, figsize=(11, 7))
    ax[0].plot(t_tra, Y_tra, "k", lw=1.1, label="measured")
    ax[0].plot(t_tra, yhat_TRA_OSA, "--", lw=0.9, label="OSA")
    ax[0].plot(t_tra, yhat_TRA_FR, ":", lw=1.2, label="free-run")
    ax[0].set_title(f"TRAIN half (first 50%) - {EXPERIMENT}")
    ax[0].set_ylabel("y (ball position)")
    ax[0].legend(loc="upper right", fontsize=8)

    ax[1].plot(t_test, Y_test, "k", lw=1.1, label="measured")
    ax[1].plot(t_test, yhat_TEST_OSA, "--", lw=0.9, label="OSA")
    ax[1].plot(t_test, yhat_TEST_FR, ":", lw=1.2, label="free-run")
    ax[1].set_title("TEST half (last 50%)")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("y (ball position)")
    ax[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig("act4_multisine_time_domain.png", dpi=120)

    # ----- frequency-domain comparison (test half) -------------------------
    f, m_meas = amplitude_spectrum(Y_test, ts)
    _, m_osa = amplitude_spectrum(yhat_TEST_OSA, ts)
    _, m_fr = amplitude_spectrum(yhat_TEST_FR, ts)
    _, m_err = amplitude_spectrum(Y_test - yhat_TEST_FR, ts)

    fig2, ax2 = plt.subplots(figsize=(11, 5))
    ax2.semilogy(f, m_meas, "k", lw=1.3, label="measured")
    ax2.semilogy(f, m_osa, "--", label="OSA")
    ax2.semilogy(f, m_fr, ":", label="free-run")
    ax2.semilogy(f, m_err, "-.", label="error (meas-FR)")
    ax2.set_title(f"Amplitude spectrum - test half ({EXPERIMENT})")
    ax2.set_xlabel("frequency [Hz]")
    ax2.set_ylabel("|Y(f)|")
    ax2.legend(loc="upper right")
    fig2.tight_layout()
    fig2.savefig("act4_multisine_spectra.png", dpi=120)

    print("\nSaved figures: act4_multisine_time_domain.png, "
          "act4_multisine_spectra.png")
    # plt.show()  # uncomment for interactive display


if __name__ == "__main__":
    main()

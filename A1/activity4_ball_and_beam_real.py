"""
Activity 4 - Linear identification on REAL data (Ball-and-Beam)
==============================================================

Activity 4 from https://helonayala.github.io/sysid/linear_activities_BLS.html
applied to the real-world ball-and-beam datasets published at:

    https://github.com/helonayala/bab_datasets

The activity is carried out on the experiments from the ball-and-beam (bab)
collection:

    * random_steps_01..04 -> rich excitation        -> TRAINING
    * rampa_positiva       -> increasing operating point sweep  -> TEST
    * rampa_negativa       -> decreasing operating point sweep  -> TEST

Part A - Linearity assessment
    Train a single global linear ARX model on the random-steps experiment and
    test it (one-step-ahead AND free-run) on the increasing/decreasing ramps.
    Conclusion: is one linear model enough across the whole operating range?

Part B - Segmented (local) linear models
    Split each ramp record into contiguous operating-point bands and fit a
    dedicated local linear ARX model per band, showing that local linearization
    tracks the data far better than a single global model.

The ARX helpers (regression_matrix, free_run, evaluate) are reused from the
Activities 1-3 script so there is a single source of truth.

Run with:  python activity4_ball_and_beam_real.py
"""

import io
import os
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import decimate

# Reuse the functions implemented for Activities 1-3
from linear_activities_BLS import regression_matrix, free_run, evaluate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/helonayala/bab_datasets/main/data/"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Original recordings are at 1 kHz (Ts = 1 ms); the ball-and-beam dynamics are
# slow relative to that, so we decimate to keep the linear ARX well conditioned.
DECIMATE = 50                      # -> effective Ts = 0.05 s

# Model structure for the global model (Part A)
N_A, N_B = 3, 3
BIAS = True                        # affine term to absorb operating-point offset

FILES = {
    "random_steps_01": "03_random_steps_01.mat",
    "random_steps_02": "03_random_steps_02.mat",
    "random_steps_03": "03_random_steps_03.mat",
    "random_steps_04": "03_random_steps_04.mat",
    "rampa_positiva":  "01_rampa_positiva.mat",
    "rampa_negativa":  "02_rampa_negativa.mat",
}


# ---------------------------------------------------------------------------
# Data loading (downloads .mat files once, caches them under ./data)
# ---------------------------------------------------------------------------
def load_experiment(name, decimate_factor=DECIMATE):
    """Load a bab experiment and return (u, y, ts).

    The .mat files contain the keys: time, u, y, ref, yf, trigger.
    The sampling time is recovered from the time vector.
    """
    fname = FILES[name]
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"  downloading {fname} ...")
        raw = urllib.request.urlopen(DATA_URL + fname, timeout=120).read()
        with open(path, "wb") as fh:
            fh.write(raw)

    d = sio.loadmat(path)
    t = d["time"].ravel().astype(float)
    u = d["u"].ravel().astype(float)
    y = d["y"].ravel().astype(float)
    ts = float(np.mean(np.diff(t)))

    if decimate_factor and decimate_factor > 1:
        u = decimate(u, decimate_factor, ftype="fir")
        y = decimate(y, decimate_factor, ftype="fir")
        ts = ts * decimate_factor

    return u, y, ts


def fit_arx(u, y, n_a, n_b, bias=BIAS):
    """Batch least squares ARX fit; returns (theta, Phi, Y)."""
    Phi, Y = regression_matrix(u, y, n_a, n_b, bias=bias)
    theta, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    return theta, Phi, Y


# ===========================================================================
# Part A - Linearity assessment with a single global linear model
# ===========================================================================
def part_a():
    print("=" * 72)
    print("ACTIVITY 4 - PART A : single global linear model across operating points")
    print("=" * 72)

    # ----- training data: concatenate the random-step experiments ----------
    u_list, y_list = [], []
    for name in ["random_steps_01", "random_steps_02",
                 "random_steps_03", "random_steps_04"]:
        u, y, ts = load_experiment(name)
        u_list.append(u)
        y_list.append(y)
    u_tra = np.concatenate(u_list)
    y_tra = np.concatenate(y_list)
    print(f"\nTraining on random steps: {u_tra.size} samples (Ts={ts:.4f}s)")

    theta, Phi_tra, Y_tra = fit_arx(u_tra, y_tra, N_A, N_B)
    rmse_tra, r2_tra = evaluate(Y_tra, Phi_tra @ theta)
    print(f"Global ARX (n_a={N_A}, n_b={N_B}, bias={BIAS}) theta = "
          f"{np.array2string(theta, precision=4)}")
    print(f"TRAIN one-step-ahead : RMSE={rmse_tra:.4e}  R2={r2_tra:.4f}")

    # ----- test on the increasing / decreasing ramps -----------------------
    tests = ["rampa_positiva", "rampa_negativa"]
    fig, axes = plt.subplots(len(tests), 1, figsize=(11, 8), sharex=False)
    for ax, name in zip(axes, tests):
        u, y, ts = load_experiment(name)
        Phi, Y = regression_matrix(u, y, N_A, N_B, bias=BIAS)
        y_osa = Phi @ theta
        y_fr = free_run(theta, u, y, N_A, N_B, bias=BIAS)

        rmse_osa, r2_osa = evaluate(Y, y_osa)
        rmse_fr, r2_fr = evaluate(Y, y_fr)
        print(f"\n{name}:")
        print(f"  one-step-ahead : RMSE={rmse_osa:.4e}  R2={r2_osa:.4f}")
        print(f"  free-run       : RMSE={rmse_fr:.4e}  R2={r2_fr:.4f}")

        t = np.arange(Y.size) * ts
        ax.plot(t, Y, "k", lw=1.2, label="measured")
        ax.plot(t, y_osa, "--", lw=1.0, label=f"OSA (R2={r2_osa:.3f})")
        ax.plot(t, y_fr, ":", lw=1.2, label=f"free-run (R2={r2_fr:.3f})")
        # keep the view on the measured range (free-run may drift on this
        # near-integrating, closed-loop-collected system)
        pad = 0.5 * (Y.max() - Y.min())
        ax.set_ylim(Y.min() - pad, Y.max() + pad)
        ax.set_title(f"Global linear model on {name}")
        ax.set_ylabel("y (ball position)")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig("act4_partA_global.png", dpi=120)
    print("\nSaved: act4_partA_global.png")
    print("\nConclusion (Part A): one-step-ahead prediction is excellent over the\n"
          "whole sweep, but the free-run drifts strongly -> a single linear model\n"
          "does NOT capture the dynamics across all operating points (the\n"
          "ball-and-beam is nonlinear / near-integrating). This motivates Part B.")


# ===========================================================================
# Part B - Segmented (local) linear models per operating envelope
# ===========================================================================
def part_b(n_segments=6, na=N_A, nb=N_B):
    print("\n" + "=" * 72)
    print("ACTIVITY 4 - PART B : segmented local linear models per operating band")
    print("=" * 72)

    for name in ["rampa_positiva", "rampa_negativa"]:
        u, y, ts = load_experiment(name)
        n = u.size
        # contiguous segments: because the ramp sweeps the operating point
        # monotonically, each time slice corresponds to one operating band.
        edges = np.linspace(0, n, n_segments + 1, dtype=int)

        print(f"\n{name}: {n} samples split into {n_segments} operating bands")
        seg_r2_local, seg_r2_global, seg_centers = [], [], []

        # one global model on the whole ramp, for comparison
        theta_g, _, _ = fit_arx(u, y, na, nb)

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(np.arange(max(na, nb), n) * ts,
                y[max(na, nb):], "k", lw=1.0, label="measured")

        for s in range(n_segments):
            a, b = edges[s], edges[s + 1]
            us, ys = u[a:b], y[a:b]
            if us.size <= max(na, nb) + 5:
                continue
            theta_l, Phi_l, Y_l = fit_arx(us, ys, na, nb)

            # local model evaluated on its own band (one-step-ahead)
            _, r2_local = evaluate(Y_l, Phi_l @ theta_l)
            # global model evaluated on the same band
            Phi_g, Y_g = regression_matrix(us, ys, na, nb, bias=BIAS)
            _, r2_global = evaluate(Y_g, Phi_g @ theta_g)

            seg_r2_local.append(r2_local)
            seg_r2_global.append(r2_global)
            seg_centers.append(0.5 * (a + b) * ts)

            t_seg = np.arange(a + max(na, nb), b) * ts
            ax.plot(t_seg, Phi_l @ theta_l, lw=1.4,
                    label=f"local seg {s + 1} (R2={r2_local:.3f})")
            print(f"  band {s + 1}: y in [{ys.min():.2f},{ys.max():.2f}]  "
                  f"local OSA R2={r2_local:.4f}  | global OSA R2={r2_global:.4f}")

        ax.set_title(f"Part B - local linear models on {name}")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("y (ball position)")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(f"act4_partB_{name}.png", dpi=120)

        # bar chart: local vs global per band
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        idx = np.arange(len(seg_r2_local))
        ax2.bar(idx - 0.2, seg_r2_local, width=0.4, label="local model")
        ax2.bar(idx + 0.2, seg_r2_global, width=0.4, label="global model")
        ax2.set_xticks(idx)
        ax2.set_xticklabels([f"band {i + 1}" for i in idx])
        ax2.set_ylabel("one-step-ahead R2")
        ax2.set_ylim(min(0, min(seg_r2_global + seg_r2_local) * 1.05), 1.02)
        ax2.set_title(f"Local vs global per-band R2 - {name}")
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(f"act4_partB_{name}_R2.png", dpi=120)
        print(f"  Saved: act4_partB_{name}.png, act4_partB_{name}_R2.png")

    print("\nConclusion (Part B): local models fit their own operating band, and\n"
          "the per-band comparison quantifies how much a dedicated linear model\n"
          "per envelope improves on a single global linear model.")


def main():
    part_a()
    part_b()
    # plt.show()  # uncomment for interactive display


if __name__ == "__main__":
    main()

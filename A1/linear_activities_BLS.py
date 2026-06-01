"""
Linear System Identification with Batch Least Squares (BLS)
===========================================================

Implementation of Activities 1, 2 and 3 from:
    https://helonayala.github.io/sysid/linear_activities_BLS.html

Built on top of the BLS example:
    https://helonayala.github.io/sysid/batch_least_squares.html
and the matReg / freeRun helpers from:
    https://github.com/helonayala/narx_narendra/blob/main/narendra.ipynb

(Activity 4, the real-world ball-and-beam study, lives in
 activity4_ball_and_beam_real.py.)

The ARX model structure used throughout is

    y(k) = -a_1 y(k-1) - ... - a_na y(k-na)
           + b_1 u(k-1) + ... + b_nb u(k-nb)

which, stacking past samples into a regression vector

    phi(k) = [ -y(k-1), ..., -y(k-na), u(k-1), ..., u(k-nb) ]

becomes linear in the parameters:  y(k) = phi(k) . theta.
With this sign convention the estimated theta recovers the true ARX
coefficients directly.

Run with:  python linear_activities_BLS.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Reproducibility for any random component
RNG = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# True system (from the BLS example) - a discretized 2nd order linear system
# ---------------------------------------------------------------------------
# y(k) = -th[0] y(k-1) - th[1] y(k-2) + th[2] u(k-1) + th[3] u(k-2)
TH_TRUE = np.array([-1.89532909e00, 9.04837418e-01, 4.83341528e-04, 4.67491667e-04])
NA_TRUE, NB_TRUE = 2, 2
N = 1000          # number of samples
TS = 0.01         # sampling time [s]


# ===========================================================================
# Activity 1 - Regression-matrix function
# ===========================================================================
def regression_matrix(u, y, n_a, n_b, bias=False):
    """Build the ARX regression matrix Phi and target vector Y.

    For each k from p = max(n_a, n_b) up to N-1 a row is created:

        [ -y(k-1), -y(k-2), ..., -y(k-na),  u(k-1), u(k-2), ..., u(k-nb) ]

    and the corresponding target is y(k). This mirrors the `matReg`
    function from the narx_narendra repository (here the y columns carry
    an explicit minus sign so that theta matches the ARX coefficients).

    Parameters
    ----------
    u, y : 1-D arrays of equal length N (input and output time series)
    n_a  : autoregressive order (number of past outputs)
    n_b  : exogenous order (number of past inputs)
    bias : if True, append a constant column of ones (affine/intercept
           term). Useful when the data has an operating-point offset.

    Returns
    -------
    Phi : (N - p, n_a + n_b [+1]) regression matrix
    Y   : (N - p,) target vector
    """
    u = np.asarray(u, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if u.shape != y.shape:
        raise ValueError("u and y must have the same length")

    n = u.shape[0]
    p = max(n_a, n_b)                      # maximum delay / initial-condition gap
    rows = n - p
    Phi = np.zeros((rows, n_a + n_b + (1 if bias else 0)))

    # past outputs (note the minus sign -> theta recovers ARX coefficients)
    for i in range(1, n_a + 1):
        Phi[:, i - 1] = -y[p - i: n - i]
    # past inputs
    for i in range(1, n_b + 1):
        Phi[:, n_a + i - 1] = u[p - i: n - i]
    if bias:
        Phi[:, -1] = 1.0

    Y = y[p:n]
    return Phi, Y


# ===========================================================================
# Activity 2 - Free-run (infinite-step-ahead) simulation function
# ===========================================================================
def free_run(theta, u, y, n_a, n_b, bias=False):
    """Free-run simulation of the identified ARX model.

    The model is iterated forward using its OWN previously simulated
    outputs as regressors (not the measured data). Only the first
    p = max(n_a, n_b) samples are seeded from the measured output to
    provide initial conditions. Mirrors the `freeRun` helper from the
    narx_narendra repository.

    Parameters
    ----------
    theta : estimated parameter vector (length n_a + n_b [+1 if bias])
    u     : input signal (length N)
    y     : measured output, used only for the p initial conditions
    n_a, n_b : model orders
    bias  : whether theta includes a trailing constant (intercept) term

    Returns
    -------
    yhat_fr : free-run output aligned with the target y[p:] (length N - p)
    """
    theta = np.asarray(theta, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    n = u.shape[0]
    p = max(n_a, n_b)
    yhat = np.zeros(n)
    yhat[:p] = y[:p]                       # seed initial conditions

    for k in range(p, n):
        phi = np.empty(n_a + n_b + (1 if bias else 0))
        for i in range(1, n_a + 1):
            phi[i - 1] = -yhat[k - i]     # uses simulated past outputs
        for i in range(1, n_b + 1):
            phi[n_a + i - 1] = u[k - i]
        if bias:
            phi[-1] = 1.0
        yhat[k] = phi @ theta

    return yhat[p:]


# ===========================================================================
# Activity 3 - Evaluation metrics and spectral comparison
# ===========================================================================
def evaluate(y_true, y_pred):
    """RMSE and R^2 using sklearn implementations."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2


def amplitude_spectrum(x, ts):
    """Single-sided amplitude spectrum of a signal.

    Returns (freqs, magnitude) with the mean removed so the DC bin does
    not dominate the plot.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x - x.mean()
    nfft = x.shape[0]
    freqs = np.fft.rfftfreq(nfft, d=ts)
    mag = np.abs(np.fft.rfft(x)) * (2.0 / nfft)
    return freqs, mag


# ===========================================================================
# Data generation (from the BLS example)
# ===========================================================================
def generate_training_data():
    """Relay-feedback excitation: produces a self-oscillating square-ish input."""
    uamp, yr = 10.0, 1.0
    u = np.zeros(N)
    y = np.zeros(N)
    u[:2] = uamp                          # kick-start the oscillation
    for k in range(2, N):
        # relay controller drives the output around +/- yr
        if y[k - 1] >= yr:
            u[k] = -uamp
        elif y[k - 1] <= -yr:
            u[k] = uamp
        else:
            u[k] = u[k - 1]
        y[k] = (-TH_TRUE[0] * y[k - 1] - TH_TRUE[1] * y[k - 2]
                + TH_TRUE[2] * u[k - 1] + TH_TRUE[3] * u[k - 2])
    return u, y


def generate_test_data():
    """Multi-sine excitation (open-loop simulation of the true model)."""
    uamp = 10.0
    k = np.arange(N)
    u = (uamp / 4 * np.sin(2 * np.pi * k * TS)
         + uamp / 4 * np.sin(np.pi / 2 * k * TS)
         + uamp / 4 * np.sin(np.pi * k * TS)
         + uamp / 4 * np.sin(np.pi / 4 * k * TS))
    y = np.zeros(N)
    for i in range(2, N):
        y[i] = (-TH_TRUE[0] * y[i - 1] - TH_TRUE[1] * y[i - 2]
                + TH_TRUE[2] * u[i - 1] + TH_TRUE[3] * u[i - 2])
    return u, y


# ===========================================================================
# Driver
# ===========================================================================
def main():
    np.set_printoptions(precision=6, suppress=True)

    u_tra, y_tra = generate_training_data()
    u_test, y_test = generate_test_data()

    # ----- Activity 1: build regression matrices and estimate via BLS ------
    Phi_TRA, y_target_TRA = regression_matrix(u_tra, y_tra, NA_TRUE, NB_TRUE)
    Phi_TEST, y_target_TEST = regression_matrix(u_test, y_test, NA_TRUE, NB_TRUE)

    th_hat, *_ = np.linalg.lstsq(Phi_TRA, y_target_TRA, rcond=None)

    print("=" * 70)
    print("ACTIVITY 1 - Batch Least Squares estimation (n_a=%d, n_b=%d)"
          % (NA_TRUE, NB_TRUE))
    print("=" * 70)
    print("Phi_TRA shape :", Phi_TRA.shape)
    print("Phi_TEST shape:", Phi_TEST.shape)
    print("theta true    :", TH_TRUE)
    print("theta hat     :", th_hat)
    print("abs error     :", np.abs(TH_TRUE - th_hat))

    # one-step-ahead (OSA) predictions
    yhat_TRA_OSA = Phi_TRA @ th_hat
    yhat_TEST_OSA = Phi_TEST @ th_hat

    # ----- Activity 2: free-run simulation --------------------------------
    yhat_TRA_FR = free_run(th_hat, u_tra, y_tra, NA_TRUE, NB_TRUE)
    yhat_TEST_FR = free_run(th_hat, u_test, y_test, NA_TRUE, NB_TRUE)

    # ----- Activity 3: metrics --------------------------------------------
    print("\n" + "=" * 70)
    print("ACTIVITY 3 - Performance metrics")
    print("=" * 70)
    rows = [
        ("TRAIN  OSA", *evaluate(y_target_TRA, yhat_TRA_OSA)),
        ("TRAIN  FR ", *evaluate(y_target_TRA, yhat_TRA_FR)),
        ("TEST   OSA", *evaluate(y_target_TEST, yhat_TEST_OSA)),
        ("TEST   FR ", *evaluate(y_target_TEST, yhat_TEST_FR)),
    ]
    print(f"{'case':<12}{'RMSE':>14}{'R2':>12}")
    for name, rmse, r2 in rows:
        print(f"{name:<12}{rmse:>14.6e}{r2:>12.6f}")

    # ---------- time-domain plots (Activities 1 & 2) ----------------------
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(y_target_TRA, "k", lw=1.2, label="measured")
    ax[0].plot(yhat_TRA_OSA, "--", label="OSA")
    ax[0].plot(yhat_TRA_FR, ":", label="free-run")
    ax[0].set_title("Training data (relay feedback)")
    ax[0].set_ylabel("y")
    ax[0].legend(loc="upper right")

    ax[1].plot(y_target_TEST, "k", lw=1.2, label="measured")
    ax[1].plot(yhat_TEST_OSA, "--", label="OSA")
    ax[1].plot(yhat_TEST_FR, ":", label="free-run")
    ax[1].set_title("Test data (multi-sine)")
    ax[1].set_xlabel("sample k")
    ax[1].set_ylabel("y")
    ax[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("act12_time_domain.png", dpi=120)

    # ---------- Activity 3: best vs worst model by R^2 (free-run, test) ----
    print("\n" + "=" * 70)
    print("ACTIVITY 3 - Model-order sweep (free-run R2 on test data)")
    print("=" * 70)
    candidates = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 2), (2, 3), (3, 3), (4, 4)]
    results = []
    for na, nb in candidates:
        Phi_tra, y_tgt_tra = regression_matrix(u_tra, y_tra, na, nb)
        th, *_ = np.linalg.lstsq(Phi_tra, y_tgt_tra, rcond=None)
        _, y_tgt_test = regression_matrix(u_test, y_test, na, nb)
        y_fr = free_run(th, u_test, y_test, na, nb)
        _, r2 = evaluate(y_tgt_test, y_fr)
        results.append((na, nb, r2, th))
        print(f"  n_a={na}, n_b={nb}  ->  free-run R2 = {r2:.6f}")

    best = max(results, key=lambda r: r[2])
    worst = min(results, key=lambda r: r[2])
    print(f"\n  BEST : n_a={best[0]}, n_b={best[1]}  (R2={best[2]:.6f})")
    print(f"  WORST: n_a={worst[0]}, n_b={worst[1]}  (R2={worst[2]:.6f})")

    # ---------- Activity 3: spectral comparison ---------------------------
    # Compare amplitude spectra of measured, OSA, FR and the errors, for the
    # best and worst models (evaluated on the test set).
    fig2, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax_s, (na, nb, r2, th), label in zip(axes, (best, worst),
                                             ("BEST", "WORST")):
        _, y_tgt = regression_matrix(u_test, y_test, na, nb)
        y_osa = regression_matrix(u_test, y_test, na, nb)[0] @ th
        y_fr = free_run(th, u_test, y_test, na, nb)
        err = y_tgt - y_fr

        f, m_meas = amplitude_spectrum(y_tgt, TS)
        _, m_osa = amplitude_spectrum(y_osa, TS)
        _, m_fr = amplitude_spectrum(y_fr, TS)
        _, m_err = amplitude_spectrum(err, TS)

        ax_s.semilogy(f, m_meas, "k", lw=1.3, label="measured")
        ax_s.semilogy(f, m_osa, "--", label="OSA")
        ax_s.semilogy(f, m_fr, ":", label="free-run")
        ax_s.semilogy(f, m_err, "-.", label="error (meas-FR)")
        ax_s.set_title(f"{label} model  n_a={na}, n_b={nb}  (free-run R2={r2:.4f})")
        ax_s.set_ylabel("|Y(f)|")
        ax_s.legend(loc="upper right")
    axes[-1].set_xlabel("frequency [Hz]")
    fig2.tight_layout()
    fig2.savefig("act3_spectra.png", dpi=120)

    print("\nSaved figures: act12_time_domain.png, act3_spectra.png")
    # plt.show()  # uncomment to display interactively


if __name__ == "__main__":
    main()

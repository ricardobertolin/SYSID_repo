"""
NARMAX on Ball-and-Beam Data
============================

Applies the NARX/FROLS nonlinear system identification approach
(from narmax_example_drone.ipynb) to the ball-and-beam dataset
used in A2, using the same experiment and pre-processing settings.

Data:  bab_datasets package, experiment 'multisine_05'
       pip install git+https://github.com/helonayala/bab_datasets.git

Run with:  python narmax_ball_and_beam.py
"""

import os
import sys
from itertools import combinations_with_replacement

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    import bab_datasets as nod
except ImportError as exc:
    raise SystemExit(
        "The 'bab_datasets' package is required.\n"
        "Install it with:\n"
        "    pip install git+https://github.com/helonayala/bab_datasets.git"
    ) from exc

# ---------------------------------------------------------------------------
# Configuration  (mirrors A2 data settings)
# ---------------------------------------------------------------------------
EXPERIMENT     = "multisine_05"
RESAMPLE_FACTOR = 50
Y_DOT_METHOD   = "savgol"
TRAIN_FRACTION = 0.5

# NARX model hyper-parameters
NY             = 2      # output lags  (matches A2 N_A)
NU             = 2      # input lags   (matches A2 N_B)
POLY_ORDER     = 3      # polynomial expansion order (1 = linear ARX)
NO_OF_TERMS    = 10     # number of terms FROLS selects


# ---------------------------------------------------------------------------
# Helper: evaluate (RMSE, R²)
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred):
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return rmse, r2


# ---------------------------------------------------------------------------
# NARX regression matrix builders
# ---------------------------------------------------------------------------
def regMatARX(y_signal_in, u_signal_in, ny, nu):
    y = np.asarray(y_signal_in, dtype=float)
    u = np.asarray(u_signal_in, dtype=float)
    max_lag = max(ny, nu)
    y_target = y[max_lag:]
    colnames = [f"y(k-{i})" for i in range(1, ny + 1)] + \
               [f"u(k-{i})" for i in range(1, nu + 1)]
    rows = []
    for k in range(max_lag, len(y)):
        row = [y[k - j] for j in range(1, ny + 1)] + \
              [u[k - j] for j in range(1, nu + 1)]
        rows.append(row)
    P0 = np.array(rows, dtype=float)
    return P0, colnames, y_target


def regMatNARX(u_signal_in, y_signal_in, nu, ny, poly_order):
    y = np.asarray(y_signal_in, dtype=float)
    u = np.asarray(u_signal_in, dtype=float)
    P0, P0_names, y_target = regMatARX(y, u, ny, nu)
    NP = len(y_target)
    n_base = P0.shape[1]

    cols = [np.ones((NP, 1))]
    names = ["constant"]
    cols.append(P0)
    names.extend(P0_names)

    if poly_order >= 2:
        for order in range(2, poly_order + 1):
            for idx_tuple in combinations_with_replacement(range(n_base), order):
                term_name = "".join(P0_names[i] for i in idx_tuple)
                names.append(term_name)
                term_col = np.prod(P0[:, list(idx_tuple)], axis=1, keepdims=True)
                cols.append(term_col)

    P = np.concatenate(cols, axis=1)
    return P, names, y_target


# ---------------------------------------------------------------------------
# FROLS algorithm
# ---------------------------------------------------------------------------
def frols_py(P, Y_in, n_terms, colnames=None, eps=1e-12):
    Y = Y_in.reshape(-1, 1)
    NP, M = P.shape
    sig_yy = float(Y.T @ Y) or eps

    sel_idx = []
    err_list = []
    g_list = []
    Q = np.empty((NP, 0))
    A = np.empty((0, 0))

    for step in range(min(n_terms, M)):
        best_err = -np.inf
        best_i = -1
        best_q = None
        best_g = 0.0

        for m in range(M):
            if m in sel_idx:
                continue
            p_m = P[:, m:m+1]
            q = p_m.copy()
            for r in range(Q.shape[1]):
                q_r = Q[:, r:r+1]
                denom = float(q_r.T @ q_r)
                if denom >= eps:
                    q -= (float(p_m.T @ q_r) / denom) * q_r
            q_norm = float(q.T @ q)
            if q_norm < eps:
                continue
            g = float(Y.T @ q) / q_norm
            err = g ** 2 * q_norm / sig_yy
            if err > best_err:
                best_err, best_i, best_q, best_g = err, m, q, g

        if best_i == -1:
            break

        sel_idx.append(best_i)
        err_list.append(best_err)
        g_list.append(best_g)
        Q = best_q if Q.shape[1] == 0 else np.hstack((Q, best_q))

        # Update upper-triangular A
        p_new = P[:, best_i:best_i+1]
        if A.size == 0:
            A = np.array([[1.0]])
        else:
            col = np.zeros((A.shape[0], 1))
            for r in range(Q.shape[1] - 1):
                q_r = Q[:, r:r+1]
                denom = float(q_r.T @ q_r)
                if denom >= eps:
                    col[r, 0] = float(p_new.T @ q_r) / denom
            A = np.block([[A, col], [np.zeros((1, A.shape[1])), np.ones((1, 1))]])

    if not sel_idx:
        return {}

    g_vec = np.array(g_list).reshape(-1, 1)
    theta = np.linalg.solve(A, g_vec).flatten()
    return {
        "th": theta,
        "selected_indices": sel_idx,
        "Psel_colnames": [colnames[i] for i in sel_idx] if colnames else [],
        "ERR_values": np.array(err_list),
    }


# ---------------------------------------------------------------------------
# NARX model class
# ---------------------------------------------------------------------------
class NARX:
    def __init__(self, ny, nu, poly_order, n_terms):
        self.ny = ny
        self.nu = nu
        self.poly_order = poly_order
        self.n_terms = n_terms
        self._max_lag = max(ny, nu)
        self._P0_names = ([f"y(k-{i})" for i in range(1, ny + 1)] +
                          [f"u(k-{i})" for i in range(1, nu + 1)])
        self._n_base = len(self._P0_names)
        # built during fit
        self.theta_ = None
        self.selected_indices_ = None
        self.selected_colnames_ = None
        self.candidate_colnames_ = None
        self.err_values_ = None

    def fit(self, u, y):
        P, names, y_t = regMatNARX(u, y, self.nu, self.ny, self.poly_order)
        self.candidate_colnames_ = names
        res = frols_py(P, y_t, self.n_terms, names)
        self.theta_            = res["th"]
        self.selected_indices_ = res["selected_indices"]
        self.selected_colnames_= res["Psel_colnames"]
        self.err_values_       = res["ERR_values"]
        return self

    # -- One-step-ahead prediction --
    def predict_osa(self, u, y):
        P, _, y_t = regMatNARX(u, y, self.nu, self.ny, self.poly_order)
        y_hat = P[:, self.selected_indices_] @ self.theta_
        return y_hat, y_t

    # -- Free-run simulation --
    def simulate(self, u, y_init):
        u = np.asarray(u, dtype=float)
        N = len(u)
        y_hat = np.zeros(N)
        y_hat[:self._max_lag] = np.asarray(y_init[:self._max_lag], dtype=float)

        # build a lookup for candidate column indices
        def _row_values(y_buf, u_arr, k):
            P0_vals = [y_buf[k - j] for j in range(1, self.ny + 1)] + \
                      [u_arr[k - j]  for j in range(1, self.nu + 1)]
            all_terms = {"constant": 1.0}
            for i, name in enumerate(self._P0_names):
                all_terms[name] = P0_vals[i]
            if self.poly_order >= 2:
                for order in range(2, self.poly_order + 1):
                    for idx_t in combinations_with_replacement(range(self._n_base), order):
                        name = "".join(self._P0_names[i] for i in idx_t)
                        all_terms[name] = np.prod([P0_vals[i] for i in idx_t])
            return np.array([all_terms[c] for c in self.candidate_colnames_])

        for k in range(self._max_lag, N):
            row = _row_values(y_hat, u, k)
            y_hat[k] = row[self.selected_indices_] @ self.theta_

        return y_hat[self._max_lag:]


# ---------------------------------------------------------------------------
# Data loading  (same settings as A2)
# ---------------------------------------------------------------------------
def load_data():
    data = nod.load_experiment(
        EXPERIMENT,
        preprocess=True,
        plot=False,
        end_idx=None,
        resample_factor=RESAMPLE_FACTOR,
        zoom_last_n=10000,
        y_dot_method=Y_DOT_METHOD,
    )
    u  = np.asarray(data.u, dtype=float).ravel()
    y  = np.asarray(data.y, dtype=float).ravel()
    ts = float(data.sampling_time)
    return u, y, ts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    np.set_printoptions(precision=4, suppress=True)

    u, y, ts = load_data()
    n     = len(u)
    split = int(round(TRAIN_FRACTION * n))

    u_tra, y_tra = u[:split], y[:split]
    u_tst, y_tst = u[split:], y[split:]

    print("=" * 70)
    print(f"NARMAX – Ball-and-Beam  ('{EXPERIMENT}', Ts={ts:.4f}s)")
    print("=" * 70)
    print(f"total samples : {n}   train : {split}   test : {n - split}")
    print(f"Model  ny={NY}  nu={NU}  poly_order={POLY_ORDER}  n_terms={NO_OF_TERMS}")

    # --- Fit on training data ---
    model = NARX(ny=NY, nu=NU, poly_order=POLY_ORDER, n_terms=NO_OF_TERMS)
    model.fit(u_tra, y_tra)

    print("\nSelected terms & parameters:")
    for name, th, err in zip(model.selected_colnames_,
                              model.theta_,
                              model.err_values_):
        print(f"  {name:<30s}  theta={th:+.6f}   ERR={err*100:.4f}%")
    print(f"  Total ERR explained : {model.err_values_.sum()*100:.6f}%")

    # --- OSA ---
    yhat_tra_osa, Y_tra = model.predict_osa(u_tra, y_tra)
    yhat_tst_osa, Y_tst = model.predict_osa(u_tst, y_tst)

    rmse_tra, r2_tra = evaluate(Y_tra, yhat_tra_osa)
    rmse_tst, r2_tst = evaluate(Y_tst, yhat_tst_osa)
    print(f"\n{'':30s}  {'RMSE':>12}  {'R²':>8}")
    print(f"  {'OSA – train':30s}  {rmse_tra:12.4e}  {r2_tra:8.4f}")
    print(f"  {'OSA – test':30s}  {rmse_tst:12.4e}  {r2_tst:8.4f}")

    # --- Free-run simulation (training data) ---
    max_lag = model._max_lag
    print(f"\nRunning free-run simulation ({len(u_tra) - max_lag} steps)...")
    yhat_tra_fr = model.simulate(u_tra, y_tra[:max_lag])
    Y_tra_fr    = y_tra[max_lag:]
    rmse_fr, r2_fr = evaluate(Y_tra_fr, yhat_tra_fr)
    print(f"  {'FR  – train':30s}  {rmse_fr:12.4e}  {r2_fr:8.4f}")

    # --- Figure 1: time-domain OSA (train + test) ---
    t_tra = np.arange(len(Y_tra)) * ts
    t_tst = (split + np.arange(len(Y_tst))) * ts

    fig1, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    axes[0].plot(t_tra, Y_tra, "k", lw=0.8, label="measured")
    axes[0].plot(t_tra, yhat_tra_osa, "--", lw=1.0,
                 label=f"NARX OSA  R²={r2_tra:.4f}")
    axes[0].set_title(f"Training half – {EXPERIMENT}")
    axes[0].set_ylabel("ball position")
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    axes[1].plot(t_tst, Y_tst, "k", lw=0.8, label="measured")
    axes[1].plot(t_tst, yhat_tst_osa, "--", lw=1.0,
                 label=f"NARX OSA  R²={r2_tst:.4f}")
    axes[1].set_title("Test half")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("ball position")
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    fig1.suptitle(
        f"NARX OSA – Ball-and-Beam  "
        f"(ny={NY}, nu={NU}, order={POLY_ORDER}, {NO_OF_TERMS} terms)",
        fontsize=11,
    )
    fig1.tight_layout()
    fname1 = os.path.join(HERE, "bab_narmax_osa.png")
    fig1.savefig(fname1, dpi=120)
    print(f"\nSaved: {fname1}")

    # --- Figure 2: Free-run simulation ---
    t_fr = np.arange(len(Y_tra_fr)) * ts

    fig2, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_fr, Y_tra_fr, "k", lw=0.8, label="measured")
    ax.plot(t_fr, yhat_tra_fr, "--", lw=1.0,
            label=f"NARX FR  R²={r2_fr:.4f}")
    ax.set_title(
        f"Free-Run Simulation – Ball-and-Beam  "
        f"(ny={NY}, nu={NU}, order={POLY_ORDER}, {NO_OF_TERMS} terms)"
    )
    ax.set_xlabel("time [s]")
    ax.set_ylabel("ball position")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig2.tight_layout()
    fname2 = os.path.join(HERE, "bab_narmax_fr.png")
    fig2.savefig(fname2, dpi=120)
    print(f"Saved: {fname2}")

    # --- Figure 3: ERR bar chart ---
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.bar(range(len(model.selected_colnames_)),
            model.err_values_ * 100,
            tick_label=model.selected_colnames_)
    ax3.set_xlabel("selected term")
    ax3.set_ylabel("ERR (%)")
    ax3.set_title("Error Reduction Ratio per selected term")
    plt.xticks(rotation=25, ha="right")
    fig3.tight_layout()
    fname3 = os.path.join(HERE, "bab_narmax_err.png")
    fig3.savefig(fname3, dpi=120)
    print(f"Saved: {fname3}")

    plt.show()


def grid_search():
    """Sweep NY, NU, POLY_ORDER and print a table sorted by FR R²."""
    NY_vals         = [1, 2, 3, 4, 5]
    NU_vals         = [1, 2, 3, 4, 5]
    POLY_ORDER_vals = [1, 2, 3]
    N_TERMS         = 10   # fixed across all runs

    u, y, ts = load_data()
    n     = len(u)
    split = int(round(TRAIN_FRACTION * n))
    u_tra, y_tra = u[:split], y[:split]
    u_tst, y_tst = u[split:], y[split:]

    total = len(NY_vals) * len(NU_vals) * len(POLY_ORDER_vals)
    print(f"Grid search: {total} configurations  (n_terms={N_TERMS} fixed)\n")

    header = f"{'ny':>4} {'nu':>4} {'ord':>4} {'#cand':>6}  "  \
             f"{'OSA_tr':>8} {'OSA_te':>8} {'FR_tr':>8}"
    print(header)
    print("-" * len(header))

    results = []
    done = 0
    for ny in NY_vals:
        for nu in NU_vals:
            for poly in POLY_ORDER_vals:
                done += 1
                try:
                    model = NARX(ny=ny, nu=nu, poly_order=poly, n_terms=N_TERMS)
                    model.fit(u_tra, y_tra)

                    # count candidates
                    n_cand = len(model.candidate_colnames_)

                    yhat_tr_osa, Y_tr = model.predict_osa(u_tra, y_tra)
                    yhat_te_osa, Y_te = model.predict_osa(u_tst, y_tst)
                    _, r2_osa_tr = evaluate(Y_tr, yhat_tr_osa)
                    _, r2_osa_te = evaluate(Y_te, yhat_te_osa)

                    max_lag = model._max_lag
                    yhat_fr = model.simulate(u_tra, y_tra[:max_lag])
                    Y_fr    = y_tra[max_lag:]
                    _, r2_fr = evaluate(Y_fr, yhat_fr)

                    results.append((ny, nu, poly, n_cand,
                                    r2_osa_tr, r2_osa_te, r2_fr))

                    print(f"{ny:>4} {nu:>4} {poly:>4} {n_cand:>6}  "
                          f"{r2_osa_tr:>8.4f} {r2_osa_te:>8.4f} {r2_fr:>8.4f}"
                          f"  [{done}/{total}]")
                except Exception as exc:
                    print(f"{ny:>4} {nu:>4} {poly:>4}  ERROR: {exc}  [{done}/{total}]")

    results.sort(key=lambda r: r[6] if np.isfinite(r[6]) else -np.inf, reverse=True)

    top10 = results[:10]

    print("\n" + "=" * 60)
    print("TOP 10 configurations by FR R²")
    print("=" * 60)
    print(f"{'ny':>4} {'nu':>4} {'ord':>4} {'#cand':>6}  "
          f"{'OSA_tr':>8} {'OSA_te':>8} {'FR_tr':>8}")
    print("-" * 60)
    for row in top10:
        ny, nu, poly, nc, r_otr, r_ote, r_fr = row
        print(f"{ny:>4} {nu:>4} {poly:>4} {nc:>6}  "
              f"{r_otr:>8.4f} {r_ote:>8.4f} {r_fr:>8.4f}")

    # --- Save CSV (all results, sorted by FR R²) ---
    import csv
    csv_path = os.path.join(HERE, "bab_narmax_grid_search.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ny", "nu", "poly_order", "n_candidates",
                         "R2_OSA_train", "R2_OSA_test", "R2_FR_train"])
        for row in results:
            writer.writerow([
                row[0], row[1], row[2], row[3],
                f"{row[4]:.6f}", f"{row[5]:.6f}",
                f"{row[6]:.6f}" if np.isfinite(row[6]) else "nan",
            ])
    print(f"\nSaved: {csv_path}")

    # --- Save formatted text report ---
    txt_path = os.path.join(HERE, "bab_narmax_grid_search.txt")
    col_w = [4, 4, 5, 7, 10, 10, 10]
    sep   = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    def row_line(vals):
        cells = [str(v).center(w) for v, w in zip(vals, col_w)]
        return "| " + " | ".join(cells) + " |"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"NARMAX Grid Search — Ball-and-Beam  (n_terms={N_TERMS} fixed)\n")
        f.write(f"Experiment: {EXPERIMENT}  Ts={ts:.4f}s  "
                f"train={split} / test={n-split} samples\n")
        f.write(f"NY ∈ {NY_vals}   NU ∈ {NU_vals}   POLY_ORDER ∈ {POLY_ORDER_vals}\n\n")

        # Full table sorted by FR R²
        f.write("All configurations — sorted by FR R² (descending)\n")
        f.write(sep + "\n")
        f.write(row_line(["ny", "nu", "ord", "#cand",
                           "OSA_tr", "OSA_te", "FR_tr"]) + "\n")
        f.write(sep + "\n")
        for r in results:
            fr_str = f"{r[6]:.4f}" if np.isfinite(r[6]) else "nan"
            f.write(row_line([r[0], r[1], r[2], r[3],
                               f"{r[4]:.4f}", f"{r[5]:.4f}", fr_str]) + "\n")
        f.write(sep + "\n\n")

        # Top 10
        f.write("TOP 10 by FR R²\n")
        f.write(sep + "\n")
        f.write(row_line(["ny", "nu", "ord", "#cand",
                           "OSA_tr", "OSA_te", "FR_tr"]) + "\n")
        f.write(sep + "\n")
        for r in top10:
            fr_str = f"{r[6]:.4f}" if np.isfinite(r[6]) else "nan"
            f.write(row_line([r[0], r[1], r[2], r[3],
                               f"{r[4]:.4f}", f"{r[5]:.4f}", fr_str]) + "\n")
        f.write(sep + "\n")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    import sys
    if "--grid" in sys.argv:
        grid_search()
    else:
        main()

"""
Physics-informed linear readout for the (Shuffle-)RSNN battery SOH model.

Implements Strategy A from physics_layer_blueprint.txt: replace the scikit
Ridge readout with a gradient-descent linear layer trained on a composite
loss

    L_total = L_data + alpha_mono * L_mono + alpha_range * L_range

where
    L_data  = MSE(X w + b, y)
    L_mono  = mean over same-cell forward-difference violations
              relu( (y_hat_{k+1} - y_hat_k) - epsilon_reg )^2
    L_range = mean( relu(y_hat - hi)^2 + relu(lo - y_hat)^2 )

Reservoir stays frozen. The readout is warm-started from the closed-form
ridge solution, then fine-tuned by Adam for a small number of epochs.

The class intentionally matches the subset of the scikit-learn Ridge API
that our pipelines use (.predict(X), an internal coef_/intercept_ pair),
so it drops into existing artifacts without further plumbing.

All maths is pure NumPy to avoid a torch dependency in the sandbox. The
problem is small (≤10k training rows, ≤300 features) so CPU numpy is
plenty fast.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def _relu(x):
    return np.maximum(x, 0.0)


def _relu_grad(x):
    return (x > 0.0).astype(float)


def _build_same_cell_forward_pairs(
    cell_ids: Sequence, cycle_order: Sequence
) -> np.ndarray:
    """Return a (K, 2) int array of (i, j) indices where j is the
    immediate successor of i within the same cell, ordered by
    ``cycle_order`` ascending.

    Forward-difference pairs are used for L_mono (O(N) per cell rather
    than O(N^2) for all pairs), following PINN4SOH's monotonicity term.
    """
    cell_ids = np.asarray(cell_ids)
    cycle_order = np.asarray(cycle_order, dtype=float)
    pairs = []
    unique_cells = np.unique(cell_ids)
    for cell in unique_cells:
        mask = cell_ids == cell
        idx = np.where(mask)[0]
        if idx.size < 2:
            continue
        order_in_cell = np.argsort(cycle_order[idx])
        idx_sorted = idx[order_in_cell]
        for k in range(idx_sorted.size - 1):
            pairs.append((int(idx_sorted[k]), int(idx_sorted[k + 1])))
    if not pairs:
        return np.zeros((0, 2), dtype=int)
    return np.asarray(pairs, dtype=int)


class PhysicsReadout:
    """Linear readout trained with a composite physics loss.

    Exposes a minimal scikit-compatible surface (``predict(X)`` and
    ``coef_``/``intercept_``) so downstream code — including artifacts
    that pickle the readout and ``predict_rows_raw`` helpers — works
    without modification.
    """

    def __init__(
        self,
        alpha_mono: float = 0.10,
        alpha_range: float = 1.00,
        epsilon_reg: float = 0.005,
        lo: float = 0.55,
        hi: float = 1.15,
        lr: float = 1e-3,
        n_epochs: int = 400,
        l2: float = 1e-3,
        verbose: bool = False,
    ):
        self.alpha_mono = float(alpha_mono)
        self.alpha_range = float(alpha_range)
        self.epsilon_reg = float(epsilon_reg)
        self.lo = float(lo)
        self.hi = float(hi)
        self.lr = float(lr)
        self.n_epochs = int(n_epochs)
        self.l2 = float(l2)
        self.verbose = bool(verbose)

        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    # ------------------------------------------------------------------
    # Warm start from closed-form ridge solution.
    # ------------------------------------------------------------------
    def _ridge_warm_start(self, X: np.ndarray, y: np.ndarray, alpha: float):
        n_features = X.shape[1]
        A = X.T @ X + alpha * np.eye(n_features)
        b = X.T @ (y - y.mean())
        w = np.linalg.solve(A, b)
        b_const = float(y.mean() - X.mean(axis=0) @ w)
        return w, b_const

    # ------------------------------------------------------------------
    # Forward prediction helper.
    # ------------------------------------------------------------------
    def _forward(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_

    # ------------------------------------------------------------------
    # Fit with optional cell/cycle metadata for the monotonicity term.
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cell_ids: Optional[Sequence] = None,
        cycle_order: Optional[Sequence] = None,
        warm_start: Optional[Tuple[np.ndarray, float]] = None,
    ) -> "PhysicsReadout":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_rows, n_features = X.shape

        # ---- Warm start --------------------------------------------------
        if warm_start is not None:
            w, b_const = warm_start
            self.coef_ = np.asarray(w, dtype=float).copy()
            self.intercept_ = float(b_const)
        else:
            w, b_const = self._ridge_warm_start(X, y, alpha=max(self.l2, 1e-6))
            self.coef_ = w
            self.intercept_ = b_const

        # ---- Build monotonicity pairs -----------------------------------
        if cell_ids is not None and cycle_order is not None:
            pairs = _build_same_cell_forward_pairs(cell_ids, cycle_order)
        else:
            pairs = np.zeros((0, 2), dtype=int)

        # ---- Adam state -------------------------------------------------
        m_w = np.zeros_like(self.coef_)
        v_w = np.zeros_like(self.coef_)
        m_b = 0.0
        v_b = 0.0
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for epoch in range(1, self.n_epochs + 1):
            y_hat = self._forward(X)
            resid = y_hat - y

            # --- L_data (MSE) ---
            grad_w = (2.0 / n_rows) * (X.T @ resid)
            grad_b = (2.0 / n_rows) * resid.sum()

            # --- L2 weight decay (keeps ridge character) ---
            grad_w += 2.0 * self.l2 * self.coef_

            # --- L_range ---
            over = _relu(y_hat - self.hi)
            under = _relu(self.lo - y_hat)
            range_grad_y = 2.0 * (over - under)
            grad_w += self.alpha_range * (X.T @ range_grad_y) / n_rows
            grad_b += self.alpha_range * range_grad_y.sum() / n_rows

            # --- L_mono (forward-difference violations per cell) ---
            if pairs.shape[0] > 0 and self.alpha_mono > 0.0:
                i_idx = pairs[:, 0]
                j_idx = pairs[:, 1]
                diff = y_hat[j_idx] - y_hat[i_idx]
                violation = _relu(diff - self.epsilon_reg)
                # d/dy_hat[j] = 2 * violation ; d/dy_hat[i] = -2 * violation
                g_pairs = 2.0 * violation / max(pairs.shape[0], 1)
                # Accumulate gradient per row in a single vector.
                d_yhat = np.zeros_like(y_hat)
                np.add.at(d_yhat, j_idx, g_pairs)
                np.add.at(d_yhat, i_idx, -g_pairs)
                grad_w += self.alpha_mono * (X.T @ d_yhat)
                grad_b += self.alpha_mono * d_yhat.sum()

            # --- Adam update -------------------------------------------
            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * (grad_w ** 2)
            m_hat = m_w / (1 - beta1 ** epoch)
            v_hat = v_w / (1 - beta2 ** epoch)
            self.coef_ -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)
            m_hat_b = m_b / (1 - beta1 ** epoch)
            v_hat_b = v_b / (1 - beta2 ** epoch)
            self.intercept_ -= float(self.lr * m_hat_b / (np.sqrt(v_hat_b) + eps))

            if self.verbose and (epoch % 50 == 0 or epoch == 1):
                mse = float(np.mean(resid ** 2))
                print(f"[PhysicsReadout] epoch {epoch:4d}  mse={mse:.5e}")

        return self

    # ------------------------------------------------------------------
    # Prediction (matches sklearn Ridge API the pipelines rely on).
    # ------------------------------------------------------------------
    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("PhysicsReadout must be fit before predict.")
        X = np.asarray(X, dtype=float)
        return self._forward(X)

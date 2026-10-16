"""
Real-Time Motor Cortex (M1) Neural Decoders for Brain-Computer Interfaces.

Implements two gold-standard invasive BCI decoding algorithms used in clinical
and preclinical systems (e.g., Neuralink Telepathy, BrainGate):
1. `KalmanFilterDecoder`: Linear state-space model tracking [px, py, vx, vy]
   from multi-channel neural firing rates with closed-form MLE parameter fitting.
2. `RidgeVelocityDecoder`: Multi-lag Optimal Linear Estimator (OLE / Wiener filter)
   with L2 regularization.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, Tuple
import numpy as np


@dataclass
class DecodingMetrics:
    """Quantitative benchmark metrics for BCI cursor decoding."""
    decoder_name: str
    vel_r2: float
    pos_r2: float
    vel_pearson_r: float
    vel_rmse: float
    pos_rmse: float
    mean_step_latency_us: float


class KalmanFilterDecoder:
    """
    State-Space Kalman Filter Neural Decoder (Wu et al., 2006; Gilja et al., 2012).

    State vector at bin t:
        x_t = [p_x(t), p_y(t), v_x(t), v_y(t)]^T  in R^4

    State transition equation:
        x_t = A * x_{t-1} + w_t,    w_t ~ N(0, W)

    Neural observation equation (centered firing rates z_t in R^N):
        z_t = C * x_t + q_t,        q_t ~ N(0, Q)
    """

    def __init__(self, reg_lambda: float = 1e-3) -> None:
        self.reg_lambda = reg_lambda
        self.A: np.ndarray | None = None
        self.W: np.ndarray | None = None
        self.C: np.ndarray | None = None
        self.Q: np.ndarray | None = None
        self.z_mean: np.ndarray | None = None
        self.x_mean: np.ndarray | None = None

    def fit(self, pos: np.ndarray, vel: np.ndarray, neural_rates: np.ndarray) -> "KalmanFilterDecoder":
        """
        Closed-form Maximum Likelihood Estimation (MLE) of state-space matrices
        (A, W, C, Q) from calibration trajectories and neural firing rates.
        """
        X = np.hstack([pos, vel])  # (T, 4)
        self.x_mean = X.mean(axis=0, keepdims=True)
        Xc = X - self.x_mean       # (T, 4)

        self.z_mean = neural_rates.mean(axis=0, keepdims=True)
        Zc = neural_rates - self.z_mean  # (T, N)

        X1 = Xc[:-1]  # (T-1, 4)
        X2 = Xc[1:]   # (T-1, 4)
        d = X.shape[1]
        n_ch = neural_rates.shape[1]

        # 1. Fit state transition matrix A: X2^T = A * X1^T
        xx1 = X1.T @ X1 + self.reg_lambda * np.eye(d)
        self.A = (X2.T @ X1) @ np.linalg.inv(xx1)

        # Process noise covariance W
        res_state = X2 - (X1 @ self.A.T)
        self.W = (res_state.T @ res_state) / max(1, len(X1) - 1) + 1e-5 * np.eye(d)

        # 2. Fit observation matrix C: Zc^T = C * Xc^T
        xx_all = Xc.T @ Xc + self.reg_lambda * np.eye(d)
        self.C = (Zc.T @ Xc) @ np.linalg.inv(xx_all)

        # Observation noise covariance Q
        res_obs = Zc - (Xc @ self.C.T)
        self.Q = (res_obs.T @ res_obs) / max(1, len(Xc) - 1) + 1e-2 * np.eye(n_ch)

        return self

    def decode(
        self,
        neural_rates: np.ndarray,
        init_state: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run causal recursive Kalman filtering step-by-step across `neural_rates` (T, N).

        Returns:
            pred_pos: (T, 2) decoded cursor position
            pred_vel: (T, 2) decoded cursor velocity
            mean_step_latency_us: average execution time per bin in microseconds
        """
        if self.A is None or self.C is None or self.W is None or self.Q is None:
            raise RuntimeError("KalmanFilterDecoder must be fitted before calling decode().")

        T, _ = neural_rates.shape
        d = self.A.shape[0]
        Zc = neural_rates - self.z_mean

        x_hat = np.zeros((T, d))
        if init_state is not None:
            x_curr = init_state - self.x_mean.ravel()
        else:
            x_curr = np.zeros(d)

        P_curr = np.eye(d) * 0.1
        I_d = np.eye(d)

        # Precompute Q^-1 * C using Woodbury / information form for fast N >> 4 update
        Q_inv_C = np.linalg.solve(self.Q, self.C)      # (N, 4)
        C_T_Q_inv_C = self.C.T @ Q_inv_C               # (4, 4)

        t0 = time.perf_counter()
        for t in range(T):
            # Predict step
            if t > 0:
                x_pred = self.A @ x_curr
                P_pred = self.A @ P_curr @ self.A.T + self.W
            else:
                x_pred = x_curr
                P_pred = P_curr

            # Update step via Matrix Inversion Lemma (4x4 inversion instead of NxN)
            P_post_inv = np.linalg.inv(P_pred) + C_T_Q_inv_C
            P_curr = np.linalg.inv(P_post_inv)
            K_t = P_curr @ Q_inv_C.T  # (4, N)

            innovation = Zc[t] - (self.C @ x_pred)
            x_curr = x_pred + K_t @ innovation
            x_hat[t] = x_curr

        elapsed_s = time.perf_counter() - t0
        mean_latency_us = (elapsed_s / max(1, T)) * 1e6

        x_hat = x_hat + self.x_mean
        pred_pos = x_hat[:, :2]
        pred_vel = x_hat[:, 2:4]
        return pred_pos, pred_vel, mean_latency_us


class RidgeVelocityDecoder:
    """
    Multi-Lag Optimal Linear Estimator (OLE / Ridge Wiener Filter).

    Predicts instantaneous cursor velocity v_t from a causal sliding window of
    neural firing rates [z_t, z_{t-1}, ..., z_{t-L+1}] with L2 regularization,
    and integrates leaky velocity to obtain cursor position.
    """

    def __init__(
        self,
        n_lags: int = 4,
        alpha: float = 50.0,
        pos_leak: float = 0.965,
    ) -> None:
        self.n_lags = n_lags
        self.alpha = alpha
        self.pos_leak = pos_leak
        self.W_vel: np.ndarray | None = None
        self.W_pos: np.ndarray | None = None

    def _build_design_matrix(self, neural_rates: np.ndarray) -> np.ndarray:
        """Construct causal lag-augmented neural feature matrix with bias column."""
        T, N = neural_rates.shape
        features = np.zeros((T, N * self.n_lags + 1))
        for lag in range(self.n_lags):
            if lag == 0:
                features[:, :N] = neural_rates
            else:
                features[lag:, lag * N : (lag + 1) * N] = neural_rates[:-lag]
                features[:lag, lag * N : (lag + 1) * N] = neural_rates[0]
        features[:, -1] = 1.0  # Bias term
        return features

    def fit(self, pos: np.ndarray, vel: np.ndarray, neural_rates: np.ndarray) -> "RidgeVelocityDecoder":
        """Fit L2-regularized linear readout weights in closed form."""
        Phi = self._build_design_matrix(neural_rates)
        reg = self.alpha * np.eye(Phi.shape[1])
        reg[-1, -1] = 0.0  # Do not regularize bias term

        gram_inv = np.linalg.inv(Phi.T @ Phi + reg)
        self.W_vel = gram_inv @ (Phi.T @ vel)
        self.W_pos = gram_inv @ (Phi.T @ pos)
        return self

    def decode(
        self,
        neural_rates: np.ndarray,
        dt: float = 0.02,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Decode cursor velocity and position from neural rates."""
        if self.W_vel is None or self.W_pos is None:
            raise RuntimeError("RidgeVelocityDecoder must be fitted before calling decode().")

        T = neural_rates.shape[0]
        t0 = time.perf_counter()
        Phi = self._build_design_matrix(neural_rates)
        pred_vel = Phi @ self.W_vel
        direct_pos = Phi @ self.W_pos

        # Blend leaky velocity integration with direct spatial readout to prevent drift
        pred_pos = np.zeros((T, 2))
        pred_pos[0] = direct_pos[0]
        for t in range(1, T):
            integrated = self.pos_leak * pred_pos[t - 1] + pred_vel[t] * dt
            pred_pos[t] = 0.75 * integrated + 0.25 * direct_pos[t]

        elapsed_s = time.perf_counter() - t0
        mean_latency_us = (elapsed_s / max(1, T)) * 1e6
        return pred_pos, pred_vel, mean_latency_us


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute variance-weighted multi-output Coefficient of Determination (R^2)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def evaluate_decoding(
    decoder_name: str,
    true_pos: np.ndarray,
    true_vel: np.ndarray,
    pred_pos: np.ndarray,
    pred_vel: np.ndarray,
    mean_step_latency_us: float,
) -> DecodingMetrics:
    """Compute comprehensive cursor decoding evaluation metrics."""
    vel_r2 = compute_r2(true_vel, pred_vel)
    pos_r2 = compute_r2(true_pos, pred_pos)

    corr_x = np.corrcoef(true_vel[:, 0], pred_vel[:, 0])[0, 1]
    corr_y = np.corrcoef(true_vel[:, 1], pred_vel[:, 1])[0, 1]
    vel_pearson_r = float(0.5 * (corr_x + corr_y))

    vel_rmse = float(np.sqrt(np.mean((true_vel - pred_vel) ** 2)))
    pos_rmse = float(np.sqrt(np.mean((true_pos - pred_pos) ** 2)))

    return DecodingMetrics(
        decoder_name=decoder_name,
        vel_r2=vel_r2,
        pos_r2=pos_r2,
        vel_pearson_r=vel_pearson_r,
        vel_rmse=vel_rmse,
        pos_rmse=pos_rmse,
        mean_step_latency_us=mean_step_latency_us,
    )

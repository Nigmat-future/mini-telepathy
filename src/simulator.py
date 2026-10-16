"""
Biophysical Motor Cortex (M1) Spike Simulator & Center-Out Task Generator.

Models extracellular neural population activity recorded by a high-channel-count
intracortical microelectrode array (e.g., Neuralink N1 array) during 2D cursor
reaching tasks:
1. Minimum-jerk reaching trajectories (Flash & Hogan, 1985) across 8 radial targets.
2. Cosine directional tuning curves (Georgopoulos et al., 1986) with speed modulation.
3. Inhomogeneous Poisson spike generation and causal exponential spike rate filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class SimulationData:
    """Container for simulated behavioral kinematics and M1 neural recordings."""
    time_s: np.ndarray          # (T,) time vector in seconds
    pos: np.ndarray             # (T, 2) cursor position [x, y] in [-1, 1] normalized units
    vel: np.ndarray             # (T, 2) cursor velocity [vx, vy] in units/s
    target_id: np.ndarray       # (T,) active target index (0..7, or -1 during center hold)
    target_coords: np.ndarray   # (8, 2) coordinates of the 8 peripheral targets
    spike_counts: np.ndarray    # (T, N) raw spike counts per time bin
    smoothed_rates: np.ndarray  # (T, N) causal exponentially smoothed firing rates (Hz)
    firing_rates_hz: np.ndarray # (T, N) underlying instantaneous firing rates (Hz)
    pref_dirs: np.ndarray       # (N,) preferred direction angle theta_p in [0, 2*pi)
    dt: float                   # bin width in seconds


def minimum_jerk_profile(tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normalized position s(tau) and normalized velocity ds/dtau
    for a minimum-jerk point-to-point movement (Flash & Hogan, 1985),
    where tau = t / T_reach in [0, 1].
    """
    tau = np.clip(tau, 0.0, 1.0)
    pos_scale = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
    vel_scale = 30.0 * (tau ** 2) - 60.0 * (tau ** 3) + 30.0 * (tau ** 4)
    return pos_scale, vel_scale


def generate_center_out_kinematics(
    n_trials: int = 32,
    n_targets: int = 8,
    radius: float = 0.80,
    reach_duration_s: float = 0.65,
    hold_duration_s: float = 0.25,
    return_duration_s: float = 0.55,
    dt: float = 0.02,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 2D Center-Out reaching kinematics (position & velocity).

    Each trial consists of:
      1. Center hold -> Outward minimum-jerk reach to peripheral target k
      2. Target hold
      3. Return reach back to center
    """
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, n_targets, endpoint=False)
    target_coords = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

    n_reach = max(2, int(round(reach_duration_s / dt)))
    n_hold = max(1, int(round(hold_duration_s / dt)))
    n_return = max(2, int(round(return_duration_s / dt)))

    tau_reach = np.linspace(0.0, 1.0, n_reach)
    s_reach, ds_reach = minimum_jerk_profile(tau_reach)

    tau_return = np.linspace(0.0, 1.0, n_return)
    s_return, ds_return = minimum_jerk_profile(tau_return)

    pos_list = []
    vel_list = []
    target_id_list = []

    # Balanced target sequence across blocks of n_targets
    target_seq = []
    while len(target_seq) < n_trials:
        block = np.arange(n_targets)
        if len(target_seq) >= n_targets:
            rng.shuffle(block)
        target_seq.extend(block.tolist())
    target_seq = target_seq[:n_trials]

    for tgt_idx in target_seq:
        tgt = target_coords[tgt_idx]

        # 1. Outward reach: (0, 0) -> tgt
        p_out = np.outer(s_reach, tgt)
        v_out = np.outer(ds_reach / reach_duration_s, tgt)
        t_out = np.full(n_reach, tgt_idx, dtype=int)

        # 2. Peripheral target hold
        p_hold = np.tile(tgt, (n_hold, 1))
        v_hold = np.zeros((n_hold, 2))
        t_hold = np.full(n_hold, tgt_idx, dtype=int)

        # 3. Return to center: tgt -> (0, 0)
        p_ret = np.outer(1.0 - s_return, tgt)
        v_ret = np.outer(-ds_return / return_duration_s, tgt)
        t_ret = np.full(n_return, -1, dtype=int)

        # 4. Short center rest
        p_rest = np.zeros((n_hold, 2))
        v_rest = np.zeros((n_hold, 2))
        t_rest = np.full(n_hold, -1, dtype=int)

        pos_list.extend([p_out, p_hold, p_ret, p_rest])
        vel_list.extend([v_out, v_hold, v_ret, v_rest])
        target_id_list.extend([t_out, t_hold, t_ret, t_rest])

    pos = np.vstack(pos_list)
    vel = np.vstack(vel_list)
    target_ids = np.concatenate(target_id_list)
    time_s = np.arange(len(pos)) * dt

    return time_s, pos, vel, target_ids, target_coords


class MotorCortexArray:
    """
    Simulates an N1-style intracortical electrode array in primary motor cortex (M1).

    Each channel/neuron i exhibits cosine directional tuning (Georgopoulos et al., 1986):
        lambda_i(t) = max(0, b_i + m_i * (v_x(t)*cos(theta_p,i) + v_y(t)*sin(theta_p,i))
                             + s_i * ||v(t)|| + noise_i(t))
    """

    def __init__(
        self,
        n_channels: int = 64,
        baseline_range_hz: Tuple[float, float] = (12.0, 26.0),
        mod_depth_range_hz: Tuple[float, float] = (20.0, 42.0),
        noise_std_hz: float = 3.5,
        smoothing_tau_s: float = 0.08,
        seed: int = 42,
    ) -> None:
        self.n_channels = n_channels
        self.noise_std_hz = noise_std_hz
        self.smoothing_tau_s = smoothing_tau_s
        self.rng = np.random.default_rng(seed)

        # Uniformly distributed preferred directions with small biological jitter
        base_angles = np.linspace(0.0, 2.0 * np.pi, n_channels, endpoint=False)
        jitter = self.rng.normal(0.0, 0.08, size=n_channels)
        self.pref_dirs = np.mod(base_angles + jitter, 2.0 * np.pi)

        # Unit preferred direction vectors: shape (N, 2)
        self.pref_vectors = np.column_stack(
            [np.cos(self.pref_dirs), np.sin(self.pref_dirs)]
        )

        # Baseline firing rates (Hz) and directional modulation depths (Hz per unit/s)
        self.baseline_hz = self.rng.uniform(*baseline_range_hz, size=n_channels)
        self.mod_depth_hz = self.rng.uniform(*mod_depth_range_hz, size=n_channels)
        self.speed_gain_hz = self.rng.uniform(1.5, 5.0, size=n_channels)

    def simulate_recordings(
        self,
        vel: np.ndarray,
        dt: float = 0.02,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Given velocity trajectory `vel` of shape (T, 2), simulate:
          - `spike_counts`: (T, N) integer Poisson spike counts per bin
          - `smoothed_rates`: (T, N) causal exponentially filtered firing rates (Hz)
          - `firing_rates_hz`: (T, N) underlying instantaneous firing rates (Hz)
        """
        n_steps = vel.shape[0]
        speed = np.linalg.norm(vel, axis=1, keepdims=True)  # (T, 1)

        # Cosine tuning projection: (T, 2) @ (2, N) -> (T, N)
        directional_drive = vel @ self.pref_vectors.T

        # Biological background rate fluctuations
        rate_noise = self.rng.normal(
            0.0, self.noise_std_hz, size=(n_steps, self.n_channels)
        )

        firing_rates_hz = (
            self.baseline_hz[None, :]
            + self.mod_depth_hz[None, :] * directional_drive
            + self.speed_gain_hz[None, :] * speed
            + rate_noise
        )
        firing_rates_hz = np.clip(firing_rates_hz, 0.5, 250.0)

        # Inhomogeneous Poisson spike counts in bin width dt
        expected_counts = firing_rates_hz * dt
        spike_counts = self.rng.poisson(expected_counts)

        # Causal exponential smoothing filter (first-order IIR low-pass on spike rate)
        instant_rates = spike_counts.astype(np.float64) / dt
        alpha = dt / (self.smoothing_tau_s + dt)
        smoothed_rates = np.zeros_like(instant_rates)
        smoothed_rates[0] = instant_rates[0]
        for t in range(1, n_steps):
            smoothed_rates[t] = (
                alpha * instant_rates[t] + (1.0 - alpha) * smoothed_rates[t - 1]
            )

        return spike_counts, smoothed_rates, firing_rates_hz

    def generate_dataset(
        self,
        n_trials: int = 32,
        dt: float = 0.02,
        seed: int = 42,
    ) -> SimulationData:
        """Generate a complete Center-Out behavioral + M1 neural recording dataset."""
        time_s, pos, vel, target_id, target_coords = generate_center_out_kinematics(
            n_trials=n_trials, dt=dt, seed=seed
        )
        spike_counts, smoothed_rates, firing_rates_hz = self.simulate_recordings(
            vel=vel, dt=dt
        )
        return SimulationData(
            time_s=time_s,
            pos=pos,
            vel=vel,
            target_id=target_id,
            target_coords=target_coords,
            spike_counts=spike_counts,
            smoothed_rates=smoothed_rates,
            firing_rates_hz=firing_rates_hz,
            pref_dirs=self.pref_dirs,
            dt=dt,
        )

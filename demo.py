"""
Mini-Telepathy End-to-End Demo & Figure Generator.

Runs a complete Motor Cortex (M1) -> 2D Cursor BCI pipeline:
  1. Simulates 64-channel intracortical spike recordings during 8-target Center-Out reaching.
  2. Calibrates a State-Space Kalman Filter Decoder and a Multi-Lag Ridge (OLE) Decoder.
  3. Evaluates real-time decoding accuracy (R^2, Pearson r, RMSE) & per-bin latency (< 50 us).
  4. Generates a 4-panel publication-grade summary figure at `assets/decoding_result.png`.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.simulator import MotorCortexArray
from src.decoder import KalmanFilterDecoder, RidgeVelocityDecoder, evaluate_decoding


def plot_decoding_dashboard(
    test_time_s: np.ndarray,
    true_pos: np.ndarray,
    true_vel: np.ndarray,
    kf_pos: np.ndarray,
    kf_vel: np.ndarray,
    ole_pos: np.ndarray,
    ole_vel: np.ndarray,
    spike_counts: np.ndarray,
    pref_dirs: np.ndarray,
    target_coords: np.ndarray,
    kf_metrics,
    ole_metrics,
    output_path: Path,
) -> None:
    """Render a 4-panel dark-mode Neuralink-style BCI decoding summary figure."""
    plt.rcParams.update({
        "figure.facecolor": "#0b0f19",
        "axes.facecolor": "#111827",
        "axes.edgecolor": "#334155",
        "axes.labelcolor": "#e2e8f0",
        "text.color": "#f8fafc",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "grid.color": "#1e293b",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Segoe UI"],
    })

    fig = plt.figure(figsize=(15, 9.5), dpi=180)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.05, 1.35], height_ratios=[1.0, 1.0], wspace=0.24, hspace=0.30)

    target_colors = plt.cm.hsv(np.linspace(0, 1, len(target_coords), endpoint=False))

    # -------------------------------------------------------------------------
    # Panel A: 2D Center-Out Cursor Trajectories (True vs Kalman Filter)
    # -------------------------------------------------------------------------
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_traj.set_title(
        f"A. 2D Center-Out Cursor Decoding (Pos R² = {kf_metrics.pos_r2:.3f})",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    # Plot peripheral targets
    for idx, (tx, ty) in enumerate(target_coords):
        circle = plt.Circle((tx, ty), 0.11, color=target_colors[idx], alpha=0.22, lw=1.5)
        circle_border = plt.Circle((tx, ty), 0.11, fill=False, edgecolor=target_colors[idx], alpha=0.85, lw=1.5, ls="--")
        ax_traj.add_patch(circle)
        ax_traj.add_patch(circle_border)
        ax_traj.text(tx, ty, f"T{idx+1}", color="#f8fafc", fontsize=8, ha="center", va="center", fontweight="bold")

    # Center home target
    center_circle = plt.Circle((0, 0), 0.08, fill=False, edgecolor="#64748b", lw=1.2, ls=":")
    ax_traj.add_patch(center_circle)

    # Trajectories
    ax_traj.plot(true_pos[:, 0], true_pos[:, 1], color="#94a3b8", lw=1.6, ls="--", alpha=0.75, label="True Motor Intent")
    ax_traj.plot(ole_pos[:, 0], ole_pos[:, 1], color="#a855f7", lw=1.2, alpha=0.55, label=f"Ridge OLE (R²={ole_metrics.pos_r2:.2f})")
    ax_traj.plot(kf_pos[:, 0], kf_pos[:, 1], color="#00f2fe", lw=2.1, alpha=0.95, label=f"Kalman Filter (R²={kf_metrics.pos_r2:.2f})")

    ax_traj.set_xlim(-1.08, 1.08)
    ax_traj.set_ylim(-1.08, 1.08)
    ax_traj.set_aspect("equal")
    ax_traj.set_xlabel("Workspace X (normalized)")
    ax_traj.set_ylabel("Workspace Y (normalized)")
    ax_traj.grid(True, linestyle="--", alpha=0.4)
    ax_traj.legend(loc="lower left", fontsize=8.5, facecolor="#0f172a", edgecolor="#334155")

    # -------------------------------------------------------------------------
    # Panel B: 64-Channel M1 Population Spike Raster (Sorted by Preferred Dir)
    # -------------------------------------------------------------------------
    ax_raster = fig.add_subplot(gs[0, 1])
    ax_raster.set_title(
        "B. 64-Channel Motor Cortex (M1) Spike Raster (Sorted by Preferred Direction θp)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    sort_idx = np.argsort(pref_dirs)
    sorted_spikes = spike_counts[:, sort_idx]
    sorted_dirs = pref_dirs[sort_idx]

    t_zeroed = test_time_s - test_time_s[0]
    n_channels = sorted_spikes.shape[1]

    for ch in range(n_channels):
        spike_bins = np.where(sorted_spikes[:, ch] > 0)[0]
        if len(spike_bins) > 0:
            col = plt.cm.hsv(sorted_dirs[ch] / (2.0 * np.pi))
            ax_raster.vlines(
                t_zeroed[spike_bins],
                ch + 0.15,
                ch + 0.85,
                color=col,
                linewidth=0.85,
                alpha=0.9,
            )

    ax_raster.set_xlim(0, t_zeroed[-1])
    ax_raster.set_ylim(0, n_channels)
    ax_raster.set_xlabel("Time in Held-Out Test Block (s)")
    ax_raster.set_ylabel("Electrode Channel # (Sorted by θp)")
    ax_raster.grid(True, axis="x", linestyle="--", alpha=0.35)

    # -------------------------------------------------------------------------
    # Panel C: N1 Array Directional Tuning Curves & Firing Rate Distribution
    # -------------------------------------------------------------------------
    ax_tuning = fig.add_subplot(gs[1, 0])
    ax_tuning.set_title(
        "C. M1 Neuron Cosine Directional Tuning Curves",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    theta_grid = np.linspace(0, 360, 180)
    theta_rad = np.deg2rad(theta_grid)
    sample_indices = np.linspace(0, n_channels - 1, 8, dtype=int)

    for idx in sample_indices:
        ch_id = sort_idx[idx]
        pd_rad = pref_dirs[ch_id]
        col = plt.cm.hsv(pd_rad / (2.0 * np.pi))
        curve = 18.0 + 30.0 * np.cos(theta_rad - pd_rad)
        curve = np.clip(curve, 0.0, None)
        ax_tuning.plot(
            theta_grid,
            curve,
            color=col,
            lw=1.8,
            alpha=0.85,
            label=f"Ch {ch_id:02d} (θp={ int(np.rad2deg(pd_rad)) }°)",
        )

    ax_tuning.set_xlim(0, 360)
    ax_tuning.set_xticks([0, 90, 180, 270, 360])
    ax_tuning.set_xticklabels(["0° (Right)", "90° (Up)", "180° (Left)", "270° (Down)", "360°"])
    ax_tuning.set_xlabel("Movement Direction θ (deg)")
    ax_tuning.set_ylabel("Firing Rate λ(θ) (spikes/s)")
    ax_tuning.grid(True, linestyle="--", alpha=0.4)
    ax_tuning.legend(loc="upper right", ncol=2, fontsize=7.5, facecolor="#0f172a", edgecolor="#334155")

    # -------------------------------------------------------------------------
    # Panel D: Real-Time Velocity Decoding Traces (Vx and Vy)
    # -------------------------------------------------------------------------
    ax_vel = fig.add_subplot(gs[1, 1])
    ax_vel.set_title(
        f"D. Real-Time Velocity Decoding (Kalman R² = {kf_metrics.vel_r2:.3f}, r = {kf_metrics.vel_pearson_r:.3f}, Latency = {kf_metrics.mean_step_latency_us:.1f} µs/bin)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    ax_vel.plot(t_zeroed, true_vel[:, 0], color="#94a3b8", lw=1.5, ls="--", alpha=0.8, label="True Vx")
    ax_vel.plot(t_zeroed, kf_vel[:, 0], color="#00f2fe", lw=2.0, alpha=0.95, label="Kalman Decoded Vx")

    ax_vel.plot(t_zeroed, true_vel[:, 1] - 3.2, color="#64748b", lw=1.5, ls="--", alpha=0.8, label="True Vy (offset)")
    ax_vel.plot(t_zeroed, kf_vel[:, 1] - 3.2, color="#38bdf8", lw=2.0, alpha=0.9, label="Kalman Decoded Vy (offset)")

    ax_vel.axhline(0.0, color="#334155", lw=0.8)
    ax_vel.axhline(-3.2, color="#334155", lw=0.8)
    ax_vel.set_xlim(0, t_zeroed[-1])
    ax_vel.set_yticks([-3.2, 0.0])
    ax_vel.set_yticklabels(["Vy (0 u/s)", "Vx (0 u/s)"])
    ax_vel.set_xlabel("Time in Held-Out Test Block (s)")
    ax_vel.set_ylabel("Cursor Velocity (norm. units/s)")
    ax_vel.grid(True, linestyle="--", alpha=0.35)
    ax_vel.legend(loc="upper right", ncol=2, fontsize=8.5, facecolor="#0f172a", edgecolor="#334155")

    fig.suptitle(
        "MINI-TELEPATHY // 64-Channel Motor Cortex (M1) Spike-to-Cursor Real-Time BCI Decoder",
        fontsize=15,
        fontweight="bold",
        color="#38bdf8",
        y=0.97,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    print("=" * 76)
    print(" 🧠 MINI-TELEPATHY: Motor Cortex (M1) Spike-to-Cursor BCI Decoding Pipeline")
    print("=" * 76)

    dt = 0.02  # 20 ms bin width (50 Hz real-time control loop)
    n_channels = 64

    # 1. Simulate 40 Center-Out reaching trials (32 calibration + 8 test across all 8 targets)
    array = MotorCortexArray(n_channels=n_channels, smoothing_tau_s=0.08, seed=42)
    sim_data = array.generate_dataset(n_trials=40, dt=dt, seed=42)

    n_total = len(sim_data.time_s)
    split_idx = int(n_total * 0.80)  # First 32 trials for calibration, last 8 trials for testing

    train_pos, test_pos = sim_data.pos[:split_idx], sim_data.pos[split_idx:]
    train_vel, test_vel = sim_data.vel[:split_idx], sim_data.vel[split_idx:]
    train_rates, test_rates = sim_data.smoothed_rates[:split_idx], sim_data.smoothed_rates[split_idx:]
    test_spikes = sim_data.spike_counts[split_idx:]
    test_time_s = sim_data.time_s[split_idx:]

    print(f"[1/3] Simulated {n_channels}-channel M1 microelectrode array @ {int(1/dt)} Hz ({dt*1000:.0f} ms bins)")
    print(f"      Calibration split : {len(train_pos)} bins ({len(train_pos)*dt:.1f} s, 32 trials)")
    print(f"      Held-out test split: {len(test_pos)} bins ({len(test_pos)*dt:.1f} s, 8 reaches)")

    # 2. Fit & decode with State-Space Kalman Filter
    kf_decoder = KalmanFilterDecoder(reg_lambda=1e-3).fit(train_pos, train_vel, train_rates)
    init_state = np.hstack([test_pos[0], test_vel[0]])
    kf_pos, kf_vel, kf_lat_us = kf_decoder.decode(test_rates, init_state=init_state)
    kf_metrics = evaluate_decoding("Kalman Filter (State-Space)", test_pos, test_vel, kf_pos, kf_vel, kf_lat_us)

    # 3. Fit & decode with Multi-Lag Ridge OLE Decoder
    ole_decoder = RidgeVelocityDecoder(n_lags=5, alpha=40.0).fit(train_pos, train_vel, train_rates)
    ole_pos, ole_vel, ole_lat_us = ole_decoder.decode(test_rates, dt=dt)
    ole_metrics = evaluate_decoding("Ridge OLE (5-Lag Wiener)", test_pos, test_vel, ole_pos, ole_vel, ole_lat_us)

    print("\n[2/3] Held-Out Test Split Benchmark Results:")
    header = f"{'Decoder Algorithm':<30} | {'Vel R²':>7} | {'Pos R²':>7} | {'Pearson r':>9} | {'Vel RMSE':>8} | {'Latency/bin':>12}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for m in (kf_metrics, ole_metrics):
        print(
            f"{m.decoder_name:<30} | {m.vel_r2:>7.3f} | {m.pos_r2:>7.3f} | "
            f"{m.vel_pearson_r:>9.3f} | {m.vel_rmse:>8.3f} | {m.mean_step_latency_us:>9.1f} µs"
        )
    print("-" * len(header))

    # 4. Save summary figure and text report
    assets_dir = Path(__file__).resolve().parent / "assets"
    fig_path = assets_dir / "decoding_result.png"
    plot_decoding_dashboard(
        test_time_s=test_time_s,
        true_pos=test_pos,
        true_vel=test_vel,
        kf_pos=kf_pos,
        kf_vel=kf_vel,
        ole_pos=ole_pos,
        ole_vel=ole_vel,
        spike_counts=test_spikes,
        pref_dirs=sim_data.pref_dirs,
        target_coords=sim_data.target_coords,
        kf_metrics=kf_metrics,
        ole_metrics=ole_metrics,
        output_path=fig_path,
    )

    report_path = assets_dir / "benchmark_summary.txt"
    report_lines = [
        header,
        "-" * len(header),
        *(
            f"{m.decoder_name:<30} | {m.vel_r2:>7.3f} | {m.pos_r2:>7.3f} | "
            f"{m.vel_pearson_r:>9.3f} | {m.vel_rmse:>8.3f} | {m.mean_step_latency_us:>9.1f} µs"
            for m in (kf_metrics, ole_metrics)
        ),
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n[3/3] Saved 4-panel visualization figure to: {fig_path}")
    print("      Open `index.html` in any browser to launch the interactive real-time BCI studio!")


if __name__ == "__main__":
    main()

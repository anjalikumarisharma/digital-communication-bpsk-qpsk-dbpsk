"""
===============================================================================
Digital Communication Laboratory - Experiment 10
BPSK, QPSK and DPSK: Phase Modulation, Ambiguity, and Constellation Analysis
===============================================================================
Programme: B.Tech in Electronics and Communication Engineering
Platform: Python 3 (NumPy, SciPy, Matplotlib, Pandas)
Laboratory Mode: Self-contained first-principles simulation without toolboxes

Description:
  Implements an end-to-end digital phase-modulated communication transceiver:
  1. Coherent BPSK and Gray-coded QPSK modulation and demodulation.
  2. Differential Encoding and Demodulation (DBPSK) for phase ambiguity resilience.
  3. Constellation diagrams: Ideal vs noisy received symbol clouds over AWGN.
  4. Carrier phase offset evaluation (0°, 10°, 30°, 60°, 90°) demonstrating
     constellation rotation, phase jitter, and catastrophic decision failure.
  5. Decision-statistic distributions (I and Q rail projections).
  6. Theoretical vs Monte Carlo simulated Bit Error Rate (BER) curves.
  7. Mandatory syllabus validation: Manually map 00, 01, 11, and 10 and verify
     corresponding detector decisions before and after phase rotation.
===============================================================================
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, List, Optional

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import pandas as pd

# Set plot styling for publication-quality visuals
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.5,
    "grid.alpha": 0.5,
    "grid.linestyle": "--",
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

# Setup Output Directory
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. CORE SYSTEM PARAMETERS
# =============================================================================
RS = 1000.0                # Symbol rate = 1000 symbols/sec
TS = 1.0 / RS              # Symbol duration = 1.0 ms
ES = 1.0                   # Normalized average symbol energy = 1.0


# =============================================================================
# 2. MODULATION & DEMODULATION FROM FIRST PRINCIPLES
# =============================================================================

def generate_random_bits(n_bits: int, seed: Optional[int] = 42) -> np.ndarray:
    """Generate independent, equiprobable binary data bits {0, 1}."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n_bits)


def modulate_bpsk(bits: np.ndarray) -> np.ndarray:
    """
    BPSK modulation:
      Bit 0 -> +1.0 (Phase 0)
      Bit 1 -> -1.0 (Phase pi)
      Average symbol energy Es = 1.0, Eb = Es = 1.0.
    """
    return np.where(bits == 0, 1.0, -1.0)


def demodulate_bpsk(symbols: np.ndarray) -> np.ndarray:
    """
    Coherent BPSK minimum-distance slicer:
      Decide bit 0 if Re(y) > 0, else bit 1.
    """
    return np.where(np.real(symbols) > 0.0, 0, 1)


def modulate_qpsk_gray(bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gray-coded QPSK mapping with normalized symbol energy Es = 1.0:
      Input: Bit sequence (length must be even).
      Grouping: Pairs of bits (b0, b1).
      Gray Mapping Rules:
        '00' -> (+1 + 1j) / sqrt(2)  [Phase  pi/4  =  45 deg]
        '01' -> (-1 + 1j) / sqrt(2)  [Phase 3pi/4  = 135 deg]
        '11' -> (-1 - 1j) / sqrt(2)  [Phase 5pi/4  = 225 deg]
        '10' -> (+1 - 1j) / sqrt(2)  [Phase 7pi/4  = 315 deg]
      Notice: Adjacent constellation points differ by exactly 1 bit.
    """
    if len(bits) % 2 != 0:
        bits = bits[:-1]

    b0 = bits[0::2]
    b1 = bits[1::2]

    # In-phase rail (b0: 0 -> +1, 1 -> -1)
    i_rail = np.where(b0 == 0, 1.0, -1.0)
    # Quadrature rail (b1: 0 -> +1, 1 -> -1)
    q_rail = np.where(b1 == 0, 1.0, -1.0)

    symbols = (i_rail + 1j * q_rail) / np.sqrt(2.0)
    dibits = np.column_stack((b0, b1))
    return symbols, dibits


def demodulate_qpsk_gray(symbols: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coherent QPSK Gray demapper:
      b0 = 0 if Re(s) > 0 else 1
      b1 = 0 if Im(s) > 0 else 1
    """
    b0 = np.where(np.real(symbols) > 0.0, 0, 1)
    b1 = np.where(np.imag(symbols) > 0.0, 0, 1)

    demod_bits = np.empty(len(symbols) * 2, dtype=int)
    demod_bits[0::2] = b0
    demod_bits[1::2] = b1
    dibits = np.column_stack((b0, b1))
    return demod_bits, dibits


def modulate_dbpsk(bits: np.ndarray) -> np.ndarray:
    """
    Differential BPSK (DBPSK) Modulation:
      Differentially encode bits: d_k = d_{k-1} ^ b_k
      s_k = +1 if d_k == 0 else -1
      Bit 0 represents NO phase change (Delta_phi = 0)
      Bit 1 represents phase change of pi (Delta_phi = pi)
    """
    n_bits = len(bits)
    d = np.zeros(n_bits + 1, dtype=int)
    d[0] = 0  # Reference starting state

    for k in range(n_bits):
        d[k + 1] = d[k] ^ bits[k]

    tx_symbols = np.where(d == 0, 1.0, -1.0)
    return tx_symbols


def demodulate_dbpsk(rx_symbols: np.ndarray) -> np.ndarray:
    """
    Differential BPSK Demodulation:
      Computes decision metric: U_k = y_k * conj(y_{k-1})
      If Re(U_k) > 0 -> bit 0 (no phase transition)
      If Re(U_k) < 0 -> bit 1 (phase transition of pi)
      Immune to constant phase offset: (s_k * e^{j*theta}) * (s_{k-1} * e^{j*theta})^*
                                     = s_k * s_{k-1}^* (phase cancels!)
    """
    u_k = rx_symbols[1:] * np.conj(rx_symbols[:-1])
    detected_bits = np.where(np.real(u_k) > 0.0, 0, 1)
    return detected_bits


# =============================================================================
# 3. CHANNEL: AWGN + CARRIER PHASE OFFSET
# =============================================================================

def apply_channel(symbols: np.ndarray, snr_db: float, phase_offset_deg: float = 0.0,
                  bits_per_symbol: int = 1, seed: Optional[int] = None) -> np.ndarray:
    """
    Add AWGN and carrier phase offset to complex baseband symbols.
    For symbol energy Es = 1.0:
      Eb = Es / bits_per_symbol
      Eb/N0 (linear) = 10^(snr_db / 10)
      N0 = Eb / (Eb/N0)
      Complex noise variance sigma^2 = N0/2 per dimension, total noise power N0.
    """
    eb = ES / bits_per_symbol
    snr_lin = 10.0 ** (snr_db / 10.0)
    n0 = eb / snr_lin
    sigma = np.sqrt(n0 / 2.0)

    # Carrier phase rotation
    phase_rad = np.deg2rad(phase_offset_deg)
    rotated_symbols = symbols * np.exp(1j * phase_rad)

    # Complex AWGN
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=len(symbols)) + 1j * rng.normal(0.0, sigma, size=len(symbols))

    return rotated_symbols + noise


# =============================================================================
# 4. MANDATORY SYLLABUS VALIDATION
# =============================================================================

def validate_qpsk_mapping() -> pd.DataFrame:
    """
    Mandatory Syllabus Validation Requirement:
    'Manually map 00, 01, 11 and 10 and check the corresponding detector decisions.'

    Tests:
      1. Ideal constellation coordinates.
      2. Demapped decision (Noiseless).
      3. Demapped decision with +30 deg phase offset.
      4. Demapped decision with +90 deg phase offset (demonstrating quadrant rotation / error).
    """
    dibits = np.array([
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0]
    ])

    bits_flat = dibits.flatten()
    symbols, _ = modulate_qpsk_gray(bits_flat)

    records = []
    phase_offsets = [0.0, 30.0, 90.0]

    for i, dibit in enumerate(dibits):
        sym = symbols[i]
        d_str = f"{dibit[0]}{dibit[1]}"
        phase_deg = np.rad2deg(np.angle(sym))
        if phase_deg < 0:
            phase_deg += 360.0

        for theta in phase_offsets:
            rx_sym = sym * np.exp(1j * np.deg2rad(theta))
            _, dem_dibits = demodulate_qpsk_gray(np.array([rx_sym]))
            dec_str = f"{dem_dibits[0, 0]}{dem_dibits[0, 1]}"
            is_correct = (dec_str == d_str)

            records.append({
                "Tx Dibit": d_str,
                "Ideal Symbol": f"{np.real(sym):+.4f} {np.imag(sym):+.4f}j",
                "Ideal Phase": f"{phase_deg:.1f}°",
                "Phase Offset": f"{theta:.0f}°",
                "Received Symbol": f"{np.real(rx_sym):+.4f} {np.imag(rx_sym):+.4f}j",
                "Rx Decision": dec_str,
                "Status": "CORRECT" if is_correct else "DECISION ERROR"
            })

    df = pd.DataFrame(records)
    return df


# =============================================================================
# 5. MONTE CARLO BER SIMULATION
# =============================================================================

def simulate_ber_phase_sweep(eb_no_db_range: np.ndarray, phase_offsets: List[float],
                             n_bits: int = 100000) -> Dict[str, np.ndarray]:
    """
    Simulate BER over Eb/N0 for:
      - Coherent BPSK (ideal)
      - Coherent QPSK (ideal)
      - DBPSK (ideal)
      - Coherent BPSK under phase offsets
      - Coherent QPSK under phase offsets
    """
    bits = generate_random_bits(n_bits, seed=123)
    if len(bits) % 2 != 0:
        bits = bits[:-1]

    results = {
        "eb_no_db": eb_no_db_range,
        "bpsk_ideal": np.zeros(len(eb_no_db_range)),
        "qpsk_ideal": np.zeros(len(eb_no_db_range)),
        "dbpsk_ideal": np.zeros(len(eb_no_db_range))
    }

    for po in phase_offsets:
        results[f"bpsk_po_{po:.0f}"] = np.zeros(len(eb_no_db_range))
        results[f"qpsk_po_{po:.0f}"] = np.zeros(len(eb_no_db_range))
        results[f"dbpsk_po_{po:.0f}"] = np.zeros(len(eb_no_db_range))

    # Pre-modulate
    bpsk_syms = modulate_bpsk(bits)
    qpsk_syms, _ = modulate_qpsk_gray(bits)
    dbpsk_syms = modulate_dbpsk(bits)

    for idx, eb_no_db in enumerate(eb_no_db_range):
        # 1. Ideal BPSK
        rx_bpsk = apply_channel(bpsk_syms, eb_no_db, phase_offset_deg=0.0, bits_per_symbol=1, seed=1000 + idx)
        results["bpsk_ideal"][idx] = np.mean(demodulate_bpsk(rx_bpsk) != bits)

        # 2. Ideal QPSK
        rx_qpsk = apply_channel(qpsk_syms, eb_no_db, phase_offset_deg=0.0, bits_per_symbol=2, seed=2000 + idx)
        q_bits, _ = demodulate_qpsk_gray(rx_qpsk)
        results["qpsk_ideal"][idx] = np.mean(q_bits != bits)

        # 3. Ideal DBPSK
        rx_dbpsk = apply_channel(dbpsk_syms, eb_no_db, phase_offset_deg=0.0, bits_per_symbol=1, seed=3000 + idx)
        results["dbpsk_ideal"][idx] = np.mean(demodulate_dbpsk(rx_dbpsk) != bits)

        # Phase offset evaluations
        for po in phase_offsets:
            # BPSK with phase offset
            rx_po_b = apply_channel(bpsk_syms, eb_no_db, phase_offset_deg=po, bits_per_symbol=1, seed=4000 + idx)
            results[f"bpsk_po_{po:.0f}"][idx] = np.mean(demodulate_bpsk(rx_po_b) != bits)

            # QPSK with phase offset
            rx_po_q = apply_channel(qpsk_syms, eb_no_db, phase_offset_deg=po, bits_per_symbol=2, seed=5000 + idx)
            q_po_bits, _ = demodulate_qpsk_gray(rx_po_q)
            results[f"qpsk_po_{po:.0f}"][idx] = np.mean(q_po_bits != bits)

            # DBPSK with phase offset (should remain invariant!)
            rx_po_d = apply_channel(dbpsk_syms, eb_no_db, phase_offset_deg=po, bits_per_symbol=1, seed=6000 + idx)
            results[f"dbpsk_po_{po:.0f}"][idx] = np.mean(demodulate_dbpsk(rx_po_d) != bits)

    return results


# =============================================================================
# 6. VISUALIZATION FUNCTIONS (SAVED TO OUTPUTS/)
# =============================================================================

def plot_constellations_ideal_noisy(filename: str = "1_constellations_ideal_and_noisy.png"):
    """Figure 1: Ideal and Noisy Constellation Diagrams for BPSK and QPSK."""
    n_syms = 2000
    bits = generate_random_bits(n_syms * 2, seed=42)
    bpsk_syms = modulate_bpsk(bits[:n_syms])
    qpsk_syms, _ = modulate_qpsk_gray(bits)

    snr_levels = [20.0, 10.0, 5.0]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))

    # (Row 1): BPSK
    axes[0, 0].scatter(np.real(bpsk_syms[:200]), np.imag(bpsk_syms[:200]), color="red", s=40, label="Symbols")
    axes[0, 0].set_title("BPSK: Ideal Constellation")

    for i, snr in enumerate(snr_levels):
        rx = apply_channel(bpsk_syms, snr, bits_per_symbol=1, seed=100 + i)
        axes[0, i + 1].scatter(np.real(rx), np.imag(rx), color="crimson", alpha=0.3, s=8)
        axes[0, i + 1].set_title(f"BPSK: AWGN ({snr:.0f} dB SNR)")

    # (Row 2): QPSK
    axes[1, 0].scatter(np.real(qpsk_syms[:200]), np.imag(qpsk_syms[:200]), color="blue", s=40, label="Symbols")
    axes[1, 0].set_title("QPSK: Ideal Gray Constellation")

    for i, snr in enumerate(snr_levels):
        rx = apply_channel(qpsk_syms, snr, bits_per_symbol=2, seed=200 + i)
        axes[1, i + 1].scatter(np.real(rx), np.imag(rx), color="navy", alpha=0.3, s=8)
        axes[1, i + 1].set_title(f"QPSK: AWGN ({snr:.0f} dB SNR)")

    for row in axes:
        for ax in row:
            ax.axhline(0, color="black", linestyle="--", alpha=0.6)
            ax.axvline(0, color="black", linestyle="--", alpha=0.6)
            ax.set_xlim(-2.0, 2.0)
            ax.set_ylim(-2.0, 2.0)
            ax.set_xlabel("In-Phase (I)")
            ax.set_ylabel("Quadrature (Q)")
            ax.grid(True)
            ax.set_aspect("equal")

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath.name}")


def plot_qpsk_gray_mapping(filename: str = "2_qpsk_gray_mapping_decision_regions.png"):
    """Figure 2: Gray mapping and decision regions in I-Q complex plane."""
    fig, ax = plt.subplots(figsize=(8.5, 8.5))

    # Decision boundaries
    ax.axhline(0, color="black", linewidth=2.0)
    ax.axvline(0, color="black", linewidth=2.0)

    # Quadrant shading (decision regions)
    ax.fill_between([0, 2], 0, 2, color="blue", alpha=0.08, label="Region '00'")
    ax.fill_between([-2, 0], 0, 2, color="orange", alpha=0.08, label="Region '01'")
    ax.fill_between([-2, 0], -2, 0, color="green", alpha=0.08, label="Region '11'")
    ax.fill_between([0, 2], -2, 0, color="purple", alpha=0.08, label="Region '10'")

    # Unit circle representing Es = 1.0
    circle = plt.Circle((0, 0), 1.0, color="gray", fill=False, linestyle=":", linewidth=1.5, label="Constant Energy $|s|=1$")
    ax.add_artist(circle)

    # Plot constellation points
    norm = 1.0 / np.sqrt(2.0)
    points = [
        (norm, norm, "00", 45, "blue"),
        (-norm, norm, "01", 135, "darkorange"),
        (-norm, -norm, "11", 225, "forestgreen"),
        (norm, -norm, "10", 315, "purple")
    ]

    for x, y, label, angle, col in points:
        ax.plot(x, y, "o", markersize=14, color=col, markeredgecolor="black", markeredgewidth=1.5)
        ax.annotate(f"'{label}'\n({angle}°)", (x, y), textcoords="offset points",
                    xytext=(15 if x > 0 else -35, 15 if y > 0 else -30),
                    fontsize=12, fontweight="bold", color=col,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=1.5))
        # Draw vector from origin
        ax.arrow(0, 0, x * 0.9, y * 0.9, head_width=0.05, head_length=0.05, fc=col, ec=col, alpha=0.7)

    ax.set_title("QPSK Gray Mapping & Euclidean Slicing Decision Boundaries\n(Adjacent symbols differ by exactly 1 bit)", fontsize=13)
    ax.set_xlabel("In-Phase Rail (Re)", fontsize=12)
    ax.set_ylabel("Quadrature Rail (Im)", fontsize=12)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_aspect("equal")

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath.name}")


def plot_phase_trajectories_rotation(filename: str = "3_phase_trajectories_and_constellation_rotation.png"):
    """Figure 3: Carrier phase offsets (0°, 10°, 30°, 60°, 90°) causing constellation rotation."""
    n_syms = 1500
    bits = generate_random_bits(n_syms * 2, seed=777)
    qpsk_syms, _ = modulate_qpsk_gray(bits)

    offsets = [0.0, 10.0, 30.0, 60.0, 90.0]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), sharey=True)

    for i, po in enumerate(offsets):
        rx = apply_channel(qpsk_syms, snr_db=18.0, phase_offset_deg=po, bits_per_symbol=2, seed=800 + i)
        axes[i].scatter(np.real(rx), np.imag(rx), color="royalblue", alpha=0.35, s=12)
        axes[i].axhline(0, color="black", linestyle="--", linewidth=1.2)
        axes[i].axvline(0, color="black", linestyle="--", linewidth=1.2)

        # Plot rotated ideal points
        rot_ideals = (np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2.0)) * np.exp(1j * np.deg2rad(po))
        axes[i].scatter(np.real(rot_ideals), np.imag(rot_ideals), color="red", marker="x", s=80, linewidth=2.5)

        axes[i].set_title(f"Phase Offset $\\Delta\\theta = {po:.0f}^\\circ$", fontsize=12)
        axes[i].set_xlabel("In-Phase (I)")
        if i == 0:
            axes[i].set_ylabel("Quadrature (Q)")
        axes[i].set_xlim(-1.8, 1.8)
        axes[i].set_ylim(-1.8, 1.8)
        axes[i].grid(True)
        axes[i].set_aspect("equal")

    fig.suptitle("Impact of Carrier Phase Offset on QPSK Constellation & Decision Boundaries (AWGN SNR = 18 dB)", fontsize=14)
    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath.name}")


def plot_decision_histograms(filename: str = "4_decision_statistic_histograms.png"):
    """Figure 4: Decision statistic distributions (I and Q rails) under phase offsets."""
    n_syms = 8000
    bits = generate_random_bits(n_syms * 2, seed=999)
    qpsk_syms, _ = modulate_qpsk_gray(bits)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Test cases: (0 deg vs 30 deg offset at 10 dB SNR)
    rx_0 = apply_channel(qpsk_syms, snr_db=10.0, phase_offset_deg=0.0, bits_per_symbol=2, seed=101)
    rx_30 = apply_channel(qpsk_syms, snr_db=10.0, phase_offset_deg=30.0, bits_per_symbol=2, seed=102)

    # I-Rail at 0 deg
    axes[0, 0].hist(np.real(rx_0[bits[0::2] == 0]), bins=50, density=True, alpha=0.6, color="blue", label="Bit 0 ($I > 0$)")
    axes[0, 0].hist(np.real(rx_0[bits[0::2] == 1]), bins=50, density=True, alpha=0.6, color="red", label="Bit 1 ($I < 0$)")
    axes[0, 0].axvline(0, color="black", linestyle="--", linewidth=1.8, label="Decision Boundary")
    axes[0, 0].set_title(r"In-Phase Rail ($\Delta\theta = 0^\circ$ Offset - Pristine Margin)")
    axes[0, 0].set_ylabel("Probability Density")
    axes[0, 0].grid(True)
    axes[0, 0].legend(loc="upper right")

    # Q-Rail at 0 deg
    axes[0, 1].hist(np.imag(rx_0[bits[1::2] == 0]), bins=50, density=True, alpha=0.6, color="blue", label="Bit 0 ($Q > 0$)")
    axes[0, 1].hist(np.imag(rx_0[bits[1::2] == 1]), bins=50, density=True, alpha=0.6, color="red", label="Bit 1 ($Q < 0$)")
    axes[0, 1].axvline(0, color="black", linestyle="--", linewidth=1.8, label="Decision Boundary")
    axes[0, 1].set_title(r"Quadrature Rail ($\Delta\theta = 0^\circ$ Offset - Pristine Margin)")
    axes[0, 1].grid(True)
    axes[0, 1].legend(loc="upper right")

    # I-Rail at 30 deg
    axes[1, 0].hist(np.real(rx_30[bits[0::2] == 0]), bins=50, density=True, alpha=0.6, color="blue", label="Bit 0")
    axes[1, 0].hist(np.real(rx_30[bits[0::2] == 1]), bins=50, density=True, alpha=0.6, color="red", label="Bit 1")
    axes[1, 0].axvline(0, color="black", linestyle="--", linewidth=1.8, label="Decision Boundary")
    axes[1, 0].set_title(r"In-Phase Rail ($\Delta\theta = 30^\circ$ Offset - Severe Cloud Spread)")
    axes[1, 0].set_xlabel("Re(y) Decision Statistic")
    axes[1, 0].set_ylabel("Probability Density")
    axes[1, 0].grid(True)
    axes[1, 0].legend(loc="upper right")

    # Q-Rail at 30 deg
    axes[1, 1].hist(np.imag(rx_30[bits[1::2] == 0]), bins=50, density=True, alpha=0.6, color="blue", label="Bit 0")
    axes[1, 1].hist(np.imag(rx_30[bits[1::2] == 1]), bins=50, density=True, alpha=0.6, color="red", label="Bit 1")
    axes[1, 1].axvline(0, color="black", linestyle="--", linewidth=1.8, label="Decision Boundary")
    axes[1, 1].set_title(r"Quadrature Rail ($\Delta\theta = 30^\circ$ Offset - Cross-Talk Spillover)")
    axes[1, 1].set_xlabel("Im(y) Decision Statistic")
    axes[1, 1].grid(True)
    axes[1, 1].legend(loc="upper right")

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath.name}")


def plot_ber_performance(filename: str = "5_ber_vs_ebno_comparison.png"):
    """Figure 5: Publication-grade semilog BER comparison over AWGN."""
    eb_no_db = np.linspace(0.0, 12.0, 13)
    phase_offsets = [10.0, 30.0]

    print("Running Monte Carlo BER simulation (100,000 bits per point)...")
    sim = simulate_ber_phase_sweep(eb_no_db, phase_offsets, n_bits=100000)

    # Theoretical curves
    eb_fine = np.linspace(0.0, 12.0, 100)
    eb_lin = 10.0 ** (eb_fine / 10.0)
    th_bpsk = 0.5 * sp.erfc(np.sqrt(eb_lin))               # Q(sqrt(2*Eb/N0))
    th_dbpsk = 0.5 * np.exp(-eb_lin)                       # 0.5 * exp(-Eb/N0)

    # BPSK with phase offset: Q(sqrt(2*Eb/N0) * cos(theta))
    th_bpsk_10 = 0.5 * sp.erfc(np.sqrt(eb_lin) * np.cos(np.deg2rad(10.0)))
    th_bpsk_30 = 0.5 * sp.erfc(np.sqrt(eb_lin) * np.cos(np.deg2rad(30.0)))

    fig, ax = plt.subplots(figsize=(11, 7.5))

    # Coherent BPSK & QPSK (Identical in AWGN!)
    ax.semilogy(eb_fine, th_bpsk, "b-", linewidth=2.0, label="Coherent BPSK / QPSK (Theory)")
    ax.semilogy(sim["eb_no_db"], sim["bpsk_ideal"], "bo", markersize=6, fillstyle="none", markeredgewidth=1.8, label="Coherent BPSK (Sim)")
    ax.semilogy(sim["eb_no_db"], sim["qpsk_ideal"], "b^", markersize=6, fillstyle="none", markeredgewidth=1.8, label="Coherent QPSK (Sim)")

    # DBPSK
    ax.semilogy(eb_fine, th_dbpsk, "g-", linewidth=1.8, label="DBPSK (Theory)")
    ax.semilogy(sim["eb_no_db"], sim["dbpsk_ideal"], "gs", markersize=6, fillstyle="none", markeredgewidth=1.8, label="DBPSK (Sim)")

    # BPSK with Phase Offset
    ax.semilogy(eb_fine, th_bpsk_10, "m--", linewidth=1.5, label="BPSK $\\Delta\\theta=10^\\circ$ (Theory)")
    ax.semilogy(sim["eb_no_db"], sim["bpsk_po_10"], "md", markersize=5, fillstyle="none", label="BPSK $\\Delta\\theta=10^\\circ$ (Sim)")

    ax.semilogy(eb_fine, th_bpsk_30, "r--", linewidth=1.5, label="BPSK $\\Delta\\theta=30^\\circ$ (Theory)")
    ax.semilogy(sim["eb_no_db"], sim["bpsk_po_30"], "rx", markersize=6, markeredgewidth=1.8, label="BPSK $\\Delta\\theta=30^\\circ$ (Sim)")

    # DBPSK with 30 deg offset (Immunity!)
    ax.semilogy(sim["eb_no_db"], sim["dbpsk_po_30"], "k+", markersize=7, markeredgewidth=1.8, label="DBPSK $\\Delta\\theta=30^\\circ$ (Sim - Phase Immune!)")

    ax.set_title("Bit Error Rate (BER) for BPSK, QPSK, and DBPSK with Carrier Phase Offsets", fontsize=13)
    ax.set_xlabel("Energy per Bit to Noise Spectral Density $E_b / N_0$ (dB)", fontsize=12)
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=12)
    ax.set_xlim(0, 12)
    ax.set_ylim(1e-5, 0.6)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(loc="lower left", framealpha=0.95)

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath.name}")


def plot_ber_vs_phase_offset(filename: str = "6_ber_degradation_vs_phase_offset.png"):
    """Figure 6: BER degradation as a function of phase offset Delta_theta in [0, 90] degrees."""
    phase_angles = np.linspace(0.0, 90.0, 19)
    snr_db = 8.0  # Moderate SNR condition
    n_bits = 80000

    bits = generate_random_bits(n_bits, seed=54321)
    if len(bits) % 2 != 0:
        bits = bits[:-1]

    bpsk_syms = modulate_bpsk(bits)
    qpsk_syms, _ = modulate_qpsk_gray(bits)
    dbpsk_syms = modulate_dbpsk(bits)

    ber_bpsk = np.zeros(len(phase_angles))
    ber_qpsk = np.zeros(len(phase_angles))
    ber_dbpsk = np.zeros(len(phase_angles))

    for i, po in enumerate(phase_angles):
        rx_b = apply_channel(bpsk_syms, snr_db, phase_offset_deg=po, bits_per_symbol=1, seed=300 + i)
        ber_bpsk[i] = np.mean(demodulate_bpsk(rx_b) != bits)

        rx_q = apply_channel(qpsk_syms, snr_db, phase_offset_deg=po, bits_per_symbol=2, seed=400 + i)
        q_bits, _ = demodulate_qpsk_gray(rx_q)
        ber_qpsk[i] = np.mean(q_bits != bits)

        rx_d = apply_channel(dbpsk_syms, snr_db, phase_offset_deg=po, bits_per_symbol=1, seed=500 + i)
        ber_dbpsk[i] = np.mean(demodulate_dbpsk(rx_d) != bits)

    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.semilogy(phase_angles, ber_bpsk, "ro-", linewidth=2.0, label=f"Coherent BPSK ($E_b/N_0={snr_db}$ dB)")
    ax.semilogy(phase_angles, ber_qpsk, "bs-", linewidth=2.0, label=f"Coherent QPSK ($E_b/N_0={snr_db}$ dB)")
    ax.semilogy(phase_angles, ber_dbpsk, "g^--", linewidth=2.2, label=f"DBPSK ($E_b/N_0={snr_db}$ dB - Invariant!)")

    # Annotations
    ax.axvline(45.0, color="purple", linestyle=":", linewidth=1.5, label=r"QPSK Decision Boundary Collapse ($45^\circ$)")
    ax.axvline(90.0, color="crimson", linestyle=":", linewidth=1.5, label=r"BPSK Complete Collapse ($90^\circ$)")

    ax.set_title(f"Bit Error Rate Degradation vs Carrier Phase Offset ($E_b/N_0 = {snr_db}$ dB)", fontsize=13)
    ax.set_xlabel("Carrier Phase Offset $\\Delta\\theta$ (Degrees)", fontsize=12)
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=12)
    ax.set_xlim(0, 90)
    ax.set_ylim(1e-4, 0.6)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(loc="center left", framealpha=0.95)

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath.name}")


def plot_time_domain_waveforms(filename: str = "7_time_domain_iq_waveforms.png"):
    """Figure 7: Time-domain In-Phase and Quadrature baseband pulse waveforms."""
    bits = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0])
    qpsk_syms, dibits = modulate_qpsk_gray(bits)
    n_syms = len(qpsk_syms)

    sps = 50
    t = np.arange(n_syms * sps) / (sps * RS)
    i_wave = np.repeat(np.real(qpsk_syms), sps)
    q_wave = np.repeat(np.imag(qpsk_syms), sps)
    phase_wave = np.repeat(np.rad2deg(np.angle(qpsk_syms)), sps)

    fig, axes = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True)

    # I-Rail
    axes[0].plot(t * 1000, i_wave, color="royalblue", linewidth=2.0)
    axes[0].set_ylabel("In-Phase $I(t)$")
    axes[0].set_title("QPSK Baseband In-Phase Component ($b_0$: 0 -> +0.707, 1 -> -0.707)")
    axes[0].set_ylim(-1.0, 1.0)
    axes[0].grid(True)

    # Q-Rail
    axes[1].plot(t * 1000, q_wave, color="darkorange", linewidth=2.0)
    axes[1].set_ylabel("Quadrature $Q(t)$")
    axes[1].set_title("QPSK Baseband Quadrature Component ($b_1$: 0 -> +0.707, 1 -> -0.707)")
    axes[1].set_ylim(-1.0, 1.0)
    axes[1].grid(True)

    # Symbol Phase
    axes[2].step(t * 1000, phase_wave, where="post", color="forestgreen", linewidth=2.0)
    axes[2].set_ylabel("Phase (Degrees)")
    axes[2].set_xlabel("Time (milliseconds)")
    axes[2].set_title("QPSK Symbol Phase Trajectory: [45°, 135°, 225°, 315°]")
    axes[2].set_ylim(-190, 190)
    axes[2].grid(True)

    for i in range(n_syms):
        d_str = f"{dibits[i, 0]}{dibits[i, 1]}"
        for ax in axes:
            ax.axvline(i * TS * 1000, color="black", linestyle=":", alpha=0.5)
        axes[0].text((i + 0.5) * TS * 1000, 0.0, f"'{d_str}'", ha="center", va="center",
                     fontsize=11, fontweight="bold", bbox=dict(boxstyle="square", fc="white", alpha=0.7))

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath.name}")


def plot_mandatory_validation(df_val: pd.DataFrame, filename: str = "8_mandatory_mapping_validation.png"):
    """Figure 8: Graphical display of mandatory QPSK mapping and detector validation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Constellation mapping with phase rotation
    norm = 1.0 / np.sqrt(2.0)
    ideals = {
        "00": (norm, norm, "blue"),
        "01": (-norm, norm, "darkorange"),
        "11": (-norm, -norm, "forestgreen"),
        "10": (norm, -norm, "purple")
    }

    for dibit, (x, y, col) in ideals.items():
        # Ideal
        ax1.plot(x, y, "o", markersize=12, color=col, markeredgecolor="black", label=f"Tx '{dibit}' (Ideal)")
        # 30 deg rotated
        rot_30 = (x + 1j * y) * np.exp(1j * np.deg2rad(30.0))
        ax1.plot(np.real(rot_30), np.imag(rot_30), "^", markersize=9, color=col, alpha=0.6)
        # 90 deg rotated (causes bit error!)
        rot_90 = (x + 1j * y) * np.exp(1j * np.deg2rad(90.0))
        ax1.plot(np.real(rot_90), np.imag(rot_90), "x", markersize=10, markeredgewidth=2, color="red")

    ax1.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax1.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax1.set_title("Mandatory Validation: Dibits '00', '01', '11', '10'\n[O = Ideal, ^ = +30°, X = +90° Rotated]", fontsize=12)
    ax1.set_xlabel("In-Phase Rail (Re)")
    ax1.set_ylabel("Quadrature Rail (Im)")
    ax1.set_xlim(-1.6, 1.6)
    ax1.set_ylim(-1.6, 1.6)
    ax1.grid(True)
    ax1.set_aspect("equal")

    # Table subplot
    ax2.axis("off")
    table_data = []
    # Filter representative rows
    for _, r in df_val.iterrows():
        table_data.append([
            r["Tx Dibit"],
            r["Ideal Phase"],
            r["Phase Offset"],
            r["Received Symbol"],
            r["Rx Decision"],
            r["Status"]
        ])
    col_labels = ["Tx Dibit", "Ideal Phase", "Offset", "Rx Symbol (I + jQ)", "Rx Decision", "Status"]
    table = ax2.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.1, 1.7)
    ax2.set_title("Detector Decisions Under Phase Offsets", fontsize=12, pad=15)

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath.name}")


# =============================================================================
# 7. MAIN DRIVER
# =============================================================================

def main():
    print("=" * 80)
    print("Digital Communication Lab - Experiment 10: BPSK, QPSK & DPSK Simulation")
    print("=" * 80)

    # 1. Mandatory Syllabus Validation
    print("\n--- 1. Mandatory Syllabus Validation: QPSK Gray Mapping & Demapper Decisions ---")
    df_val = validate_qpsk_mapping()
    print(df_val.to_string(index=False))

    # 2. Visualizations
    print("\n--- 2. Generating Publication Visualizations (outputs/) ---")
    plot_constellations_ideal_noisy()
    plot_qpsk_gray_mapping()
    plot_phase_trajectories_rotation()
    plot_decision_histograms()
    plot_ber_vs_phase_offset()
    plot_time_domain_waveforms()
    plot_mandatory_validation(df_val)
    plot_ber_performance()

    print("\n" + "=" * 80)
    print("Experiment 10 simulation completed successfully!")
    print(f"All outputs generated in: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()

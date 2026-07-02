#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Postprocess a direct Contini run into a normalized flame transfer function using cross spectra."
    )
    parser.add_argument("--config", type=Path, default=Path("contini_flame_unsteady.cfg"))
    parser.add_argument("--history", type=Path, default=Path("history.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("chirp_ftf.csv"))
    parser.add_argument("--output-plot", type=Path, default=Path("chirp_ftf.png"))
    parser.add_argument(
        "--window-start-iter",
        type=int,
        default=None,
        help="First time iteration included in the analysis window. Defaults to WINDOW_START_ITER from the cfg.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Number of samples in the analysis window. Defaults to ITER_AVERAGE_OBJ from the cfg, else all remaining samples.",
    )
    parser.add_argument(
        "--segment-length",
        type=int,
        default=256,
        help="Segment length for the Welch-style cross spectral estimate.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.75,
        help="Segment overlap fraction in [0, 1).",
    )
    parser.add_argument(
        "--coherence-threshold",
        type=float,
        default=0.8,
        help="Threshold used to highlight trustworthy frequency bands.",
    )
    parser.add_argument("--min-freq", type=float, default=1.0, help="Minimum plotted/exported frequency in Hz.")
    parser.add_argument("--max-freq", type=float, default=180.0, help="Maximum plotted/exported frequency in Hz.")
    return parser.parse_args()


def read_config(path: Path) -> dict[str, str]:
    config: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("%", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        config[key.strip().upper()] = value.strip()
    return config


def parse_float(config: dict[str, str], key: str, default: float = 0.0) -> float:
    value = config.get(key.upper())
    if value is None:
        return default
    return float(value.split()[0].strip())


def parse_int(config: dict[str, str], key: str, default: int = 0) -> int:
    value = config.get(key.upper())
    if value is None:
        return default
    return int(float(value.split()[0].strip()))


def strip_quotes(text: str) -> str:
    return text.strip().strip('"').strip()


def read_history(path: Path) -> tuple[list[str], list[list[float]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        try:
            header = [strip_quotes(entry) for entry in next(reader)]
        except StopIteration as exc:
            raise RuntimeError(f"History file is empty: {path}") from exc

        rows: list[list[float]] = []
        for row in reader:
            if not row:
                continue
            cleaned = [entry.strip() for entry in row if entry.strip()]
            if not cleaned:
                continue
            rows.append([float(entry) for entry in cleaned])

    if not rows:
        raise RuntimeError(f"No history samples found in {path}")

    return header, rows


def find_column(header: list[str], *candidates: str) -> int:
    normalized = {name.upper().replace(" ", ""): idx for idx, name in enumerate(header)}
    for candidate in candidates:
        key = candidate.upper().replace(" ", "")
        if key in normalized:
            return normalized[key]
    raise RuntimeError(f"None of the columns {candidates} were found in the history header: {header}")


def build_time_vector(header: list[str], rows: list[list[float]], time_step: float) -> np.ndarray:
    try:
        time_iter_idx = find_column(header, "Time_Iter")
        iter_values = np.array([row[time_iter_idx] for row in rows], dtype=float)
    except RuntimeError:
        iter_values = np.arange(len(rows), dtype=float)
    return iter_values * time_step


def forcing_signal(config: dict[str, str], time: np.ndarray) -> np.ndarray:
    chirp_amp = parse_float(config, "INLET_CHIRP_AMPLITUDE", 0.0)
    if abs(chirp_amp) > 0.0:
        freq_start = parse_float(config, "INLET_CHIRP_FREQ_START")
        freq_end = parse_float(config, "INLET_CHIRP_FREQ_END")
        duration = parse_float(config, "INLET_CHIRP_DURATION")
        start_time = parse_float(config, "INLET_CHIRP_START_TIME", 0.0)
        phase0 = parse_float(config, "INLET_CHIRP_PHASE", 0.0)
        chirp_method = config.get("INLET_CHIRP_METHOD", "LINEAR").strip().upper()
        if duration <= 0.0:
            raise ValueError("INLET_CHIRP_DURATION must be positive when chirp forcing is enabled.")

        signal = np.ones_like(time)
        active = (time >= start_time) & (time <= start_time + duration)
        t_rel = time[active] - start_time
        if chirp_method == "LOGARITHMIC":
            if freq_start <= 0.0 or freq_end <= 0.0:
                raise ValueError("LOGARITHMIC inlet chirp requires positive INLET_CHIRP_FREQ_START and INLET_CHIRP_FREQ_END.")
            growth = math.log(freq_end / freq_start) / duration
            phase = phase0 + 2.0 * math.pi * freq_start * (np.exp(growth * t_rel) - 1.0) / growth
        else:
            chirp_rate = (freq_end - freq_start) / duration
            phase = phase0 + 2.0 * math.pi * (freq_start * t_rel + 0.5 * chirp_rate * t_rel * t_rel)
        signal[active] = 1.0 + chirp_amp * np.sin(phase)
        return signal

    sine_amp = parse_float(config, "INLET_SINE_AMPLITUDE", 0.0)
    sine_freq = parse_float(config, "INLET_SINE_FREQUENCY", 0.0)
    sine_phase = parse_float(config, "INLET_SINE_PHASE", 0.0)
    omega_t = 2.0 * math.pi * sine_freq * time + sine_phase
    return 1.0 + sine_amp * np.sin(omega_t)


def hann_window(length: int) -> np.ndarray:
    return np.hanning(length)


def welch_transfer_function(
    u: np.ndarray,
    q: np.ndarray,
    dt: float,
    segment_length: int,
    overlap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if segment_length < 8:
        raise ValueError("--segment-length must be at least 8.")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("--overlap must be in [0, 1).")
    if len(u) != len(q):
        raise ValueError("Input and output signals must have the same length.")
    if len(u) < segment_length:
        raise ValueError("Windowed signal is shorter than --segment-length.")

    step = max(1, int(round(segment_length * (1.0 - overlap))))
    starts = range(0, len(u) - segment_length + 1, step)
    window = hann_window(segment_length)
    window_power = np.sum(window * window)

    suu = None
    sqq = None
    squ = None
    freq = None
    n_segments = 0

    for start in starts:
        stop = start + segment_length
        u_seg = u[start:stop]
        q_seg = q[start:stop]

        u_seg = u_seg - np.mean(u_seg)
        q_seg = q_seg - np.mean(q_seg)

        u_win = u_seg * window
        q_win = q_seg * window

        u_hat = np.fft.rfft(u_win)
        q_hat = np.fft.rfft(q_win)

        if freq is None:
            freq = np.fft.rfftfreq(segment_length, d=dt)
            suu = np.zeros_like(u_hat, dtype=np.complex128)
            sqq = np.zeros_like(q_hat, dtype=np.complex128)
            squ = np.zeros_like(q_hat, dtype=np.complex128)

        scale = 1.0 / window_power
        suu += scale * np.conj(u_hat) * u_hat
        sqq += scale * np.conj(q_hat) * q_hat
        squ += scale * q_hat * np.conj(u_hat)
        n_segments += 1

    if n_segments == 0 or freq is None or suu is None or sqq is None or squ is None:
        raise RuntimeError("No valid spectral segments were built.")

    suu /= n_segments
    sqq /= n_segments
    squ /= n_segments

    valid = np.abs(suu) > 1.0e-18
    h = np.full_like(squ, np.nan + 1j * np.nan, dtype=np.complex128)
    h[valid] = squ[valid] / suu[valid]

    coherence = np.zeros_like(freq, dtype=float)
    denom = (sqq.real * suu.real)
    coh_valid = denom > 1.0e-30
    coherence[coh_valid] = np.abs(squ[coh_valid]) ** 2 / denom[coh_valid]
    coherence = np.clip(coherence, 0.0, 1.0)

    return freq, h, coherence, suu.real, sqq.real, n_segments


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    header, rows = read_history(args.history)

    time_step = parse_float(config, "TIME_STEP")
    window_start_iter = args.window_start_iter
    if window_start_iter is None:
        window_start_iter = parse_int(config, "WINDOW_START_ITER", 0)

    window_size = args.window_size
    if window_size is None:
        cfg_window = parse_int(config, "ITER_AVERAGE_OBJ", 0)
        if cfg_window > 0:
            window_size = cfg_window

    try:
        heat_release_idx = find_column(header, "HeatReleaseGlobal", "HEAT_RELEASE_GLOBAL")
    except RuntimeError as exc:
        raise RuntimeError(
            "Heat-release column not found in the selected history file. "
            "This usually means you passed an adjoint history instead of a direct-run history."
        ) from exc

    time = build_time_vector(header, rows, time_step)
    heat_release = np.array([row[heat_release_idx] for row in rows], dtype=float)
    inlet_signal = forcing_signal(config, time)

    if window_start_iter < 0:
        raise ValueError("--window-start-iter must be non-negative.")

    available = len(time) - window_start_iter
    if available <= 1:
        raise ValueError("The requested window starts beyond the available history samples.")

    if window_size is None or window_size <= 0 or window_size > available:
        window_size = available

    window_slice = slice(window_start_iter, window_start_iter + window_size)
    time_window = time[window_slice]
    q_window = heat_release[window_slice]
    u_window = inlet_signal[window_slice]

    q_mean = float(np.mean(q_window))
    u_mean = float(np.mean(u_window))
    if abs(q_mean) == 0.0:
        raise ZeroDivisionError("Mean heat release in the selected window is zero.")
    if abs(u_mean) == 0.0:
        raise ZeroDivisionError("Mean inlet signal in the selected window is zero.")

    q_norm = (q_window - q_mean) / q_mean
    u_norm = (u_window - u_mean) / u_mean

    segment_length = args.segment_length
    if len(u_norm) < segment_length:
        segment_length = max(8, 2 ** int(math.floor(math.log2(len(u_norm)))))
    if len(u_norm) < 8:
        raise ValueError("Need at least 8 samples in the selected analysis window.")

    freq, ftf, coherence, suu, sqq, n_segments = welch_transfer_function(
        u_norm, q_norm, time_step, segment_length, args.overlap
    )

    freq_mask = freq >= args.min_freq
    if args.max_freq is not None:
        freq_mask &= freq <= args.max_freq
    if len(freq_mask) > 0:
        freq_mask[0] = False

    freq_sel = freq[freq_mask]
    ftf_sel = ftf[freq_mask]
    coh_sel = coherence[freq_mask]
    suu_sel = suu[freq_mask]
    sqq_sel = sqq[freq_mask]

    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frequency_hz",
                "ftf_real",
                "ftf_imag",
                "ftf_amp",
                "ftf_phase_deg",
                "coherence",
                "suu",
                "sqq",
                "trusted",
            ]
        )
        for f, value, coh, suu_i, sqq_i in zip(freq_sel, ftf_sel, coh_sel, suu_sel, sqq_sel):
            writer.writerow(
                [
                    f,
                    value.real,
                    value.imag,
                    abs(value),
                    np.angle(value, deg=True),
                    coh,
                    suu_i,
                    sqq_i,
                    int(coh >= args.coherence_threshold),
                ]
            )

    trusted = coh_sel >= args.coherence_threshold

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(freq_sel, np.abs(ftf_sel), lw=1.2, color="tab:blue", label="All")
    if np.any(trusted):
        axes[0].plot(freq_sel[trusted], np.abs(ftf_sel[trusted]), lw=0, marker="o", ms=3, color="tab:orange", label="Trusted")
    axes[0].set_ylabel(r"$|FTF|$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(freq_sel, np.angle(ftf_sel, deg=True), lw=1.2, color="tab:blue")
    if np.any(trusted):
        axes[1].plot(freq_sel[trusted], np.angle(ftf_sel[trusted], deg=True), lw=0, marker="o", ms=3, color="tab:orange")
    axes[1].set_ylabel("Phase [deg]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(freq_sel, coh_sel, lw=1.2, color="tab:green")
    axes[2].axhline(args.coherence_threshold, color="tab:red", ls="--", lw=1.0)
    axes[2].set_xlabel("Frequency [Hz]")
    axes[2].set_ylabel("Coherence")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Normalized Flame Transfer Function from Cross Spectra")
    fig.tight_layout()
    fig.savefig(args.output_plot, dpi=200, bbox_inches="tight")

    trusted_count = int(np.count_nonzero(trusted))
    print(f"Config: {args.config}")
    print(f"History: {args.history}")
    print(f"Window start iter: {window_start_iter}")
    print(f"Window size: {window_size}")
    print(f"Segment length: {segment_length}")
    print(f"Overlap: {args.overlap:.3f}")
    print(f"Number of segments: {n_segments}")
    print(f"Mean heat release: {q_mean:.16e}")
    print(f"Mean inlet signal: {u_mean:.16e}")
    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.output_plot}")
    print(f"Trusted frequency bins: {trusted_count} / {len(freq_sel)} with coherence >= {args.coherence_threshold:.3f}")


if __name__ == "__main__":
    main()

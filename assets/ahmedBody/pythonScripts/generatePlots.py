"""Generate plots from the single Ahmed-body OpenFOAM case."""
from io import StringIO
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


CASE_DIR = Path(__file__).resolve().parents[1]
IMAGE_DIR = CASE_DIR / "report" / "images"
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})


def latest_directory(directory: Path) -> Path | None:
    directories = [path for path in directory.iterdir() if path.is_dir()]
    return max(directories, key=lambda path: float(path.name)) if directories else None


def plot_profiles() -> bool:
    output_dir = CASE_DIR / "postProcessing" / "velocityProfiles"
    latest_dir = latest_directory(output_dir) if output_dir.is_dir() else None
    files = sorted(latest_dir.glob("*.xy")) if latest_dir else []
    if not files:
        print("No U velocity profiles found; run the OpenFOAM case first.")
        return False

    fig, (streamwise, vertical) = plt.subplots(1, 2, figsize=(10, 4))
    for filepath in files:
        data = np.loadtxt(filepath, comments="#")
        if data.ndim != 2 or data.shape[1] < 4:
            print(f"Skipping {filepath}: expected y Ux Uy Uz columns.")
            continue
        label = re.sub(r"_", "/", filepath.stem)
        y = data[:, 0]
        streamwise.plot(data[:, 1], y, label=label)
        vertical.plot(data[:, 2], y, label=label)

    for axis, title, xlabel in (
        (streamwise, "Mean streamwise velocity", r"$\langle U_x\rangle$ [m/s]"),
        (vertical, "Mean vertical velocity", r"$\langle U_y\rangle$ [m/s]"),
    ):
        axis.set_xlabel(xlabel)
        axis.set_ylabel("y [m]")
        axis.set_title(title)
        axis.grid(True)
        axis.legend()
    fig.savefig(IMAGE_DIR / "velocity_profiles.png", dpi=300)
    plt.close(fig)
    return True


def plot_psd() -> bool:
    probe_dir = CASE_DIR / "postProcessing" / "probes1"
    probe_dir = latest_directory(probe_dir) if probe_dir.is_dir() else None
    filepath = probe_dir / "U" if probe_dir else None
    if not filepath or not filepath.exists():
        print("No probe data found; run the OpenFOAM case first.")
        return False

    probe_text = re.sub(r"[()]", " ", filepath.read_text())
    data = np.loadtxt(StringIO(probe_text), comments="#")
    if data.ndim != 2 or data.shape[0] < 2:
        print(f"Skipping {filepath}: not enough samples.")
        return False
    time = data[:, 0]
    dt = np.mean(np.diff(time))
    if dt <= 0:
        raise ValueError(f"Probe times in {filepath} are not increasing.")

    num_probes = (data.shape[1] - 1) // 3
    columns = min(num_probes, 5)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), squeeze=False)
    for index in range(columns):
        signal = data[:, 1 + 3 * index]
        frequencies, spectrum = welch(
            signal - np.mean(signal), fs=1.0 / dt, nperseg=min(512, len(signal))
        )
        axis = axes.flat[index]
        axis.plot(frequencies, spectrum, "k-")
        axis.set_title(f"Probe {index + 1}")
        axis.set_xlabel("f [Hz]")
        axis.set_ylabel(r"$S_{U_x}$")
        axis.grid(True)
    for axis in axes.flat[columns:]:
        axis.set_visible(False)
    fig.savefig(IMAGE_DIR / "velocity_psd.png", dpi=300)
    plt.close(fig)
    return True


if __name__ == "__main__":
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    profiles_created = plot_profiles()
    psd_created = plot_psd()
    if not profiles_created or not psd_created:
        raise SystemExit(1)
    print(f"Plots saved to {IMAGE_DIR}")
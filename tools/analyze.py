import argparse
import json
import os
from pathlib import Path
from typing import Any

import librosa
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

NFFT = 4096
HOP = NFFT // 4


def analyze(file: Path) -> dict[str, Any]:
    y, sr = librosa.load(file)

    if y.ndim != 1:
        raise Exception(
            f"Loaded audio file had {y.ndim} channels, we only support mono"
        )

    S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP))
    spec_bw = librosa.feature.spectral_bandwidth(S=S, sr=sr).flatten()
    times = librosa.times_like(spec_bw, sr=sr, hop_length=HOP, n_fft=NFFT)

    return {
        "sample_rate": sr,
        "spectrogram": S,
        "times": times,
        "spectral_centroid": librosa.feature.spectral_centroid(
            S=S, sr=sr
        ).flatten(),
        "spectral_bandwidth": spec_bw,
        "spectral_flatness": librosa.feature.spectral_flatness(S=S).flatten(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file")
    args = parser.parse_args()

    audio_file = Path(args.audio_file)
    audio_name = audio_file.stem
    script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    output_dir = script_dir / "output" / audio_name
    output_dir.mkdir(parents=True, exist_ok=True)

    features = analyze(audio_file)

    S_mean = np.mean(features["spectrogram"] * 2, axis=1)
    S_mean_db = librosa.power_to_db(S_mean, ref=np.max)
    freqs = librosa.fft_frequencies(sr=features["sample_rate"], n_fft=NFFT)
    S_db = librosa.amplitude_to_db(features["spectrogram"], ref=np.max)
    active = (S_db > -15).any(axis=1)

    stats = {
        "min_frequency": freqs[active].min(),
        "max_frequency": freqs[active].max(),
        "mean_spectral_centroid": features["spectral_centroid"].mean().item(),
        "mean_spectral_bandwidth": features["spectral_bandwidth"].mean().item(),
        "mean_spectral_flatness": features["spectral_flatness"].mean().item(),
    }
    with open(output_dir / f"stats_{audio_name}.json", "w") as fp:
        json.dump(stats, fp, indent=2)

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    freq_color = "sienna"
    f_fig, f_ax = plt.subplots(figsize=(10, 5))
    f_ax.plot(freqs, S_mean_db, linewidth=1, color=freq_color)
    f_ax.set_xlabel("Hz")
    f_ax.set_ylabel("dB")
    f_ax.set_xscale("log")
    f_ax.set_xlim(20, features["sample_rate"] / 2)
    f_ax.set_ylim(-25, 0)
    f_ax.fill_between(freqs, S_mean_db, S_mean_db.min(), color=freq_color)
    f_ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    f_fig.savefig(fig_dir / "frequency_content.png", bbox_inches="tight")

    s_fig, s_ax = plt.subplots(figsize=(10, 5))
    librosa.display.specshow(
        S_db, y_axis="log", x_axis="time", ax=s_ax, n_fft=NFFT, hop_length=HOP
    )
    s_ax.plot(
        features["times"],
        features["spectral_centroid"].T,
        label="Spectral centroid",
        color="white",
    )
    s_ax.fill_between(
        features["times"],
        np.maximum(
            0, features["spectral_centroid"] - features["spectral_bandwidth"]
        ),
        np.minimum(
            features["spectral_centroid"] + features["spectral_bandwidth"],
            features["sample_rate"] / 2,
        ),
        alpha=0.5,
        label="Bandwidth",
    )
    s_ax.legend(loc="lower right")
    s_fig.savefig(fig_dir / f"c_bw_{audio_name}.png", bbox_inches="tight")

    fl_fig, fl_ax = plt.subplots(figsize=(10, 5))
    fl_ax.plot(
        features["times"], features["spectral_flatness"], color="darkgreen"
    )
    fl_ax.set_ylabel("Flatness")
    fl_ax.set_xlabel("Time")
    fl_ax.xaxis.set_major_formatter(librosa.display.TimeFormatter())
    fl_fig.savefig(fig_dir / f"fl_{audio_name}.png", bbox_inches="tight")


if __name__ == "__main__":
    main()

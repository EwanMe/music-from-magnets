import argparse
import json
import os
from pathlib import Path
from typing import Any

import librosa
import matplotlib.pyplot as plt
import numpy as np


def analyze(file: Path) -> dict[str, Any]:
    y, sr = librosa.load(file)

    if y.ndim != 1:
        raise Exception(
            f"Loaded audio file had {y.ndim} channels, we only support mono"
        )

    S = np.abs(librosa.stft(y))
    spec_bw = librosa.feature.spectral_bandwidth(S=S, sr=sr).flatten()
    times = librosa.times_like(spec_bw)

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
    feature_means = {
        "mean_spectral_centroid": features["spectral_centroid"].mean().item(),
        "mean_spectral_bandwidth": features["spectral_bandwidth"].mean().item(),
        "mean_spectral_flatness": features["spectral_flatness"].mean().item(),
    }
    with open(output_dir / f"means_{audio_name}.json", "w") as fp:
        json.dump(feature_means, fp, indent=2)

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    librosa.display.specshow(
        librosa.amplitude_to_db(features["spectrogram"], ref=np.max),
        y_axis="log",
        x_axis="time",
        ax=ax,
    )
    ax.plot(
        features["times"],
        features["spectral_centroid"].T,
        label="Spectral centroid",
        color="white",
    )
    ax.fill_between(
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
    ax.legend(loc="lower right")
    fig.savefig(fig_dir / f"c_bw_{audio_name}.png", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        features["times"],
        features["spectral_flatness"],
    )
    ax.set_ylabel("Flatness")
    ax.set_xlabel("Time")
    ax.xaxis.set_major_formatter(librosa.display.TimeFormatter())
    fig.savefig(fig_dir / f"fl_{audio_name}.png", bbox_inches="tight")


if __name__ == "__main__":
    main()

import numpy as np
import numpy.typing as npt
import sounddevice as sd


class Wavetable:

    def __init__(self, sample_rate: int, num_waves: int, device: int) -> None:
        self.sample_rate = sample_rate
        self.num_waves = num_waves
        sd.default.device = device

        self.lowest_freq: float | None = None
        self.N: int = 100000  # magic
        self._buffer = np.zeros(self.N)
        self.waves = np.zeros((self.num_waves, self.N))

        slew_duration = 0.003
        self.slew_samples = int(slew_duration * self.sample_rate)
        self.hanning = np.hanning(2 * self.slew_samples)

    def __enter__(self) -> None:
        sd.play(self._buffer, samplerate=self.sample_rate, loop=True)

    def __exit__(self, exc_type, exc, tb) -> None:
        sd.stop()

    def _get_summed_wave(self) -> npt.NDArray[np.float64]:
        return np.sum(self.waves, axis=0) / len(self.waves)

    def add_wave(self, index: int, frequency: float, amplitude: float) -> None:
        if self.lowest_freq is None or frequency < self.lowest_freq:
            self.lowest_freq = frequency

        self.waves[index] = amplitude * np.sin(
            2 * np.pi * frequency * np.arange(self.N) / self.sample_rate
        )

        summed_wave = self._get_summed_wave()

        if self.slew_samples >= len(summed_wave) // 2:
            raise ValueError("Slew duration is longer than buffer.")

        # Apply window to start and end of wave to avoid artifacts
        summed_wave[: self.slew_samples] *= self.hanning[: self.slew_samples]
        summed_wave[-self.slew_samples :] *= self.hanning[self.slew_samples :]

        # Replace all samples in playback buffer
        for i, elem in enumerate(summed_wave):
            self._buffer[i] = elem

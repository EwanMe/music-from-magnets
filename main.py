import argparse
import enum
import threading
import time

from flatspin.model import PinwheelSpinIceDiamond, SquareSpinIceClosed, SpinIce
from flatspin.plotting import plot_vectors
import keyboard
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mido.backends.rtmidi
import mido
import numpy as np

import wavetable


def remap_range(
    # Doesn't handle value outside of range
    value: int | float,
    old_min: int | float,
    old_max: int | float,
    new_min: int | float,
    new_max: int | float,
) -> float:
    return new_min + (value - old_min) * (new_max - new_min) / (
        old_max - old_min
    )


class Instrument:
    class Arrow(enum.Enum):
        UP = 1
        DOWN = 2
        LEFT = 3
        RIGHT = 4

    def __init__(
        self,
        width: int,
        height: int,
        *,
        ice_type: str,
        input_type: str,
        animation: bool,
        use_opencl: bool,
        disorder: float,
    ):
        plt.rcParams["animation.frame_format"] = "svg"

        self.a = 0.0
        self.b = 0.0
        self.temperature = 0

        self.animation = animation
        self.lock = threading.Lock()
        self.updated = True

        asi_args = {
            "size": (width, height),
            "use_opencl": use_opencl,
            "disorder": disorder,
        }
        if ice_type == "pinwheel":
            self.model = PinwheelSpinIceDiamond(**asi_args)
        elif ice_type == "square":
            self.model = SquareSpinIceClosed(**asi_args)
        else:
            raise Exception("Invalid spin ice type: " + ice_type)

        if self.animation:
            self.model.set_h_ext([self.a, self.b])

            # Plot lattice
            self.fig, (self.vec_ax, self.norm_ax) = plt.subplots(1, 2)
            self.model.plot(ax=self.norm_ax)

            # Plot vectors
            self.grid = self.model.grid(cell_size=(1.5, 1.5))
            self.UV = self.grid.add_values(self.model.vectors, method="sum")
            x, y = self.grid.centers()
            XY = np.column_stack([x, y])
            plot_vectors(XY, self.UV, normalize=True, ax=self.vec_ax)

        if input_type == "joystick":
            devices = [
                device["name"]
                for device in mido.backends.rtmidi.get_devices()
                if device["is_input"] is True
            ]

            dev_sel = 0
            if len(devices) == 0:
                raise Exception("No MIDI input devices found.")
            if len(devices) > 1:
                dev_list = [f"({i}) {d}" for i, d in enumerate(devices)]
                dev_sel = int(
                    input(
                        "Multiple MIDI input devices found, please select one:\n"
                        + "\n".join(dev_list)
                        + "\n"
                    )
                )
            self.in_port: mido.ports.BaseInput = mido.open_input(
                devices[dev_sel], callback=self._handle_midi
            )
        elif input_type == "keyboard":
            keyboard.add_hotkey("up", self._on_press, [self.Arrow.UP])
            keyboard.add_hotkey("down", self._on_press, [self.Arrow.DOWN])
            keyboard.add_hotkey("left", self._on_press, [self.Arrow.LEFT])
            keyboard.add_hotkey("right", self._on_press, [self.Arrow.RIGHT])
            keyboard.add_hotkey("space", self._flip_mag)
        else:
            raise Exception("Invalid input type: " + input_type)

        devices = wavetable.get_available_output_devices()
        dev_sel = devices[0][0]
        if len(devices) == 0:
            raise Exception("No audio output devices found.")
        if len(devices) > 1:
            dev_name_list = [f"({i}) {device["name"]}" for i, device in devices]

            dev_sel = int(
                input(
                    "Multiple audio output devices found, please select one:\n"
                    + "\n".join(dev_name_list)
                    + "\n"
                )
            )
        self._audio_running = True
        self._audio_thread = threading.Thread(
            target=self._play_audio, args=(dev_sel,)
        )
        self._audio_thread.start()

    def _flip_mag(self):
        self.model.polarize()

    def _on_press(self, key: Arrow):
        match key:
            case self.Arrow.UP:
                self.a = 0.5
            case self.Arrow.DOWN:
                self.a = -0.5
            case self.Arrow.LEFT:
                self.b = -0.5
            case self.Arrow.RIGHT:
                self.b = 0.5

    def run(self):
        def _step():
            yield self.model
            while True:
                self.model.set_h_ext([self.a, self.b])
                self.model.set_temperature(self.temperature)
                with self.lock:
                    self.updated = True
                self.model.step()
                yield self.model

        def _animate(model: SpinIce):
            self.vec_ax.set_title(str(model.h_ext[0]))
            self.norm_ax.set_title(str(model.temperature))

            # Update lattice
            model.plot(ax=self.norm_ax, replace=True)

            # Update vectors
            self.UV = self.grid.add_values(self.model.vectors, method="sum")
            x, y = self.grid.centers()
            XY = np.column_stack([x, y])
            plot_vectors(
                XY, self.UV, normalize=True, ax=self.vec_ax, replace=True
            )

        if self.animation:
            anim = FuncAnimation(
                self.fig,
                _animate,
                frames=_step(),
                interval=1,
                blit=False,
                cache_frame_data=False,
            )
            plt.show()
            plt.close()
            self._audio_running = False
            self._audio_thread.join()
        else:
            while True:
                print(self.a, self.b)
                time.sleep(1)

    def _handle_midi(self, msg: mido.Message):
        h_ext_min = -0.1
        h_ext_max = 0.1
        midi_min = 0
        midi_max = 127
        temp_min = 0
        temp_max = 600

        value = remap_range(msg.value, midi_min, midi_max, h_ext_min, h_ext_max)
        if msg.is_cc(1):
            self.b = value
        elif msg.is_cc(2):
            self.a = value
        elif msg.is_cc(11):
            self.temperature = remap_range(
                msg.value, midi_min, midi_max, temp_min, temp_max
            )

    def _play_audio(self, output_device: int) -> None:
        def field_to_wave(x, y) -> tuple[float, float]:
            r = np.sqrt(np.pow(x, 2) + np.pow(y, 2))
            phi = np.atan2(x, y)
            f = remap_range(phi, -2 * np.pi, 2 * np.pi, 400, 1000)
            amp = remap_range(r, 0, 4, 0, 0.8)
            return f, amp

        def update_freqs():
            for i, (x, y) in enumerate(self.UV.reshape(-1, 2)):
                x = remap_range(x, 0, 4, -1, 1)
                y = remap_range(y, 0, 4, -1, 1)
                freq, amp = field_to_wave(x, y)
                # Bug: amp above 1
                wt.add_wave(i, freq, amp)

        wt = wavetable.Wavetable(
            44150, np.prod(self.grid.size), device=output_device
        )
        with wt:
            while self._audio_running:
                if self.updated:
                    with self.lock:
                        update_freqs()
                        self.updated = False
                time.sleep(0.01)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dimension", type=int, default=8)
    parser.add_argument(
        "-t", "--type", choices=["pinwheel", "square"], default="pinwheel"
    )
    parser.add_argument(
        "-i", "--input", choices=["joystick", "keyboard"], default="joystick"
    )
    parser.add_argument("-a", "--animation", action="store_true")
    parser.add_argument("--disorder", type=float, default=0.05)
    parser.add_argument("-cl", "--use-opencl", action="store_true")
    args = parser.parse_args()

    dimension: int = args.dimension
    ice_type: str = args.type
    input_type: str = args.input
    animation: bool = args.animation
    disorder: float = args.disorder
    use_opencl: bool = args.use_opencl

    Instrument(
        dimension,
        dimension,
        ice_type=ice_type,
        input_type=input_type,
        animation=animation,
        use_opencl=use_opencl,
        disorder=disorder,
    ).run()


if __name__ == "__main__":
    main()

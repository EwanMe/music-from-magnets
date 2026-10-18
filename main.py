import argparse
import enum
import time

from flatspin.model import PinwheelSpinIceDiamond, SquareSpinIceClosed, SpinIce
import keyboard
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mido.backends.rtmidi
import mido

import wavetable


def remap_range(
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
    ):
        plt.rcParams["animation.frame_format"] = "svg"

        self.a = 0.0
        self.b = 0.0
        self.animation = animation

        if ice_type == "pinwheel":
            self.model = PinwheelSpinIceDiamond(size=(width, height))
        elif ice_type == "square":
            self.model = SquareSpinIceClosed(size=(width, height))
        else:
            raise Exception("Invalid spin ice type: " + ice_type)

        if self.animation:
            self.fig, self.ax = plt.subplots()
            self.model.set_h_ext([self.a, self.b])
            self.model.plot()

        if input_type == "joystick":
            device_name = next(
                device["name"]
                for device in mido.backends.rtmidi.get_devices()
                if device["is_input"] is True
            )
            self.in_port: mido.ports.BaseInput = mido.open_input(
                device_name, callback=self._handle_midi
            )
        elif input_type == "keyboard":
            keyboard.add_hotkey("up", self._on_press, [self.Arrow.UP])
            keyboard.add_hotkey("down", self._on_press, [self.Arrow.DOWN])
            keyboard.add_hotkey("left", self._on_press, [self.Arrow.LEFT])
            keyboard.add_hotkey("right", self._on_press, [self.Arrow.RIGHT])
            keyboard.add_hotkey("space", self._flip_mag)
        else:
            raise Exception("Invalid input type: " + input_type)

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
                # Update ext field here
                self.model.set_h_ext([self.a, self.b])
                self.model.relax()
                yield self.model

        def _animate(model: SpinIce):
            self.ax.set_title(str(model.h_ext[0]))
            model.plot(ax=self.ax, replace=True)

        if self.animation:
            anim = FuncAnimation(
                self.fig,
                _animate,
                frames=_step(),
                interval=30,
                blit=False,
                cache_frame_data=False,
            )
            plt.show()
            plt.close()
        else:
            while True:
                print(self.a, self.b)
                time.sleep(1)

    def _handle_midi(self, msg: mido.Message):
        h_ext_min = -0.5
        h_ext_max = 0.5
        midi_min = 0
        midi_max = 127

        value = remap_range(msg.value, midi_min, midi_max, h_ext_min, h_ext_max)
        if msg.is_cc(1):
            self.b = value
        elif msg.is_cc(2):
            self.a = value


def main() -> None:

    wt = wavetable.Wavetable(44100)
    wt.add_wave(440, 1)

    with wt:
        time.sleep(1)
        wt.add_wave(2 * 440, 0.7)
        time.sleep(1)
    return

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dimension", type=int, default=8)
    parser.add_argument(
        "-t", "--type", choices=["pinwheel", "square"], default="pinwheel"
    )
    parser.add_argument(
        "-i", "--input", choices=["joystick", "keyboard"], default="joystick"
    )
    parser.add_argument("-a", "--animation", action="store_true", default=True)
    args = parser.parse_args()

    dimension: int = args.dimension
    ice_type: str = args.type
    input_type: str = args.input
    animation: bool = args.animation

    Instrument(
        dimension,
        dimension,
        ice_type=ice_type,
        input_type=input_type,
        animation=animation,
    ).run()


if __name__ == "__main__":
    main()

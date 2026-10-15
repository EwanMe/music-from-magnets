import argparse
import enum
import threading

from flatspin.model import PinwheelSpinIceDiamond, SquareSpinIceClosed, SpinIce
import keyboard
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mido.backends.rtmidi
import mido


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


class Input:
    class Arrow(enum.Enum):
        UP = 1
        DOWN = 2
        LEFT = 3
        RIGHT = 4

    def __init__(
        self, width: int, height: int, *, ice_type: str, input_type: str
    ):
        plt.rcParams["animation.frame_format"] = "svg"

        self.fig, self.ax = plt.subplots()

        if ice_type == "pinwheel":
            self.model = PinwheelSpinIceDiamond(size=(width, height))
        elif ice_type == "square":
            self.model = SquareSpinIceClosed(size=(width, height))
        else:
            raise Exception("Invalid spin ice type: " + ice_type)

        self.a = 0.0
        self.b = 0.0

        self.model.set_h_ext([self.a, self.b])
        self.model.plot()

        if input_type == "joystick":
            self.t = threading.Thread(target=self._handle_midi)
            self.run = True
            self.t.start()
        elif input_type == "keyboard":
            keyboard.add_hotkey("up", self._on_press, [self.Arrow.UP])
            keyboard.add_hotkey("down", self._on_press, [self.Arrow.DOWN])
            keyboard.add_hotkey("left", self._on_press, [self.Arrow.LEFT])
            keyboard.add_hotkey("right", self._on_press, [self.Arrow.RIGHT])
            keyboard.add_hotkey("space", self._flip_mag)
        else:
            raise Exception("Invalid input type: " + input_type)

    def __del__(self):
        self.t.join(timeout=1.0)

    def _flip_mag(self):
        self.model.polarize()

    def _on_press(self, key: Arrow):
        match key:
            case self.Arrow.UP:
                self.a = 1
            case self.Arrow.DOWN:
                self.a = -1
            case self.Arrow.LEFT:
                self.b = -1
            case self.Arrow.RIGHT:
                self.b = 1
        self.model.set_h_ext([self.a, self.b])
        print(self.a, self.b)

    def draw(self):
        anim = FuncAnimation(
            self.fig,
            self._animate,
            frames=self._step(),
            interval=30,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()
        plt.close()

        self.run = False

    def _step(self):
        yield self.model
        while True:
            # Update ext field here
            self.model.set_h_ext([self.a, self.b])
            self.model.relax()
            yield self.model

    def _animate(self, model: SpinIce):
        self.ax.set_title(str(model.h_ext[0]))
        model.plot(ax=self.ax, replace=True)

    def _handle_midi(self):
        device_name = next(
            device["name"]
            for device in mido.backends.rtmidi.get_devices()
            if device["is_input"] is True
        )
        in_port = mido.backends.rtmidi.Input(device_name)

        h_ext_max = 0.5

        while self.run:
            # x value never changed
            msg: mido.Message = in_port.receive()
            if msg.is_cc(1):
                value = remap_range(msg.value, 0, 127, -h_ext_max, h_ext_max)
                self.b = value
            elif msg.is_cc(2):
                value = remap_range(msg.value, 0, 127, -h_ext_max, h_ext_max)
                self.a = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dimension", type=int, default=8)
    parser.add_argument(
        "-t", "--type", choices=["pinwheel", "square"], default="pinwheel"
    )
    parser.add_argument(
        "-i", "--input", choices=["joystick", "keyboard"], default="joystick"
    )
    args = parser.parse_args()

    dimension: int = args.dimension
    ice_type: str = args.type
    input_type: str = args.input

    inp = Input(dimension, dimension, ice_type=ice_type, input_type=input_type)
    inp.draw()


if __name__ == "__main__":
    main()

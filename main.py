import argparse
import enum
import json
import socket
import threading
import time

from flatspin.model import PinwheelSpinIceDiamond, SquareSpinIceClosed, SpinIce
from flatspin.plotting import plot_vectors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mido
import numpy as np

import utils
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
        *,
        width: int,
        height: int,
        ice_type: str,
        animation: bool,
        use_opencl: bool,
        disorder: float,
        is_server: bool,
    ):
        plt.rcParams["animation.frame_format"] = "svg"

        self.a = 0.0
        self.b = 0.0
        self.temperature = 0

        self.animation = animation
        self.lock = threading.Lock()
        self.updated = True
        self.is_server = is_server

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

        self._running = True
        if self.is_server:
            srv = socket.create_server(("127.0.0.1", 5555))
            print("Waiting for connection")
            self.conn, _ = srv.accept()
            print("Connected")
            self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            self._midi_thread = threading.Thread(target=self._receive_midi)
            self._midi_thread.start()

            self._audio_thread = threading.Thread(target=self._send_audio_ctrl)
            self._audio_thread.start()
        else:
            self.in_port: mido.ports.BaseInput = mido.open_input(
                utils.user_select_midi_device(), callback=self._handle_midi
            )

            # User interaction for device selection must happen outside thread
            self._audio_thread = threading.Thread(
                target=self._play_audio,
                args=(utils.user_select_sound_device(),),
            )
            self._audio_thread.start()

    def run(self):
        def _step():
            yield self.model
            while True:
                self.model.set_h_ext([self.a, self.b])
                self.model.set_temperature(self.temperature)
                with self.lock:
                    self.updated = True
                self.model.relax()
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
            self._running = False
            self._audio_thread.join()
            if self.is_server:
                self._midi_thread.join()
        else:
            while True:
                print(self.a, self.b)
                time.sleep(1)

    def _receive_midi(self):
        sock_file = self.conn.makefile()

        while self._running:
            # This line blocks and prevents thread join
            raw = json.loads(sock_file.readline())
            msg = mido.Message(
                type=raw["type"],
                control=raw["control"],
                value=raw["value"],
                channel=raw["channel"],
            )
            self._handle_midi(msg)

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

    def _send_audio_ctrl(self) -> None:
        while self._running:
            if self.updated:
                with self.lock:
                    self.conn.sendall(
                        (json.dumps(self._get_freqs()) + "\n").encode()
                    )

    def _get_freqs(self) -> list[tuple[int, float, float]]:
        def field_to_wave(x, y) -> tuple[float, float]:
            r = np.sqrt(np.pow(x, 2) + np.pow(y, 2))
            phi = np.atan2(x, y)
            f = remap_range(phi, -2 * np.pi, 2 * np.pi, 400, 1000)
            amp = remap_range(r, 0, 4, 0, 0.8)
            return f, amp

        freqs: list[tuple[int, float, float]] = []
        for i, (x, y) in enumerate(self.UV.reshape(-1, 2)):
            x = remap_range(x, 0, 4, -1, 1)
            y = remap_range(y, 0, 4, -1, 1)
            freq, amp = field_to_wave(x, y)
            freqs.append((i, freq, amp))
        return freqs

    def _play_audio(self, output_device: int) -> None:
        wt = wavetable.Wavetable(
            44150, np.prod(self.grid.size), device=output_device
        )
        with wt:
            while self._running:
                if self.updated:
                    with self.lock:
                        freqs = self._get_freqs()
                        for i, freq, amp in freqs:
                            wt.add_wave(i, freq, amp)
                        self.updated = False
                time.sleep(0.01)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dimension", type=int, default=8)
    parser.add_argument(
        "-t", "--type", choices=["pinwheel", "square"], default="pinwheel"
    )
    parser.add_argument("-a", "--animation", action="store_true")
    parser.add_argument("--disorder", type=float, default=0.05)
    parser.add_argument("-cl", "--use-opencl", action="store_true")
    parser.add_argument("-s", "--server", action="store_true")
    args = parser.parse_args()

    dimension: int = args.dimension
    ice_type: str = args.type
    animation: bool = args.animation
    disorder: float = args.disorder
    use_opencl: bool = args.use_opencl
    is_server: bool = args.server

    Instrument(
        width=dimension,
        height=dimension,
        ice_type=ice_type,
        animation=animation,
        use_opencl=use_opencl,
        disorder=disorder,
        is_server=is_server,
    ).run()


if __name__ == "__main__":
    main()

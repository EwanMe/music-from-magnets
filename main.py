import argparse
import enum
import json
import random
import socket
import threading
import time

from flatspin.model import PinwheelSpinIceDiamond, SquareSpinIceClosed, SpinIce
from flatspin.plotting import plot_vectors
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mido
import numpy as np

matplotlib.use("TkAgg")


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


def cart_to_polar(x, y) -> tuple[float, float]:
    r = np.sqrt(np.pow(x, 2) + np.pow(y, 2))
    phi = np.atan2(y, x)
    return r, phi


class Instrument:

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

        self.x = 0.0
        self.y = 0.0
        self.noise = 0

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

        self.model.set_h_ext([self.x, self.y])

        # Plot lattice
        self.fig, (self.vec_ax, self.norm_ax) = plt.subplots(1, 2)

        # Plot vectors
        self.grid = self.model.grid(cell_size=(1.5, 1.5))
        self.UV = self.grid.add_values(self.model.vectors, method="sum")
        x, y = self.grid.centers()
        XY = np.column_stack([x, y])

        if self.animation:
            self.model.plot(ax=self.norm_ax)
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
            import utils

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
        num_spins = self.model.N

        def noisify(x: float) -> float:
            return x + random.random() * self.noise

        def _step():
            yield self.model
            while True:
                v = [
                    [noisify(self.x), noisify(self.y)] for _ in range(num_spins)
                ]
                self.model.set_h_ext(v)
                self.model.set_temperature(self.noise)
                with self.lock:
                    self.updated = True
                self.model.relax()
                yield self.model

        def _animate(model: SpinIce):
            r, phi = cart_to_polar(*(model.h_ext[0]))
            self.vec_ax.set_title(f"r={r:.3f}, phi={phi:.3f}")
            self.norm_ax.set_title(f"noise={self.noise:.5f}")

            # Update lattice
            model.plot(ax=self.norm_ax, replace=True)

            # Update vectors
            self.UV = self.grid.add_values(self.model.vectors, method="sum")
            x, y = self.grid.centers()
            XY = np.column_stack([x, y])
            plot_vectors(
                XY, self.UV, normalize=True, ax=self.vec_ax, replace=True
            )

        anim = FuncAnimation(
            self.fig,
            _animate,
            frames=_step(),
            interval=1,
            blit=False,
            cache_frame_data=False,
        )
        if self.animation:
            plt.show()
            plt.close()

        self._running = False
        if self.is_server:
            self.conn.shutdown(socket.SHUT_RDWR)
            self._midi_thread.join()
        self._audio_thread.join()

    def _receive_midi(self):
        # Set timeout to handle thread teardown
        # self.conn.settimeout(2)

        while self._running:
            # This line blocks and prevents thread join
            try:
                sock_file = self.conn.makefile()
                raw = json.loads(sock_file.readline())
                msg = mido.Message(
                    type=raw["type"],
                    control=raw["control"],
                    value=raw["value"],
                    channel=raw["channel"],
                )
                self._handle_midi(msg)
            except socket.timeout:
                continue

    def _handle_midi(self, msg: mido.Message):
        h_ext_min = -0.08
        h_ext_max = 0.08
        midi_min = 0
        midi_max = 127
        noise_min = 0
        noise_max = np.abs(h_ext_max)

        value = remap_range(msg.value, midi_min, midi_max, h_ext_min, h_ext_max)
        if msg.is_cc(1):
            self.y = value
        elif msg.is_cc(2):
            self.x = value
        elif msg.is_cc(11):
            self.noise = remap_range(
                msg.value, midi_min, midi_max, noise_min, noise_max
            )

    def _send_audio_ctrl(self) -> None:
        while self._running:
            if self.updated:
                with self.lock:
                    self.conn.sendall(
                        (json.dumps(self._get_freqs()) + "\n").encode()
                    )

    def _get_freqs(self) -> list[tuple[int, float, float]]:
        def field_to_wave(i: int, x: float, y: float) -> tuple[float, float]:
            r, phi = cart_to_polar(x, y)

            freq = (
                scale[int(remap_range(phi, -np.pi, np.pi, 0, scale_len))]
                * (i + 1)
                * 0.5
            )
            amp = remap_range(r, 0, 4, 0, 0.8)
            return freq, amp

        c_maj = [
            261.6256,  # C
            293.6648,  # D
            329.6276,  # E
            349.2282,  # F
            391.9954,  # G
            440.0000,  # A
            493.8833,  # B
        ]
        c_maj_by_fifths = [
            261.6256,  # C
            391.9954,  # G
            293.6648,  # D
            440.0000,  # A
            329.6276,  # E
            493.8833,  # B
            349.2282,  # F
        ]
        c_min_by_fifths = [
            261.6256,  # C
            391.9954,  # G
            293.6648,  # D
            415.3047,  # Ab
            311.1270,  # Eb
            466.1638,  # Bb
            349.2282,  # F
        ]
        c_doric_by_fifths = [
            261.6256,  # C
            391.9954,  # G
            293.6648,  # D
            415.3047,  # Ab
            311.1270,  # Eb
            493.8833,  # B
            349.2282,  # F
        ]
        # Circle of fifths, only major key ring tonics
        fifths = [
            261.6256,  # C
            391.9954,  # G
            293.6648,  # D
            440.0000,  # A
            329.6276,  # E
            493.8833,  # B
            369.9944,  # F♯
            277.1826,  # C♯
        ]
        # Circle of fourths, only major key ring tonics
        fourths = [
            261.6256,  # C
            349.2282,  # F
            466.1638,  # Bb
            311.1270,  # Eb
            415.3047,  # Ab
            277.1826,  # Db
            369.9944,  # Gb
        ]

        scale = c_doric_by_fifths
        scale_len = len(scale)

        r, _ = cart_to_polar(*(self.model.h_ext[0]))

        freqs: list[tuple[int, float, float]] = []
        for i, (x, y) in enumerate(self.UV.reshape(-1, 2)):
            x = remap_range(x, 0, 4, -1, 1)
            y = remap_range(y, 0, 4, -1, 1)
            freq, amp = field_to_wave(i, x, y)
            freqs.append((i, freq, amp))
        return freqs

    def _play_audio(self, output_device: int) -> None:
        import wavetable

        wt = wavetable.Wavetable(
            44100, np.prod(self.grid.size), device=output_device
        )
        with wt:
            while self._running:
                if self.updated:
                    with self.lock:
                        wt.add_waves(self._get_freqs())
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

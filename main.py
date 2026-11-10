import configargparse
import json
import os
from pathlib import Path
import random
import socket
import threading
import time
import sys

from flatspin.model import PinwheelSpinIceDiamond, SquareSpinIceClosed, SpinIce
from flatspin.plotting import plot_vectors
import flatspin.plotting
import matplotlib
import matplotlib.axes
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
        use_opencl: bool,
        disorder: float,
        is_server: bool,
        use_default_devices: bool,
        scale: dict[str, float],
        plot_scale: bool,
        plots: str,
    ) -> None:
        plt.rcParams["animation.frame_format"] = "svg"

        self.x = 0.0
        self.y = 0.0
        self.noise = 0

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

        self.scale_len = len(scale)
        self.name_scale = list(scale.keys())
        self.freq_scale = list(scale.values())
        self.plots = plots
        self.plot_scale = plot_scale

        self.model.set_h_ext([self.x, self.y])

        if self.plots == "vector":
            self.fig, self.vec_ax = plt.subplots()
        elif self.plots == "both":
            self.fig, (self.vec_ax, self.norm_ax) = plt.subplots(1, 2)
        else:
            raise Exception("Unknown plot type: " + self.plots)

        # Plot vectors
        self.grid = self.model.grid(cell_size=(1.5, 1.5))
        self.UV = self.grid.add_values(self.model.vectors, method="sum")
        x, y = self.grid.centers()
        XY = np.column_stack([x, y])

        if self.plots == "both":
            # Plot lattice
            self.model.plot(ax=self.norm_ax)
            plot_vectors(XY, self.UV, normalize=True, ax=self.vec_ax)

        if self.plot_scale:
            start_deg = -180
            centering = 0.5
            theta = np.deg2rad(
                start_deg
                + (np.arange(self.scale_len) + centering) * 360 / self.scale_len
            )
            colors = flatspin.plotting.vector_rgba(np.cos(theta), np.sin(theta))

            xmin, xmax = self.vec_ax.get_xlim()
            ymin, ymax = self.vec_ax.get_ylim()
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            radius = min(xmax - xmin, ymax - ymin) / 2

            self.vec_ax.pie(
                [1] * self.scale_len,
                center=(center_x, center_y),
                startangle=start_deg,
                colors=colors,
                counterclock=True,
                radius=radius,
                wedgeprops=dict(width=0.3 * radius),
                labels=scale.keys(),
                labeldistance=0.85,
                textprops={"fontsize": 10},
            )

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

            midi_dev = None
            audio_dev = None
            if not use_default_devices:
                midi_dev = utils.user_select_midi_device()
                audio_dev = utils.user_select_sound_device()

            self.in_port: mido.ports.BaseInput = mido.open_input(
                midi_dev, callback=self._handle_midi
            )

            # User interaction for device selection must happen outside thread
            self._audio_thread = threading.Thread(
                target=self._play_audio, args=(audio_dev,)
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
            r, phi = cart_to_polar(self.x, self.y)

            if self.plots == "both":
                self.norm_ax.set_title(f"noise={self.noise:.5f}")

                # Update lattice
                model.plot(ax=self.norm_ax, replace=True)

            if not self.plot_scale:
                self.vec_ax.set_title(f"r={r:.3f}, phi={phi:.3f}")

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
        h_ext_max = 0.08
        midi_min = 0
        midi_max = 127
        noise_min = 0
        noise_max = np.abs(h_ext_max)

        value = remap_range(
            msg.value, midi_min, midi_max, -h_ext_max, h_ext_max
        )
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

            # Map phi from radians to an index in the scale notes array
            idx = int(
                np.floor(remap_range(phi, -np.pi, np.pi, 0, self.scale_len))
                % self.scale_len
            )
            freq = self.freq_scale[idx]
            amp = remap_range(r, 0, 4, 0, 0.8)
            return freq, amp

        freqs: list[tuple[int, float, float]] = []
        for i, (x, y) in enumerate(self.UV.reshape(-1, 2)):
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
    parser = configargparse.ArgParser()
    parser.add_argument("-c", "--config-file", is_config_file=True)
    parser.add_argument("-d", "--dimension", type=int, default=8)
    parser.add_argument(
        "-t", "--type", choices=["pinwheel", "square"], default="pinwheel"
    )
    parser.add_argument("--disorder", type=float, default=0.05)
    parser.add_argument("-cl", "--use-opencl", action="store_true")
    parser.add_argument("-s", "--server", action="store_true")
    parser.add_argument("-dd", "--default-devices", action="store_true")
    parser.add_argument("--plot-scale", action="store_true")
    parser.add_argument("--plots", choices=["vector", "both"], default="both")
    parser.add_argument("--scale", default="c_maj_by_fifths")
    args = parser.parse_args()

    dimension: int = args.dimension
    ice_type: str = args.type
    disorder: float = args.disorder
    use_opencl: bool = args.use_opencl
    is_server: bool = args.server
    use_default_devices: bool = args.default_devices
    plot_scale: bool = args.plot_scale
    plots: str = args.plots
    scale_name: str = args.scale

    scales_file = (
        Path(os.path.dirname(os.path.realpath(__file__))) / "scales.json"
    )
    with open(scales_file, "r", encoding="utf-8") as fp:
        if sys.version_info < (3, 7):
            # Python <3.7 does not decode key-values ordered by default
            from collections import OrderedDict

            scales = json.load(fp, object_pairs_hook=OrderedDict)
        else:
            scales = json.load(fp)

    Instrument(
        width=dimension,
        height=dimension,
        ice_type=ice_type,
        use_opencl=use_opencl,
        disorder=disorder,
        is_server=is_server,
        use_default_devices=use_default_devices,
        scale=scales[scale_name],
        plot_scale=plot_scale,
        plots=plots,
    ).run()


if __name__ == "__main__":
    main()

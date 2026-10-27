from typing import Any

import mido.backends.rtmidi
import sounddevice as sd


def user_select_midi_device():
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
    return devices[dev_sel]


def get_available_output_devices() -> list[tuple[int, Any]]:
    return [
        (i, device)
        for i, device in enumerate(sd.query_devices())
        if device["max_output_channels"] > 0
    ]


def user_select_sound_device():
    devices = get_available_output_devices()
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

    return dev_sel

import json
import socket
import threading
import time

import mido

import utils
import wavetable

s = socket.create_connection(("127.0.0.1", 5555))
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

lock = threading.Lock()
updated = False
packet = (0, 0, 0)


def send_midi():
    with mido.open_input(utils.user_select_midi_device()) as port:
        for msg in port:
            s.sendall((json.dumps(msg.dict()) + "\n").encode())


def receive_updates():
    global lock
    global updated
    global packet

    sock_file = s.makefile()
    while True:
        with lock:
            packet = json.loads(sock_file.readline())
            updated = True


threading.Thread(target=send_midi, daemon=True).start()
threading.Thread(target=receive_updates, daemon=True).start()

size = 16 * 16
wt = wavetable.Wavetable(44100, size, device=utils.user_select_sound_device())
with wt:
    while True:
        if updated:
            with lock:
                for i, freq, amp in packet:
                    wt.add_wave(i, freq, amp)
                updated = False
        time.sleep(0.01)

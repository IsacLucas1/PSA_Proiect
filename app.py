import os
import signal
import shutil
import time
import threading
import numpy as np
import sounddevice as sd
from scipy.signal import lfilter
from scipy.io.wavfile import write as write_wav
from pydub import AudioSegment
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import tkinter as tk
from tkinter import filedialog

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

SAMPLE_RATE = 44100
BLOCK_SIZE = 2048
CHANNELS = 2
UPLOAD_FOLDER = 'uploads'

if os.path.exists(UPLOAD_FOLDER): shutil.rmtree(UPLOAD_FOLDER)
os.makedirs(UPLOAD_FOLDER)

# --- GLOBAL ---
tracks = {}
track_id_counter = 0
is_master_playing = False
master_volume = 1.0
connected_clients = 0
current_viz_data = [0] * 32
is_recording = False
recording_buffer = []


# --- DSP MATH ---
def make_shelf(w0, gain_db, type_shelf):
    A = 10.0 ** (gain_db / 40.0)
    sin_w0 = np.sin(w0)
    cos_w0 = np.cos(w0)
    alpha = sin_w0 / 2.0 * np.sqrt(2.0)
    if type_shelf == 'low':
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
    else:
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
    return np.array([b0, b1, b2]) / a0, np.array([a0, a1, a2]) / a0


class AudioTrack:
    def __init__(self, filepath, name):
        self.filename_display = name
        self.playing = False
        self.volume = 1.0
        self.pan = 0.5
        self.position = 0

        # Parametri Noi
        self.speed = 1.0  # 1.0 = Normal, 0.5 = Slow, 2.0 = Fast
        self.bass_gain = 0.0
        self.treble_gain = 0.0
        self.reverb_amount = 0.0

        self.zi_bass = np.zeros((2, 2), dtype=np.float32)
        self.zi_treble = np.zeros((2, 2), dtype=np.float32)

        # Reverb Buffer
        self.delay_len = int(SAMPLE_RATE * 0.4)
        self.delay_buffer = np.zeros((self.delay_len, CHANNELS), dtype=np.float32)
        self.delay_ptr = 0

        try:
            seg = AudioSegment.from_file(filepath)
            seg = seg.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS)
            y = np.array(seg.get_array_of_samples(), dtype=np.float32)
            if seg.channels == 2:
                y = y.reshape((-1, 2))
            else:
                y = y.reshape((-1, 1))
                y = np.column_stack((y, y))
            self.data = y / 32768.0
            self.loaded = True
        except Exception as e:
            print(f"Eroare: {e}")
            self.data = np.zeros((100, 2), dtype=np.float32)
            self.loaded = False

    def get_chunk(self, output_size):
        if not self.playing or not self.loaded:
            return np.zeros((output_size, CHANNELS), dtype=np.float32)

        # --- LOGICA DE SPEED / RESAMPLING ---
        # Calculăm câte sample-uri "reale" trebuie să citim din fișier
        # pentru a umple bufferul de output, ținând cont de viteză.
        # Ex: Dacă viteza e 2.0, citim dublu (4096) și le înghesuim în (2048).

        needed_source_frames = int(output_size * self.speed)

        start = self.position
        end = start + needed_source_frames

        # Extragem bucata din sursă (cu Loop logic)
        if end > len(self.data):
            first_part = self.data[start:]
            remain = needed_source_frames - len(first_part)

            # Caz extrem: loop-ul e mai mic decât bufferul
            if remain > len(self.data):
                # Umplem cu zerouri pt simplitate (sau am putea repeta de mai multe ori)
                second_part = np.zeros((remain, CHANNELS), dtype=np.float32)
                if len(self.data) > 0:
                    second_part[:len(self.data)] = self.data[:len(second_part)]
            else:
                second_part = self.data[:remain]

            raw_chunk = np.vstack((first_part, second_part))
            self.position = remain
        else:
            raw_chunk = self.data[start:end].copy()
            self.position = end

        # --- RE-EȘANTIONARE (INTERPOLARE) ---
        # Dacă viteza nu e 1.0, trebuie să redimensionăm raw_chunk la output_size
        if abs(self.speed - 1.0) > 0.01:
            # Creăm axa timpului pentru input și output
            x_old = np.linspace(0, 1, len(raw_chunk))
            x_new = np.linspace(0, 1, output_size)

            # Interpolăm canalul Stânga
            resampled_l = np.interp(x_new, x_old, raw_chunk[:, 0])
            # Interpolăm canalul Dreapta
            resampled_r = np.interp(x_new, x_old, raw_chunk[:, 1])

            chunk = np.column_stack((resampled_l, resampled_r)).astype(np.float32)
        else:
            chunk = raw_chunk

        return self.apply_dsp(chunk)

    def apply_dsp(self, chunk):
        # 1. REVERB (Delay Network)
        if self.reverb_amount > 0.05:
            n = len(chunk)
            # Logica de buffer circular simplificată
            delayed_sig = np.zeros_like(chunk)

            # Verificăm dacă citirea depășește bufferul
            if self.delay_ptr + n <= self.delay_len:
                delayed_sig = self.delay_buffer[self.delay_ptr: self.delay_ptr + n]
                # Feedback: scriem înapoi (input + 60% din ecou)
                feedback = chunk + (delayed_sig * 0.6)
                self.delay_buffer[self.delay_ptr: self.delay_ptr + n] = feedback
                self.delay_ptr += n
            else:
                # Wrap around
                part1 = self.delay_len - self.delay_ptr
                part2 = n - part1
                delayed_sig[:part1] = self.delay_buffer[self.delay_ptr:]
                delayed_sig[part1:] = self.delay_buffer[:part2]

                feedback = chunk + (delayed_sig * 0.6)
                self.delay_buffer[self.delay_ptr:] = feedback[:part1]
                self.delay_buffer[:part2] = feedback[part1:]
                self.delay_ptr = part2

            if self.delay_ptr >= self.delay_len: self.delay_ptr = 0

            # Mixăm semnalul
            chunk = chunk + (delayed_sig * self.reverb_amount)

        # 2. BASS
        if abs(self.bass_gain) > 0.1:
            w0 = 2 * np.pi * 200 / SAMPLE_RATE
            b, a = make_shelf(w0, self.bass_gain, 'low')
            for ch in range(CHANNELS):
                chunk[:, ch], self.zi_bass[:, ch] = lfilter(b, a, chunk[:, ch], zi=self.zi_bass[:, ch])

        # 3. TREBLE
        if abs(self.treble_gain) > 0.1:
            w0 = 2 * np.pi * 3000 / SAMPLE_RATE
            b, a = make_shelf(w0, self.treble_gain, 'high')
            for ch in range(CHANNELS):
                chunk[:, ch], self.zi_treble[:, ch] = lfilter(b, a, chunk[:, ch], zi=self.zi_treble[:, ch])

        # 4. VOLUM & PAN
        processed = chunk * self.volume
        left_gain = (1.0 - self.pan) * 2
        right_gain = self.pan * 2
        processed[:, 0] *= min(left_gain, 1.0)
        processed[:, 1] *= min(right_gain, 1.0)
        return processed


# --- FFT ---
def compute_fft(audio_chunk):
    mono = (audio_chunk[:, 0] + audio_chunk[:, 1]) / 2
    windowed = mono * np.hanning(len(mono))
    spectrum = np.fft.rfft(windowed)
    magnitude = np.abs(spectrum)
    n_bins = 32
    if len(magnitude) < n_bins: return [0] * 32
    step = len(magnitude) // n_bins
    bars = []
    for i in range(n_bins):
        segment = magnitude[i * step: (i + 1) * step]
        if len(segment) > 0:
            bars.append(float(np.mean(segment)))
        else:
            bars.append(0.0)
    return bars


def audio_callback(outdata, frames, time_info, status):
    global current_viz_data, is_master_playing, recording_buffer
    if status: print(status)
    mix_buffer = np.zeros((frames, CHANNELS), dtype=np.float32)
    active = [t for t in tracks.values() if t.playing]
    if active and is_master_playing:
        for t in active:
            mix_buffer += t.get_chunk(frames)
        mix_buffer *= master_volume
        np.clip(mix_buffer, -1.0, 1.0, outdata)
        if is_recording: recording_buffer.append(outdata.copy())
        raw_bars = compute_fft(mix_buffer)
        current_viz_data = [min(x * 0.2, 1.0) for x in raw_bars]
    else:
        outdata[:] = 0
        current_viz_data = [0] * 32
        if is_recording: recording_buffer.append(np.zeros((frames, CHANNELS), dtype=np.float32))


def background_viz():
    while True:
        try:
            if is_master_playing:
                socketio.emit('viz_update', {'data': current_viz_data})
            else:
                socketio.emit('viz_update', {'data': [0] * 32})
            socketio.sleep(0.04)
        except:
            break


stream = sd.OutputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=CHANNELS, callback=audio_callback)
stream.start()
socketio.start_background_task(background_viz)


@app.route('/')
def index(): return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    global track_id_counter
    if 'file' not in request.files: return jsonify({'error': 'no file'}), 400
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, f"{track_id_counter}_{filename}")
    file.save(filepath)
    new_track = AudioTrack(filepath, filename)
    if new_track.loaded:
        curr = track_id_counter
        tracks[curr] = new_track
        socketio.emit('new_track_added', {'id': curr, 'name': filename})
        track_id_counter += 1
        return jsonify({'id': curr, 'name': filename})
    return jsonify({'error': 'bad file'}), 500


@socketio.on('control_update')
def handle_control(data):
    if 'master_volume' in data:
        global master_volume
        master_volume = float(data['master_volume'])
        socketio.emit('sync_control', data, include_self=False)
        return

    t_id = int(data['id'])
    if t_id in tracks:
        track = tracks[t_id]
        if 'volume' in data: track.volume = float(data['volume'])
        if 'pan' in data: track.pan = float(data['pan'])
        if 'bass' in data: track.bass_gain = float(data['bass'])
        if 'treble' in data: track.treble_gain = float(data['treble'])
        if 'reverb' in data: track.reverb_amount = float(data['reverb'])
        if 'speed' in data: track.speed = float(data['speed'])  # NOU
        if 'playing' in data:
            track.playing = data['playing']
            if track.playing:
                global is_master_playing
                is_master_playing = True

        socketio.emit('sync_control', data, include_self=False)


@socketio.on('reset_track')
def handle_reset(data):
    t_id = int(data['id'])
    if t_id in tracks:
        track = tracks[t_id]
        track.volume = 1.0
        track.pan = 0.5
        track.bass_gain = 0.0
        track.treble_gain = 0.0
        track.reverb_amount = 0.0
        track.speed = 1.0

        track.zi_bass = np.zeros((2, 2), dtype=np.float32)
        track.zi_treble = np.zeros((2, 2), dtype=np.float32)
        track.delay_buffer = np.zeros((track.delay_len, CHANNELS), dtype=np.float32)

        socketio.emit('sync_reset_track', {'id': t_id})


@socketio.on('remove_track')
def handle_remove(data):
    t_id = int(data['id'])
    if t_id in tracks:
        del tracks[t_id]
        socketio.emit('sync_remove_track', {'id': t_id})


@socketio.on('remove_all_tracks')
def handle_remove_all():
    global tracks
    tracks.clear()
    socketio.emit('sync_remove_all')


@socketio.on('reset_all_effects')
def handle_reset_all():
    for track in tracks.values():
        track.volume = 1.0
        track.pan = 0.5
        track.bass_gain = 0.0
        track.treble_gain = 0.0
        track.reverb_amount = 0.0
        track.speed = 1.0
    socketio.emit('sync_reset_all')


def save_file_dialog_thread(audio_data):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("Wave files", "*.wav")],
                                             title="Salvează Mixajul")
    root.destroy()
    if file_path:
        try:
            int16_audio = (audio_data * 32767).astype(np.int16)
            write_wav(file_path, SAMPLE_RATE, int16_audio)
            socketio.emit('rec_status', {'status': 'saved', 'filename': os.path.basename(file_path)})
        except:
            socketio.emit('rec_status', {'status': 'error'})
    else:
        socketio.emit('rec_status', {'status': 'cancelled'})


@socketio.on('toggle_recording')
def handle_record(data):
    global is_recording, recording_buffer
    should_record = data['record']
    if should_record:
        recording_buffer = []
        is_recording = True
        socketio.emit('rec_status', {'status': 'recording'})
    else:
        is_recording = False
        socketio.emit('rec_status', {'status': 'saving'})
        if len(recording_buffer) > 0:
            full_audio = np.concatenate(recording_buffer, axis=0)
            threading.Thread(target=save_file_dialog_thread, args=(full_audio,)).start()
        else:
            socketio.emit('rec_status', {'status': 'empty'})


def check_for_shutdown():
    time.sleep(2.0)
    if connected_clients <= 0:
        stream.abort()
        os.kill(os.getpid(), signal.SIGINT)


@socketio.on('connect')
def handle_connect():
    global connected_clients
    connected_clients += 1
    for t_id, track in tracks.items():
        emit('new_track_added', {'id': t_id, 'name': track.filename_display})
        emit('sync_control', {
            'id': t_id,
            'volume': track.volume,
            'pan': track.pan,
            'bass': track.bass_gain,
            'treble': track.treble_gain,
            'reverb': track.reverb_amount,
            'speed': track.speed,
            'playing': track.playing
        })


@socketio.on('disconnect')
def handle_disconnect():
    global connected_clients
    connected_clients -= 1
    threading.Thread(target=check_for_shutdown).start()


if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)
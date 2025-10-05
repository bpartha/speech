import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.fft import fft
import time

# --- CONFIG ---
fs = 44100  # sampling rate
duration = 3  # seconds
chunk_size = 2048  # samples per update frame

# --- RECORD OR LOAD AUDIO ---
print("Recording 3 seconds of audio... 🎤")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()
print("Recording complete!")

# Flatten the array
audio = audio.flatten()

# --- SETUP PLOTS ---
plt.ion()  # interactive mode
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

# Time-domain plot
time_axis = np.linspace(0, chunk_size/fs, chunk_size)
line1, = ax1.plot(time_axis, np.zeros(chunk_size))
ax1.set_title("Time Domain (Waveform)")
ax1.set_ylim(-1, 1)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")

# Frequency-domain plot
freq_axis = np.linspace(0, fs/2, chunk_size//2)
line2, = ax2.plot(freq_axis, np.zeros(chunk_size//2))
ax2.set_title("Frequency Domain (Live Spectrum)")
ax2.set_xlim(0, 2000)  # up to 2 kHz
ax2.set_ylim(0, 50)
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Magnitude")

plt.tight_layout()

# --- ANIMATE ---
for i in range(0, len(audio)-chunk_size, chunk_size):
    chunk = audio[i:i+chunk_size]

    # Update waveform
    line1.set_ydata(chunk)

    # FFT for this chunk
    fft_data = np.abs(fft(chunk))
    line2.set_ydata(fft_data[:chunk_size//2])

    plt.pause(0.01)

plt.ioff()
plt.show()
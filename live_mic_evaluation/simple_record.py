#!/usr/bin/env python3
"""
Extremely simple audio recording script. 
Records audio and saves it without any processing or playback.
"""
import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np
import time
import os
from datetime import datetime

# Parameters
DURATION = 5  # seconds
SAMPLE_RATE = 16000  # Hz
DEVICE_ID = 1  # MacBook Pro Microphone

print("Simple Audio Recorder")
print(f"Recording {DURATION} seconds from device {DEVICE_ID}")
print("Starting in 3 seconds...")

# Simple countdown
for i in range(3, 0, -1):
    print(f"{i}...")
    time.sleep(1)

print("Recording NOW! Speak clearly...")

# Record audio
audio_data = sd.rec(
    int(DURATION * SAMPLE_RATE), 
    samplerate=SAMPLE_RATE, 
    channels=1,
    dtype='float32',
    device=DEVICE_ID
)

# Show progress during recording
for i in range(DURATION):
    print(f"Recording: {i+1}/{DURATION} seconds")
    time.sleep(1)

# Wait for recording to complete
sd.wait()
print("Recording finished!")

# Create output folder
output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(output_folder, exist_ok=True)

# Create filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(output_folder, f"recording_{timestamp}.wav")

# Convert to int16 format for WAV file
audio_int16 = np.int16(audio_data * 32767)

# Save to file
wavfile.write(output_file, SAMPLE_RATE, audio_int16)
print(f"Audio saved to: {output_file}")

# Print audio stats
max_amplitude = np.max(np.abs(audio_data))
rms_level = np.sqrt(np.mean(audio_data**2))

print("\nAudio Statistics:")
print(f"Peak amplitude: {max_amplitude:.4f}")
print(f"RMS level: {rms_level:.4f}")

if rms_level < 0.01:
    print("WARNING: Very low audio level detected. Check your microphone.")
elif max_amplitude > 0.9:
    print("WARNING: Audio may be clipping. Try reducing microphone volume.")
else:
    print("Audio levels look good!")

print("\nDone.")
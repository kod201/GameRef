#!/bin/bash
# Script to check system status and run basic audio recording

echo "===== SYSTEM CHECK REPORT ====="
echo "Current directory: $(pwd)"
echo "Date: $(date)"
echo "Python version: $(python --version 2>&1)"
echo 

echo "===== DIRECTORY STRUCTURE ====="
echo "Checking for model file..."
if [ -f "models/ggml-tiny.en.bin" ]; then
  echo "✅ Model file found: models/ggml-tiny.en.bin ($(du -h models/ggml-tiny.en.bin | cut -f1))"
else
  echo "❌ Model file not found at models/ggml-tiny.en.bin"
  echo "Searching for model file elsewhere..."
  find . -name "ggml-tiny.en.bin" -type f
fi
echo

echo "Checking for Whisper.cpp executable..."
if [ -f "build/whisper.cpp/main" ]; then
  echo "✅ Whisper.cpp executable found: build/whisper.cpp/main"
else
  echo "❌ Whisper.cpp executable not found at build/whisper.cpp/main"
  echo "Build directory contents:"
  ls -la build/whisper.cpp/ 2>/dev/null || echo "build/whisper.cpp/ directory not found"
fi
echo

echo "===== AUDIO DEVICES ====="
python -c "import sounddevice as sd; [print(f'ID: {i}, Name: {d[\"name\"]}, Inputs: {d[\"max_input_channels\"]}') for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]" 2>/dev/null || echo "Failed to list audio devices"
echo

echo "===== RECORDING TEST ====="
echo "Recording 3 seconds of audio from device 1..."
if [ -d "output" ]; then
  echo "Output directory exists"
else
  echo "Creating output directory"
  mkdir -p output
fi

# Record and save a simple test WAV file using Python
python -c "
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import time

print('Starting recording in 3 seconds...')
for i in range(3, 0, -1):
    print(f'{i}...')
    time.sleep(1)
print('Recording now! Speak into your microphone...')

# Record 3 seconds
audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype='float32', device=1)
sd.wait()

print('Recording complete!')

# Save to WAV
wav.write('output/test_recording.wav', 16000, np.int16(audio * 32767))
print('Saved to output/test_recording.wav')

# Calculate audio levels
max_level = np.max(np.abs(audio))
rms_level = np.sqrt(np.mean(audio**2))
print(f'Max level: {max_level:.4f}')
print(f'RMS level: {rms_level:.4f}')
" 2>&1

echo
echo "Checking if recording was saved..."
if [ -f "output/test_recording.wav" ]; then
  echo "✅ Recording saved successfully: $(du -h output/test_recording.wav | cut -f1)"
else
  echo "❌ Recording not saved"
fi

echo
echo "===== SUMMARY ====="
echo "Check complete. See above for results."
echo
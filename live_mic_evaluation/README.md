# Live Microphone Evaluation with Speech Recognition

A Python-based system for evaluating audio input devices, measuring audio quality metrics, and transcribing speech using Whisper.cpp with the tiny.en model.

## Features

- Record audio from any connected microphone device
- Measure audio quality metrics:
  - RMS level
  - Peak level
  - Clipping percentage
  - Silent segments percentage
  - Signal-to-Noise Ratio (estimated)
- Transcribe speech using Whisper.cpp with the tiny.en model
- Generate audio visualizations (waveform, spectrogram, volume levels)
- Save recordings and results for further analysis

## Requirements

- Python 3.9+
- Dependencies: numpy, sounddevice, scipy, librosa, matplotlib, pydub, cmake
- For Whisper.cpp: Git, CMake, and a C++ compiler

## Installation

1. Clone or download this repository
2. Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### List Available Audio Input Devices

To see a list of all available audio input devices:

```bash
python main.py --list
# or
./main.py --list  # if you've set execute permissions
```

### Basic Recording and Transcription

Record audio for 5 seconds using the default microphone and transcribe it:

```bash
python main.py
```

### Advanced Options

```bash
python main.py --device 1 --duration 10 --visualize --model tiny.en
```

### All Available Options

```
--device, -d       Audio input device ID (default: system default)
--list, -l         List available audio input devices and exit
--duration, -t     Recording duration in seconds (default: 5.0)
--sample-rate, -sr Sample rate in Hz (default: 16000 for Whisper)
--channels, -c     Number of channels (default: 1)
--output, -o       Output directory for results (default: ./output)
--visualize, -v    Generate audio visualizations
--skip-transcription, -s  Skip speech transcription
--model, -m        Whisper model to use (default: tiny.en)
```

## Output

All results are stored in an output directory, organized by timestamp:

- `recorded_audio.wav`: The recorded audio file
- `evaluation_report.json`: JSON report with audio metrics and transcription
- `audio_visualization.png`: Visualizations of audio (if enabled)

## Whisper.cpp Integration

The first time you run speech recognition, the system will:

1. Clone the Whisper.cpp repository
2. Build the Whisper.cpp library
3. Download the specified model (tiny.en by default)

This setup process may take a few minutes the first time.

## Example Output

Here's an example of the JSON report generated:

```json
{
  "timestamp": "20250412_123456",
  "audio_file": "recorded_audio.wav",
  "audio_metrics": {
    "rms_level": 0.1234,
    "peak_level": 0.8765,
    "clipping_percentage": 0.0,
    "length_seconds": 5.0,
    "silent_percentage": 12.34,
    "sample_rate": 16000,
    "snr_db": 34.56
  },
  "transcription": {
    "text": "This is the transcribed speech from the audio.",
    "success": true
  }
}
```

## Notes

- For best results, ensure your microphone is working properly and the environment isn't too noisy
- The tiny.en model is optimized for English speech and is relatively small (~75MB)

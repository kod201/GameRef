#!/usr/bin/env python3
"""
Direct test of Whisper.cpp executable from command line.
This bypasses the Python integration and directly calls the Whisper.cpp executable.
"""
import os
import subprocess
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Test Whisper.cpp directly")
    parser.add_argument("--device", "-d", type=int, default=None, 
                        help="Audio device ID (default: system default)")
    parser.add_argument("--duration", "-t", type=float, default=5.0,
                        help="Recording duration in seconds (default: 5.0)")
    parser.add_argument("--sample-rate", "-sr", type=int, default=16000,
                        help="Sample rate in Hz (default: 16000)")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available input devices")
    return parser.parse_args()

def list_devices():
    """List all audio input devices."""
    print("Available audio input devices:")
    for i, device in enumerate(sd.query_devices()):
        if device['max_input_channels'] > 0:
            print(f"ID: {i}, Name: {device['name']}, Channels: {device['max_input_channels']}")

def record_audio(device_id, duration, sample_rate=16000):
    """Record audio from specified device."""
    print(f"Recording for {duration} seconds...")
    print("Speak into your microphone...")
    
    # Create array to hold audio data
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32',
        device=device_id
    )
    
    # Display a countdown
    for i in range(int(duration), 0, -1):
        print(f"\rRecording: {i} seconds left...", end='', flush=True)
        time.sleep(1)
    
    sd.wait()  # Wait until recording is finished
    print("\nRecording finished!")
    
    return audio_data.flatten()

def main():
    args = parse_args()
    
    # List devices if requested
    if args.list:
        list_devices()
        return 0
    
    # Show selected device
    if args.device is not None:
        print(f"Using audio device ID: {args.device}")
    else:
        print("Using default audio device")
    
    try:
        # Create output directory
        base_dir = Path(__file__).parent.absolute()
        output_dir = base_dir / "output"
        output_dir.mkdir(exist_ok=True)
        
        # Paths to files and executables
        whisper_executable = base_dir / "build" / "whisper.cpp" / "main"
        model_path = base_dir / "models" / "ggml-tiny.en.bin"
        
        # Check if whisper executable exists
        if not whisper_executable.exists():
            print(f"Error: Whisper executable not found at {whisper_executable}")
            print("Running ls of build directory to debug:")
            if (base_dir / "build").exists():
                print(list((base_dir / "build").glob("**/*")))
            return 1
        
        # Check if model exists
        if not model_path.exists():
            print(f"Error: Model file not found at {model_path}")
            print("Running ls of models directory to debug:")
            if (base_dir / "models").exists():
                print(list((base_dir / "models").glob("**/*")))
            return 1
        
        print(f"Using whisper executable: {whisper_executable}")
        print(f"Using model: {model_path}")
        
        # Record audio
        audio_data = record_audio(args.device, args.duration, args.sample_rate)
        
        # Save audio to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_file = output_dir / f"test_recording_{timestamp}.wav"
        
        audio_data_int16 = np.int16(audio_data * 32767)
        wav.write(wav_file, args.sample_rate, audio_data_int16)
        print(f"Audio saved to: {wav_file}")
        
        # Run Whisper.cpp directly
        print("\nRunning Whisper.cpp directly...")
        
        cmd = [
            str(whisper_executable),
            "-m", str(model_path),
            "-f", str(wav_file),
            "-l", "en",
            "-t", str(os.cpu_count() or 4)
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        print("\n--- Whisper.cpp Output ---")
        print(process.stdout)
        
        if process.stderr:
            print("--- Errors ---")
            print(process.stderr)
        
        print("\nProcess completed.")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
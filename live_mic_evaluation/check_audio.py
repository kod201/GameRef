#!/usr/bin/env python3
"""
Simple audio check utility to verify microphone functionality
"""
import sys
import time
import argparse
import numpy as np
import sounddevice as sd

def parse_args():
    parser = argparse.ArgumentParser(description="Simple audio check utility")
    parser.add_argument("--device", "-d", type=int, default=None, 
                        help="Audio device ID to check (default: system default)")
    parser.add_argument("--duration", "-t", type=float, default=5.0,
                        help="Check duration in seconds (default: 5.0)")
    parser.add_argument("--sample-rate", "-sr", type=int, default=16000,
                        help="Sample rate in Hz (default: 16000)")
    return parser.parse_args()

def audio_callback(indata, frames, time_info, status):
    """Print a simple volume meter during recording."""
    volume_norm = np.linalg.norm(indata) * 10
    print(f"\rVolume: {'#' * int(volume_norm)} {volume_norm:.2f}", end="")

def main():
    args = parse_args()
    
    # List available devices
    print("Available audio input devices:")
    for i, device in enumerate(sd.query_devices()):
        if device['max_input_channels'] > 0:
            print(f"ID: {i}, Name: {device['name']}, Channels: {device['max_input_channels']}")
    
    # Run the audio check
    device_id = args.device
    duration = args.duration
    sample_rate = args.sample_rate
    
    print(f"\nChecking microphone (ID: {device_id if device_id is not None else 'default'})")
    print(f"Sample rate: {sample_rate} Hz, Duration: {duration} s")
    print("Speak now to see if your microphone is working...")
    
    try:
        # Start recording with a callback that shows volume
        with sd.InputStream(device=device_id, channels=1, samplerate=sample_rate,
                           callback=audio_callback):
            # Wait for the specified duration
            time.sleep(duration)
            
    except KeyboardInterrupt:
        print("\nCheck stopped.")
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    print("\nAudio check completed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
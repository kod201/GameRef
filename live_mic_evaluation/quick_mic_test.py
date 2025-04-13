#!/usr/bin/env python3
"""
Quick microphone test - records and immediately plays back audio
"""
import sounddevice as sd
import numpy as np
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Quick microphone test")
    parser.add_argument("--device", "-d", type=int, default=None, 
                        help="Audio device ID (default: system default)")
    parser.add_argument("--duration", "-t", type=float, default=3.0,
                        help="Recording duration in seconds (default: 3.0)")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available audio devices")
    return parser.parse_args()

def list_devices():
    """List all audio devices."""
    print("Available audio devices:")
    print("-" * 40)
    
    input_devices = []
    output_devices = []
    
    for i, device in enumerate(sd.query_devices()):
        device_info = f"ID: {i}, Name: {device['name']}"
        
        if device['max_input_channels'] > 0:
            input_devices.append(f"{device_info}, Input channels: {device['max_input_channels']}")
            
        if device['max_output_channels'] > 0:
            output_devices.append(f"{device_info}, Output channels: {device['max_output_channels']}")
    
    print("INPUT DEVICES:")
    for device in input_devices:
        print(f"  {device}")
        
    print("\nOUTPUT DEVICES:")
    for device in output_devices:
        print(f"  {device}")

def main():
    args = parse_args()
    
    if args.list:
        list_devices()
        return
    
    # Default to system default device if none specified
    device_id = args.device
    duration = args.duration
    
    print(f"QUICK MICROPHONE TEST")
    print(f"Using device: {device_id if device_id is not None else 'default'}")
    print(f"Recording for {duration} seconds, then will play back immediately")
    print("-" * 40)
    print("Recording starting in:")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    print("GO! Speak now!")
    
    # Record audio
    recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='float32', device=device_id)
    
    # Show progress
    for i in range(int(duration)):
        time.sleep(1)
        print(f"Recording: {i+1}/{int(duration)} seconds")
    
    # Wait for recording to complete
    sd.wait()
    print("Recording complete!")
    
    # Normalize audio to avoid playback issues
    max_val = np.max(np.abs(recording))
    if max_val > 0:
        normalized = recording / max_val * 0.7  # Scale to 70% volume
    else:
        normalized = recording
        print("Warning: Very low or no audio detected")
    
    print("Playing back recording...")
    
    # Play back the recording
    sd.play(normalized, samplerate=44100, device=device_id)
    sd.wait()
    
    print("Playback complete!")
    
    # Show a simple volume meter
    volume = np.sqrt(np.mean(recording**2))
    bars = int(volume * 50)
    print("\nRecorded volume level:")
    print(f"[{'#' * bars}{' ' * (50-bars)}] {volume:.4f}")
    
    if volume < 0.01:
        print("Very low audio level detected. Check your microphone.")
    elif volume < 0.05:
        print("Low audio level. Try speaking louder or adjusting your microphone.")
    elif volume > 0.5:
        print("Good audio level!")
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()
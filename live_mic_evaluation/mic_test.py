#!/usr/bin/env python3
"""
Microphone test utility to verify audio input devices are working correctly.
"""
import argparse
import numpy as np
import sounddevice as sd
import time
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Test audio input devices")
    parser.add_argument("--device", "-d", type=int, 
                        help="Audio device ID to test (default: system default)")
    parser.add_argument("--duration", "-t", type=float, default=10.0,
                        help="Test duration in seconds (default: 10.0)")
    parser.add_argument("--sample-rate", "-sr", type=int, default=16000,
                        help="Sample rate in Hz (default: 16000)")
    return parser.parse_args()

def print_audio_levels(indata, frames, time, status):
    """Callback to print audio levels during recording."""
    volume_norm = np.linalg.norm(indata) * 10
    bars = int(volume_norm)
    
    # Print a VU meter with bars
    sys.stdout.write("\r" + "▐" + "█" * bars + " " * (40 - bars) + "▌ ")
    
    # Add numerical level indicator
    level_db = 20 * np.log10(volume_norm + 1e-6)
    sys.stdout.write(f" {level_db:.1f} dB ")
    
    # Add warning indicators
    if volume_norm > 30:
        sys.stdout.write("(LOUD)")
    elif volume_norm < 2:
        sys.stdout.write("(quiet)")
    else:
        sys.stdout.write("(good)")
    
    sys.stdout.flush()

def test_microphone(device=None, duration=10, sample_rate=16000):
    """Run a live test of the microphone with visual feedback."""
    try:
        print(f"\nTesting microphone (ID: {device if device is not None else 'default'})")
        print("Speak into your microphone to see the audio levels.")
        print("Press Ctrl+C to stop the test.\n")
        
        # Start the stream with a callback function
        with sd.InputStream(callback=print_audio_levels, 
                           device=device, 
                           channels=1, 
                           samplerate=sample_rate):
            
            # Show a countdown timer
            for i in range(int(duration), 0, -1):
                time.sleep(1)
                # Don't overwrite the level indicator
                if i % 5 == 0:
                    sys.stdout.write(f"\n{i} seconds left... ")
                    sys.stdout.flush()
            
            print("\nTest completed.")
    
    except KeyboardInterrupt:
        print("\nMicrophone test stopped by user.")
    except Exception as e:
        print(f"\nError testing microphone: {e}")
        print("Please check your microphone connections and system permissions.")
        return False
    
    return True

def main():
    args = parse_args()
    
    # List available devices
    print("Available audio input devices:")
    for i, device in enumerate(sd.query_devices()):
        if device['max_input_channels'] > 0:
            print(f"ID: {i}, Name: {device['name']}, Channels: {device['max_input_channels']}")
    
    # Test the selected microphone
    test_microphone(args.device, args.duration, args.sample_rate)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
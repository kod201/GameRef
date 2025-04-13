#!/usr/bin/env python3
"""
Simple audio recording and visualization script.
Does not require Whisper.cpp - only tests microphone input and visualization.
"""
import os
import sys
import time
import argparse
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Record and visualize audio")
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

def visualize_audio(audio_data, sample_rate):
    """Generate and display visualizations of the audio."""
    print("\nGenerating visualizations...")
    
    # Create figure with subplots
    plt.figure(figsize=(10, 8))
    
    # Plot waveform
    plt.subplot(2, 1, 1)
    plt.title('Waveform')
    time_axis = np.linspace(0, len(audio_data)/sample_rate, len(audio_data))
    plt.plot(time_axis, audio_data)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Plot frequency spectrum using FFT
    plt.subplot(2, 1, 2)
    plt.title('Frequency Spectrum')
    spectrum = np.abs(np.fft.rfft(audio_data))
    freq = np.fft.rfftfreq(len(audio_data), 1/sample_rate)
    plt.plot(freq, spectrum)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.xscale('log')
    
    # Ensure the plots don't overlap
    plt.tight_layout()
    
    # Save to output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_file = os.path.join(output_dir, f"visualization_{timestamp}.png")
    wav_file = os.path.join(output_dir, f"recording_{timestamp}.wav")
    
    plt.savefig(viz_file)
    print(f"Visualization saved to: {viz_file}")
    
    # Save the audio file
    audio_data_int16 = np.int16(audio_data * 32767)
    wav.write(wav_file, sample_rate, audio_data_int16)
    print(f"Audio saved to: {wav_file}")
    
    # Show the plot
    plt.show()

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
        # Record audio
        audio_data = record_audio(args.device, args.duration, args.sample_rate)
        
        # Visualize the recorded audio
        visualize_audio(audio_data, args.sample_rate)
        
        print("\nProcess completed successfully.")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
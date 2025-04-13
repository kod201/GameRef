#!/usr/bin/env python3
"""
Python-based Transcription Tool using SpeechRecognition library
"""
import os
import sys
import time
import argparse
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from datetime import datetime
import speech_recognition as sr

def parse_args():
    parser = argparse.ArgumentParser(description="Python Transcription Tool")
    parser.add_argument("--device", "-d", type=int, default=1, 
                        help="Audio device ID (default: 1 - MacBook Pro Microphone)")
    parser.add_argument("--duration", "-t", type=float, default=5.0,
                        help="Recording duration in seconds (default: 5.0)")
    parser.add_argument("--sample-rate", "-sr", type=int, default=16000,
                        help="Sample rate in Hz (default: 16000)")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available audio devices")
    parser.add_argument("--continuous", "-c", action="store_true",
                        help="Enable continuous recording and transcription mode")
    return parser.parse_args()

def list_devices():
    """List all audio devices."""
    print("Available audio input devices:")
    print("-" * 50)
    for i, device in enumerate(sd.query_devices()):
        if device['max_input_channels'] > 0:
            print(f"ID: {i}, Name: {device['name']}, Channels: {device['max_input_channels']}")

def record_and_transcribe(device_id, duration, sample_rate, chunk_number=None):
    """Record audio and transcribe it."""
    # Create identifier for this recording
    chunk_id = f"chunk_{chunk_number}" if chunk_number is not None else "recording"
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    print(f"\n--- {timestamp}: Recording {chunk_id} ({duration}s) ---")
    print("Speak clearly into your microphone...")
    
    # Record audio
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.int16,
        device=device_id
    )
    
    # Display a countdown
    for i in range(int(duration), 0, -1):
        print(f"\rRecording: {i} seconds left...", end="", flush=True)
        time.sleep(1)
    
    # Wait until recording is finished
    sd.wait()
    print("\nRecording finished!")
    
    # Create temporary WAV file
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    wav_file = os.path.join(output_dir, f"{chunk_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
    
    # Save audio to WAV file
    wav.write(wav_file, sample_rate, audio_data)
    print(f"Audio saved to: {wav_file}")
    
    # Perform transcription
    print("Transcribing audio...")
    
    try:
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        # Load audio file
        with sr.AudioFile(wav_file) as source:
            # Read the audio data
            audio = recognizer.record(source)
            
            # Attempt transcription with Google's API
            text = recognizer.recognize_google(audio)
            
            print("\n--- Transcription Result ---")
            print(text)
            print("-" * 30)
            
            return text, wav_file
    
    except sr.UnknownValueError:
        print("Could not understand audio (no speech detected)")
        return "", wav_file
    except sr.RequestError as e:
        print(f"Error with speech recognition service; {e}")
        return "", wav_file
    except Exception as e:
        print(f"Error during transcription: {e}")
        return "", wav_file

def continuous_mode(device_id, duration, sample_rate):
    """Run in continuous recording and transcription mode."""
    print("\n=== CONTINUOUS TRANSCRIPTION MODE ===")
    print("Press Ctrl+C to stop at any time")
    print("=" * 50)
    
    chunk_number = 1
    transcription_history = []
    
    try:
        while True:
            # Record and transcribe
            text, _ = record_and_transcribe(device_id, duration, sample_rate, chunk_number)
            
            # Add to history if there's text
            if text:
                transcription_history.append(text)
                
                # Show recent history every few chunks
                if chunk_number % 3 == 0:
                    print("\n--- Recent Conversation ---")
                    for i, t in enumerate(transcription_history[-5:]):  # Last 5 transcriptions
                        print(f"{i+1}. {t}")
                    print("-" * 30)
            
            # Increment chunk number
            chunk_number += 1
            
            # Small pause between recordings
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n\nContinuous transcription stopped by user")
        
        # Save full transcript
        if transcription_history:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
            transcript_file = os.path.join(output_dir, f"full_transcript_{timestamp}.txt")
            
            with open(transcript_file, 'w') as f:
                f.write("\n".join(transcription_history))
                
            print(f"Full transcript saved to: {transcript_file}")

def main():
    try:
        # Parse arguments
        args = parse_args()
        
        # Check if SpeechRecognition is installed
        try:
            import speech_recognition
        except ImportError:
            print("The 'SpeechRecognition' package is required but not installed.")
            print("Please install it with: pip install SpeechRecognition")
            return 1
            
        # List devices if requested
        if args.list:
            list_devices()
            return 0
            
        # Run in continuous mode if requested
        if args.continuous:
            continuous_mode(args.device, args.duration, args.sample_rate)
            return 0
            
        # Single recording and transcription
        record_and_transcribe(args.device, args.duration, args.sample_rate)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
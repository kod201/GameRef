#!/usr/bin/env python3
"""
Simple Live Transcription with Whisper.cpp

A simplified version of live transcription that records audio in chunks
and then processes them, providing immediate feedback.
"""
import os
import sys
import time
import argparse
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
from datetime import datetime
import threading
import tempfile
import queue

from whisper_integration import WhisperManager

def parse_args():
    parser = argparse.ArgumentParser(description="Simple Live Transcription")
    parser.add_argument("--device", "-d", type=int, default=None,
                        help="Audio device ID (default: system default)")
    parser.add_argument("--chunk", "-c", type=float, default=5.0,
                        help="Chunk length in seconds (default: 5.0)")
    parser.add_argument("--model", "-m", type=str, default="tiny.en",
                        help="Whisper model name (default: tiny.en)")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available input devices")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show verbose output")
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
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32',
        device=device_id
    )
    
    # Show a progress bar
    bar_length = 30
    for i in range(int(duration * 10)):
        progress = i / (duration * 10)
        block = int(round(bar_length * progress))
        text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.1f}%"
        sys.stdout.write(text)
        sys.stdout.flush()
        time.sleep(0.1)
    
    sd.wait()
    sys.stdout.write("\rRecording complete!                                        \n")
    return audio_data

def main():
    args = parse_args()
    
    # List devices if requested
    if args.list:
        list_devices()
        return 0
    
    # Set up whisper
    print("Initializing Whisper.cpp...")
    whisper = WhisperManager(model_name=args.model)
    
    if not whisper.ensure_ready():
        print("Failed to initialize Whisper. Check the model and try again.")
        return 1
    
    print("\n------ SIMPLE LIVE TRANSCRIPTION ------")
    print("Press Ctrl+C to exit")
    print("Recording and transcribing in chunks of", args.chunk, "seconds")
    print("----------------------------------------\n")
    
    # Create output directory for saving temporary files
    temp_dir = tempfile.mkdtemp()
    
    chunk_num = 1
    
    try:
        while True:
            print(f"\n--- Chunk {chunk_num} ---")
            
            # Record audio chunk
            audio_data = record_audio(args.device, args.chunk)
            
            # Save to temporary file
            temp_file = os.path.join(temp_dir, f"chunk_{chunk_num}.wav")
            wavfile.write(temp_file, 16000, np.int16(audio_data * 32767))
            
            # Process with whisper
            print("Transcribing...")
            result = whisper.transcribe_audio(temp_file)
            
            # Show results
            if result["success"]:
                if "text" in result:
                    transcript = result["text"].strip()
                elif "transcript" in result:
                    transcript = result["transcript"].strip()
                else:
                    transcript = "No text detected"
                
                print("\nTranscription:")
                print("-" * 40)
                print(transcript)
                print("-" * 40)
                
                # Show debug info if verbose
                if args.verbose and "segments" in result:
                    print("\nDetailed segments:")
                    for i, segment in enumerate(result["segments"]):
                        print(f"  [{segment.get('t0', '?')}-{segment.get('t1', '?')}] {segment.get('text', '')}")
            else:
                print("\nTranscription failed.")
                if args.verbose:
                    print(result.get("error", "Unknown error"))
            
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except:
                pass
                
            chunk_num += 1
            
    except KeyboardInterrupt:
        print("\nStopping transcription...")
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
    
    # Final cleanup
    try:
        import shutil
        shutil.rmtree(temp_dir)
    except:
        pass
        
    return 0

if __name__ == "__main__":
    sys.exit(main())